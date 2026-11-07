"""
Build ppbench nurikabe dataset for TRM training with proper train/val/test splits.

Nurikabe puzzle format (10x10):
  - Input:  10x10 grid with number clues (island sizes) and '.' (unknown cells)
  - Output: 10x10 grid with number clues (preserved), '#' (shaded/sea), '+' (unshaded/island)
  - The model predicts the full solved grid from the input grid

Filters applied:
  - pid == "nurikabe"
  - width == 10 and height == 10 (square grid, 2D RoPE compatible)
  - No '-' tokens (filters out nurikabe variants with special cells)

Augmentations (all puzzle-agnostic):
  - Trajectory: random intermediate solve states as input (same solved output)
  - Dihedral: 8 symmetry transforms (rotations + reflections) applied to both input and output

Splits:
  - test:  Golden-300 nurikabe puzzles (no augmentation) — final benchmark
  - val:   Held-out puzzles from full dataset (no augmentation) — training monitoring
  - train: Remaining puzzles (with augmentation) — training data

Usage (writes <output-dir> and <output-dir>-golden):
  python dataset/build_ppbench_nurikabe.py --output-dir data/ppbench-nurikabe-aug10 --num-augs 10 --val-size 100 --max-clue 30
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from ppbench import load_dataset
from ppbench.gym_env import PuzzleEnv
from common import PuzzleDatasetMetadata

IGNORE_LABEL_ID = -100
GRID_SIZE = 10
SEQ_LEN = GRID_SIZE * GRID_SIZE  # 100

# --- Dihedral transforms (8 symmetries of a square) ---

def dihedral_transform_2d(grid_2d: np.ndarray, tid: int) -> np.ndarray:
    """Apply one of the 8 dihedral symmetries to a 2D array."""
    if tid == 0:
        return grid_2d                              # identity
    elif tid == 1:
        return np.rot90(grid_2d, k=1)              # 90° CCW
    elif tid == 2:
        return np.rot90(grid_2d, k=2)              # 180°
    elif tid == 3:
        return np.rot90(grid_2d, k=3)              # 270° CCW
    elif tid == 4:
        return np.fliplr(grid_2d)                   # horizontal flip
    elif tid == 5:
        return np.flipud(grid_2d)                   # vertical flip
    elif tid == 6:
        return grid_2d.T                            # transpose (main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(grid_2d, k=1))   # anti-diagonal reflection
    return grid_2d


def apply_dihedral(flat_grid: list, tid: int) -> list:
    """Apply dihedral transform to a flat grid (list of tokens), return flat list."""
    arr = np.array(flat_grid).reshape(GRID_SIZE, GRID_SIZE)
    transformed = dihedral_transform_2d(arr, tid)
    return transformed.flatten().tolist()


# --- Puzzle parsing ---

def get_puzzle_encoding(record: dict) -> str:
    """Extract canonical puzzle encoding from URL for deduplication."""
    url = record["puzzlink_url"]
    return url.split("?", 1)[1] if "?" in url else url


def parse_nurikabe_grid(raw_state: str) -> list:
    """Parse a pzprv3 nurikabe state into a flat list of tokens.

    pzprv3 nurikabe format:
        pzprv3
        nurikabe
        <height>
        <width>
        <height rows of width space-separated tokens>

    Returns flat list of length height*width, or None on failure.
    """
    lines = raw_state.strip().split("\n")
    if len(lines) < 4:
        return None

    if lines[0] != "pzprv3" or lines[1] != "nurikabe":
        return None

    try:
        height = int(lines[2])
        width = int(lines[3])
    except ValueError:
        return None

    if height != GRID_SIZE or width != GRID_SIZE:
        return None

    grid_lines = [l.strip() for l in lines[4:] if l.strip()]
    if len(grid_lines) != height:
        return None

    flat = []
    for gl in grid_lines:
        tokens = gl.split(" ")
        if len(tokens) != width:
            return None
        flat.extend(tokens)

    return flat


def is_clean_nurikabe(flat_grid: list, max_clue: int = None) -> bool:
    """Check that a parsed nurikabe grid contains only valid tokens.

    Rejects grids with:
      - '-' tokens (nurikabe variants with special cells)
      - Non-integer tokens
      - Clue values above max_clue (if specified)
    """
    valid_tokens = {".", "#", "+"}
    for tok in flat_grid:
        if tok in valid_tokens:
            continue
        # Must be a positive integer (clue number)
        try:
            val = int(tok)
            if val < 1:
                return False
            if max_clue is not None and val > max_clue:
                return False
        except ValueError:
            return False
    return True


def is_fully_solved(flat_grid: list) -> bool:
    """Check that a solved grid has no remaining '.' tokens."""
    return "." not in flat_grid


# --- Vocabulary ---

def build_vocab(max_clue: int) -> dict:
    """Build vocabulary: <pad>, '.', '#', '+', '1', '2', ..., '<max_clue>'.

    Token IDs:
      0: <pad>
      1: '.'  (unknown cell in input)
      2: '#'  (shaded/sea in output)
      3: '+'  (unshaded/island in output)
      4: '1', 5: '2', ..., (max_clue+3): '<max_clue>'
    """
    vocab = {"<pad>": 0, ".": 1, "#": 2, "+": 3}
    for i in range(1, max_clue + 1):
        vocab[str(i)] = i + 3
    return vocab


# --- Processing ---

def process_records(records: list, num_augs: int, use_dihedral: bool, max_clue_cap: int, desc: str):
    """Process nurikabe records into (input, label) pairs.

    Augmentation strategy:
      - Always include the base (initial -> solved) pair
      - For each of num_augs augmentations:
        1. Sample a random trajectory state as input
        2. If use_dihedral: apply a random dihedral transform to both input and output

    Args:
        max_clue_cap: Maximum allowed clue value. Puzzles with larger clues are skipped.

    Returns: (inputs_parsed, labels_parsed, max_clue_seen, skipped_count)
    """
    inputs_parsed = []
    labels_parsed = []
    max_clue = 0
    skipped = 0

    for record in tqdm(records, desc=desc):
        try:
            moves = record.get("solution", {}).get("moves_full", [])
            if not moves:
                skipped += 1
                continue

            env = PuzzleEnv(puzzle_url=record["puzzlink_url"])
            obs, _ = env.reset()

            initial_grid = parse_nurikabe_grid(obs)
            if initial_grid is None:
                skipped += 1
                continue

            if not is_clean_nurikabe(initial_grid, max_clue=max_clue_cap):
                skipped += 1
                continue

            # Build trajectory
            trajectory = [initial_grid]
            for move in moves:
                obs, _, _, _, _ = env.step(move)
                grid = parse_nurikabe_grid(obs)
                if grid is None:
                    break
                trajectory.append(grid)

            if len(trajectory) != len(moves) + 1:
                skipped += 1
                continue

            solved = trajectory[-1]

            if not is_fully_solved(solved):
                skipped += 1
                continue

            if not is_clean_nurikabe(solved, max_clue=max_clue_cap):
                skipped += 1
                continue

            # Track max clue value
            for tok in solved:
                if tok not in (".", "#", "+"):
                    max_clue = max(max_clue, int(tok))

            # Always include base pair (identity transform)
            inputs_parsed.append(trajectory[0])
            labels_parsed.append(solved)

            # Augmentations
            for _ in range(num_augs):
                # Random trajectory state
                start_idx = np.random.randint(0, max(1, len(trajectory) - 1))
                inp = trajectory[start_idx]
                out = solved

                # Random dihedral transform
                if use_dihedral:
                    tid = np.random.randint(0, 8)
                    inp = apply_dihedral(inp, tid)
                    out = apply_dihedral(out, tid)

                inputs_parsed.append(inp)
                labels_parsed.append(out)

        except Exception as e:
            skipped += 1

    return inputs_parsed, labels_parsed, max_clue, skipped


def save_split(split_name: str, output_dir: str, inputs_parsed: list, labels_parsed: list,
               vocab: dict):
    """Tokenize and save a data split as numpy arrays."""
    num_examples = len(inputs_parsed)

    inputs_np = np.zeros((num_examples, SEQ_LEN), dtype=np.int32)
    labels_np = np.full((num_examples, SEQ_LEN), IGNORE_LABEL_ID, dtype=np.int32)
    puzzle_ids_np = np.ones(num_examples, dtype=np.int32)

    valid = 0
    tokenization_errors = 0
    for i in range(num_examples):
        if len(inputs_parsed[i]) != SEQ_LEN or len(labels_parsed[i]) != SEQ_LEN:
            continue
        try:
            inputs_np[valid] = [vocab[c] for c in inputs_parsed[i]]
            labels_np[valid] = [vocab[c] for c in labels_parsed[i]]
            valid += 1
        except KeyError as e:
            tokenization_errors += 1

    if tokenization_errors > 0:
        print(f"  WARNING: {tokenization_errors} examples had tokens outside vocab")

    # Truncate to valid
    inputs_np = inputs_np[:valid]
    labels_np = labels_np[:valid]
    puzzle_ids_np = puzzle_ids_np[:valid]

    group_indices = np.arange(valid + 1, dtype=np.int32)
    puzzle_indices = np.arange(valid + 1, dtype=np.int32)

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    np.save(os.path.join(split_dir, "all__inputs.npy"), inputs_np)
    np.save(os.path.join(split_dir, "all__labels.npy"), labels_np)
    np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), puzzle_ids_np)
    np.save(os.path.join(split_dir, "all__group_indices.npy"), group_indices)
    np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), puzzle_indices)

    metadata = PuzzleDatasetMetadata(
        seq_len=SEQ_LEN,
        vocab_size=len(vocab),
        pad_id=vocab["<pad>"],
        ignore_label_id=IGNORE_LABEL_ID,
        blank_identifier_id=0,
        num_puzzle_identifiers=2,
        total_groups=valid,
        mean_puzzle_examples=1.0,
        total_puzzles=valid,
        sets=["all"],
    )

    with open(os.path.join(split_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    print(f"  {split_name}: {valid} examples saved")
    return valid


def build_datasets(dataset_name: str, output_dir: str, num_augs: int, val_size: int,
                   seed: int, use_dihedral: bool, max_clue: int):
    rng = np.random.RandomState(seed)

    # --- 1. Load golden test set ---
    golden_records = [r for r in load_dataset("golden") if r["pid"] == "nurikabe"]
    golden_encodings = {get_puzzle_encoding(r) for r in golden_records}
    print(f"Golden nurikabe puzzles: {len(golden_records)}")

    # --- 2. Load full dataset, filter, remove golden ---
    all_records = [r for r in load_dataset(dataset_name) if r["pid"] == "nurikabe"]
    pool_records = [r for r in all_records if get_puzzle_encoding(r) not in golden_encodings]
    print(f"Full dataset nurikabe: {len(all_records)}")
    print(f"After removing golden: {len(pool_records)}")

    # --- 3. Leakage verification ---
    pool_encodings = {get_puzzle_encoding(r) for r in pool_records}
    leaked = golden_encodings & pool_encodings
    assert len(leaked) == 0, f"DATA LEAKAGE: {len(leaked)} golden puzzles found in pool!"

    # --- 4. Split pool into train and val ---
    indices = rng.permutation(len(pool_records))
    actual_val_size = min(val_size, len(pool_records) // 5)
    val_indices = indices[:actual_val_size]
    train_indices = indices[actual_val_size:]

    val_records = [pool_records[i] for i in val_indices]
    train_records = [pool_records[i] for i in train_indices]
    print(f"Train pool: {len(train_records)}")
    print(f"Val pool:   {len(val_records)}")

    # Verify no overlap
    val_encodings = {get_puzzle_encoding(r) for r in val_records}
    train_encodings = {get_puzzle_encoding(r) for r in train_records}
    assert len(val_encodings & train_encodings) == 0, "Leakage between train and val!"
    assert len(val_encodings & golden_encodings) == 0, "Leakage between val and golden!"

    # --- 5. Process splits ---
    # Note: 10x10 filter, '-' filter, and max_clue filter all happen inside process_records
    aug_desc = f"trajectory"
    if use_dihedral:
        aug_desc += " + dihedral"
    print(f"\nProcessing (train augs={num_augs} [{aug_desc}], max_clue={max_clue}, val/test augs=0)...")

    test_in, test_lbl, test_max, test_skip = process_records(
        golden_records, num_augs=0, use_dihedral=False, max_clue_cap=max_clue, desc="Test (golden)")
    val_in, val_lbl, val_max, val_skip = process_records(
        val_records, num_augs=0, use_dihedral=False, max_clue_cap=max_clue, desc="Val")
    train_in, train_lbl, train_max, train_skip = process_records(
        train_records, num_augs=num_augs, use_dihedral=use_dihedral, max_clue_cap=max_clue, desc="Train")

    global_max_clue = max(test_max, val_max, train_max)
    print(f"\nFiltered/skipped: test={test_skip}, val={val_skip}, train={train_skip}")
    print(f"Max clue value in accepted puzzles: {global_max_clue} (cap={max_clue})")
    print(f"Processed: test={len(test_in)}, val={len(val_in)}, train={len(train_in)}")

    # --- 6. Build vocabulary from data ---
    vocab = build_vocab(global_max_clue)
    print(f"Vocab size: {len(vocab)} (pad + '.' + '#' + '+' + 1..{global_max_clue})")

    # --- 7. Save ---
    print(f"\nSaving to {output_dir}...")
    save_split("train", output_dir, train_in, train_lbl, vocab)
    save_split("test", output_dir, val_in, val_lbl, vocab)

    golden_dir = output_dir + "-golden"
    print(f"\nSaving golden test set to {golden_dir}...")
    save_split("test", golden_dir, test_in, test_lbl, vocab)
    save_split("train", golden_dir, test_in, test_lbl, vocab)

    # --- 8. Save mappings ---
    for d in [output_dir, golden_dir]:
        with open(os.path.join(d, "identifiers.json"), "w") as f:
            json.dump({"1": "nurikabe"}, f)
        with open(os.path.join(d, "vocab.json"), "w") as f:
            json.dump(vocab, f)

    # --- 9. Summary ---
    print(f"\n{'='*70}")
    print(f"Nurikabe dataset built successfully!")
    print(f"  Grid:   {GRID_SIZE}x{GRID_SIZE} (seq_len={SEQ_LEN})")
    print(f"  Vocab:  {len(vocab)} tokens (max clue={global_max_clue})")
    print(f"  Train:  {output_dir}/train/   (augs={num_augs}, dihedral={use_dihedral})")
    print(f"  Val:    {output_dir}/test/    (no augs)")
    print(f"  Golden: {golden_dir}/test/    (no augs)")
    print(f"{'='*70}")
    print(f"\nTo train:")
    print(f"  python pretrain.py \\")
    print(f"    data_paths=\"['{output_dir}']\" \\")
    print(f"    data_paths_test=\"['{output_dir}']\" \\")
    print(f"    arch.pos_encodings=rope2d \\")
    print(f"    ...  # seq_len={SEQ_LEN}, vocab_size={len(vocab)} set automatically from data")
    print(f"\nFor final golden benchmark:")
    print(f"  data_paths_test=\"['{golden_dir}']\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ppbench nurikabe dataset for TRM")
    parser.add_argument("--dataset", type=str, default="full",
                        help="Source dataset: 'full' or 'golden'")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--num-augs", type=int, default=100,
                        help="Augmentations per training puzzle")
    parser.add_argument("--val-size", type=int, default=100,
                        help="Puzzles held out for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splits")
    parser.add_argument("--no-dihedral", action="store_true",
                        help="Disable dihedral augmentation (trajectory only)")
    parser.add_argument("--max-clue", type=int, default=50,
                        help="Maximum clue value allowed. Puzzles with larger clues are filtered out. "
                             "Golden set max is 20; default 50 is generous. Determines vocab size.")
    args = parser.parse_args()

    build_datasets(args.dataset, args.output_dir, args.num_augs, args.val_size,
                   args.seed, use_dihedral=not args.no_dihedral, max_clue=args.max_clue)