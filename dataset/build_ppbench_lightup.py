"""
Build ppbench lightup (akari) dataset for TRM training with proper train/val/test splits.

Lightup puzzle format:
  Single grid (simplest format — no two-grid merging needed).
  Initial: '-' (black cell), '0'-'4' (numbered black), '.' (empty white)
  Solved:  '-' (unchanged), '0'-'4' (unchanged), '#' (light), '+' (illuminated)

  The model predicts the full solved grid from the initial grid.
  Light placements (#) are the real decisions (~17 per puzzle).
  Illuminated cells (+) are deterministic given light positions but provide
  useful auxiliary gradient signal about light propagation.

Vocabulary (10 tokens):
  0: <pad>
  1: '.'  (empty white cell, unsolved)
  2: '-'  (black cell, no number)
  3: '0'  (black cell, 0 adjacent lights)
  4: '1'  (black cell, 1 adjacent light)
  5: '2'  (black cell, 2 adjacent lights)
  6: '3'  (black cell, 3 adjacent lights)
  7: '4'  (black cell, 4 adjacent lights)
  8: '#'  (light placed here)
  9: '+'  (illuminated by a light)

Usage (writes <output-dir> and <output-dir>-golden):
  python dataset/build_ppbench_lightup.py --output-dir data/ppbench-lightup-aug10 --num-augs 10 --val-size 100
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

# --- Vocabulary ---
VOCAB = {
    "<pad>": 0,
    ".": 1,   # empty white cell
    "-": 2,   # black cell (no number)
    "0": 3,   # black cell, 0 adjacent lights
    "1": 4,   # black cell, 1 adjacent light
    "2": 5,   # black cell, 2 adjacent lights
    "3": 6,   # black cell, 3 adjacent lights
    "4": 7,   # black cell, 4 adjacent lights
    "#": 8,   # light
    "+": 9,   # illuminated
}

# Black cell tokens (given in input, unchanged in output)
BLACK_TOKENS = {"-", "0", "1", "2", "3", "4"}


# --- Dihedral transforms ---
# All lightup tokens are invariant under dihedral transforms
# (no directional tokens like shakashaka's triangles)

def dihedral_transform_2d(grid_2d: np.ndarray, tid: int) -> np.ndarray:
    """Apply one of 8 dihedral symmetries to a 2D array."""
    if tid == 0: return grid_2d
    elif tid == 1: return np.rot90(grid_2d, k=1)
    elif tid == 2: return np.rot90(grid_2d, k=2)
    elif tid == 3: return np.rot90(grid_2d, k=3)
    elif tid == 4: return np.fliplr(grid_2d)
    elif tid == 5: return np.flipud(grid_2d)
    elif tid == 6: return grid_2d.T
    elif tid == 7: return np.fliplr(np.rot90(grid_2d, k=1))
    return grid_2d


def apply_dihedral(flat_grid: list, tid: int) -> list:
    """Apply dihedral transform to a flat grid. No token remapping needed."""
    arr = np.array(flat_grid).reshape(GRID_SIZE, GRID_SIZE)
    transformed = dihedral_transform_2d(arr, tid)
    return transformed.flatten().tolist()


# --- Puzzle parsing ---

def get_puzzle_encoding(record: dict) -> str:
    """Extract canonical puzzle encoding from URL for deduplication."""
    url = record["puzzlink_url"]
    return url.split("?", 1)[1] if "?" in url else url


def parse_lightup_grid(raw_state: str) -> list:
    """Parse a pzprv3 lightup state into a flat list of tokens.

    Lightup uses a SINGLE grid (no two-grid format).
    Format:
      pzprv3 / lightup / height / width
      <height rows of width space-separated tokens>

    Returns flat list of length height*width, or None on failure.
    """
    lines = raw_state.strip().split("\n")
    if len(lines) < 4 or lines[1] != "lightup":
        return None

    try:
        height = int(lines[2])
        width = int(lines[3])
    except ValueError:
        return None

    if height != GRID_SIZE or width != GRID_SIZE:
        return None

    grid_lines = [l.strip() for l in lines[4:] if l.strip()]

    # Take only the first `height` lines (lightup is single-grid)
    if len(grid_lines) < height:
        return None
    grid_lines = grid_lines[:height]

    flat = []
    for gl in grid_lines:
        tokens = gl.split(" ")
        if len(tokens) != width:
            return None
        for tok in tokens:
            if tok not in VOCAB:
                return None
            flat.append(tok)

    return flat


def is_fully_solved(flat_grid: list) -> bool:
    """Check no unsolved '.' cells remain."""
    return "." not in flat_grid


# --- Processing ---

def process_records(records: list, num_augs: int, use_dihedral: bool, desc: str):
    """Process lightup records into (input, label) pairs.

    Returns: (inputs_parsed, labels_parsed, skipped_count)
    """
    inputs_parsed = []
    labels_parsed = []
    skipped = 0

    for record in tqdm(records, desc=desc):
        try:
            moves = record.get("solution", {}).get("moves_full", [])
            if not moves:
                skipped += 1
                continue

            env = PuzzleEnv(puzzle_url=record["puzzlink_url"])
            obs, _ = env.reset()

            initial_grid = parse_lightup_grid(obs)
            if initial_grid is None:
                skipped += 1
                continue

            # Build trajectory
            trajectory = [initial_grid]
            for move in moves:
                obs, _, _, _, _ = env.step(move)
                grid = parse_lightup_grid(obs)
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

            # Always include base pair (identity)
            inputs_parsed.append(trajectory[0])
            labels_parsed.append(solved)

            # Augmentations
            for _ in range(num_augs):
                start_idx = np.random.randint(0, max(1, len(trajectory) - 1))
                inp = trajectory[start_idx]
                out = solved

                if use_dihedral:
                    tid = np.random.randint(0, 8)
                    inp = apply_dihedral(inp, tid)
                    out = apply_dihedral(out, tid)

                inputs_parsed.append(inp)
                labels_parsed.append(out)

        except Exception as e:
            skipped += 1

    return inputs_parsed, labels_parsed, skipped


# --- Save ---

def save_split(split_name: str, output_dir: str, inputs_parsed: list, labels_parsed: list,
               mask_clues: bool = False):
    """Tokenize and save a data split as numpy arrays.

    If mask_clues=True, black cells (where input == label, i.e., '-' and '0'-'4')
    have their label set to IGNORE_LABEL_ID.
    """
    num_examples = len(inputs_parsed)

    inputs_np = np.zeros((num_examples, SEQ_LEN), dtype=np.int32)
    labels_np = np.full((num_examples, SEQ_LEN), IGNORE_LABEL_ID, dtype=np.int32)
    puzzle_ids_np = np.ones(num_examples, dtype=np.int32)

    valid = 0
    tokenization_errors = 0
    total_masked = 0
    total_cells = 0

    for i in range(num_examples):
        if len(inputs_parsed[i]) != SEQ_LEN or len(labels_parsed[i]) != SEQ_LEN:
            continue
        try:
            inp_tokens = [VOCAB[c] for c in inputs_parsed[i]]
            lbl_tokens = [VOCAB[c] for c in labels_parsed[i]]

            inputs_np[valid] = inp_tokens

            if mask_clues:
                for j in range(SEQ_LEN):
                    if inp_tokens[j] == lbl_tokens[j]:
                        labels_np[valid, j] = IGNORE_LABEL_ID
                        total_masked += 1
                    else:
                        labels_np[valid, j] = lbl_tokens[j]
                    total_cells += 1
            else:
                labels_np[valid] = lbl_tokens
                total_cells += SEQ_LEN

            valid += 1
        except KeyError:
            tokenization_errors += 1

    if tokenization_errors > 0:
        print(f"  WARNING: {tokenization_errors} examples had tokens outside vocab")

    if mask_clues and total_cells > 0:
        pct = 100 * total_masked / total_cells
        print(f"  Clue masking: {total_masked}/{total_cells} cells masked ({pct:.1f}%)")

    # Truncate
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
        vocab_size=len(VOCAB),
        pad_id=VOCAB["<pad>"],
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


# --- Main ---

def build_datasets(dataset_name: str, output_dir: str, num_augs: int, val_size: int,
                   seed: int, use_dihedral: bool, mask_clues: bool):
    rng = np.random.RandomState(seed)

    # 1. Load golden test set
    golden_records = [r for r in load_dataset("golden") if r["pid"] == "lightup"]
    golden_encodings = {get_puzzle_encoding(r) for r in golden_records}
    print(f"Golden lightup: {len(golden_records)}")

    # 2. Load full dataset, remove golden
    all_records = [r for r in load_dataset(dataset_name) if r["pid"] == "lightup"]
    pool_records = [r for r in all_records if get_puzzle_encoding(r) not in golden_encodings]
    print(f"Full dataset: {len(all_records)}")
    print(f"After removing golden: {len(pool_records)}")

    # 3. Leakage check
    pool_encodings = {get_puzzle_encoding(r) for r in pool_records}
    leaked = golden_encodings & pool_encodings
    assert len(leaked) == 0, f"DATA LEAKAGE: {len(leaked)} golden puzzles in pool!"

    # 4. Split pool
    indices = rng.permutation(len(pool_records))
    actual_val_size = min(val_size, len(pool_records) // 5)
    val_records = [pool_records[i] for i in indices[:actual_val_size]]
    train_records = [pool_records[i] for i in indices[actual_val_size:]]
    print(f"Train pool: {len(train_records)}")
    print(f"Val pool:   {len(val_records)}")

    # Verify no overlap
    val_enc = {get_puzzle_encoding(r) for r in val_records}
    train_enc = {get_puzzle_encoding(r) for r in train_records}
    assert not (val_enc & train_enc), "Leakage: train/val overlap!"
    assert not (val_enc & golden_encodings), "Leakage: val/golden overlap!"

    # 5. Process
    aug_desc = "trajectory" + (" + dihedral" if use_dihedral else "")
    print(f"\nProcessing (train augs={num_augs} [{aug_desc}], val/test augs=0)...")

    test_in, test_lbl, test_skip = process_records(
        golden_records, num_augs=0, use_dihedral=False, desc="Test (golden)")
    val_in, val_lbl, val_skip = process_records(
        val_records, num_augs=0, use_dihedral=False, desc="Val")
    train_in, train_lbl, train_skip = process_records(
        train_records, num_augs=num_augs, use_dihedral=use_dihedral, desc="Train")

    print(f"\nSkipped: test={test_skip}, val={val_skip}, train={train_skip}")
    print(f"Processed: test={len(test_in)}, val={len(val_in)}, train={len(train_in)}")

    # 6. Save
    print(f"\nSaving to {output_dir} (mask_clues={mask_clues})...")
    save_split("train", output_dir, train_in, train_lbl, mask_clues=mask_clues)
    save_split("test", output_dir, val_in, val_lbl, mask_clues=mask_clues)

    golden_dir = output_dir + "-golden"
    print(f"\nSaving golden to {golden_dir}...")
    save_split("test", golden_dir, test_in, test_lbl, mask_clues=mask_clues)
    save_split("train", golden_dir, test_in, test_lbl, mask_clues=mask_clues)

    # 7. Save mappings
    for d in [output_dir, golden_dir]:
        with open(os.path.join(d, "identifiers.json"), "w") as f:
            json.dump({"1": "lightup"}, f)
        with open(os.path.join(d, "vocab.json"), "w") as f:
            json.dump(VOCAB, f)

    # 8. Summary
    print(f"\n{'='*70}")
    print(f"Lightup dataset built!")
    print(f"  Grid:   {GRID_SIZE}x{GRID_SIZE} (seq_len={SEQ_LEN})")
    print(f"  Vocab:  {len(VOCAB)} tokens")
    print(f"  Train:  {output_dir}/train/   (augs={num_augs}, dihedral={use_dihedral})")
    print(f"  Val:    {output_dir}/test/    (no augs)")
    print(f"  Golden: {golden_dir}/test/    (no augs)")
    print(f"{'='*70}")
    print(f"\nTo train:")
    print(f"  python pretrain.py \\")
    print(f"    data_paths=\"['{output_dir}']\" \\")
    print(f"    data_paths_test=\"['{output_dir}']\" \\")
    print(f"    arch.pos_encodings=rope2d \\")
    print(f"    ...  # vocab_size={len(VOCAB)}, seq_len={SEQ_LEN}")
    print(f"\nFor golden benchmark:")
    print(f"  data_paths_test=\"['{golden_dir}']\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ppbench lightup dataset for TRM")
    parser.add_argument("--dataset", type=str, default="full")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-augs", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dihedral", action="store_true",
                        help="Disable dihedral augmentation (trajectory only)")
    parser.add_argument("--mask-clues", action="store_true",
                        help="Mask black cells in labels (focus loss on white cells)")
    args = parser.parse_args()

    build_datasets(args.dataset, args.output_dir, args.num_augs, args.val_size,
                   args.seed, use_dihedral=not args.no_dihedral, mask_clues=args.mask_clues)