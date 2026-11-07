"""
Build ppbench tapa dataset for TRM training with proper train/val/test splits.

Tapa puzzle format:
  Single grid (like lightup). Shade cells to form a connected wall.
  Numbers in clue cells indicate lengths of consecutive shaded blocks
  among the 8 surrounding cells. Question marks (encoded as '-' in pzprv3)
  represent unknown block lengths.

  Initial: '.' (empty), '-' (? clue), number/compound clues, clue combos with '-'
  Solved:  clue cells unchanged, '.' becomes '#' (shaded) or '+' (unshaded)

  The model predicts '#' or '+' for non-clue cells.
  Clue cells are given (unchanged between input and output).

Vocabulary (dynamic, depends on clue patterns found in data):
  0: <pad>
  1: '.'  (empty, unsolved, input only)
  2: '#'  (shaded)
  3: '+'  (unshaded)
  4+: clue pattern tokens (one per unique normalized pattern)

  Clue patterns are normalized by sorting block lengths, with '-' (wildcard)
  sorted after all numbers. E.g., "3,1" and "1,3" both become (1,3).
  "3,-" and "-,3" both become ("-",3) -> sorted as (3,"?").

Usage (writes <output-dir> and <output-dir>-golden):
  python dataset/build_ppbench_tapa.py --output-dir data/ppbench-tapa-aug10 --num-augs 10 --val-size 50
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

# --- Special tokens ---
PAD_TOKEN = 0
EMPTY_TOKEN = 1    # '.' (unsolved)
SHADED_TOKEN = 2   # '#'
UNSHADED_TOKEN = 3 # '+'
CLUE_OFFSET = 4    # clue tokens start here

# Characters that represent wildcards (question marks) in pzprv3
WILDCARD_CHAR = "-"
# Sentinel for sorting: wildcards sort after all numbers
WILDCARD_SORT_KEY = 999


# --- Clue normalization ---

def normalize_clue(raw_token: str) -> tuple:
    """Normalize a raw clue token into a canonical sorted tuple.

    Clue parts are comma-separated. Each part is either a number or '-' (wildcard).
    We sort numerically with wildcards last, so "3,-" and "-,3" both become ("?", 3)
    displayed as (3, "?") when sorted ascending.

    Returns a tuple of strings, e.g., ("1", "3") or ("3", "?") or ("?", "?").
    """
    parts = raw_token.split(",")
    normalized = []
    for p in parts:
        p = p.strip()
        if p == WILDCARD_CHAR:
            normalized.append("?")
        else:
            normalized.append(p)

    # Sort: numbers first (by value), then wildcards
    def sort_key(x):
        if x == "?":
            return WILDCARD_SORT_KEY
        return int(x)

    return tuple(sorted(normalized, key=sort_key))


def is_clue_token(tok: str) -> bool:
    """Check if a token is a tapa clue (number, wildcard, or compound)."""
    if tok in ('.', '#', '+'):
        return False
    parts = tok.split(",")
    for p in parts:
        p = p.strip()
        if p == WILDCARD_CHAR:
            continue
        try:
            int(p)
        except ValueError:
            return False
    return True


# --- Dihedral transforms ---
# Tapa clue patterns are invariant under dihedral transforms:
# - Block lengths don't change when the grid is rotated/reflected
# - '#' and '+' are invariant
# - '.' is invariant
# No token remapping needed (unlike shakashaka triangles or heyawake boundaries)

def dihedral_transform_2d(grid_2d: np.ndarray, tid: int) -> np.ndarray:
    """Apply one of 8 dihedral symmetries to a 2D array."""
    if tid == 0: return grid_2d.copy()
    elif tid == 1: return np.rot90(grid_2d, k=1)
    elif tid == 2: return np.rot90(grid_2d, k=2)
    elif tid == 3: return np.rot90(grid_2d, k=3)
    elif tid == 4: return np.fliplr(grid_2d)
    elif tid == 5: return np.flipud(grid_2d)
    elif tid == 6: return grid_2d.T
    elif tid == 7: return np.fliplr(np.rot90(grid_2d, k=1))
    return grid_2d.copy()


def apply_dihedral(flat_grid: list, tid: int) -> list:
    """Apply dihedral transform to a flat grid. No token remapping needed for tapa."""
    if tid == 0:
        return flat_grid
    arr = np.array(flat_grid).reshape(GRID_SIZE, GRID_SIZE)
    transformed = dihedral_transform_2d(arr, tid)
    return transformed.flatten().tolist()


# --- Puzzle parsing ---

def get_puzzle_encoding(record: dict) -> str:
    """Extract canonical puzzle encoding from URL for deduplication."""
    url = record["puzzlink_url"]
    return url.split("?", 1)[1] if "?" in url else url


def parse_tapa_grid(raw_state: str) -> list:
    """Parse a pzprv3 tapa state into a flat list of raw tokens.

    Tapa uses a SINGLE grid.
    Format:
      pzprv3 / tapa / height / width
      <height rows of width space-separated tokens>

    Returns flat list of length height*width, or None on failure.
    """
    lines = raw_state.strip().split("\n")
    if len(lines) < 4 or lines[1] != "tapa":
        return None

    try:
        height = int(lines[2])
        width = int(lines[3])
    except ValueError:
        return None

    if height != GRID_SIZE or width != GRID_SIZE:
        return None

    grid_lines = [l.strip() for l in lines[4:] if l.strip()]
    if len(grid_lines) < height:
        return None
    grid_lines = grid_lines[:height]

    flat = []
    for gl in grid_lines:
        tokens = gl.split(" ")
        if len(tokens) != width:
            return None
        flat.extend(tokens)

    return flat


def is_fully_solved(flat_grid: list) -> bool:
    """Check no unsolved '.' cells remain."""
    return "." not in flat_grid


# --- Vocab building ---

def build_vocab_from_records(records: list) -> dict:
    """Scan all records to discover all clue patterns and build vocabulary.

    First pass: collect all unique normalized clue patterns.
    Second pass: assign token IDs.

    Returns vocab dict mapping normalized_pattern_str -> token_id.
    """
    all_patterns = set()

    for record in tqdm(records, desc="Scanning vocab"):
        try:
            env = PuzzleEnv(puzzle_url=record["puzzlink_url"])
            obs, _ = env.reset()
            flat = parse_tapa_grid(obs)
            if flat is None:
                continue

            for tok in flat:
                if is_clue_token(tok):
                    pattern = normalize_clue(tok)
                    all_patterns.add(pattern)
        except Exception:
            continue

    # Sort patterns for deterministic ordering
    def pattern_sort_key(p):
        # Sort by length first, then lexicographically
        return (len(p), p)

    sorted_patterns = sorted(all_patterns, key=pattern_sort_key)

    # Build vocab
    vocab = {
        "<pad>": PAD_TOKEN,
        ".": EMPTY_TOKEN,
        "#": SHADED_TOKEN,
        "+": UNSHADED_TOKEN,
    }

    for i, pattern in enumerate(sorted_patterns):
        pattern_str = ",".join(pattern)
        vocab[pattern_str] = CLUE_OFFSET + i

    return vocab


def tokenize_cell(raw_token: str, vocab: dict) -> int:
    """Convert a raw cell token to its vocab ID."""
    if raw_token == ".":
        return EMPTY_TOKEN
    if raw_token == "#":
        return SHADED_TOKEN
    if raw_token == "+":
        return UNSHADED_TOKEN

    if is_clue_token(raw_token):
        pattern = normalize_clue(raw_token)
        pattern_str = ",".join(pattern)
        if pattern_str in vocab:
            return vocab[pattern_str]

    return None  # Unknown token


# --- Processing ---

def process_records(records: list, vocab: dict, num_augs: int,
                    use_dihedral: bool, desc: str):
    """Process tapa records into (input, label) pairs.

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

            initial_grid = parse_tapa_grid(obs)
            if initial_grid is None:
                skipped += 1
                continue

            # Check all tokens are in vocab
            initial_tokens = [tokenize_cell(tok, vocab) for tok in initial_grid]
            if None in initial_tokens:
                skipped += 1
                continue

            # Build trajectory
            trajectory_raw = [initial_grid]
            trajectory_tok = [initial_tokens]
            for move in moves:
                obs, _, _, _, _ = env.step(move)
                grid = parse_tapa_grid(obs)
                if grid is None:
                    break
                tokens = [tokenize_cell(tok, vocab) for tok in grid]
                if None in tokens:
                    break
                trajectory_raw.append(grid)
                trajectory_tok.append(tokens)

            if len(trajectory_tok) != len(moves) + 1:
                skipped += 1
                continue

            solved_tok = trajectory_tok[-1]
            solved_raw = trajectory_raw[-1]
            if not is_fully_solved(solved_raw):
                skipped += 1
                continue

            # Always include base pair (identity)
            inputs_parsed.append(trajectory_tok[0])
            labels_parsed.append(solved_tok)

            # Augmentations
            for _ in range(num_augs):
                start_idx = np.random.randint(0, max(1, len(trajectory_tok) - 1))
                inp = trajectory_tok[start_idx]
                out = solved_tok

                if use_dihedral:
                    tid = np.random.randint(0, 8)
                    inp = apply_dihedral(inp, tid)
                    out = apply_dihedral(out, tid)

                inputs_parsed.append(inp)
                labels_parsed.append(out)

        except Exception:
            skipped += 1

    return inputs_parsed, labels_parsed, skipped


# --- Save ---

def save_split(split_name: str, output_dir: str, inputs_parsed: list,
               labels_parsed: list, vocab_size: int, mask_clues: bool = False):
    """Tokenize and save a data split as numpy arrays.

    If mask_clues=True, cells where input == label (clue cells and walls)
    have their label set to IGNORE_LABEL_ID.
    """
    num_examples = len(inputs_parsed)

    inputs_np = np.zeros((num_examples, SEQ_LEN), dtype=np.int32)
    labels_np = np.full((num_examples, SEQ_LEN), IGNORE_LABEL_ID, dtype=np.int32)
    puzzle_ids_np = np.ones(num_examples, dtype=np.int32)

    valid = 0
    total_masked = 0
    total_cells = 0

    for i in range(num_examples):
        if len(inputs_parsed[i]) != SEQ_LEN or len(labels_parsed[i]) != SEQ_LEN:
            continue

        inp_tokens = inputs_parsed[i]
        lbl_tokens = labels_parsed[i]

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
        vocab_size=vocab_size,
        pad_id=PAD_TOKEN,
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

def build_datasets(dataset_name: str, output_dir: str, num_augs: int,
                   val_size: int, seed: int, use_dihedral: bool,
                   mask_clues: bool):
    rng = np.random.RandomState(seed)

    # 1. Load golden test set
    golden_records = [r for r in load_dataset("golden") if r["pid"] == "tapa"]
    golden_encodings = {get_puzzle_encoding(r) for r in golden_records}
    print(f"Golden tapa: {len(golden_records)}")

    # 2. Load full dataset, remove golden
    all_records = [r for r in load_dataset(dataset_name) if r["pid"] == "tapa"]
    pool_records = [r for r in all_records
                    if get_puzzle_encoding(r) not in golden_encodings]
    print(f"Full dataset: {len(all_records)}")
    print(f"After removing golden: {len(pool_records)}")

    # 3. Leakage check
    pool_encodings = {get_puzzle_encoding(r) for r in pool_records}
    leaked = golden_encodings & pool_encodings
    assert len(leaked) == 0, f"DATA LEAKAGE: {len(leaked)} golden puzzles in pool!"

    # 4. Build vocab from ALL records (so golden and pool share same vocab)
    print(f"\nBuilding vocabulary from all records...")
    all_for_vocab = all_records + [r for r in golden_records
                                    if get_puzzle_encoding(r) not in pool_encodings]
    vocab = build_vocab_from_records(all_for_vocab)
    vocab_size = len(vocab)
    print(f"Vocab: {vocab_size} tokens ({vocab_size - CLUE_OFFSET} clue patterns)")

    # 5. Split pool
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

    # 6. Process
    aug_desc = "trajectory" + (" + dihedral" if use_dihedral else "")
    print(f"\nProcessing (train augs={num_augs} [{aug_desc}], val/test augs=0)...")

    test_in, test_lbl, test_skip = process_records(
        golden_records, vocab, num_augs=0,
        use_dihedral=False, desc="Test (golden)")
    val_in, val_lbl, val_skip = process_records(
        val_records, vocab, num_augs=0,
        use_dihedral=False, desc="Val")
    train_in, train_lbl, train_skip = process_records(
        train_records, vocab, num_augs=num_augs,
        use_dihedral=use_dihedral, desc="Train")

    print(f"\nSkipped: test={test_skip}, val={val_skip}, train={train_skip}")
    print(f"Processed: test={len(test_in)}, val={len(val_in)}, train={len(train_in)}")

    # 7. Save
    print(f"\nSaving to {output_dir} (mask_clues={mask_clues})...")
    save_split("train", output_dir, train_in, train_lbl, vocab_size,
               mask_clues=mask_clues)
    save_split("test", output_dir, val_in, val_lbl, vocab_size,
               mask_clues=mask_clues)

    golden_dir = output_dir + "-golden"
    print(f"\nSaving golden to {golden_dir}...")
    save_split("test", golden_dir, test_in, test_lbl, vocab_size,
               mask_clues=mask_clues)
    save_split("train", golden_dir, test_in, test_lbl, vocab_size,
               mask_clues=mask_clues)

    # 8. Save mappings
    for d in [output_dir, golden_dir]:
        with open(os.path.join(d, "identifiers.json"), "w") as f:
            json.dump({"1": "tapa"}, f)
        with open(os.path.join(d, "vocab.json"), "w") as f:
            json.dump(vocab, f, indent=2)

    # 9. Summary
    print(f"\n{'='*70}")
    print(f"Tapa dataset built!")
    print(f"  Grid:   {GRID_SIZE}x{GRID_SIZE} (seq_len={SEQ_LEN})")
    print(f"  Vocab:  {vocab_size} tokens ({vocab_size - CLUE_OFFSET} clue patterns)")
    print(f"  Clue patterns: {list(k for k in vocab if k not in ('<pad>', '.', '#', '+'))}")
    print(f"  Train:  {output_dir}/train/   (augs={num_augs}, dihedral={use_dihedral})")
    print(f"  Val:    {output_dir}/test/    (no augs)")
    print(f"  Golden: {golden_dir}/test/    (no augs)")
    print(f"{'='*70}")
    print(f"\nTo train:")
    print(f"  python pretrain.py \\")
    print(f"    data_paths=\"['{output_dir}']\" \\")
    print(f"    data_paths_test=\"['{output_dir}']\" \\")
    print(f"    arch.pos_encodings=rope2d \\")
    print(f"    ...  # vocab_size={vocab_size}, seq_len={SEQ_LEN}")
    print(f"\nFor golden benchmark:")
    print(f"  data_paths_test=\"['{golden_dir}']\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ppbench tapa dataset for TRM")
    parser.add_argument("--dataset", type=str, default="full")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-augs", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dihedral", action="store_true",
                        help="Disable dihedral augmentation (trajectory only)")
    parser.add_argument("--mask-clues", action="store_true",
                        help="Mask clue/wall cells in labels (focus loss on shading decisions)")
    args = parser.parse_args()

    build_datasets(args.dataset, args.output_dir, args.num_augs, args.val_size,
                   args.seed, use_dihedral=not args.no_dihedral,
                   mask_clues=args.mask_clues)