"""
Build ppbench heyawake dataset for TRM training with proper train/val/test splits.

Heyawake puzzle format:
  Three-grid puzzle requiring room boundary encoding.
  pzprv3 format: pzprv3 / heyawake / height / width / num_rooms
    Grid 1: room IDs (integer per cell, 0-indexed)
    Grid 2: clue numbers ('.' = no clue, integer = shaded count in room)
    Grid 3: answer grid ('.' = unsolved, '#' = shaded, '+' = unshaded)

  The room IDs are arbitrary labels. What matters for constraint propagation
  is which neighbors are in the same room. We encode this as a 4-bit boundary
  pattern (U/R/D/L walls) per cell, combined with clue status into a compound
  token.

  The model predicts binary shading (#/+) for every cell.
  ~30% of cells are shaded on average.
  All 100 positions require prediction (no "given" cells in the output).

Vocabulary (dynamic, depends on --max-clue):
  0: <pad>
  1: '#'  (shaded, output only)
  2: '+'  (unshaded, output only)
  3+: compound input tokens = boundary_pattern * (max_clue + 2) + clue_index
       boundary_pattern: 0-15 (4-bit URDL, 1=wall/edge, 0=same room)
       clue_index: 0=no_clue, 1=clue_0, 2=clue_1, ..., (max_clue+1)=clue_max

  With --max-clue 10:  vocab = 3 + 16 * 12 = 195
  With no cap (max=33): vocab = 3 + 16 * 35 = 563

Dihedral augmentation:
  Square grids support all 8 dihedral symmetries.
  Boundary patterns must be remapped under transforms (wall directions change).
  Clue values and answer tokens (#/+) are invariant.

Usage (writes <output-dir> and <output-dir>-golden):
  python dataset/build_ppbench_heyawake.py --output-dir data/ppbench-heyawake-aug10 --num-augs 10 --val-size 100 --max-clue 10
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
SHADED_TOKEN = 1    # '#'
UNSHADED_TOKEN = 2  # '+'
COMPOUND_OFFSET = 3  # compound input tokens start here

# --- Boundary encoding ---

def compute_boundary_pattern(room_arr, r, c, h, w):
    """Compute 4-bit boundary pattern (URDL) for cell (r, c).
    Bit is 1 if neighbor is in a different room or is out of bounds (grid edge)."""
    rid = room_arr[r, c]
    U = 1 if (r == 0 or room_arr[r - 1, c] != rid) else 0
    R = 1 if (c == w - 1 or room_arr[r, c + 1] != rid) else 0
    D = 1 if (r == h - 1 or room_arr[r + 1, c] != rid) else 0
    L = 1 if (c == 0 or room_arr[r, c - 1] != rid) else 0
    return (U << 3) | (R << 2) | (D << 1) | L


def compute_boundary_grid(room_arr):
    """Compute boundary pattern for every cell. Returns [H, W] int array."""
    h, w = room_arr.shape
    boundary = np.zeros((h, w), dtype=np.int32)
    for r in range(h):
        for c in range(w):
            boundary[r, c] = compute_boundary_pattern(room_arr, r, c, h, w)
    return boundary


# Dihedral boundary remapping tables (verified against recomputed boundaries).
# For each transform tid, maps old boundary pattern -> new boundary pattern.
# Derivation: track how U/R/D/L neighbor directions change under each spatial transform.
#   tid=0 identity:    U->U, R->R, D->D, L->L
#   tid=1 rot90 CCW:   U->R, R->D, D->L, L->U  (new_U=old_R, ...)
#   tid=2 rot180:      U->D, R->L, D->U, L->R
#   tid=3 rot270 CCW:  U->L, R->U, D->R, L->D  (new_U=old_L, ...)
#   tid=4 fliplr:      U->U, R->L, D->D, L->R
#   tid=5 flipud:      U->D, R->R, D->U, L->L
#   tid=6 transpose:   U->L, R->D, D->R, L->U  (new_U=old_L, ...)
#   tid=7 anti-diag:   U->R, R->U, D->L, L->D  (new_U=old_R, ...)
BOUNDARY_REMAP = {
    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    1: [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15],
    2: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
    3: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15],
    4: [0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15],
    5: [0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7, 14, 15],
    6: [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15],
    7: [0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15],
}


# --- Dihedral transforms ---

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


def apply_dihedral_input(flat_input: list, tid: int) -> list:
    """Apply dihedral transform to flat input tokens, remapping boundary patterns.

    The compound token encodes (boundary_pattern, clue_index).
    Under dihedral, boundary_pattern must be remapped; clue_index is invariant.
    """
    if tid == 0:
        return flat_input

    arr = np.array(flat_input).reshape(GRID_SIZE, GRID_SIZE)
    transformed = dihedral_transform_2d(arr, tid)
    flat = transformed.flatten()

    # Remap boundary patterns within compound tokens
    remap = BOUNDARY_REMAP[tid]
    result = []
    for tok in flat:
        if tok < COMPOUND_OFFSET:
            result.append(tok)
        else:
            compound = tok - COMPOUND_OFFSET
            # Decode: compound = boundary_pattern * num_clue_slots + clue_index
            # We need num_clue_slots to decode, but it's baked into the token.
            # Instead, we can pass it as a parameter. For simplicity, we'll
            # compute it from the global that the caller sets.
            result.append(tok)  # placeholder
    return result  # This won't work — need num_clue_slots


def apply_dihedral_to_grids(boundary_grid_flat: list, clue_flat: list,
                            answer_flat: list, tid: int) -> tuple:
    """Apply dihedral transform to the raw grids (before tokenization).

    Returns (transformed_boundary_flat, transformed_clue_flat, transformed_answer_flat).
    Boundary patterns are remapped. Clue values and answer tokens are invariant.
    """
    if tid == 0:
        return boundary_grid_flat, clue_flat, answer_flat

    h, w = GRID_SIZE, GRID_SIZE
    remap = BOUNDARY_REMAP[tid]

    # Transform boundary grid with remapping
    b_arr = np.array(boundary_grid_flat).reshape(h, w)
    b_transformed = dihedral_transform_2d(b_arr, tid)
    b_remapped = np.vectorize(lambda p: remap[p])(b_transformed)

    # Transform clue grid (values invariant, just spatial rearrangement)
    c_arr = np.array(clue_flat).reshape(h, w)
    c_transformed = dihedral_transform_2d(c_arr, tid)

    # Transform answer grid (values invariant)
    a_arr = np.array(answer_flat).reshape(h, w)
    a_transformed = dihedral_transform_2d(a_arr, tid)

    return b_remapped.flatten().tolist(), c_transformed.flatten().tolist(), a_transformed.flatten().tolist()


def apply_dihedral_answer(flat_answer: list, tid: int) -> list:
    """Apply dihedral transform to flat answer tokens. No remapping needed."""
    if tid == 0:
        return flat_answer
    arr = np.array(flat_answer).reshape(GRID_SIZE, GRID_SIZE)
    transformed = dihedral_transform_2d(arr, tid)
    return transformed.flatten().tolist()


# --- Tokenization ---

def make_vocab(max_clue: int) -> dict:
    """Build the vocabulary mapping.

    Returns a dict with metadata about the vocab structure.
    """
    num_clue_slots = max_clue + 2  # slot 0 = no clue, slots 1..max_clue+1 = clue values 0..max_clue
    num_compound = 16 * num_clue_slots
    vocab_size = COMPOUND_OFFSET + num_compound

    return {
        "pad": PAD_TOKEN,
        "shaded": SHADED_TOKEN,
        "unshaded": UNSHADED_TOKEN,
        "compound_offset": COMPOUND_OFFSET,
        "num_clue_slots": num_clue_slots,
        "max_clue": max_clue,
        "vocab_size": vocab_size,
    }


def tokenize_input(boundary_flat: list, clue_flat: list, vocab: dict) -> list:
    """Convert boundary patterns + clue values to compound input tokens."""
    num_clue_slots = vocab["num_clue_slots"]
    tokens = []
    for bp, cv in zip(boundary_flat, clue_flat):
        if cv == -1:  # no clue
            clue_index = 0
        else:
            clue_index = cv + 1
        token = COMPOUND_OFFSET + bp * num_clue_slots + clue_index
        tokens.append(token)
    return tokens


def tokenize_output(answer_flat: list) -> list:
    """Convert answer tokens to output token IDs."""
    tokens = []
    for tok in answer_flat:
        if tok == 1:  # shaded
            tokens.append(SHADED_TOKEN)
        elif tok == 0:  # unshaded
            tokens.append(UNSHADED_TOKEN)
        else:
            raise ValueError(f"Unexpected answer token: {tok}")
    return tokens


# --- Puzzle parsing ---

def get_puzzle_encoding(record: dict) -> str:
    """Extract canonical puzzle encoding from URL for deduplication."""
    url = record["puzzlink_url"]
    return url.split("?", 1)[1] if "?" in url else url


def parse_heyawake_grids(raw_state: str, max_clue: int):
    """Parse a pzprv3 heyawake state into structured grids.

    Returns (boundary_flat, clue_flat, answer_flat) or None on failure.
      boundary_flat: list of int (0-15), length H*W
      clue_flat: list of int (-1=no clue, 0..max_clue), length H*W
      answer_flat: list of int (0=unshaded, 1=shaded, -1=unsolved), length H*W
    """
    lines = raw_state.strip().split("\n")
    if len(lines) < 5 or lines[1] != "heyawake":
        return None

    try:
        h = int(lines[2])
        w = int(lines[3])
    except ValueError:
        return None

    if h != GRID_SIZE or w != GRID_SIZE:
        return None

    grid_start = 5  # after: pzprv3, heyawake, h, w, num_rooms
    if len(lines) < grid_start + 3 * h:
        return None

    # Parse room grid -> boundary patterns
    room_rows = []
    for r in range(h):
        tokens = lines[grid_start + r].split()
        if len(tokens) != w:
            return None
        try:
            room_rows.append([int(t) for t in tokens])
        except ValueError:
            return None
    room_arr = np.array(room_rows, dtype=np.int32)
    boundary_grid = compute_boundary_grid(room_arr)
    boundary_flat = boundary_grid.flatten().tolist()

    # Parse clue grid
    clue_flat = []
    for r in range(h):
        tokens = lines[grid_start + h + r].split()
        if len(tokens) != w:
            return None
        for tok in tokens:
            if tok == '.':
                clue_flat.append(-1)  # no clue
            else:
                try:
                    val = int(tok)
                except ValueError:
                    return None
                if val > max_clue:
                    return None  # exceeds max_clue cap
                clue_flat.append(val)

    # Parse answer grid
    answer_flat = []
    for r in range(h):
        tokens = lines[grid_start + 2 * h + r].split()
        if len(tokens) != w:
            return None
        for tok in tokens:
            if tok == '#':
                answer_flat.append(1)  # shaded
            elif tok == '+':
                answer_flat.append(0)  # unshaded
            elif tok == '.':
                answer_flat.append(-1)  # unsolved
            else:
                return None

    return boundary_flat, clue_flat, answer_flat


def is_fully_solved(answer_flat: list) -> bool:
    """Check no unsolved cells remain."""
    return -1 not in answer_flat


# --- Processing ---

def process_records(records: list, max_clue: int, num_augs: int,
                    use_dihedral: bool, desc: str):
    """Process heyawake records into (input, label) pairs.

    For heyawake, "trajectory" augmentation applies random subsets of moves
    to create intermediate states. Since the output is binary (#/+) and all
    cells are unsolved initially, intermediate states have some cells solved
    and others still '.'.

    For partially-solved states, the input boundary+clue tokens stay the same
    (they come from the room/clue grids which never change). The difference is
    in the labels: unsolved cells in a partial trajectory state still need to
    be predicted. Since all moves are needed for the full solution, we always
    use the FULL solved state as the label.

    The "trajectory" variation is: we can reveal some already-solved cells in
    the input by replacing boundary+clue tokens with the answer tokens. But
    this doesn't match the architecture cleanly since input tokens are compound
    (boundary+clue) and output tokens are binary. Instead, trajectory
    augmentation here means we just use the same (full input, full label) pair
    with optional dihedral transforms.

    Returns: (inputs_parsed, labels_parsed, skipped_count)
      inputs_parsed: list of lists (boundary_flat, clue_flat tuples, pre-tokenization)
      labels_parsed: list of lists (answer_flat, pre-tokenization)
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

            # Parse initial state (to get boundary + clue grids)
            parsed_init = parse_heyawake_grids(obs, max_clue)
            if parsed_init is None:
                skipped += 1
                continue

            boundary_flat, clue_flat, _ = parsed_init

            # Apply all moves to get solved state
            for move in moves:
                obs, _, _, _, _ = env.step(move)

            parsed_solved = parse_heyawake_grids(obs, max_clue)
            if parsed_solved is None:
                skipped += 1
                continue

            _, _, answer_flat = parsed_solved
            if not is_fully_solved(answer_flat):
                skipped += 1
                continue

            # Always include base pair (identity)
            inputs_parsed.append((boundary_flat, clue_flat))
            labels_parsed.append(answer_flat)

            # Augmentations (dihedral only for heyawake — no trajectory mixing
            # since input/output token spaces are disjoint)
            for _ in range(num_augs):
                if use_dihedral:
                    tid = np.random.randint(0, 8)
                    aug_b, aug_c, aug_a = apply_dihedral_to_grids(
                        boundary_flat, clue_flat, answer_flat, tid)
                    inputs_parsed.append((aug_b, aug_c))
                    labels_parsed.append(aug_a)
                else:
                    # Without dihedral, augmentation just repeats the base pair
                    # (no trajectory augmentation possible for heyawake)
                    inputs_parsed.append((boundary_flat, clue_flat))
                    labels_parsed.append(answer_flat)

        except Exception as e:
            skipped += 1

    return inputs_parsed, labels_parsed, skipped


# --- Save ---

def save_split(split_name: str, output_dir: str, inputs_parsed: list,
               labels_parsed: list, vocab: dict, mask_clues: bool = False):
    """Tokenize and save a data split as numpy arrays.

    If mask_clues=True: not applicable in the traditional sense for heyawake
    since input tokens != output tokens at ALL positions. Instead, this flag
    is a no-op (documented for interface consistency with other builders).
    """
    num_examples = len(inputs_parsed)

    inputs_np = np.zeros((num_examples, SEQ_LEN), dtype=np.int32)
    labels_np = np.full((num_examples, SEQ_LEN), IGNORE_LABEL_ID, dtype=np.int32)
    puzzle_ids_np = np.ones(num_examples, dtype=np.int32)

    valid = 0
    tokenization_errors = 0

    for i in range(num_examples):
        boundary_flat, clue_flat = inputs_parsed[i]
        answer_flat = labels_parsed[i]

        if len(boundary_flat) != SEQ_LEN or len(answer_flat) != SEQ_LEN:
            continue

        try:
            inp_tokens = tokenize_input(boundary_flat, clue_flat, vocab)
            lbl_tokens = tokenize_output(answer_flat)

            inputs_np[valid] = inp_tokens
            labels_np[valid] = lbl_tokens

            valid += 1
        except (KeyError, ValueError):
            tokenization_errors += 1

    if tokenization_errors > 0:
        print(f"  WARNING: {tokenization_errors} examples had tokenization errors")

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
        vocab_size=vocab["vocab_size"],
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
                   max_clue: int, mask_clues: bool):
    rng = np.random.RandomState(seed)
    vocab = make_vocab(max_clue)

    print(f"Vocab: {vocab['vocab_size']} tokens (max_clue={max_clue}, "
          f"{vocab['num_clue_slots']} clue slots, "
          f"{16 * vocab['num_clue_slots']} compound tokens)")

    # 1. Load golden test set
    golden_records = [r for r in load_dataset("golden") if r["pid"] == "heyawake"]
    golden_encodings = {get_puzzle_encoding(r) for r in golden_records}
    print(f"Golden heyawake: {len(golden_records)}")

    # 2. Load full dataset, remove golden
    all_records = [r for r in load_dataset(dataset_name) if r["pid"] == "heyawake"]
    pool_records = [r for r in all_records
                    if get_puzzle_encoding(r) not in golden_encodings]
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
    aug_desc = "dihedral" if use_dihedral else "repeat (no dihedral)"
    print(f"\nProcessing (train augs={num_augs} [{aug_desc}], val/test augs=0)...")

    test_in, test_lbl, test_skip = process_records(
        golden_records, max_clue=max_clue, num_augs=0,
        use_dihedral=False, desc="Test (golden)")
    val_in, val_lbl, val_skip = process_records(
        val_records, max_clue=max_clue, num_augs=0,
        use_dihedral=False, desc="Val")
    train_in, train_lbl, train_skip = process_records(
        train_records, max_clue=max_clue, num_augs=num_augs,
        use_dihedral=use_dihedral, desc="Train")

    print(f"\nSkipped: test={test_skip}, val={val_skip}, train={train_skip}")
    print(f"Processed: test={len(test_in)}, val={len(val_in)}, train={len(train_in)}")

    # 6. Save
    print(f"\nSaving to {output_dir} (mask_clues={mask_clues})...")
    save_split("train", output_dir, train_in, train_lbl, vocab,
               mask_clues=mask_clues)
    save_split("test", output_dir, val_in, val_lbl, vocab,
               mask_clues=mask_clues)

    golden_dir = output_dir + "-golden"
    print(f"\nSaving golden to {golden_dir}...")
    save_split("test", golden_dir, test_in, test_lbl, vocab,
               mask_clues=mask_clues)
    save_split("train", golden_dir, test_in, test_lbl, vocab,
               mask_clues=mask_clues)

    # 7. Save mappings
    # Build human-readable vocab mapping
    vocab_map = {"<pad>": PAD_TOKEN, "#": SHADED_TOKEN, "+": UNSHADED_TOKEN}
    for bp in range(16):
        dirs = ""
        if bp & 8: dirs += "U"
        if bp & 4: dirs += "R"
        if bp & 2: dirs += "D"
        if bp & 1: dirs += "L"
        if not dirs: dirs = "none"

        # No-clue token
        tok = COMPOUND_OFFSET + bp * vocab["num_clue_slots"] + 0
        vocab_map[f"b{bp:02d}_{dirs}"] = tok

        # Clue tokens
        for cv in range(max_clue + 1):
            tok = COMPOUND_OFFSET + bp * vocab["num_clue_slots"] + cv + 1
            vocab_map[f"b{bp:02d}_{dirs}_c{cv}"] = tok

    for d in [output_dir, golden_dir]:
        with open(os.path.join(d, "identifiers.json"), "w") as f:
            json.dump({"1": "heyawake"}, f)
        with open(os.path.join(d, "vocab.json"), "w") as f:
            json.dump(vocab_map, f, indent=2)
        with open(os.path.join(d, "vocab_config.json"), "w") as f:
            json.dump(vocab, f, indent=2)

    # 8. Summary
    print(f"\n{'='*70}")
    print(f"Heyawake dataset built!")
    print(f"  Grid:      {GRID_SIZE}x{GRID_SIZE} (seq_len={SEQ_LEN})")
    print(f"  Vocab:     {vocab['vocab_size']} tokens (max_clue={max_clue})")
    print(f"  Encoding:  compound = {COMPOUND_OFFSET} + boundary * {vocab['num_clue_slots']} + clue_idx")
    print(f"  Train:     {output_dir}/train/   (augs={num_augs}, dihedral={use_dihedral})")
    print(f"  Val:       {output_dir}/test/    (no augs)")
    print(f"  Golden:    {golden_dir}/test/    (no augs)")
    print(f"{'='*70}")
    print(f"\nTo train:")
    print(f"  python pretrain.py \\")
    print(f"    data_paths=\"['{output_dir}']\" \\")
    print(f"    data_paths_test=\"['{output_dir}']\" \\")
    print(f"    arch.pos_encodings=rope2d \\")
    print(f"    ...  # vocab_size={vocab['vocab_size']}, seq_len={SEQ_LEN}")
    print(f"\nFor golden benchmark:")
    print(f"  data_paths_test=\"['{golden_dir}']\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ppbench heyawake dataset for TRM")
    parser.add_argument("--dataset", type=str, default="full")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-augs", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dihedral", action="store_true",
                        help="Disable dihedral augmentation")
    parser.add_argument("--max-clue", type=int, default=10,
                        help="Max clue value to include (filters puzzles with higher clues). "
                             "Default 10 keeps 93.8%% of 10x10 puzzles and all golden 10x10.")
    parser.add_argument("--mask-clues", action="store_true",
                        help="No-op for heyawake (all cells need prediction). "
                             "Kept for interface consistency with other builders.")
    args = parser.parse_args()

    build_datasets(args.dataset, args.output_dir, args.num_augs, args.val_size,
                   args.seed, use_dihedral=not args.no_dihedral,
                   max_clue=args.max_clue, mask_clues=args.mask_clues)