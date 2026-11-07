"""
Build ppbench sudoku dataset for TRM training with proper train/val/test splits.

Splits:
  - test:  Golden-300 sudoku puzzles (15 puzzles, no augmentation) — final benchmark
  - val:   Held-out puzzles from full dataset (no augmentation) — smooth training monitoring
  - train: Remaining puzzles from full dataset (with augmentation) — training data

Data leakage prevention:
  - Golden puzzles are identified by their puzzle encoding (URL suffix)
  - Val puzzles are sampled from the non-golden pool before augmentation
  - No puzzle appears in more than one split

Usage (writes <output-dir> and <output-dir>-golden):
  python dataset/build_ppbench_sudoku.py --output-dir data/ppbench-sudoku-aug10 --num-augs 10 --val-size 100
"""

import os
import json
import argparse
import numpy as np
import random
from tqdm import tqdm
from ppbench import load_dataset
from ppbench.gym_env import PuzzleEnv
from common import PuzzleDatasetMetadata

IGNORE_LABEL_ID = -100


def get_puzzle_encoding(record: dict) -> str:
    """Extract the canonical puzzle encoding from the URL for deduplication.
    
    The encoding is the part after '?' in puzzlink_url, e.g. 'sudoku/9/9/h836m8m9j...'
    This is invariant to http vs https and domain differences.
    """
    url = record["puzzlink_url"]
    return url.split("?", 1)[1] if "?" in url else url


def get_sudoku_dimensions(raw_state: str) -> int:
    """Extract grid size N from pzprv3 header."""
    lines = raw_state.strip().split("\n")
    if len(lines) >= 3 and lines[0] == "pzprv3" and lines[1] == "sudoku":
        return int(lines[2])
    return -1


def extract_9x9_grid(raw_state: str) -> list:
    """Parse a 9x9 pzprv3 sudoku state into a flat 81-element list.
    
    The pzprv3 format for sudoku has 18 grid lines:
      - Lines 0-8: clue grid (given digits as numbers, blanks as '.')
      - Lines 9-17: answer grid (filled digits as numbers, given cells as '.')
    
    We merge them: if the clue cell has a digit, use it; otherwise use the answer cell.
    """
    lines = raw_state.strip().split("\n")
    grid_lines = [line.strip() for line in lines if " " in line.strip()]

    if len(grid_lines) != 18:
        raise ValueError(f"Expected 18 grid lines for 9x9 sudoku, got {len(grid_lines)}")

    top_half = [line.split(" ") for line in grid_lines[:9]]
    bottom_half = [line.split(" ") for line in grid_lines[9:]]

    valid_digits = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
    merged = []
    for r in range(9):
        for c in range(9):
            clue = top_half[r][c]
            answer = bottom_half[r][c]
            if clue in valid_digits:
                merged.append(clue)
            elif answer in valid_digits:
                merged.append(answer)
            else:
                merged.append(".")
    return merged


def process_records(records: list, num_augs: int, desc: str):
    """Process puzzle records into (input, label) pairs with optional trajectory augmentation.
    
    For each puzzle:
      1. Replay all solution moves to build a trajectory of board states
      2. Always include the base state (empty puzzle -> solved) as one example
      3. If num_augs > 0, sample random intermediate states as additional inputs
         (all mapping to the same solved output)
    
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

            if get_sudoku_dimensions(obs) != 9:
                skipped += 1
                continue

            # Build trajectory: [initial_state, after_move_1, ..., solved_state]
            trajectory = [extract_9x9_grid(obs)]
            for move in moves:
                obs, _, _, _, _ = env.step(move)
                trajectory.append(extract_9x9_grid(obs))

            solved = trajectory[-1]

            # Always include base: initial -> solved
            inputs_parsed.append(trajectory[0])
            labels_parsed.append(solved)

            # Trajectory augmentation: random intermediate states as input
            for _ in range(num_augs):
                start_idx = np.random.randint(0, max(1, len(trajectory) - 1))
                inp_state = trajectory[start_idx]

                digits = ['1','2','3','4','5','6','7','8','9']
                shuffled = digits.copy()
                random.shuffle(shuffled)
                mapping = {d: s for d, s in zip(digits, shuffled)}

                aug_in = [mapping.get(c, c) for c in inp_state]
                aug_out = [mapping.get(c, c) for c in solved]

                inputs_parsed.append(aug_in)
                labels_parsed.append(aug_out)

        except Exception as e:
            skipped += 1

    return inputs_parsed, labels_parsed, skipped


# --- Vocabulary ---
VOCAB = {
    "<pad>": 0,
    ".": 1,
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
    "6": 7, "7": 8, "8": 9, "9": 10,
}
SEQ_LEN = 81


def save_split(split_name: str, output_dir: str, inputs_parsed: list, labels_parsed: list):
    """Tokenize and save a data split as numpy arrays."""
    num_examples = len(inputs_parsed)

    inputs_np = np.zeros((num_examples, SEQ_LEN), dtype=np.int32)
    labels_np = np.full((num_examples, SEQ_LEN), IGNORE_LABEL_ID, dtype=np.int32)
    puzzle_ids_np = np.ones(num_examples, dtype=np.int32)

    valid = 0
    for i in range(num_examples):
        if len(inputs_parsed[i]) != SEQ_LEN or len(labels_parsed[i]) != SEQ_LEN:
            continue
        inputs_np[valid] = [VOCAB[c] for c in inputs_parsed[i]]
        labels_np[valid] = [VOCAB[c] for c in labels_parsed[i]]
        valid += 1

    # Truncate to valid examples
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


def build_datasets(dataset_name: str, output_dir: str, num_augs: int, val_size: int, seed: int):
    rng = np.random.RandomState(seed)

    # --- 1. Load golden test set (benchmark, never touched) ---
    golden_records = [r for r in load_dataset("golden") if r["pid"] == "sudoku"]
    golden_encodings = {get_puzzle_encoding(r) for r in golden_records}
    print(f"Golden (test) sudoku puzzles: {len(golden_records)}")

    # --- 2. Load target dataset, filter to sudoku, remove golden puzzles ---
    all_records = [r for r in load_dataset(dataset_name) if r["pid"] == "sudoku"]
    pool_records = [r for r in all_records if get_puzzle_encoding(r) not in golden_encodings]
    print(f"Full dataset sudoku: {len(all_records)}")
    print(f"After removing golden: {len(pool_records)}")

    # --- 3. Leakage verification ---
    pool_encodings = {get_puzzle_encoding(r) for r in pool_records}
    leaked = golden_encodings & pool_encodings
    assert len(leaked) == 0, f"DATA LEAKAGE: {len(leaked)} golden puzzles found in training pool!"

    # --- 4. Split pool into train and val ---
    indices = rng.permutation(len(pool_records))
    actual_val_size = min(val_size, len(pool_records) // 5)  # cap at 20% of pool
    val_indices = indices[:actual_val_size]
    train_indices = indices[actual_val_size:]

    val_records = [pool_records[i] for i in val_indices]
    train_records = [pool_records[i] for i in train_indices]
    print(f"Train puzzles: {len(train_records)}")
    print(f"Val puzzles:   {len(val_records)}")

    # Verify no overlap between val and train
    val_encodings = {get_puzzle_encoding(r) for r in val_records}
    train_encodings = {get_puzzle_encoding(r) for r in train_records}
    assert len(val_encodings & train_encodings) == 0, "Leakage between train and val!"
    assert len(val_encodings & golden_encodings) == 0, "Leakage between val and golden!"

    # --- 5. Process each split ---
    print(f"\nProcessing splits (train augs={num_augs}, val/test augs=0)...")
    test_in, test_lbl, test_skip = process_records(golden_records, num_augs=0, desc="Test (golden)")
    val_in, val_lbl, val_skip = process_records(val_records, num_augs=0, desc="Val")
    train_in, train_lbl, train_skip = process_records(train_records, num_augs=num_augs, desc="Train")

    print(f"\nSkipped non-9x9: test={test_skip}, val={val_skip}, train={train_skip}")

    # --- 6. Save ---
    # The TRM training code uses "train" and "test" splits.
    # We save val as "test" (what the training loop evaluates on),
    # and save golden as a separate "golden" split for final benchmark.
    #
    # Directory structure:
    #   output_dir/
    #     train/           <- training data (augmented)
    #     test/            <- val split (un-augmented, used by training loop)
    #     golden/          <- golden-300 test (un-augmented, for final benchmark)
    #
    # To train: data_paths=['output_dir'] data_paths_test=['output_dir']
    #   -> training loop loads train/ for training, test/ for eval
    #
    # For final benchmark evaluation: data_paths_test=['output_dir_golden']
    #   where output_dir_golden points to a dir with the golden split as "test"

    print(f"\nSaving to {output_dir}...")
    save_split("train", output_dir, train_in, train_lbl)
    save_split("test", output_dir, val_in, val_lbl)

    # Also save golden as a standalone dataset for final evaluation
    golden_dir = output_dir + "-golden"
    print(f"\nSaving golden test set to {golden_dir}...")
    save_split("test", golden_dir, test_in, test_lbl)
    # Golden needs a train split too (even if unused) for metadata consistency
    save_split("train", golden_dir, test_in, test_lbl)

    # --- 7. Save mappings ---
    for d in [output_dir, golden_dir]:
        with open(os.path.join(d, "identifiers.json"), "w") as f:
            json.dump({"1": "sudoku"}, f)
        with open(os.path.join(d, "vocab.json"), "w") as f:
            json.dump(VOCAB, f)

    # --- 8. Summary ---
    print(f"\n{'='*60}")
    print(f"Dataset built successfully!")
    print(f"  Train:  {output_dir}/train/   ({len(train_records)} puzzles, {num_augs} augs each)")
    print(f"  Val:    {output_dir}/test/    ({len(val_records)} puzzles, no augs)")
    print(f"  Golden: {golden_dir}/test/    ({len(golden_records)} puzzles, no augs)")
    print(f"{'='*60}")
    print(f"\nTo train:")
    print(f"  data_paths=\"['{output_dir}']\"")
    print(f"  data_paths_test=\"['{output_dir}']\"")
    print(f"\nFor final golden benchmark:")
    print(f"  data_paths_test=\"['{golden_dir}']\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ppbench sudoku dataset for TRM")
    parser.add_argument("--dataset", type=str, default="full",
                        help="Source dataset: 'full' (62k, needs HF download) or 'golden' (300, bundled)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for the dataset")
    parser.add_argument("--num-augs", type=int, default=100,
                        help="Number of trajectory augmentations per training puzzle")
    parser.add_argument("--val-size", type=int, default=100,
                        help="Number of puzzles to hold out for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for val/train split")
    args = parser.parse_args()

    build_datasets(args.dataset, args.output_dir, args.num_augs, args.val_size, args.seed)