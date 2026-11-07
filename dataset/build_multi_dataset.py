"""
Merge multiple single-puzzle TRM datasets into a unified multi-puzzle dataset.

This script:
  1. Loads individual puzzle datasets (already built by their respective builders)
  2. Remaps token IDs into a unified vocabulary (puzzle-specific ranges)
  3. Pads shorter sequences (e.g., sudoku 81 -> 100) with pad tokens
  4. Assigns distinct puzzle identifiers per puzzle type
  5. Outputs a single dataset in the standard TRM numpy format

The unified vocabulary assigns non-overlapping token ranges to each puzzle type:
  0: <pad> (shared)
  Then each puzzle's original tokens (excluding pad) are offset into their own range.

Usage:
  python dataset/build_multi_dataset.py --output-dir data/ppbench-multi-aug10 \
    --puzzle-dirs sudoku=data/ppbench-sudoku-aug10 lightup=data/ppbench-lightup-aug10 ... \
    --golden-dirs sudoku=data/ppbench-sudoku-aug10-golden lightup=data/ppbench-lightup-aug10-golden ...
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple

IGNORE_LABEL_ID = -100


def load_split(dataset_path: str, split: str) -> dict:
    """Load a single split's numpy arrays and metadata."""
    split_dir = os.path.join(dataset_path, split)
    metadata = json.load(open(os.path.join(split_dir, "dataset.json")))

    data = {
        "inputs": np.load(os.path.join(split_dir, "all__inputs.npy")),
        "labels": np.load(os.path.join(split_dir, "all__labels.npy")),
        "puzzle_identifiers": np.load(os.path.join(split_dir, "all__puzzle_identifiers.npy")),
    }

    return data, metadata


def build_unified_vocab(puzzle_configs: List[dict]) -> Tuple[dict, dict, int]:
    """Build unified vocabulary from multiple puzzle configs.

    Returns:
        vocab_info: dict mapping puzzle_name -> {start, count, remap_table}
        unified_vocab: dict mapping token_name -> unified_id (for saving)
        total_vocab_size: int
    """
    vocab_info = {}
    unified_vocab = {"<pad>": 0}
    offset = 1  # 0 is pad

    for config in puzzle_configs:
        name = config["name"]
        orig_vocab_size = config["vocab_size"]
        orig_vocab = config.get("vocab", {})  # token_name -> orig_id

        # Number of non-pad tokens
        n_tokens = orig_vocab_size - 1

        # Build remap table: orig_id -> unified_id
        # orig_id 0 (pad) -> 0 (unified pad)
        # orig_id 1..N -> offset..offset+N-1
        remap = np.zeros(orig_vocab_size, dtype=np.int32)
        remap[0] = 0  # pad -> pad
        for orig_id in range(1, orig_vocab_size):
            remap[orig_id] = offset + (orig_id - 1)

        # Add to unified vocab for human-readable mapping
        if orig_vocab:
            for tok_name, orig_id in orig_vocab.items():
                if tok_name == "<pad>":
                    continue
                unified_id = int(remap[orig_id])
                unified_vocab[f"{name}/{tok_name}"] = unified_id

        vocab_info[name] = {
            "start": offset,
            "count": n_tokens,
            "end": offset + n_tokens - 1,
            "remap": remap,
            "orig_vocab_size": orig_vocab_size,
        }

        offset += n_tokens

    return vocab_info, unified_vocab, offset


def remap_array(arr: np.ndarray, remap: np.ndarray, ignore_value: int = IGNORE_LABEL_ID) -> np.ndarray:
    """Remap token IDs in an array using the remap table.
    Preserves IGNORE_LABEL_ID values."""
    result = arr.copy()
    mask = arr != ignore_value
    result[mask] = remap[arr[mask]]
    return result


def pad_sequences(arr: np.ndarray, target_len: int, pad_value: int, source_grid_size: int = None) -> np.ndarray:
    """Pad sequences from current length to target_len.
    
    If source_grid_size is provided, performs spatial 2D padding:
    embeds the source_grid_size x source_grid_size grid into a 
    target_grid_size x target_grid_size grid (where target_grid_size^2 = target_len),
    padding the right columns and bottom rows. This preserves 2D spatial structure
    for positional encodings like rope2d.
    
    If source_grid_size is None, falls back to sequential padding (appends at end).
    """
    current_len = arr.shape[1]
    if current_len >= target_len:
        return arr[:, :target_len]

    target_grid_size = int(np.sqrt(target_len))
    
    if source_grid_size is not None and source_grid_size * source_grid_size == current_len and target_grid_size * target_grid_size == target_len:
        # Spatial 2D padding: embed smaller grid in larger grid
        n = arr.shape[0]
        result = np.full((n, target_len), pad_value, dtype=arr.dtype)
        
        # Reshape to 2D, pad, flatten back
        source_2d = arr.reshape(n, source_grid_size, source_grid_size)
        for i in range(source_grid_size):
            for j in range(source_grid_size):
                result[:, i * target_grid_size + j] = source_2d[:, i, j]
        
        return result
    else:
        # Sequential padding (append at end)
        pad_width = target_len - current_len
        return np.pad(arr, ((0, 0), (0, pad_width)), constant_values=pad_value)


def merge_split(puzzle_datasets: Dict[str, Tuple[dict, dict]],
                vocab_info: dict,
                puzzle_id_map: dict,
                unified_seq_len: int) -> Tuple[dict, int]:
    """Merge multiple puzzle datasets for a single split.

    Returns merged data dict and total example count.
    """
    all_inputs = []
    all_labels = []
    all_puzzle_ids = []

    for name, (data, metadata) in puzzle_datasets.items():
        n_examples = len(data["inputs"])
        if n_examples == 0:
            continue

        info = vocab_info[name]
        puzzle_id = puzzle_id_map[name]

        # Remap token IDs
        inputs_remapped = remap_array(data["inputs"], info["remap"])
        labels_remapped = remap_array(data["labels"], info["remap"])

        # Pad sequences to unified length (spatial 2D padding for grid puzzles)
        source_seq_len = metadata['seq_len']
        source_grid_size = int(np.sqrt(source_seq_len))
        if source_grid_size * source_grid_size == source_seq_len:
            grid_size = source_grid_size
        else:
            grid_size = None  # not a square grid, use sequential padding
        
        inputs_padded = pad_sequences(inputs_remapped, unified_seq_len, pad_value=0, source_grid_size=grid_size)
        labels_padded = pad_sequences(labels_remapped, unified_seq_len, pad_value=IGNORE_LABEL_ID, source_grid_size=grid_size)

        # Assign puzzle identifiers
        puzzle_ids = np.full(n_examples, puzzle_id, dtype=np.int32)

        all_inputs.append(inputs_padded)
        all_labels.append(labels_padded)
        all_puzzle_ids.append(puzzle_ids)

        print(f"    {name}: {n_examples} examples, "
              f"seq_len {metadata['seq_len']} -> {unified_seq_len}, "
              f"tokens remapped to {info['start']}-{info['end']}, "
              f"puzzle_id={puzzle_id}")

    if not all_inputs:
        return None, 0

    merged = {
        "inputs": np.concatenate(all_inputs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "puzzle_identifiers": np.concatenate(all_puzzle_ids, axis=0),
    }

    return merged, len(merged["inputs"])


def save_merged_split(split_name: str, output_dir: str, merged_data: dict,
                      total_vocab_size: int, unified_seq_len: int,
                      num_puzzle_identifiers: int):
    """Save a merged split to disk."""
    n = len(merged_data["inputs"])

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    np.save(os.path.join(split_dir, "all__inputs.npy"), merged_data["inputs"].astype(np.int32))
    np.save(os.path.join(split_dir, "all__labels.npy"), merged_data["labels"].astype(np.int32))
    np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), merged_data["puzzle_identifiers"].astype(np.int32))
    np.save(os.path.join(split_dir, "all__group_indices.npy"), np.arange(n + 1, dtype=np.int32))
    np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), np.arange(n + 1, dtype=np.int32))

    metadata = {
        "seq_len": unified_seq_len,
        "vocab_size": total_vocab_size,
        "pad_id": 0,
        "ignore_label_id": IGNORE_LABEL_ID,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": num_puzzle_identifiers,
        "total_groups": n,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": n,
        "sets": ["all"],
    }

    with open(os.path.join(split_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  {split_name}: {n} examples saved")


def save_per_puzzle_eval_sets(puzzle_datasets: Dict[str, Tuple[dict, dict]],
                              vocab_info: dict, puzzle_id_map: dict,
                              unified_seq_len: int, total_vocab_size: int,
                              num_puzzle_identifiers: int,
                              base_output_dir: str, label: str):
    """Save individual per-puzzle eval sets in unified vocab format.

    Creates: base_output_dir-{puzzle_name}/test/ and base_output_dir-{puzzle_name}/train/
    Each is a standalone dataset compatible with data_paths_test.
    """
    created_dirs = []

    for name, (data, metadata) in puzzle_datasets.items():
        n = len(data["inputs"])
        if n == 0:
            continue

        info = vocab_info[name]
        puzzle_id = puzzle_id_map[name]

        # Remap and pad
        inputs_remapped = remap_array(data["inputs"], info["remap"])
        labels_remapped = remap_array(data["labels"], info["remap"])
        
        source_seq_len = metadata['seq_len']
        source_grid_size = int(np.sqrt(source_seq_len))
        grid_size = source_grid_size if source_grid_size * source_grid_size == source_seq_len else None
        
        inputs_padded = pad_sequences(inputs_remapped, unified_seq_len, pad_value=0, source_grid_size=grid_size)
        labels_padded = pad_sequences(labels_remapped, unified_seq_len, pad_value=IGNORE_LABEL_ID, source_grid_size=grid_size)
        puzzle_ids = np.full(n, puzzle_id, dtype=np.int32)

        single_data = {
            "inputs": inputs_padded,
            "labels": labels_padded,
            "puzzle_identifiers": puzzle_ids,
        }

        out_dir = f"{base_output_dir}-{label}-{name}"
        os.makedirs(out_dir, exist_ok=True)

        # Save as both test/ and train/ (train needed for the dataset loader to not crash)
        for split in ["test", "train"]:
            save_merged_split(split, out_dir, single_data,
                              total_vocab_size, unified_seq_len, num_puzzle_identifiers)

        created_dirs.append((name, out_dir, n))
        print(f"    {name}: {n} examples -> {out_dir}")

    return created_dirs


def parse_dir_args(dir_args: list) -> dict:
    """Parse 'name=path' arguments into a dict."""
    result = {}
    for arg in dir_args:
        if "=" not in arg:
            raise ValueError(f"Expected 'name=path' format, got: {arg}")
        name, path = arg.split("=", 1)
        result[name] = path
    return result


def main():
    parser = argparse.ArgumentParser(description="Merge multi-puzzle TRM datasets")
    parser.add_argument("--puzzle-dirs", nargs="+", required=True,
                        help="Puzzle datasets as name=path pairs, e.g., sudoku=data/ppbench-sudoku-aug10")
    parser.add_argument("--golden-dirs", nargs="+", default=None,
                        help="Golden test datasets as name=path pairs (optional)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--unified-seq-len", type=int, default=100,
                        help="Unified sequence length (pad shorter sequences)")
    args = parser.parse_args()

    puzzle_dirs = parse_dir_args(args.puzzle_dirs)
    golden_dirs = parse_dir_args(args.golden_dirs) if args.golden_dirs else {}

    puzzle_names = list(puzzle_dirs.keys())
    print(f"Merging {len(puzzle_names)} puzzle types: {puzzle_names}")
    print(f"Unified seq_len: {args.unified_seq_len}")

    # 1. Load metadata from each puzzle to build unified vocab
    print(f"\n--- Loading metadata ---")
    puzzle_configs = []
    for name in puzzle_names:
        path = puzzle_dirs[name]
        meta = json.load(open(os.path.join(path, "train", "dataset.json")))

        # Try to load vocab.json for human-readable mapping
        vocab_file = os.path.join(path, "vocab.json")
        vocab = json.load(open(vocab_file)) if os.path.exists(vocab_file) else {}

        config = {
            "name": name,
            "vocab_size": meta["vocab_size"],
            "seq_len": meta["seq_len"],
            "vocab": vocab,
        }
        puzzle_configs.append(config)
        print(f"  {name}: vocab_size={meta['vocab_size']}, seq_len={meta['seq_len']}, "
              f"puzzles={meta['total_puzzles']}")

    # 2. Build unified vocabulary
    print(f"\n--- Building unified vocabulary ---")
    vocab_info, unified_vocab, total_vocab_size = build_unified_vocab(puzzle_configs)

    print(f"  Total unified vocab size: {total_vocab_size}")
    for name, info in vocab_info.items():
        print(f"  {name}: tokens {info['start']}-{info['end']} ({info['count']} tokens)")

    # 3. Assign puzzle identifiers (1-indexed, 0 is blank)
    puzzle_id_map = {name: i + 1 for i, name in enumerate(puzzle_names)}
    num_puzzle_identifiers = len(puzzle_names) + 1  # +1 for blank id 0

    print(f"\n  Puzzle identifiers: {puzzle_id_map}")

    # 4. Load and merge train split
    print(f"\n--- Merging train split ---")
    train_datasets = {}
    for name in puzzle_names:
        data, meta = load_split(puzzle_dirs[name], "train")
        train_datasets[name] = (data, meta)

    merged_train, n_train = merge_split(
        train_datasets, vocab_info, puzzle_id_map, args.unified_seq_len)

    # 5. Load and merge test split (val sets)
    print(f"\n--- Merging test split (val) ---")
    test_datasets = {}
    for name in puzzle_names:
        data, meta = load_split(puzzle_dirs[name], "test")
        test_datasets[name] = (data, meta)

    merged_test, n_test = merge_split(
        test_datasets, vocab_info, puzzle_id_map, args.unified_seq_len)

    # 6. Save merged dataset
    print(f"\n--- Saving ---")
    os.makedirs(args.output_dir, exist_ok=True)

    if merged_train is not None:
        save_merged_split("train", args.output_dir, merged_train,
                          total_vocab_size, args.unified_seq_len, num_puzzle_identifiers)
    if merged_test is not None:
        save_merged_split("test", args.output_dir, merged_test,
                          total_vocab_size, args.unified_seq_len, num_puzzle_identifiers)

    # 7. Save golden test set if provided
    golden_output = args.output_dir + "-golden"
    golden_per_puzzle_dirs = []
    if golden_dirs:
        print(f"\n--- Merging golden test sets ---")
        golden_datasets = {}
        for name in puzzle_names:
            if name in golden_dirs:
                data, meta = load_split(golden_dirs[name], "test")
                golden_datasets[name] = (data, meta)
            else:
                print(f"  WARNING: no golden set for {name}")

        merged_golden, n_golden = merge_split(
            golden_datasets, vocab_info, puzzle_id_map, args.unified_seq_len)

        if merged_golden is not None:
            os.makedirs(golden_output, exist_ok=True)
            save_merged_split("test", golden_output, merged_golden,
                              total_vocab_size, args.unified_seq_len, num_puzzle_identifiers)
            save_merged_split("train", golden_output, merged_golden,
                              total_vocab_size, args.unified_seq_len, num_puzzle_identifiers)

        # Save per-puzzle golden sets (unified vocab, for per-type metrics)
        print(f"\n--- Saving per-puzzle golden sets (unified vocab) ---")
        golden_per_puzzle_dirs = save_per_puzzle_eval_sets(
            golden_datasets, vocab_info, puzzle_id_map,
            args.unified_seq_len, total_vocab_size, num_puzzle_identifiers,
            args.output_dir, "golden")

    # 7b. Save per-puzzle val sets (unified vocab, for per-type metrics)
    print(f"\n--- Saving per-puzzle val sets (unified vocab) ---")
    val_per_puzzle_dirs = save_per_puzzle_eval_sets(
        test_datasets, vocab_info, puzzle_id_map,
        args.unified_seq_len, total_vocab_size, num_puzzle_identifiers,
        args.output_dir, "val")

    # 8. Save mappings
    all_output_dirs = [args.output_dir]
    if golden_dirs:
        all_output_dirs.append(golden_output)
    all_output_dirs += [d for _, d, _ in val_per_puzzle_dirs]
    all_output_dirs += [d for _, d, _ in golden_per_puzzle_dirs]

    for d in all_output_dirs:
        if os.path.exists(d):
            with open(os.path.join(d, "identifiers.json"), "w") as f:
                json.dump({str(v): k for k, v in puzzle_id_map.items()}, f, indent=2)
            with open(os.path.join(d, "vocab.json"), "w") as f:
                json.dump(unified_vocab, f, indent=2)
            with open(os.path.join(d, "vocab_info.json"), "w") as f:
                json.dump({
                    name: {"start": info["start"], "end": info["end"],
                           "count": info["count"], "orig_vocab_size": info["orig_vocab_size"]}
                    for name, info in vocab_info.items()
                }, f, indent=2)

    # 9. Summary
    print(f"\n{'='*70}")
    print(f"Multi-puzzle dataset built!")
    print(f"  Puzzle types: {len(puzzle_names)}")
    print(f"  Unified vocab: {total_vocab_size} tokens")
    print(f"  Unified seq_len: {args.unified_seq_len}")
    print(f"  Train: {n_train} examples")
    print(f"  Val: {n_test} examples (per-puzzle sets also saved)")
    if golden_dirs:
        print(f"  Golden: {n_golden} examples (per-puzzle sets also saved)")
    print(f"  Output: {args.output_dir}")

    per_type = {}
    for name in puzzle_names:
        n = len(train_datasets[name][0]["inputs"])
        per_type[name] = n
    print(f"\n  Per-type train examples: {per_type}")
    print(f"  Puzzle identifiers: {puzzle_id_map}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()