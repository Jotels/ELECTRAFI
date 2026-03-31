#!/usr/bin/env python3
import os
import csv
import json
import argparse
import random
import re

# Default directory and output
DEFAULT_ROOT_DIR = "/gnome_ecd"
DEFAULT_OUTPUT_CSV = "GNOME_TASKS.csv"


def find_chgcars(root_dir):
    """
    Recursively find all CHGCAR-like files:
    - CHGCAR
    - *.CHGCAR
    - *.CHGCAR.lz4
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            upper = fn.upper()
            if (
                upper == "CHGCAR" or
                upper.endswith(".CHGCAR") or
                upper.endswith(".CHGCAR.LZ4") or
                upper.endswith(".CHGCAR.gz") or
                upper.endswith(".chgcar.LZ4") or
                upper.endswith(".chgcar")
            ):
                yield os.path.join(dirpath, fn)


def extract_id_from_path(path):
    """
    Extract an integer ID from a CHGCAR path.
    Assumes filenames like '1810.CHGCAR' or '1810.CHGCAR.lz4'.

    Returns:
        int or None
    """
    base = os.path.basename(path)
    # Grab leading integer before first dot, if present
    m = re.match(r"^(\d+)\.", base)
    if not m:
        return None
    return int(m.group(1))


def filter_by_json_test(chgcar_paths, json_path):
    """
    From a list of CHGCAR paths, keep only those whose extracted ID
    is in the 'test' list of the JSON file.

    JSON format:
    {
      "train": [...],
      "validation": [...],
      "test": [1810, 42, ...]
    }
    """
    with open(json_path, "r") as f:
        splits = json.load(f)

    test_ids = set(splits.get("test", []))
    if not test_ids:
        print(f"⚠️  No 'test' entries found in {json_path}; returning empty list.")
        return []

    filtered = []
    for p in chgcar_paths:
        idx = extract_id_from_path(p)
        if idx is not None and idx in test_ids:
            filtered.append(p)

    return filtered


def write_tasks_csv(chgcar_paths, output_csv):
    """
    Write the CSV with headers and all performed_* columns set to False.
    """
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "CHGCAR_PATH",
            "performed_default",
            "performed_true_init",
            "performed_ml_init",
        ])
        for path in chgcar_paths:
            writer.writerow([path, "False", "False", "False"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a task CSV for CHGCAR(/.lz4) files."
    )
    parser.add_argument(
        "--root",
        dest="root_dir",
        default=DEFAULT_ROOT_DIR,
        help=f"Root directory to search for CHGCAR files (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--out",
        dest="output_csv",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of random CHGCARs to select. "
             "Use -1 to use all matching files.",
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default=None,
        help="Optional JSON with 'train'/'validation'/'test' lists. "
             "If provided, only CHGCARs whose ID is in 'test' are used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )

    args = parser.parse_args()

    # 1) Find all CHGCAR-like files under root_dir
    all_paths = sorted(find_chgcars(args.root_dir))
    if not all_paths:
        print(f"❌ No CHGCAR/CHGCAR.lz4 files found under {args.root_dir}")
        raise SystemExit(1)

    print(f"Found {len(all_paths)} CHGCAR-like files under {args.root_dir}")

    # 2) Optionally filter by JSON 'test' split
    if args.splits_json is not None:
        before = len(all_paths)
        all_paths = filter_by_json_test(all_paths, args.splits_json)
        print(
            f"Filtered by test split in {args.splits_json}: "
            f"{before} → {len(all_paths)} files"
        )
        if not all_paths:
            print("❌ No files matched the 'test' IDs from the JSON.")
            raise SystemExit(1)

    # 3) Sample N or use all
    if args.num == -1 or args.num >= len(all_paths):
        selected = all_paths
        print(f"Using all {len(selected)} files.")
    else:
        random.seed(args.seed)
        selected = random.sample(all_paths, args.num)
        print(f"Randomly selected {len(selected)} files out of {len(all_paths)}.")

    # 4) Write CSV
    write_tasks_csv(selected, args.output_csv)
    print(f"✅ Wrote {len(selected)} entries to {args.output_csv}")
