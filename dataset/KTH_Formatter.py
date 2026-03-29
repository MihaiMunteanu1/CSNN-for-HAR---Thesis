"""
KTH Dataset Formatter for CSNN Simulator

This script organizes raw KTH action recognition dataset videos into the
folder structure expected by the dataset::Video class in the simulator.

Expected raw KTH naming: person01_boxing_d1_uncomp.avi
                         person{NN}_{action}_{scenario}_uncomp.avi

Output structure:
    output_dir/
    ├── train/
    │   ├── boxing/
    │   ├── handclapping/
    │   ├── handwaving/
    │   ├── jogging/
    │   ├── running/
    │   └── walking/
    └── test/
        ├── boxing/
        ├── handclapping/
        ├── handwaving/
        ├── jogging/
        ├── running/
        └── walking/

Usage:
    python KTH_Formatter.py --input <raw_kth_folder> --output <output_folder>

Train/Test split (standard KTH protocol):
    - Train: subjects 01-16
    - Test:  subjects 17-25
"""

import os
import sys
import shutil
import argparse
import re

# KTH action classes (6 classes)
KTH_ACTIONS = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

# Standard KTH train/test split by subject number
TRAIN_SUBJECTS = list(range(1, 17))   # subjects 01-16
TEST_SUBJECTS = list(range(17, 26))   # subjects 17-25


def parse_kth_filename(filename):
    """
    Parse a KTH video filename to extract subject number and action.
    
    Expected format: person{NN}_{action}_{scenario}_uncomp.avi
    Examples:
        person01_boxing_d1_uncomp.avi
        person12_handclapping_d2_uncomp.avi
    
    Returns:
        (subject_number: int, action: str) or None if parsing fails
    """
    match = re.match(r'person(\d+)_(\w+?)_d\d+_uncomp\.avi', filename, re.IGNORECASE)
    if match:
        subject = int(match.group(1))
        action = match.group(2).lower()
        return subject, action

    match = re.match(r'person(\d+)_(\w+?)_d\d+\.avi', filename, re.IGNORECASE)
    if match:
        subject = int(match.group(1))
        action = match.group(2).lower()
        return subject, action

    match = re.match(r'person(\d+)_(\w+)', filename, re.IGNORECASE)
    if match:
        subject = int(match.group(1))
        remaining = match.group(2).lower()
        for action in KTH_ACTIONS:
            if remaining.startswith(action):
                return subject, action

    return None


def organize_kth_dataset(input_dir, output_dir, copy_mode="copy"):
    """
    Organize raw KTH videos into train/test folder structure.
    
    Args:
        input_dir: Path to raw KTH videos (can be flat or organized by action)
        output_dir: Path to output directory
        copy_mode: "copy" to copy files, "symlink" to create symbolic links
    """
    for split in ["train", "test"]:
        for action in KTH_ACTIONS:
            os.makedirs(os.path.join(output_dir, split, action), exist_ok=True)

    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.avi'):
                video_files.append(os.path.join(root, f))

    if not video_files:
        print(f"ERROR: No .avi video files found in '{input_dir}'")
        print("Make sure the KTH dataset videos are in the input directory.")
        sys.exit(1)

    print(f"Found {len(video_files)} video files in '{input_dir}'")

    stats = {"train": {a: 0 for a in KTH_ACTIONS}, "test": {a: 0 for a in KTH_ACTIONS}}
    skipped = []

    for video_path in sorted(video_files):
        filename = os.path.basename(video_path)
        result = parse_kth_filename(filename)

        if result is None:
            skipped.append(filename)
            continue

        subject, action = result

        if action not in KTH_ACTIONS:
            skipped.append(filename)
            continue

        if subject in TRAIN_SUBJECTS:
            split = "train"
        elif subject in TEST_SUBJECTS:
            split = "test"
        else:
            skipped.append(filename)
            continue

        dest_path = os.path.join(output_dir, split, action, filename)

        if not os.path.exists(dest_path):
            if copy_mode == "symlink":
                os.symlink(os.path.abspath(video_path), dest_path)
            else:
                shutil.copy2(video_path, dest_path)

        stats[split][action] += 1

    print("\n" + "=" * 60)
    print("KTH Dataset Organization Summary")
    print("=" * 60)
    
    for split in ["train", "test"]:
        total = sum(stats[split].values())
        print(f"\n{split.upper()} set ({total} videos):")
        for action in KTH_ACTIONS:
            print(f"  {action:15s}: {stats[split][action]:4d} videos")

    if skipped:
        print(f"\nSkipped {len(skipped)} files (could not parse):")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    total_organized = sum(sum(stats[s].values()) for s in ["train", "test"])
    print(f"\nTotal organized: {total_organized} videos")
    print(f"Output directory: {output_dir}")
    print("\nTo run the KTH experiment:")
    print(f'  export INPUT_PATH="{os.path.abspath(output_dir)}/"')


def main():
    parser = argparse.ArgumentParser(
        description="Organize raw KTH dataset videos into train/test folder structure for CSNN Simulator"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to raw KTH video files (can be flat directory or organized by action)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output directory (will be created if it doesn't exist)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["copy", "symlink"],
        default="copy",
        help="File transfer mode: 'copy' (default) or 'symlink'"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory '{args.input}' does not exist.")
        sys.exit(1)

    organize_kth_dataset(args.input, args.output, args.mode)


if __name__ == "__main__":
    main()
