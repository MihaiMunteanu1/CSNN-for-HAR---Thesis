"""
Splits (by subject id):
  train : 11, 12, 13, 14, 15, 16, 17, 18                    (8 subjects)
  val   : 19, 20, 21, 23, 24, 25, 01, 04                    (8 subjects)
  test  : 02, 03, 05, 06, 07, 08, 09, 10, 22                (9 subjects)

"""

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


TRAIN_SUBJECTS = {11, 12, 13, 14, 15, 16, 17, 18}
VAL_SUBJECTS   = {19, 20, 21, 23, 24, 25,  1,  4}
TEST_SUBJECTS  = { 2,  3,  5,  6,  7,  8,  9, 10, 22}

KTH_CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]


def split_for_subject(subj):
    if subj in TRAIN_SUBJECTS: return "train"
    if subj in VAL_SUBJECTS:   return "val"
    if subj in TEST_SUBJECTS:  return "test"
    return None


def parse_subject(filename):
    m = re.match(r"person(\d+)_", filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    src = Path(args.input).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve()

    if not src.is_dir():
        sys.exit(f"Input folder does not exist: {src}")
    if dst == src:
        sys.exit("Output must differ from input.")

    union = TRAIN_SUBJECTS | VAL_SUBJECTS | TEST_SUBJECTS
    assert union == set(range(1, 26)), f"Splits miss subjects: {set(range(1,26)) - union}"
    assert not (TRAIN_SUBJECTS & VAL_SUBJECTS), "train/val overlap"
    assert not (TRAIN_SUBJECTS & TEST_SUBJECTS), "train/test overlap"
    assert not (VAL_SUBJECTS   & TEST_SUBJECTS), "val/test overlap"

    print(f"Input  : {src}")
    print(f"Output : {dst}")
    print()

    counts = Counter()
    missing = Counter()
    skipped = 0

    for top in src.iterdir():
        if not top.is_dir() or top.name not in ("train", "test"):
            continue
        for action_dir in top.iterdir():
            if not action_dir.is_dir():
                continue
            action = action_dir.name
            if action not in KTH_CLASSES:
                print(f"  WARN: unknown action folder {action_dir}", file=sys.stderr)
                continue
            for video in sorted(action_dir.iterdir()):
                if not video.is_file():
                    continue
                subj = parse_subject(video.name)
                if subj is None:
                    missing["unparseable"] += 1
                    print(f"  WARN: cannot parse subject from {video.name}", file=sys.stderr)
                    continue
                split = split_for_subject(subj)
                if split is None:
                    missing[f"subject_{subj}"] += 1
                    continue

                out_path = dst / split / action / video.name
                counts[split] += 1

                if args.dry_run:
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() and out_path.stat().st_size == video.stat().st_size:
                    skipped += 1
                    continue
                shutil.copy2(video, out_path)

    print("Summary:")
    for split in ("train", "val", "test"):
        print(f"  {split:5s} : {counts[split]:4d} videos")
    print(f"  total : {sum(counts.values())} videos")
    if skipped:
        print(f"  (skipped {skipped} that already existed at destination)")
    if missing:
        print(f"  WARN missing: {dict(missing)}")
    print("Done.")


if __name__ == "__main__":
    main()

