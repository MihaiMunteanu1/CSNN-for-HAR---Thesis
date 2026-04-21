#!/usr/bin/env python3
"""
Visualize qualitative SVM classification results for the KTH_3D experiment.

Reads the `qualitative_results_L<idx>.json` file produced by
`analysis::SvmQualitative` during a run of KTH_3D and, for every test sample,
extracts the actual frames from the original video file on disk. The frames
are organised on disk so that it is easy to visually inspect:

  - which videos the network classified correctly
  - which videos were misclassified, grouped by (true_label -> predicted_label)

Directory layout produced:

    <output>/
      confusion_matrix.png        # heatmap of normalized confusion matrix
      correct/
        <action>/
          <video_stem>__s<sample_idx>__g<group_idx>/
            frame_<N>.png
            grid.png              # all frames stitched into one image
      misclassified/
        <true>_as_<pred>/
          <video_stem>__s<sample_idx>__g<group_idx>/
            frame_<N>.png
            grid.png
      summary.md                  # textual summary per true class

Usage
-----
    python visualize_qualitative.py \
        --json /path/to/result/seed_7/qualitative_results_L2.json \
        --hog-json /path/to/hog/hog_person_data_5.json \
        --videos-root /home/mmuntean/kth_organized \
        --output /path/to/qualitative_out \
        [--max-per-pair 10] \
        [--no-grid] \
        [--no-heatmap]

Dependencies
------------
    opencv-python (cv2), numpy. Optional: matplotlib (for the heatmap).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not installed. Run: pip install opencv-python",
          file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: pip install numpy", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_hog(path: Path) -> Dict[str, Any]:
    """Load the HOG person-detection JSON.

    The file has this structure (produced by src/tool/extract_bboxes_kth.py):

        {
          "config": {...},
          "videos": {
            "train/boxing/person01_boxing_d1.avi": {
               "total_video_frames": N,
               "groups": [
                  [ {"frame_idx": K, "bboxes": [...]}, ... ],
                  ...
               ]
            },
            ...
          }
        }
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_frame_indices(
    hog_data: Dict[str, Any],
    video_key: str,
    group_idx: int,
) -> List[int]:
    """Return the list of frame indices for a given (video_key, group_idx).

    Falls back to an empty list if the key or group is missing, which causes
    the caller to skip that sample with a warning.
    """
    videos = hog_data.get("videos", {})
    entry = videos.get(video_key)
    if entry is None:
        return []

    groups = entry.get("groups", [])
    if not groups:
        return []

    # Mirror the wrap-around logic of VideoKTH_3D::next() so Python results
    # match what the C++ side actually read from disk.
    real_idx = group_idx % len(groups)
    group = groups[real_idx]
    return [fb["frame_idx"] for fb in group if "frame_idx" in fb]


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, frame_indices: List[int]) -> List[np.ndarray]:
    """Read specific frame indices from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames: List[np.ndarray] = []
    try:
        for idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def make_grid(frames: List[np.ndarray], cols: Optional[int] = None) -> Optional[np.ndarray]:
    """Stitch frames into a single image for quick inspection."""
    if not frames:
        return None

    # All frames should already be the same size (KTH videos are 160x120),
    # but normalize defensively.
    h, w = frames[0].shape[:2]
    frames = [cv2.resize(f, (w, h)) if f.shape[:2] != (h, w) else f for f in frames]

    n = len(frames)
    if cols is None:
        cols = n
    rows = (n + cols - 1) // cols

    # Pad with black frames if needed
    total = rows * cols
    if n < total:
        black = np.zeros_like(frames[0])
        frames = frames + [black] * (total - n)

    rows_imgs = []
    for r in range(rows):
        row = np.hstack(frames[r * cols:(r + 1) * cols])
        rows_imgs.append(row)
    return np.vstack(rows_imgs)


# ---------------------------------------------------------------------------
# Confusion matrix heatmap (optional)
# ---------------------------------------------------------------------------

def save_heatmap(
    matrix: List[List[int]],
    classes: List[str],
    output_path: Path,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping heatmap. "
              "Install with: pip install matplotlib")
        return False

    mat = np.array(matrix, dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    norm = mat / row_sums

    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 2, 1.2 * len(classes) + 2))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    for i in range(len(classes)):
        for j in range(len(classes)):
            txt_color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{int(mat[i, j])}\n{norm[i, j]:.2f}",
                    ha="center", va="center", color=txt_color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sanitize(name: str) -> str:
    """Make a string safe to use as a path component."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize qualitative KTH_3D SVM results.",
    )
    parser.add_argument("--json", required=True, type=Path,
                        help="Path to qualitative_results_L<idx>.json produced by SvmQualitative")
    parser.add_argument("--hog-json", required=True, type=Path,
                        help="Path to hog_person_data_<N>.json used by the experiment")
    parser.add_argument("--videos-root", required=True, type=Path,
                        help="Root folder containing train/ and test/ subdirs of videos")
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory where visualisations will be written")
    parser.add_argument("--max-per-pair", type=int, default=10,
                        help="Maximum number of videos to extract for each (true,pred) pair")
    parser.add_argument("--no-grid", action="store_true",
                        help="Skip writing the stitched grid.png per sample")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip the matplotlib confusion-matrix heatmap")
    args = parser.parse_args()

    if not args.json.exists():
        print(f"ERROR: results JSON not found: {args.json}", file=sys.stderr)
        return 1
    if not args.hog_json.exists():
        print(f"ERROR: HOG JSON not found: {args.hog_json}", file=sys.stderr)
        return 1
    if not args.videos_root.is_dir():
        print(f"ERROR: videos root is not a directory: {args.videos_root}", file=sys.stderr)
        return 1

    results = load_results(args.json)
    hog_data = load_hog(args.hog_json)

    classes = results.get("classes", [])
    matrix = results.get("confusion_matrix", [])
    samples = results.get("samples", [])
    accuracy = results.get("accuracy", 0.0)
    total = results.get("total", 0)
    correct = results.get("correct", 0)

    print(f"Loaded {len(samples)} samples from {args.json}")
    print(f"Experiment: {results.get('experiment', '?')}  "
          f"layer_index: {results.get('layer_index', '?')}")
    print(f"Accuracy: {accuracy:.4f}  ({correct}/{total})")
    print(f"Classes: {classes}")

    args.output.mkdir(parents=True, exist_ok=True)

    # --- Save the heatmap ---
    if not args.no_heatmap and matrix and classes:
        heatmap_path = args.output / "confusion_matrix.png"
        if save_heatmap(matrix, classes, heatmap_path):
            print(f"Wrote {heatmap_path}")

    # --- Group samples by (true, predicted) and count how many we've saved ---
    per_pair_counter: Dict[tuple, int] = defaultdict(int)
    skipped_no_video = 0
    skipped_budget = 0
    extracted = 0

    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        true_label = sample["true_label"]
        pred_label = sample["predicted_label"]
        video_key = sample["video_key"]
        group_idx = int(sample["group_idx"])
        sample_idx = int(sample["sample_idx"])
        correct_flag = bool(sample["correct"])

        summary[true_label][pred_label] += 1

        pair_key = (true_label, pred_label)
        if per_pair_counter[pair_key] >= args.max_per_pair:
            skipped_budget += 1
            continue

        if not video_key:
            skipped_no_video += 1
            continue

        # Resolve frame indices from the HOG JSON
        frame_indices = resolve_frame_indices(hog_data, video_key, group_idx)
        if not frame_indices:
            print(f"  WARN: no HOG frames for {video_key} group {group_idx}, skipping")
            skipped_no_video += 1
            continue

        # Locate the actual video file on disk
        video_path = args.videos_root / video_key
        if not video_path.exists():
            print(f"  WARN: video not found on disk: {video_path}")
            skipped_no_video += 1
            continue

        frames = extract_frames(video_path, frame_indices)
        if not frames:
            print(f"  WARN: could not read frames from {video_path}")
            skipped_no_video += 1
            continue

        # Build output dir
        if correct_flag:
            category_dir = args.output / "correct" / sanitize(true_label)
        else:
            category_dir = (args.output / "misclassified" /
                            f"{sanitize(true_label)}_as_{sanitize(pred_label)}")

        video_stem = sanitize(Path(video_key).stem)
        sample_dir = category_dir / f"{video_stem}__s{sample_idx}__g{group_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for frame_no, frame in zip(sorted(frame_indices), frames):
            cv2.imwrite(str(sample_dir / f"frame_{frame_no:04d}.png"), frame)

        if not args.no_grid:
            grid = make_grid(frames)
            if grid is not None:
                cv2.imwrite(str(sample_dir / "grid.png"), grid)

        per_pair_counter[pair_key] += 1
        extracted += 1

    # --- Summary file ---
    summary_path = args.output / "summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Qualitative results — {results.get('experiment', '?')}\n\n")
        f.write(f"- Layer index: {results.get('layer_index', '?')}\n")
        f.write(f"- Accuracy: **{accuracy:.4f}** ({correct}/{total})\n")
        f.write(f"- Samples extracted: {extracted}\n")
        f.write(f"- Samples skipped (budget): {skipped_budget}\n")
        f.write(f"- Samples skipped (missing video/frames): {skipped_no_video}\n\n")
        f.write("## Per-class breakdown\n\n")
        for true_cls in classes:
            row = summary.get(true_cls, {})
            total_true = sum(row.values())
            if total_true == 0:
                continue
            hits = row.get(true_cls, 0)
            acc = hits / total_true if total_true else 0.0
            f.write(f"### {true_cls}  —  acc {acc:.2%} ({hits}/{total_true})\n")
            for pred_cls in classes:
                cnt = row.get(pred_cls, 0)
                if cnt == 0:
                    continue
                marker = "✓" if pred_cls == true_cls else "✗"
                f.write(f"- {marker} predicted as **{pred_cls}**: {cnt}\n")
            f.write("\n")

    print()
    print(f"Extracted frames for {extracted} samples")
    print(f"Skipped (budget):            {skipped_budget}")
    print(f"Skipped (missing video/HOG): {skipped_no_video}")
    print(f"Wrote {summary_path}")
    print(f"Done. Browse {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# python3 src/tool/visualize_qualitative.py \
#         --json result/seed_7/qualitative_results_L2.json \
#                --hog-json ../hog/hog_person_data_5.json \
#                --videos-root /home/mmuntean/kth_organized \
#                --output result/seed_7/qualitative
