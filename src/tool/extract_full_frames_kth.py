"""
Pre-extract FULL frames (no crop) + per-sample bboxes for KTH.

Difference from extract_frames_kth.py:
- Saves the WHOLE frame (no crop to bbox)
- Saves per-frame bboxes in metadata so HOGSampler3D can restrict
  spatial sampling to the person region
- One sample per group (no augmentation), for both train and test

Output:
  <output>.npy   uint8 array, shape (N, T, H, W)
  <output>.json  metadata with one entry per sample, each containing
                 a 'bboxes' list of length T (each [x, y, w, h] in
                 OUTPUT-resolution pixel coordinates)

"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


KTH_CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
CLASS_TO_IDX = {c: i for i, c in enumerate(KTH_CLASSES)}
VALID_SPLITS = ("train", "val", "test")


def parse_kth_filename(filename):
    name = Path(filename).name
    match = re.match(r"person(\d+)_(\w+)", name, re.IGNORECASE)
    if match:
        subject = int(match.group(1))
        rest = match.group(2).lower()
        for action in KTH_CLASSES:
            if rest.startswith(action):
                return subject, action
    return None, None


def split_from_video_key(video_key):
    head = video_key.replace("\\", "/").split("/", 1)[0]
    return head if head in VALID_SPLITS else None


def rescale_bbox(bbox, src_w, src_h, out_w, out_h):
    sx = out_w / float(src_w)
    sy = out_h / float(src_h)
    x = int(round(bbox["x"] * sx))
    y = int(round(bbox["y"] * sy))
    w = int(round(bbox["w"] * sx))
    h = int(round(bbox["h"] * sy))
    x = max(0, min(x, out_w - 1))
    y = max(0, min(y, out_h - 1))
    w = max(1, min(w, out_w - x))
    h = max(1, min(h, out_h - y))
    return x, y, w, h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video_root", default="/home/mmuntean/kth_organized")
    parser.add_argument("--frame_size_width", type=int, default=160)
    parser.add_argument("--frame_size_height", type=int, default=120)
    parser.add_argument("--report_every", type=int, default=25)
    args = parser.parse_args()

    print(f"Loading bbox JSON: {args.bbox_json}")
    with open(args.bbox_json, "r") as f:
        in_data = json.load(f)

    config = in_data.get("config", {})
    src_w = config.get("frame_width", 160)
    src_h = config.get("frame_height", 120)
    T = config.get("temporal_kernel", 19)
    H = args.frame_size_height
    W = args.frame_size_width

    print(f"Source frame: {src_w}x{src_h} | T={T} | output frame: {W}x{H}")
    print(f"Video root: {args.video_root}")
    print(f"Mode: FULL FRAME (no crop) | 1 sample per group | no augmentation")

    frames_list = []
    samples_meta = []

    videos = in_data.get("videos", {})
    n_videos = len(videos)

    for vi, (video_key, video_data) in enumerate(videos.items()):
        subject, action = parse_kth_filename(video_key)
        if subject is None or action is None:
            continue
        split = split_from_video_key(video_key)
        if split is None:
            print(f"  WARN: skipping {video_key} (no train/val/test prefix)")
            continue
        label_idx = CLASS_TO_IDX[action]

        video_path = Path(args.video_root) / video_key
        if not video_path.exists():
            continue

        groups = video_data.get("groups", [])
        if not groups:
            continue

        needed = set()
        for g in groups:
            for fd in g:
                needed.add(int(fd["frame_idx"]))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        loaded = {}
        idx = 0
        max_needed = max(needed) if needed else -1
        while True:
            ret, frame = cap.read()
            if not ret or idx > max_needed:
                break
            if idx in needed:
                if frame.shape[1] != src_w or frame.shape[0] != src_h:
                    frame = cv2.resize(frame, (src_w, src_h))
                if W != src_w or H != src_h:
                    frame = cv2.resize(frame, (W, H))
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                loaded[idx] = frame.astype(np.uint8)
            idx += 1
        cap.release()

        for gi, group in enumerate(groups):
            clip = np.zeros((T, H, W), dtype=np.uint8)
            bboxes_per_t = []
            ok = True
            for fi, frame_data in enumerate(group):
                f_idx = int(frame_data["frame_idx"])
                bbox = frame_data.get("selected_bbox") or \
                       (frame_data.get("bboxes") or [None])[0]
                if bbox is None or f_idx not in loaded:
                    ok = False
                    break
                clip[fi] = loaded[f_idx]
                bx, by, bw, bh = rescale_bbox(bbox, src_w, src_h, W, H)
                bboxes_per_t.append([bx, by, bw, bh])
            if not ok or len(bboxes_per_t) != T:
                continue

            frames_list.append(clip)
            samples_meta.append({
                "video_key": video_key,
                "subject": subject,
                "action": action,
                "label_idx": label_idx,
                "group_idx": gi,
                "frame_indices": [int(fd["frame_idx"]) for fd in group],
                "bboxes": bboxes_per_t,
                "split": split,
            })

        if (vi + 1) % args.report_every == 0 or vi == n_videos - 1:
            print(f"  [{vi+1:4d}/{n_videos}] {video_key} | total samples: {len(frames_list)}")

    if not frames_list:
        raise SystemExit("No samples extracted. Check --video_root and --bbox_json.")

    arr = np.stack(frames_list, axis=0)
    split_counts = {s: sum(1 for m in samples_meta if m["split"] == s)
                    for s in VALID_SPLITS}

    print()
    print("Statistics:")
    print(f"  total samples : {arr.shape[0]}")
    for s in VALID_SPLITS:
        print(f"  {s:5s}         : {split_counts[s]}")
    print(f"  shape         : {arr.shape}  dtype={arr.dtype}")
    print(f"  size          : {arr.nbytes / 1e6:.1f} MB")

    out_npy = args.output + ".npy"
    out_json = args.output + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(out_npy)), exist_ok=True)

    print(f"\nSaving frames to {out_npy} ...")
    np.save(out_npy, arr)

    meta_doc = {
        "config": {
            "N": int(arr.shape[0]),
            "T": int(arr.shape[1]),
            "H": int(arr.shape[2]),
            "W": int(arr.shape[3]),
            "C": 1,
            "frame_size_width": int(W),
            "frame_size_height": int(H),
            "source_frame_width": int(src_w),
            "source_frame_height": int(src_h),
            "source_json": os.path.basename(args.bbox_json),
            "mode": "full_frame_with_bboxes",
        },
        "samples": samples_meta,
    }
    print(f"Saving metadata to {out_json} ...")
    with open(out_json, "w") as f:
        json.dump(meta_doc, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

