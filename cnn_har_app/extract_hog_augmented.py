"""
Pre-compute HOG features with on-line video augmentation, output as .npz.

Pipeline:
  1. Read bbox metadata JSON (output of src/tool/extract_bboxes_kth.py).
  2. For each video, load the needed frames once.
  3. For each (sample = group of T frames):
       - Always emit the original variant (no augmentation).
       - For TRAIN videos only, emit (num_aug - 1) augmented variants:
           * horizontal flip (always one of the variants)
           * random bbox jitter: dx ∈ [-5, +5], dy ∈ [-3, +3], scale ∈ [0.9, 1.1]
           * random brightness/contrast: alpha ∈ [0.85, 1.15], beta ∈ [-15, +15]
       - For each variant: crop bbox → resize 64×128 → flip/brightness → HOG.
  4. Save all samples to .npz with arrays:
       - features : (N, T*3780) float32
       - bboxes   : (N, T, 4)    float32 — (cx, cy, w, h) normalized to frame
       - labels   : (N,)         int64
       - metadata : (N,)         object (dicts: video_key, subject, action, ...)

Usage:
    python3 extract_hog_augmented.py \
        --bbox_json ../hog/hog_person_data_new_7.json \
        --output ../hog/hog_aug_7.npz \
        --num_aug 4
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
TRAIN_SUBJECTS = set(range(1, 17))

HOG_WIN_W = 64
HOG_WIN_H = 128
HOG_FEAT_PER_FRAME = 3780  # cv2 default HOG on 64x128


def parse_kth_filename(filename):
    name = Path(filename).name
    match = re.match(r'person(\d+)_(\w+)', name, re.IGNORECASE)
    if match:
        subject = int(match.group(1))
        rest = match.group(2).lower()
        for action in KTH_CLASSES:
            if rest.startswith(action):
                return subject, action
    return None, None


def jitter_bbox(bbox, frame_w, frame_h, dx, dy, scale):
    cx = bbox["x"] + bbox["w"] / 2.0
    cy = bbox["y"] + bbox["h"] / 2.0
    new_w = max(1, int(bbox["w"] * scale))
    new_h = max(1, int(bbox["h"] * scale))
    new_x = int(cx - new_w / 2.0 + dx)
    new_y = int(cy - new_h / 2.0 + dy)
    new_x = max(0, min(new_x, frame_w - 1))
    new_y = max(0, min(new_y, frame_h - 1))
    new_w = max(1, min(new_w, frame_w - new_x))
    new_h = max(1, min(new_h, frame_h - new_y))
    return {"x": new_x, "y": new_y, "w": new_w, "h": new_h}


def compute_hog_with_aug(frame_bgr, bbox, aug, hog_desc, frame_w, frame_h):
    """
    Apply aug to (bbox, frame), crop, resize 64x128, flip + brightness, HOG.

    Returns:
        (hog_vec_flat (3780,) float32, post_aug_bbox dict) or (None, None) on failure.
    """
    box = jitter_bbox(bbox, frame_w, frame_h, aug["dx"], aug["dy"], aug["scale"])

    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return None, None

    crop = cv2.resize(crop, (HOG_WIN_W, HOG_WIN_H))
    if aug["flip"]:
        crop = cv2.flip(crop, 1)
    if aug["alpha"] != 1.0 or aug["beta"] != 0:
        crop = cv2.convertScaleAbs(crop, alpha=aug["alpha"], beta=aug["beta"])

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    feat = hog_desc.compute(gray)
    if feat is None:
        return None, None

    # Bbox stored in conceptual (post-flip) coords so the bbox stream is
    # consistent with the HOG content for the model.
    if aug["flip"]:
        post_box = dict(box)
        post_box["x"] = frame_w - box["x"] - box["w"]
    else:
        post_box = box

    return feat.flatten().astype(np.float32), post_box


def load_video_frames(video_path, frame_indices, frame_w, frame_h):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    needed = set(frame_indices)
    found = {}
    idx = 0
    max_needed = max(needed) if needed else -1
    while True:
        ret, frame = cap.read()
        if not ret or idx > max_needed:
            break
        if idx in needed:
            found[idx] = cv2.resize(frame, (frame_w, frame_h))
        idx += 1
    cap.release()
    return found


def make_aug_specs(num_aug, rng):
    """Build a list of augmentation specs. First is always 'orig' (identity)."""
    specs = [{
        "name": "orig", "flip": False, "scale": 1.0,
        "dx": 0, "dy": 0, "alpha": 1.0, "beta": 0.0,
    }]
    if num_aug <= 1:
        return specs

    # Always include a pure horizontal flip.
    specs.append({
        "name": "flip", "flip": True, "scale": 1.0,
        "dx": 0, "dy": 0, "alpha": 1.0, "beta": 0.0,
    })

    for j in range(num_aug - 2):
        specs.append({
            "name": f"jit{j}",
            "flip": bool(rng.random() < 0.5),
            "scale": float(rng.uniform(0.92, 1.08)),
            "dx": int(rng.integers(-5, 6)),
            "dy": int(rng.integers(-3, 4)),
            "alpha": float(rng.uniform(0.85, 1.15)),
            "beta": float(rng.uniform(-15, 15)),
        })
    return specs


def process_video(video_key, video_data, video_root, frame_w, frame_h,
                  num_aug, hog_desc, rng):
    """
    Yields (sample_features (T*3780,), sample_bboxes (T, 4), label_idx, meta_dict)
    for each (group, augmentation) tuple of this video.

    Bboxes are normalized to [0, 1] (cx, cy, w, h).
    """
    subject, action = parse_kth_filename(video_key)
    if subject is None or action is None:
        return
    label_idx = CLASS_TO_IDX[action]
    is_train = subject in TRAIN_SUBJECTS

    video_path = Path(video_root) / video_key
    if not video_path.exists():
        return

    groups = video_data.get("groups", [])
    if not groups:
        return

    needed_frames = set()
    for group in groups:
        for fd in group:
            needed_frames.add(int(fd["frame_idx"]))

    frames = load_video_frames(video_path, sorted(needed_frames), frame_w, frame_h)
    if not frames:
        return

    aug_specs = make_aug_specs(num_aug if is_train else 1, rng)

    for ai, aug in enumerate(aug_specs):
        for gi, group in enumerate(groups):
            T = len(group)
            feats = np.zeros((T, HOG_FEAT_PER_FRAME), dtype=np.float32)
            boxes = np.zeros((T, 4), dtype=np.float32)
            ok = True

            for fi, frame_data in enumerate(group):
                idx = int(frame_data["frame_idx"])
                bbox = frame_data.get("selected_bbox") or (frame_data.get("bboxes") or [None])[0]
                if bbox is None or idx not in frames:
                    ok = False
                    break

                hog_vec, post_box = compute_hog_with_aug(
                    frames[idx], bbox, aug, hog_desc, frame_w, frame_h
                )
                if hog_vec is None:
                    ok = False
                    break

                feats[fi] = hog_vec
                cx = (post_box["x"] + post_box["w"] / 2.0) / frame_w
                cy = (post_box["y"] + post_box["h"] / 2.0) / frame_h
                boxes[fi] = [cx, cy, post_box["w"] / frame_w, post_box["h"] / frame_h]

            if not ok:
                continue

            meta = {
                "video_key": video_key,
                "subject": subject,
                "action": action,
                "label_idx": label_idx,
                "group_idx": gi,
                "aug_idx": ai,
                "aug_name": aug["name"],
                "frame_indices": [int(fd["frame_idx"]) for fd in group],
                "split": "train" if is_train else "test",
            }
            yield feats.reshape(-1), boxes, label_idx, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox_json", type=str, required=True,
                        help="Input JSON from extract_bboxes_kth.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz path with pre-computed augmented HOG features")
    parser.add_argument("--video_root", type=str, default="/home/mmuntean/kth_organized")
    parser.add_argument("--num_aug", type=int, default=4,
                        help="Variants per train video (1 orig + N-1 augs). Test always 1.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_every", type=int, default=25)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hog_desc = cv2.HOGDescriptor()

    print(f"Loading bbox JSON: {args.bbox_json}")
    with open(args.bbox_json, "r") as f:
        in_data = json.load(f)

    config = in_data.get("config", {})
    frame_w = config.get("frame_width", 160)
    frame_h = config.get("frame_height", 120)
    T = config.get("temporal_kernel", 7)

    print(f"Frame size: {frame_w}x{frame_h} | T={T} | num_aug per train: {args.num_aug}")
    print(f"Video root: {args.video_root}")

    videos = in_data.get("videos", {})
    n_videos = len(videos)

    feats_list = []
    boxes_list = []
    labels_list = []
    meta_list = []

    n_train_videos = 0
    n_test_videos = 0

    for vi, (video_key, video_data) in enumerate(videos.items()):
        n_made = 0
        for feats, boxes, label, meta in process_video(
            video_key, video_data, args.video_root,
            frame_w, frame_h, args.num_aug, hog_desc, rng,
        ):
            feats_list.append(feats)
            boxes_list.append(boxes)
            labels_list.append(label)
            meta_list.append(meta)
            n_made += 1

        if n_made > 0:
            if meta_list[-1]["split"] == "train":
                n_train_videos += 1
            else:
                n_test_videos += 1

        if (vi + 1) % args.report_every == 0 or vi == n_videos - 1:
            print(f"  [{vi+1:4d}/{n_videos}] last={video_key} -> {n_made} variant(s) "
                  f"| total samples so far: {len(feats_list)}")

    if not feats_list:
        raise SystemExit("No samples produced. Check video_root paths and bbox JSON.")

    features = np.stack(feats_list, axis=0)
    bboxes = np.stack(boxes_list, axis=0)
    labels = np.asarray(labels_list, dtype=np.int64)
    metadata = np.array(meta_list, dtype=object)

    n_train = sum(1 for m in meta_list if m["split"] == "train")
    n_test = len(meta_list) - n_train

    print()
    print("Statistics:")
    print(f"  total samples : {len(meta_list)}")
    print(f"  train samples : {n_train}  (from {n_train_videos} videos)")
    print(f"  test samples  : {n_test}   (from {n_test_videos} videos)")
    print(f"  features shape: {features.shape}  ({features.nbytes / 1e6:.1f} MB)")
    print(f"  bboxes shape  : {bboxes.shape}")

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"\nSaving to {out_path} ...")
    np.savez(
        out_path,
        features=features,
        bboxes=bboxes,
        labels=labels,
        metadata=metadata,
        config=np.array(
            {
                **config,
                "hog_precomputed": True,
                "augmented": True,
                "num_aug_per_train": args.num_aug,
                "source_json": os.path.basename(args.bbox_json),
                "seed": args.seed,
            },
            dtype=object,
        ),
    )
    print("Done.")


if __name__ == "__main__":
    main()


# cd ~/csnn_simulator_v2/cnn_har_app
# python3 extract_hog_augmented.py \
#         --bbox_json ../hog/hog_person_data_new_7.json \
#         --output ../hog/hog_aug_7.npz \
#         --num_aug 6 \
#                   --video_root /home/mmuntean/kth_organized
# sau num_aug 4-6-8

#apoi python3 train.py --data_path ../hog/hog_aug_7.npz
