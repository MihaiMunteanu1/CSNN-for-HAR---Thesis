"""
KTH frame extractor — saves cropped person frames as .jpg files.

Strategy: extract CONSECUTIVE frame windows (required for temporal STDP
learning where the 3D filter spans multiple frames). Each window contains
`seq_length` consecutive frames (default 5). Up to `num_sequences` windows
per video (default 10), distributed evenly across available runs of
consecutive person-detected frames.

For each frame in a window:
  1. Smooth the bbox temporally (alpha=0.65)
  2. Crop with aspect-aware padding around the person
  3. Resize to target_width x target_height
  4. Save as .jpg with zero-padded frame index

Output structure:
  kth_cropped/
    train/
      boxing/
        person01_boxing_d1_frame0042.jpg
        person01_boxing_d1_frame0043.jpg   <- consecutive in video
        person01_boxing_d1_frame0044.jpg
        person01_boxing_d1_frame0045.jpg
        person01_boxing_d1_frame0046.jpg   <- end of window 1
        person01_boxing_d1_frame0089.jpg   <- start of window 2
        ...
    test/
      ...
    metadata.json   (bbox + source info per frame)

ImageSequenceKTH will sort files lexicographically (= by frame index with
zero-padding), then chunk them into non-overlapping groups of
temporal_depth. Since we saved non-overlapping windows of exactly
seq_length consecutive frames, each group corresponds to one clean
consecutive-frame window.
"""

import argparse
import json
import os
import cv2
import numpy as np
import math

# ---------------------------------------------------------------------------
# Detection helpers (reused from extract_bboxes_kth.py)
# ---------------------------------------------------------------------------

_HOG_SCALE_FACTOR = 3
_HOG_MIN_W = 240
_HOG_MIN_H = 180


def clip_bbox(b, frame_w, frame_h):
    x = max(0, min(int(b["x"]), frame_w - 1))
    y = max(0, min(int(b["y"]), frame_h - 1))
    w = max(1, min(int(b["w"]), frame_w - x))
    h = max(1, min(int(b["h"]), frame_h - y))
    out = dict(b)
    out["x"], out["y"], out["w"], out["h"] = x, y, w, h
    return out


def passes_bbox_quality(b, frame_w, frame_h, min_area_ratio, min_aspect, max_aspect):
    w, h = b["w"], b["h"]
    if w <= 1 or h <= 1:
        return False
    if (w * h) < (frame_w * frame_h * min_area_ratio):
        return False
    ar = w / float(h + 1e-9)
    return min_aspect <= ar <= max_aspect


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, a["w"] * a["h"])
    area_b = max(1, b["w"] * b["h"])
    union = area_a + area_b - inter
    return inter / float(union + 1e-9)


def center_distance(a, b):
    acx, acy = a["x"] + a["w"] / 2.0, a["y"] + a["h"] / 2.0
    bcx, bcy = b["x"] + b["w"] / 2.0, b["y"] + b["h"] / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def normalize_confidences(boxes, key="confidence"):
    if not boxes:
        return []
    vals = [float(bb.get(key, 0.0)) for bb in boxes]
    vmin, vmax = min(vals), max(vals)
    out = []
    for bb in boxes:
        s = float(bb.get(key, 0.0))
        if vmax - vmin < 1e-9:
            ns = 0.5
        else:
            ns = (s - vmin) / (vmax - vmin + 1e-9)
        c = dict(bb)
        c["_norm_conf"] = ns
        out.append(c)
    return out


def score_box(box, prev_box, frame_w, frame_h):
    conf = float(box.get("_norm_conf", 0.5))
    area_ratio = (box["w"] * box["h"]) / float(frame_w * frame_h + 1e-9)
    if area_ratio < 0.005:
        area_term = -0.6
    elif area_ratio > 0.70:
        area_term = -0.5
    else:
        area_term = 0.25
    temporal_term = 0.0
    if prev_box is not None:
        iou = bbox_iou(box, prev_box)
        dist = center_distance(box, prev_box) / float(max(frame_w, frame_h) + 1e-9)
        temporal_term = 0.8 * iou - 0.35 * dist
    src = box.get("source", "")
    src_bias = 0.10 if src == "hog" else 0.0
    return conf + area_term + temporal_term + src_bias


def merge_hog_mog2(hog_boxes, mog_boxes, prev_box, frame_w, frame_h, iou_merge_th=0.35):
    hog_n = normalize_confidences(hog_boxes)
    mog_n = normalize_confidences(mog_boxes)
    fused = []
    used_m = set()
    for h in hog_n:
        best_j, best_iou = -1, 0.0
        for j, m in enumerate(mog_n):
            if j in used_m:
                continue
            iou = bbox_iou(h, m)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_merge_th:
            m = mog_n[best_j]
            used_m.add(best_j)
            merged = {
                "x": int(round(0.6 * h["x"] + 0.4 * m["x"])),
                "y": int(round(0.6 * h["y"] + 0.4 * m["y"])),
                "w": int(round(0.5 * h["w"] + 0.5 * m["w"])),
                "h": int(round(0.5 * h["h"] + 0.5 * m["h"])),
                "confidence": 0.6 * float(h.get("confidence", 0.0)) + 0.4 * float(m.get("confidence", 0.0)),
                "source": "hog+mog2"
            }
            fused.append(merged)
        else:
            fused.append(h)
    for j, m in enumerate(mog_n):
        if j not in used_m:
            fused.append(m)
    if not fused:
        return []
    best, best_score = None, -1e18
    for b in fused:
        s = score_box(b, prev_box, frame_w, frame_h)
        if s > best_score:
            best_score = s
            best = b
    return [best] if best is not None else []


def smooth_bbox(prev_box, cur_box, alpha=0.65):
    if prev_box is None:
        return cur_box
    return {
        "x": int(round(alpha * prev_box["x"] + (1.0 - alpha) * cur_box["x"])),
        "y": int(round(alpha * prev_box["y"] + (1.0 - alpha) * cur_box["y"])),
        "w": int(round(alpha * prev_box["w"] + (1.0 - alpha) * cur_box["w"])),
        "h": int(round(alpha * prev_box["h"] + (1.0 - alpha) * cur_box["h"])),
        "confidence": float(cur_box.get("confidence", 0.0)),
        "source": cur_box.get("source", "fused")
    }


def detect_persons_in_frame(frame_gray, hog, frame_w, frame_h, hit_threshold):
    det_w = max(frame_w * _HOG_SCALE_FACTOR, _HOG_MIN_W)
    det_h = max(frame_h * _HOG_SCALE_FACTOR, _HOG_MIN_H)
    det_frame = cv2.resize(frame_gray, (det_w, det_h))
    if det_frame.dtype != np.uint8:
        det_frame = cv2.normalize(det_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    found_locations, weights = hog.detectMultiScale(
        det_frame, hitThreshold=hit_threshold,
        winStride=(12, 12), padding=(4, 4), scale=1.10
    )
    sx = frame_w / float(det_w)
    sy = frame_h / float(det_h)
    bboxes = []
    for i, (x, y, w, h) in enumerate(found_locations):
        bboxes.append({
            "x": int(x * sx), "y": int(y * sy),
            "w": int(w * sx), "h": int(h * sy),
            "confidence": float(weights[i]) if i < len(weights) else 0.0,
            "source": "hog"
        })
    return bboxes


def detect_persons_mog2(frame_gray, fgbg, frame_w, frame_h, min_area):
    fgmask = fgbg.apply(frame_gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return []
    x, y, w, h = cv2.boundingRect(c)
    return [{
        "x": int(max(0, min(x, frame_w - 1))),
        "y": int(max(0, min(y, frame_h - 1))),
        "w": int(max(1, min(w, frame_w - x))),
        "h": int(max(1, min(h, frame_h - y))),
        "confidence": float(cv2.contourArea(c)),
        "source": "mog2"
    }]


# ---------------------------------------------------------------------------
# Full-video scan
# ---------------------------------------------------------------------------

def scan_video_frames(video_path, hog, args):
    """
    Scan every frame of the video. For each frame, run HOG + MOG2 fusion,
    apply temporal smoothing, and return list of (frame_idx, smoothed_bbox)
    for frames where a person was detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open {video_path}")
        return []

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=16, detectShadows=False
    )

    detected = []
    idx = 0
    prev_selected = None
    miss_streak = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            idx += 1
            continue

        frame_resized = cv2.resize(frame, (args.frame_width, args.frame_height))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        hog_boxes = detect_persons_in_frame(
            gray, hog, args.frame_width, args.frame_height, args.hit_threshold
        )
        hog_filtered = [
            clip_bbox(b, args.frame_width, args.frame_height)
            for b in hog_boxes
            if passes_bbox_quality(
                clip_bbox(b, args.frame_width, args.frame_height),
                args.frame_width, args.frame_height,
                args.min_bbox_area_ratio, args.min_bbox_aspect, args.max_bbox_aspect
            )
        ]

        mog_boxes = detect_persons_mog2(
            gray, fgbg, args.frame_width, args.frame_height, args.mog2_min_area
        )
        mog_filtered = [
            clip_bbox(b, args.frame_width, args.frame_height)
            for b in mog_boxes
            if passes_bbox_quality(
                clip_bbox(b, args.frame_width, args.frame_height),
                args.frame_width, args.frame_height,
                args.min_bbox_area_ratio, args.min_bbox_aspect, args.max_bbox_aspect
            )
        ]

        selected = merge_hog_mog2(
            hog_filtered, mog_filtered,
            prev_selected, args.frame_width, args.frame_height,
            iou_merge_th=0.35
        )

        if selected:
            selected_box = smooth_bbox(prev_selected, selected[0], alpha=0.65)
            selected_box = clip_bbox(selected_box, args.frame_width, args.frame_height)
            prev_selected = selected_box
            miss_streak = 0
            detected.append((idx, selected_box))
        else:
            # Allow up to 2 missed frames before breaking the run (reuse prev_selected)
            if prev_selected is not None and miss_streak < 2:
                detected.append((idx, prev_selected))
                miss_streak += 1
            else:
                prev_selected = None
                miss_streak += 1

        idx += 1

    cap.release()
    return detected


# ---------------------------------------------------------------------------
# Consecutive-sequence selection
# ---------------------------------------------------------------------------

def select_consecutive_sequences(detected_frames, seq_length, num_sequences):
    """
    Select up to `num_sequences` non-overlapping windows of exactly
    `seq_length` CONSECUTIVE frames each.

    A "run" is a maximal stretch of frames with consecutive indices
    (idx, idx+1, idx+2, ...) where the person was detected. We split each
    run into non-overlapping windows of length `seq_length`. All candidate
    windows across all runs are collected, then distributed evenly.

    Consecutive frame indexing is CRITICAL: the 3D conv filter_depth spans
    multiple frames, so learning real motion patterns requires frames at
    natural temporal spacing (~33ms apart at 30fps).
    """
    if len(detected_frames) < seq_length:
        return []

    # 1. Find maximal runs of consecutive frame indices
    runs = []
    current = [detected_frames[0]]
    for i in range(1, len(detected_frames)):
        if detected_frames[i][0] == current[-1][0] + 1:
            current.append(detected_frames[i])
        else:
            if len(current) >= seq_length:
                runs.append(current)
            current = [detected_frames[i]]
    if len(current) >= seq_length:
        runs.append(current)

    if not runs:
        return []

    # 2. Extract all non-overlapping windows from all runs
    all_windows = []
    for run in runs:
        n_windows = len(run) // seq_length
        for w in range(n_windows):
            all_windows.append(run[w * seq_length : (w + 1) * seq_length])

    if not all_windows:
        return []

    # 3. If fewer available than requested, return all
    if len(all_windows) <= num_sequences:
        return all_windows

    # 4. Distribute evenly across the video timeline
    step = len(all_windows) / num_sequences
    chosen = [all_windows[int(i * step)] for i in range(num_sequences)]
    return chosen


# ---------------------------------------------------------------------------
# Crop + resize (aspect-aware padding)
# ---------------------------------------------------------------------------

def crop_and_resize(frame, bbox, target_w, target_h, padding_ratio=0.15):
    """
    Crop around bbox with aspect-aware padding, then resize to
    target_w x target_h. Aspect-aware padding avoids stretching the
    person when the bbox aspect (~1:2 for standing person) differs from
    the target aspect (4:3 for 80x60).
    """
    fh, fw = frame.shape[:2]
    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    # Base padding
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)
    ex1 = bx - pad_x
    ey1 = by - pad_y
    ex2 = bx + bw + pad_x
    ey2 = by + bh + pad_y

    # Aspect-aware expansion: match target aspect ratio before clipping
    ew = ex2 - ex1
    eh = ey2 - ey1
    target_aspect = target_w / float(target_h)
    current_aspect = ew / float(eh) if eh > 0 else target_aspect

    if current_aspect < target_aspect:
        # Too narrow -> expand horizontally
        new_ew = int(eh * target_aspect)
        extra = (new_ew - ew) // 2
        ex1 -= extra
        ex2 += extra
    elif current_aspect > target_aspect:
        # Too wide -> expand vertically
        new_eh = int(ew / target_aspect)
        extra = (new_eh - eh) // 2
        ey1 -= extra
        ey2 += extra

    # Clip to frame boundaries
    cx1 = max(0, ex1)
    cy1 = max(0, ey1)
    cx2 = min(fw, ex2)
    cy2 = min(fh, ey2)

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = frame

    resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized


# ---------------------------------------------------------------------------
# Process one video
# ---------------------------------------------------------------------------

def process_video(video_path, hog, args):
    """
    Returns list of dicts:
      {"frame_idx": int, "bbox": dict, "image": np.array}
    Each entry corresponds to one cropped frame; entries are produced in
    temporal order, grouped into consecutive windows.
    """
    detected = scan_video_frames(video_path, hog, args)
    if not detected:
        return []

    windows = select_consecutive_sequences(
        detected, args.seq_length, args.num_sequences
    )
    if not windows:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    results = []
    for window in windows:
        for frame_idx, bbox in window:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_resized = cv2.resize(frame, (args.frame_width, args.frame_height))
            cropped = crop_and_resize(
                frame_resized, bbox,
                args.crop_width, args.crop_height,
                padding_ratio=args.padding_ratio
            )

            results.append({
                "frame_idx": frame_idx,
                "bbox": {
                    "x": bbox["x"], "y": bbox["y"],
                    "w": bbox["w"], "h": bbox["h"],
                    "confidence": float(bbox.get("confidence", 0.0)),
                    "source": bbox.get("source", "unknown")
                },
                "image": cropped
            })

    cap.release()
    return results


# ---------------------------------------------------------------------------
# Parse video filename -> person, action, scenario
# ---------------------------------------------------------------------------

def parse_video_name(video_name):
    """
    KTH filenames: person01_boxing_d1_uncomp.avi
    Returns (person_id, action, scenario) or None.
    """
    base = os.path.splitext(video_name)[0]
    parts = base.split("_")
    if len(parts) < 3:
        return None

    person_str = parts[0]
    action = parts[1]
    scenario = parts[2]

    try:
        person_id = int(person_str.replace("person", ""))
    except ValueError:
        return None

    return person_id, action, scenario


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_dataset(args):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    metadata = {
        "config": {
            "frame_width": args.frame_width,
            "frame_height": args.frame_height,
            "crop_width": args.crop_width,
            "crop_height": args.crop_height,
            "seq_length": args.seq_length,
            "num_sequences": args.num_sequences,
            "padding_ratio": args.padding_ratio,
        },
        "videos": {}
    }

    total_saved = 0

    for split in ["train", "test"]:
        split_path = os.path.join(args.input_path, split)
        if not os.path.isdir(split_path):
            print(f"WARNING: Missing {split_path}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split}/")
        print(f"{'='*60}")

        for action in sorted(os.listdir(split_path)):
            action_path = os.path.join(split_path, action)
            if not os.path.isdir(action_path):
                continue

            out_dir = os.path.join(args.output_dir, split, action)
            os.makedirs(out_dir, exist_ok=True)

            videos = sorted([
                v for v in os.listdir(action_path)
                if os.path.isfile(os.path.join(action_path, v))
            ])
            print(f"  {split}/{action}/ ({len(videos)} videos)")

            action_saved = 0

            for video_name in videos:
                video_path = os.path.join(action_path, video_name)
                parsed = parse_video_name(video_name)
                if parsed is None:
                    print(f"    WARNING: Cannot parse {video_name}, skipping.")
                    continue

                person_id, action_label, scenario = parsed
                results = process_video(video_path, hog, args)

                if not results:
                    print(f"    WARNING: No sequences extracted from {video_name}")
                    continue

                video_key = f"{split}/{action}/{video_name}"
                video_meta = {
                    "person_id": person_id,
                    "action": action_label,
                    "scenario": scenario,
                    "num_frames_extracted": len(results),
                    "num_sequences": len(results) // args.seq_length,
                    "frames": []
                }

                for r in results:
                    # Zero-padded frame index so lex sort == temporal sort
                    fname = f"person{person_id:02d}_{action_label}_{scenario}_frame{r['frame_idx']:04d}.jpg"
                    out_path = os.path.join(out_dir, fname)

                    cv2.imwrite(out_path, r["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])

                    video_meta["frames"].append({
                        "filename": fname,
                        "frame_idx": r["frame_idx"],
                        "bbox": r["bbox"]
                    })

                metadata["videos"][video_key] = video_meta
                action_saved += len(results)

            total_saved += action_saved
            print(f"    Saved {action_saved} frames for {action}")

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Total frames saved: {total_saved}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metadata: {meta_path}")
    print(f"{'='*60}")


def build_parser():
    p = argparse.ArgumentParser(
        description="KTH frame extractor: crops person + saves consecutive windows"
    )
    p.add_argument("--input_path", type=str,
                   default="/home/mmuntean/kth_organized/",
                   help="Path to kth_organized/ with train/ and test/ subdirs")
    p.add_argument("--output_dir", type=str,
                   default="/home/mmuntean/kth_cropped/",
                   help="Output directory for cropped frames")

    # Sequence selection (replaces num_frames + min_gap)
    p.add_argument("--seq_length", type=int, default=5,
                   help="Number of CONSECUTIVE frames per sequence (= temporal_depth in SNN)")
    p.add_argument("--num_sequences", type=int, default=10,
                   help="Max number of non-overlapping sequences per video")

    # Frame / crop sizes (matched to KTH_3D pipeline: 80x60 input)
    p.add_argument("--frame_width", type=int, default=80,
                   help="Video resize width for detection")
    p.add_argument("--frame_height", type=int, default=60,
                   help="Video resize height for detection")
    p.add_argument("--crop_width", type=int, default=80,
                   help="Output crop width (matches SNN input)")
    p.add_argument("--crop_height", type=int, default=60,
                   help="Output crop height (matches SNN input)")
    p.add_argument("--padding_ratio", type=float, default=0.15,
                   help="Padding around bbox as fraction of bbox size")

    # Detection params (same defaults as extract_bboxes_kth.py)
    p.add_argument("--hit_threshold", type=float, default=-0.75)
    p.add_argument("--mog2_min_area", type=int, default=180)
    p.add_argument("--min_bbox_area_ratio", type=float, default=0.008)
    p.add_argument("--min_bbox_aspect", type=float, default=0.22)
    p.add_argument("--max_bbox_aspect", type=float, default=1.6)

    return p


def main():
    args = build_parser().parse_args()

    print(f"Input path:    {args.input_path}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Sequences:     {args.num_sequences} per video, {args.seq_length} frames each (consecutive)")
    print(f"Detection:     {args.frame_width}x{args.frame_height}")
    print(f"Crop size:     {args.crop_width}x{args.crop_height}")
    print(f"Padding:       {args.padding_ratio} (aspect-aware)")

    os.makedirs(args.output_dir, exist_ok=True)
    process_dataset(args)


if __name__ == "__main__":
    main()