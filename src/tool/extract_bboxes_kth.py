"""
KTH person detection -> JSON groups + preview images.
"""

import argparse
import json
import os
import cv2
import numpy as np
import math

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


def select_best_bbox(bboxes):
    if not bboxes:
        return None
    return max(bboxes, key=lambda bb: (float(bb.get("confidence", 0.0)), bb["w"] * bb["h"]))


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
    # score = confidence + size prior + temporal consistency
    conf = float(box.get("_norm_conf", 0.5))
    area_ratio = (box["w"] * box["h"]) / float(frame_w * frame_h + 1e-9)

    # person prior: penalize too tiny/too huge
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
    src_bias = 0.10 if src == "hog" else 0.0  # slight preference for HOG when comparable

    return conf + area_term + temporal_term + src_bias

def merge_hog_mog2(hog_boxes, mog_boxes, prev_box, frame_w, frame_h, iou_merge_th=0.35):
    # Normalize confidence per source
    hog_n = normalize_confidences(hog_boxes)
    mog_n = normalize_confidences(mog_boxes)

    # If both exist, try pair-merge by IoU
    fused = []
    used_m = set()
    for h in hog_n:
        best_j = -1
        best_iou = 0.0
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
            # Weighted merge (favor HOG slightly in position, keep larger size tendency)
            fx = int(round(0.6 * h["x"] + 0.4 * m["x"]))
            fy = int(round(0.6 * h["y"] + 0.4 * m["y"]))
            fw = int(round(0.5 * h["w"] + 0.5 * m["w"]))
            fh = int(round(0.5 * h["h"] + 0.5 * m["h"]))
            merged = {
                "x": fx, "y": fy, "w": fw, "h": fh,
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

    best = None
    best_score = -1e18
    for b in fused:
        s = score_box(b, prev_box, frame_w, frame_h)
        if s > best_score:
            best_score = s
            best = b

    return [best] if best is not None else []

def smooth_bbox(prev_box, cur_box, alpha=0.65):
    """
    EMA smoothing for stable temporal boxes
    """
    if prev_box is None:
        return cur_box
    out = {
        "x": int(round(alpha * prev_box["x"] + (1.0 - alpha) * cur_box["x"])),
        "y": int(round(alpha * prev_box["y"] + (1.0 - alpha) * cur_box["y"])),
        "w": int(round(alpha * prev_box["w"] + (1.0 - alpha) * cur_box["w"])),
        "h": int(round(alpha * prev_box["h"] + (1.0 - alpha) * cur_box["h"])),
        "confidence": float(cur_box.get("confidence", 0.0)),
        "source": cur_box.get("source", "fused")
    }
    return out

def detect_persons_in_frame(frame_gray, hog, frame_width, frame_height, hit_threshold):
    det_w = max(frame_width * _HOG_SCALE_FACTOR, _HOG_MIN_W)
    det_h = max(frame_height * _HOG_SCALE_FACTOR, _HOG_MIN_H)

    det_frame = cv2.resize(frame_gray, (det_w, det_h))
    if det_frame.dtype != np.uint8:
        det_frame = cv2.normalize(det_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    found_locations, weights = hog.detectMultiScale(
        det_frame,
        hitThreshold=hit_threshold,
        winStride=(12, 12),
        padding=(4, 4),
        scale=1.10
    )

    sx = frame_width / float(det_w)
    sy = frame_height / float(det_h)

    bboxes = []
    for i, (x, y, w, h) in enumerate(found_locations):
        bboxes.append({
            "x": int(x * sx),
            "y": int(y * sy),
            "w": int(w * sx),
            "h": int(h * sy),
            "confidence": float(weights[i]) if i < len(weights) else 0.0,
            "source": "hog"
        })
    return bboxes


def detect_persons_mog2(frame_gray, fgbg, frame_width, frame_height, min_area):
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
        "x": int(max(0, min(x, frame_width - 1))),
        "y": int(max(0, min(y, frame_height - 1))),
        "w": int(max(1, min(w, frame_width - x))),
        "h": int(max(1, min(h, frame_height - y))),
        "confidence": float(cv2.contourArea(c)),
        "source": "mog2"
    }]


def scan_video_frames(video_path, hog, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open {video_path}")
        return [], False

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=16, detectShadows=False
    )

    all_frames = []
    idx = 0
    used_any_mog2 = False
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

        # HOG candidates
        hog_boxes = detect_persons_in_frame(
            gray, hog, args.frame_width, args.frame_height, args.hit_threshold
        )
        hog_filtered = []
        for b in hog_boxes:
            bb = clip_bbox(b, args.frame_width, args.frame_height)
            if passes_bbox_quality(
                    bb, args.frame_width, args.frame_height,
                    args.min_bbox_area_ratio, args.min_bbox_aspect, args.max_bbox_aspect
            ):
                hog_filtered.append(bb)

        # MOG2 candidates
        mog_boxes = detect_persons_mog2(
            gray, fgbg, args.frame_width, args.frame_height, args.mog2_min_area
        )
        mog_filtered = []
        for b in mog_boxes:
            bb = clip_bbox(b, args.frame_width, args.frame_height)
            if passes_bbox_quality(
                    bb, args.frame_width, args.frame_height,
                    args.min_bbox_area_ratio, args.min_bbox_aspect, args.max_bbox_aspect
            ):
                mog_filtered.append(bb)

        # Fusion
        selected = merge_hog_mog2(
            hog_filtered,
            mog_filtered if args.mog2_fallback else [],
            prev_selected,
            args.frame_width,
            args.frame_height,
            iou_merge_th=0.35
        )

        if selected:
            # smooth single selected box
            selected_box = smooth_bbox(prev_selected, selected[0], alpha=0.65)
            selected_box = clip_bbox(selected_box, args.frame_width, args.frame_height)
            bboxes = [selected_box]
            prev_selected = selected_box
            miss_streak = 0
            if selected_box.get("source", "").find("mog2") != -1:
                used_any_mog2 = True
        else:
            # short temporal carry
            if prev_selected is not None and miss_streak < args.max_carry:
                bboxes = [prev_selected]
                miss_streak += 1
            else:
                bboxes = []
                prev_selected = None
                miss_streak += 1

        all_frames.append((idx, bboxes))
        idx += 1

    cap.release()
    return all_frames, used_any_mog2

def select_centered_groups(all_frames, args):
    total = len(all_frames)
    if total == 0:
        return []

    if args.temporal_kernel < 3 or args.temporal_kernel % 2 == 0:
        raise ValueError("temporal_kernel must be odd and >= 3 (e.g. 3,5,7).")

    g = args.frame_gap
    half = args.temporal_kernel // 2

    offsets = [o * g for o in range(-half, half + 1)]

    candidates = []

    for i in range(total):
        idxs = [i + off for off in offsets]
        if idxs[0] < 0 or idxs[-1] >= total:
            continue

        group = []
        ok = True
        center_conf = 0.0

        for pos, fi in enumerate(idxs):
            frame_idx, bboxes = all_frames[fi]

            filtered = []
            for b in bboxes:
                bb = clip_bbox(b, args.frame_width, args.frame_height)
                if passes_bbox_quality(
                        bb, args.frame_width, args.frame_height,
                        args.min_bbox_area_ratio, args.min_bbox_aspect, args.max_bbox_aspect
                ):
                    filtered.append(bb)

            best = select_best_bbox(filtered)
            if best is None:
                ok = False
                break

            if pos == half:
                center_conf = float(best.get("confidence", 0.0))

            group.append({
                "frame_idx": frame_idx,
                "bboxes": filtered,
                "selected_bbox": best,
                "is_center": (pos == half)
            })

        if ok:
            candidates.append((center_conf, i, group))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    blocked = set()

    for conf, center_i, grp in candidates:
        if center_i in blocked:
            continue
        selected.append(grp)

        for off in range(-half * g, half * g + 1):
            blocked.add(center_i + off)

        if len(selected) >= args.num_groups:
            break

    if len(selected) < args.num_groups:
        used_centers = {g_[half]["frame_idx"] for g_ in selected}
        for conf, center_i, grp in candidates:
            cf = grp[half]["frame_idx"]
            if cf in used_centers:
                continue
            selected.append(grp)
            used_centers.add(cf)
            if len(selected) >= args.num_groups:
                break

    selected.sort(key=lambda g_: g_[half]["frame_idx"])
    return selected


def save_preview_images(video_path, groups, action, video_name, out_dir, frame_width, frame_height, num_previews=3):
    os.makedirs(os.path.join(out_dir, action), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    saved = 0
    for group_idx, group in enumerate(groups):
        if saved >= num_previews:
            break

        frames_bgr = []
        for frame_data in group:
            fid = frame_data["frame_idx"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()

            if not ret or frame is None:
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (frame_width, frame_height))

            # prefer selected_bbox
            boxes = [frame_data["selected_bbox"]] if frame_data.get("selected_bbox") else frame_data.get("bboxes", [])
            for bbox in boxes:
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                src = bbox.get("source", "hog")
                color = (255, 80, 0) if src == "hog" else (0, 200, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, src, (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            label = f"f#{fid}" + (" [C]" if frame_data.get("is_center") else "")
            cv2.putText(frame, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            frames_bgr.append(frame)

        if not frames_bgr:
            continue

        strip = np.hstack(frames_bgr)
        bar = np.zeros((20, strip.shape[1], 3), dtype=np.uint8)
        txt = f"{action} | {video_name} | group {group_idx}"
        cv2.putText(bar, txt, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        preview = np.vstack([bar, strip])

        out_path = os.path.join(out_dir, action, f"{os.path.splitext(video_name)[0]}_group{group_idx}.jpg")
        cv2.imwrite(out_path, preview)
        saved += 1

    cap.release()


def process_dataset(input_path, split, hog, args):
    split_path = os.path.join(input_path, split)
    if not os.path.isdir(split_path):
        print(f"WARNING: Missing {split_path}, skipping.")
        return {}

    results = {}
    for action in sorted(os.listdir(split_path)):
        if args.only_action and action != args.only_action:
            continue
        action_path = os.path.join(split_path, action)
        if not os.path.isdir(action_path):
            continue

        videos = sorted(os.listdir(action_path))
        print(f"  Processing {split}/{action}/ ({len(videos)} videos)")
        previews_saved_for_action = 0

        for video_name in videos:
            video_path = os.path.join(action_path, video_name)
            if not os.path.isfile(video_path):
                continue

            rel_key = f"{split}/{action}/{video_name}"
            all_frames, used_fallback = scan_video_frames(video_path, hog, args)
            if not all_frames:
                continue

            groups = select_centered_groups(all_frames, args)
            if not groups:
                continue

            detection_frames = sum(1 for _, b in all_frames if b)
            results[rel_key] = {
                "total_video_frames": len(all_frames),
                "detection_frames": detection_frames,
                "num_groups": len(groups),
                "detector": "mog2_fallback" if used_fallback else "hog",
                "groups": groups
            }


        action_videos = len([k for k in results if k.startswith(f"{split}/{action}/")])
        print(f"    Done: {len(videos)} videos, {action_videos} with valid groups")

    return results


def build_parser():
    p = argparse.ArgumentParser("KTH HOG bbox extractor")

    p.add_argument(
        "--input_path",
        type=str,
        default="",
        help="Path to organized KTH dataset root (must contain train/, val/, test/)"
    )
    p.add_argument("--temporal_kernel", type=int, default=19)
    p.add_argument("--num_groups", type=int, default=10)
    p.add_argument("--frame_gap", type=int, default=2)

    p.add_argument("--frame_width", type=int, default=160)
    p.add_argument("--frame_height", type=int, default=120)

    p.add_argument("--hit_threshold", type=float, default=-0.75)
    p.add_argument("--mog2_fallback", action="store_true", default=True)
    p.add_argument("--mog2_min_area", type=int, default=720)

    p.add_argument("--max_carry", type=int, default=2)
    p.add_argument("--min_bbox_area_ratio", type=float, default=0.008) #0.012 was before
    p.add_argument("--min_bbox_aspect", type=float, default=0.22) #0.2 before
    p.add_argument("--max_bbox_aspect", type=float, default=1.6) #1.4 before

    p.add_argument("--only_action", type=str, default="")
    p.add_argument("--merge_into", type=str, default="")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--preview_dir", type=str, default="")
    p.add_argument("--preview_videos_per_action", type=int, default=1)
    p.add_argument("--preview_groups_per_video", type=int, default=1)

    return p


def main():
    args = build_parser().parse_args()

    if args.temporal_kernel < 3 or args.temporal_kernel % 2 == 0:
        raise ValueError("temporal_kernel must be odd and >= 3")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    if not args.input_path:
        args.input_path = "/home/mmuntean/kth_organized_tvt/"

    if not args.output:
        suffix = ("_only_" + args.only_action) if args.only_action else ""
        args.output = "hog/hog_person_data_tvt_elassal_" + str(args.temporal_kernel) + suffix + ".json"
    if not args.preview_dir:
        args.preview_dir = "../hog/hog_previews_tvt_" + str(args.temporal_kernel)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.preview_dir, exist_ok=True)

    print(f"Input path:      {args.input_path}")
    print(f"Output JSON:     {args.output}")
    print(f"Temporal kernel: {args.temporal_kernel}")
    print(f"Groups/video:    {args.num_groups}")
    print(f"Frame gap:       {args.frame_gap}")
    print(f"Frame size:      {args.frame_width}x{args.frame_height}")
    if args.only_action:
        print(f"Only action:     {args.only_action}")
    if args.merge_into:
        print(f"Merge into:      {args.merge_into}")
    print(f"Detection res:   {args.frame_width * _HOG_SCALE_FACTOR}x{args.frame_height * _HOG_SCALE_FACTOR}")
    print(f"MOG2 fallback:   {args.mog2_fallback}")
    print(f"Preview dir:     {args.preview_dir if args.preview_dir else '(disabled)'}")
    print()

    output_data = {
        "config": {
            "temporal_kernel": args.temporal_kernel,
            "num_groups": args.num_groups,
            "frame_gap": args.frame_gap,
            "frame_width": args.frame_width,
            "frame_height": args.frame_height
        },
        "videos": {}
    }

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(args.input_path, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping '{split}/' (folder missing under {args.input_path})")
            continue
        print(f"Processing {split}/...")
        split_results = process_dataset(args.input_path, split, hog, args)
        output_data["videos"].update(split_results)

    if args.merge_into:
        with open(args.merge_into, "r", encoding="utf-8") as f:
            base = json.load(f)
        base_videos = base.get("videos", {})

        new_actions = {k.split("/")[1] for k in output_data["videos"] if len(k.split("/")) >= 2}
        if args.only_action:
            new_actions.add(args.only_action)

        removed = [k for k in base_videos if len(k.split("/")) >= 2 and k.split("/")[1] in new_actions]
        for k in removed:
            del base_videos[k]
        base_videos.update(output_data["videos"])
        base["videos"] = base_videos

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2)

        print(f"\nMerge done for action(s): {sorted(new_actions)}")
        print(f"  removed {len(removed)} old keys, added {len(output_data['videos'])} new keys")
        print(f"  total videos in merged JSON: {len(base_videos)}")
        print(f"Saved merged JSON: {args.output}")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nDone. Videos with valid groups: {len(output_data['videos'])}")
        print(f"Saved JSON: {args.output}")


if __name__ == "__main__":
    main()