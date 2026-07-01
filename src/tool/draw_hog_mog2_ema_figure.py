"""
Figura pentru licenta: Fuziunea HOG+MOG2 si netezirea EMA a bbox-urilor.

Produce o imagine compusa cu doua sectiuni:
  (A) Pipeline-ul de fuziune HOG+MOG2 pe un cadru exemplu:
      1. Cadru original (grayscale)
      2. Detectia HOG (People Detector) cu bbox-uri HOG
      3. Detectia MOG2 (Background Subtraction) cu masca fg si conturul
      4. Bbox-ul fuzionat HOG+MOG2 (media ponderata)

  (B) Graficul EMA: centrul bbox-ului brut vs netezit (smooth_bbox, alpha=0.65)
      pe o secventa de ~50+ cadre dintr-un video.

"""

import argparse
import os
import math

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec

# --- Constante HOG (din extract_bboxes_kth.py) ---
_HOG_SCALE_FACTOR = 3
_HOG_MIN_W = 240
_HOG_MIN_H = 180


# ─────────────────────────────────────────────────────────
# Detectie HOG (identica cu extract_bboxes_kth.py)
# ─────────────────────────────────────────────────────────
def detect_hog(frame_gray, hog_descriptor, fw, fh, hit_threshold=-0.75):
    det_w = max(fw * _HOG_SCALE_FACTOR, _HOG_MIN_W)
    det_h = max(fh * _HOG_SCALE_FACTOR, _HOG_MIN_H)
    det_frame = cv2.resize(frame_gray, (det_w, det_h))
    if det_frame.dtype != np.uint8:
        det_frame = cv2.normalize(det_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    locations, weights = hog_descriptor.detectMultiScale(
        det_frame, hitThreshold=hit_threshold,
        winStride=(12, 12), padding=(4, 4), scale=1.10
    )

    sx = fw / float(det_w)
    sy = fh / float(det_h)

    bboxes = []
    for i, (x, y, w, h) in enumerate(locations):
        bboxes.append({
            "x": int(x * sx), "y": int(y * sy),
            "w": int(w * sx), "h": int(h * sy),
            "confidence": float(weights[i]) if i < len(weights) else 0.0,
            "source": "hog"
        })
    return bboxes


# ─────────────────────────────────────────────────────────
# Detectie MOG2 (identica cu extract_bboxes_kth.py)
# ─────────────────────────────────────────────────────────
def detect_mog2(frame_gray, fgbg, fw, fh, min_area=720):
    fgmask = fgbg.apply(frame_gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], fgmask

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return [], fgmask

    x, y, w, h = cv2.boundingRect(c)
    bbox = {
        "x": int(max(0, min(x, fw - 1))),
        "y": int(max(0, min(y, fh - 1))),
        "w": int(max(1, min(w, fw - x))),
        "h": int(max(1, min(h, fh - y))),
        "confidence": float(cv2.contourArea(c)),
        "source": "mog2"
    }
    return [bbox], fgmask


# ─────────────────────────────────────────────────────────
# Fuziune HOG+MOG2 (simplificata pt vizualizare)
# ─────────────────────────────────────────────────────────
def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"]+a["w"], a["y"]+a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, a["w"] * a["h"])
    area_b = max(1, b["w"] * b["h"])
    return inter / float(area_a + area_b - inter + 1e-9)


def fuse_boxes(hog_boxes, mog_boxes, iou_th=0.35):
    """Merge cel mai bun HOG cu cel mai bun MOG2 daca IoU > threshold."""
    if not hog_boxes and not mog_boxes:
        return None, None, None

    best_hog = max(hog_boxes, key=lambda b: b["confidence"]) if hog_boxes else None
    best_mog = max(mog_boxes, key=lambda b: b["confidence"]) if mog_boxes else None

    if best_hog and best_mog and bbox_iou(best_hog, best_mog) >= iou_th:
        fused = {
            "x": int(round(0.6 * best_hog["x"] + 0.4 * best_mog["x"])),
            "y": int(round(0.6 * best_hog["y"] + 0.4 * best_mog["y"])),
            "w": int(round(0.5 * best_hog["w"] + 0.5 * best_mog["w"])),
            "h": int(round(0.5 * best_hog["h"] + 0.5 * best_mog["h"])),
            "confidence": 0.6 * best_hog["confidence"] + 0.4 * best_mog["confidence"],
            "source": "hog+mog2"
        }
        return best_hog, best_mog, fused
    elif best_hog:
        return best_hog, best_mog, best_hog
    elif best_mog:
        return best_hog, best_mog, best_mog
    return None, None, None


# ─────────────────────────────────────────────────────────
# EMA smoothing (din extract_bboxes_kth.py)
# ─────────────────────────────────────────────────────────
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


def bbox_center(b):
    return b["x"] + b["w"] / 2.0, b["y"] + b["h"] / 2.0


# ─────────────────────────────────────────────────────────
# Gasire automata video
# ─────────────────────────────────────────────────────────
def find_video(dataset_root, action="boxing", split="train"):
    action_dir = os.path.join(dataset_root, split, action)
    if not os.path.isdir(action_dir):
        return None
    videos = sorted([f for f in os.listdir(action_dir) if f.endswith(".avi")])
    if videos:
        return os.path.join(action_dir, videos[0])
    return None


# ─────────────────────────────────────────────────────────
# Desenare utilitare
# ─────────────────────────────────────────────────────────
def draw_bbox_on_ax(ax, bbox, color, lw=2.0, label=None, ls="--"):
    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                            edgecolor=color, linewidth=lw,
                            linestyle=ls, label=label))


def show_gray_frame(ax, img, title, fontsize=10):
    ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.2)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Figura HOG+MOG2 fusion + EMA (licenta)")
    ap.add_argument("--dataset-root", default="C:/Users/munte/Desktop/kth_organized_tvt",
                    help="Root dir KTH organized (train/val/test sub-dirs)")
    ap.add_argument("--video", default="",
                    help="Cale directa la un .avi. Daca gol, se alege automat boxing/train.")
    ap.add_argument("--action", default="boxing")
    ap.add_argument("--split", default="train")
    ap.add_argument("--target-frame", type=int, default=40,
                    help="Cadrul pe care se deseneaza panoul de fuziune (A). "
                         "Alege un cadru cu miscare clara (~30-60).")
    ap.add_argument("--n-ema-frames", type=int, default=80,
                    help="Cate cadre sa proceseze pentru graficul EMA (B).")
    ap.add_argument("--frame-width", type=int, default=160)
    ap.add_argument("--frame-height", type=int, default=120)
    ap.add_argument("--ema-alpha", type=float, default=0.65)
    ap.add_argument("--hit-threshold", type=float, default=-0.75)
    ap.add_argument("--out", default="licenta/figures/hog_mog2_ema_pipeline.png")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    fw, fh = args.frame_width, args.frame_height

    # --- Gasire video ---
    if args.video:
        video_path = args.video
    else:
        video_path = find_video(args.dataset_root, args.action, args.split)
    if not video_path or not os.path.isfile(video_path):
        raise SystemExit(f"Video negasit! Cauta in: {args.dataset_root}")
    print(f"Video: {video_path}")

    # --- Setup detectori ---
    hog_desc = cv2.HOGDescriptor()
    hog_desc.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    fgbg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=16, detectShadows=False)

    # --- Procesare cadre ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Nu pot deschide: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_process = min(args.n_ema_frames, total_frames)
    print(f"Total cadre video: {total_frames}, procesez: {n_process}")

    raw_centers = []   # centrele bbox-ului brut (detectia cadru-cu-cadru)
    ema_centers = []   # centrele bbox-ului netezit EMA
    raw_bboxes_per_frame = []  # (hog_boxes, mog_boxes, fused, gray, fgmask) pt fiecare cadru

    prev_smooth = None

    for idx in range(n_process):
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (fw, fh))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        hog_boxes = detect_hog(gray, hog_desc, fw, fh, args.hit_threshold)
        mog_boxes, fgmask = detect_mog2(gray, fgbg, fw, fh)

        best_hog, best_mog, fused = fuse_boxes(hog_boxes, mog_boxes)

        if fused:
            smoothed = smooth_bbox(prev_smooth, fused, alpha=args.ema_alpha)
            prev_smooth = smoothed

            cx_raw, cy_raw = bbox_center(fused)
            cx_ema, cy_ema = bbox_center(smoothed)

            raw_centers.append((idx, cx_raw, cy_raw))
            ema_centers.append((idx, cx_ema, cy_ema))
        else:
            # Fara detectie pe acest cadru
            if prev_smooth:
                cx_ema, cy_ema = bbox_center(prev_smooth)
                ema_centers.append((idx, cx_ema, cy_ema))
            smoothed = prev_smooth

        raw_bboxes_per_frame.append({
            "idx": idx, "gray": gray.copy(), "fgmask": fgmask.copy(),
            "hog_boxes": hog_boxes, "mog_boxes": mog_boxes,
            "fused": fused, "smoothed": smoothed
        })

    cap.release()
    print(f"Cadre procesate: {len(raw_bboxes_per_frame)}")
    print(f"Detectii brute: {len(raw_centers)}, netezite: {len(ema_centers)}")

    # --- Alegere cadru target (panou A) ---
    tf = args.target_frame
    if tf >= len(raw_bboxes_per_frame):
        tf = len(raw_bboxes_per_frame) // 2
    target = raw_bboxes_per_frame[tf]

    # Daca pe cadrul ales nu s-a detectat nimic, cauta primul cu detectie
    if target["fused"] is None:
        for d in raw_bboxes_per_frame:
            if d["fused"] is not None:
                target = d
                tf = d["idx"]
                break
    print(f"Cadru selectat pt panou A: {tf}")

    # ═══════════════════════════════════════════════
    # CONSTRUIRE FIGURA
    # ═══════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 10), facecolor="#0D1117")

    # Layout: randuri 2, proportie [3 : 2]
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[3, 2],
                                 hspace=0.35, top=0.92, bottom=0.07,
                                 left=0.04, right=0.96)

    # ─── Randul 1: 4 panouri fuziune ───
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0],
                                              wspace=0.12)

    ax_colors = {
        "text": "#E6EDF3",
        "bg": "#0D1117",
        "border": "#30363D",
    }

    def style_ax(ax):
        ax.set_facecolor("#161B22")
        for s in ax.spines.values():
            s.set_color(ax_colors["border"])
            s.set_linewidth(1.5)
        ax.tick_params(colors=ax_colors["text"])

    # Panel 1: Cadru original
    ax1 = fig.add_subplot(gs_top[0])
    style_ax(ax1)
    ax1.imshow(target["gray"], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax1.set_title("(a) Cadru original\n(grayscale)", fontsize=11,
                   fontweight="bold", color=ax_colors["text"], pad=10)
    ax1.set_xticks([]); ax1.set_yticks([])
    # Adauga label cadru
    ax1.text(3, fh - 5, f"frame #{tf}", fontsize=8, color="#58A6FF",
             fontweight="bold", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#0D1117", alpha=0.8, edgecolor="#30363D"))

    # Panel 2: Detectie HOG
    ax2 = fig.add_subplot(gs_top[1])
    style_ax(ax2)
    ax2.imshow(target["gray"], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax2.set_title("(b) Detecție HOG\n(People Detector SVM)", fontsize=11,
                   fontweight="bold", color=ax_colors["text"], pad=10)
    ax2.set_xticks([]); ax2.set_yticks([])

    hog_color = "#FF6B35"  # portocaliu vibrant
    for b in target["hog_boxes"]:
        draw_bbox_on_ax(ax2, b, hog_color, lw=2.5, ls="-")
    if target["hog_boxes"]:
        best_h = max(target["hog_boxes"], key=lambda bb: bb["confidence"])
        ax2.text(best_h["x"], max(8, best_h["y"] - 4),
                 f'HOG (conf={best_h["confidence"]:.2f})',
                 fontsize=7.5, color=hog_color, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="#0D1117", alpha=0.85,
                           edgecolor=hog_color, linewidth=0.8))
    else:
        ax2.text(fw // 2, fh // 2, "Nicio detecție HOG", ha="center", va="center",
                 fontsize=9, color="#FF6B35", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D1117", alpha=0.8))

    # Panel 3: Detectie MOG2 (masca foreground + bbox)
    ax3 = fig.add_subplot(gs_top[2])
    style_ax(ax3)
    # Overlay: cadru gri + masca foreground colorata
    fgmask_vis = target["fgmask"]
    overlay = cv2.cvtColor(target["gray"], cv2.COLOR_GRAY2RGB)
    # Coloreaza foreground-ul cu verde
    fg_colored = np.zeros_like(overlay)
    fg_colored[:, :, 1] = fgmask_vis  # canal verde
    overlay_blend = cv2.addWeighted(overlay, 0.6, fg_colored, 0.5, 0)
    ax3.imshow(overlay_blend, interpolation="nearest")
    ax3.set_title("(c) Detecție MOG2\n(Background Subtraction)", fontsize=11,
                   fontweight="bold", color=ax_colors["text"], pad=10)
    ax3.set_xticks([]); ax3.set_yticks([])

    mog_color = "#3FB950"  # verde
    for b in target["mog_boxes"]:
        draw_bbox_on_ax(ax3, b, mog_color, lw=2.5, ls="-")
    if target["mog_boxes"]:
        best_m = max(target["mog_boxes"], key=lambda bb: bb["confidence"])
        ax3.text(best_m["x"], max(8, best_m["y"] - 4),
                 f'MOG2 (area={int(best_m["confidence"])})',
                 fontsize=7.5, color=mog_color, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="#0D1117", alpha=0.85,
                           edgecolor=mog_color, linewidth=0.8))
    else:
        ax3.text(fw // 2, fh // 2, "Nicio detecție MOG2", ha="center", va="center",
                 fontsize=9, color=mog_color, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D1117", alpha=0.8))

    # Nota: MOG2 foloseste ultimele N cadre pentru modelul de fundal
    ax3.text(3, fh - 5, "history = 120 cadre\nvarThreshold = 16",
             fontsize=7, color="#8B949E", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#0D1117", alpha=0.85,
                       edgecolor="#30363D", linewidth=0.8))

    # Panel 4: Fuziunea HOG+MOG2
    ax4 = fig.add_subplot(gs_top[3])
    style_ax(ax4)
    ax4.imshow(target["gray"], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax4.set_title("(d) Bbox fuzionat\n(HOG+MOG2, medie ponderată)", fontsize=11,
                   fontweight="bold", color=ax_colors["text"], pad=10)
    ax4.set_xticks([]); ax4.set_yticks([])

    fused_color = "#D2A8FF"  # mov deschis
    # Deseneaza ghost-uri (HOG si MOG2) subtile
    if target["hog_boxes"]:
        best_h = max(target["hog_boxes"], key=lambda bb: bb["confidence"])
        draw_bbox_on_ax(ax4, best_h, hog_color, lw=1.0, ls=":")
    if target["mog_boxes"]:
        best_m = max(target["mog_boxes"], key=lambda bb: bb["confidence"])
        draw_bbox_on_ax(ax4, best_m, mog_color, lw=1.0, ls=":")
    # Bbox fuzionat (linie solida, groasa)
    if target["fused"]:
        draw_bbox_on_ax(ax4, target["fused"], fused_color, lw=3.0, ls="-",
                         label="Fuzionat HOG+MOG2")
        fc = target["fused"]
        ax4.text(fc["x"], max(8, fc["y"] - 4),
                 f'FUSED ({fc["source"]})',
                 fontsize=7.5, color=fused_color, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="#0D1117", alpha=0.85,
                           edgecolor=fused_color, linewidth=0.8))
    # Legenda
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=hog_color, lw=1.5, linestyle=":", label="HOG"),
        Line2D([0], [0], color=mog_color, lw=1.5, linestyle=":", label="MOG2"),
        Line2D([0], [0], color=fused_color, lw=3, linestyle="-", label="Fuzionat"),
    ]
    ax4.legend(handles=legend_elems, loc="lower right", fontsize=7.5,
               framealpha=0.9, facecolor="#161B22", edgecolor="#30363D",
               labelcolor=ax_colors["text"])

    # Sageti intre panouri (→) pe background-ul figurii
    for i, (left_ax, right_ax) in enumerate([(ax1, ax2), (ax2, ax3), (ax3, ax4)]):
        left_ax.annotate(
            "", xy=(-0.03, 0.5), xycoords=right_ax.transAxes,
            xytext=(1.03, 0.5), textcoords=left_ax.transAxes,
            arrowprops=dict(
                arrowstyle="->,head_width=0.4,head_length=0.3",
                color="#58A6FF", linewidth=2.0,
                connectionstyle="arc3,rad=0",
            ),
            annotation_clip=False,
        )

    # ─── Randul 2: Graficul EMA ───
    ax_ema = fig.add_subplot(gs_main[1])
    ax_ema.set_facecolor("#161B22")
    for s in ax_ema.spines.values():
        s.set_color(ax_colors["border"])
        s.set_linewidth(1.5)
    ax_ema.tick_params(colors=ax_colors["text"], labelsize=9)

    if raw_centers and ema_centers:
        raw_idx = [r[0] for r in raw_centers]
        raw_cx = [r[1] for r in raw_centers]
        raw_cy = [r[2] for r in raw_centers]

        ema_idx = [e[0] for e in ema_centers]
        ema_cx = [e[1] for e in ema_centers]
        ema_cy = [e[2] for e in ema_centers]

        # Centru X
        ax_ema.plot(raw_idx, raw_cx, color="#FF6B35", alpha=0.5, linewidth=1.2,
                     linestyle="-", marker=".", markersize=3, label="Centru X brut (detecție)")
        ax_ema.plot(ema_idx, ema_cx, color="#FF6B35", alpha=1.0, linewidth=2.5,
                     linestyle="-", label=f"Centru X netezit (EMA α={args.ema_alpha})")

        # Centru Y
        ax_ema.plot(raw_idx, raw_cy, color="#58A6FF", alpha=0.5, linewidth=1.2,
                     linestyle="-", marker=".", markersize=3, label="Centru Y brut (detecție)")
        ax_ema.plot(ema_idx, ema_cy, color="#58A6FF", alpha=1.0, linewidth=2.5,
                     linestyle="-", label=f"Centru Y netezit (EMA α={args.ema_alpha})")

        # Linie verticala la target frame
        ax_ema.axvline(x=tf, color="#D2A8FF", linestyle="--", linewidth=1.5, alpha=0.7,
                        label=f"Cadru selectat (#{tf})")

        # Fill intre raw si ema (arata reducerea zgomotului)
        # Interpoleaza ema pe aceleasi indici ca raw
        raw_idx_set = set(raw_idx)
        common_idx = sorted(raw_idx_set.intersection(set(ema_idx)))
        if len(common_idx) > 2:
            raw_cx_common = [raw_cx[raw_idx.index(i)] for i in common_idx]
            ema_cx_common = [ema_cx[ema_idx.index(i)] for i in common_idx]
            ax_ema.fill_between(common_idx, raw_cx_common, ema_cx_common,
                                 alpha=0.12, color="#FF6B35")
            raw_cy_common = [raw_cy[raw_idx.index(i)] for i in common_idx]
            ema_cy_common = [ema_cy[ema_idx.index(i)] for i in common_idx]
            ax_ema.fill_between(common_idx, raw_cy_common, ema_cy_common,
                                 alpha=0.12, color="#58A6FF")

    ax_ema.set_xlabel("Index cadru", fontsize=11, color=ax_colors["text"], fontweight="bold")
    ax_ema.set_ylabel("Coordonata centru bbox (px)", fontsize=11,
                       color=ax_colors["text"], fontweight="bold")
    ax_ema.set_title("(e) Netezire temporală EMA – centrul bbox-ului brut vs. netezit\n"
                      f"(smooth_bbox, α = {args.ema_alpha})",
                      fontsize=12, fontweight="bold", color=ax_colors["text"], pad=12)
    ax_ema.legend(loc="upper right", fontsize=8, framealpha=0.9,
                   facecolor="#161B22", edgecolor="#30363D",
                   labelcolor=ax_colors["text"], ncol=2)
    ax_ema.grid(True, alpha=0.15, color="#30363D")

    # Adauga formula EMA ca text
    formula = (r"$\hat{b}_t = \alpha \cdot \hat{b}_{t-1} + (1 - \alpha) \cdot b_t$"
               f"\n" + r"$\alpha = " + f"{args.ema_alpha}" + r"$")
    ax_ema.text(0.02, 0.95, formula,
                transform=ax_ema.transAxes, fontsize=11, color="#D2A8FF",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0D1117",
                          alpha=0.9, edgecolor="#D2A8FF", linewidth=1.2))

    # ─── Titlu general ───
    fig.suptitle(
        "Pipeline-ul de detecție a persoanei: Fuziune HOG + MOG2 cu netezire EMA",
        fontsize=15, fontweight="bold", color="#E6EDF3", y=0.97
    )
    # Subtitlu cu video
    fig.text(0.5, 0.935, f"Video: {os.path.basename(video_path)}  |  "
             f"Rezoluție: {fw}×{fh}  |  α = {args.ema_alpha}",
             ha="center", fontsize=9, color="#8B949E", style="italic")

    # ─── Salvare ───
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"\nSalvat: {args.out}")
    print(f"  DPI: {args.dpi}")
    print(f"  Dimensiune: ~{os.path.getsize(args.out) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
