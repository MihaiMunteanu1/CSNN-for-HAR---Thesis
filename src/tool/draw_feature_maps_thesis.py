"""
Vizualizare Feature Maps Conv1 & Conv2 — figura de licenta.

Produce doua figuri premium (dark-mode, stil consistent cu draw_hog_mog2_ema_figure.py):
  1. conv1_feature_maps.png  —  feature maps dupa stratul Conv1 (N filtre)
  2. conv2_feature_maps.png  —  feature maps dupa stratul Conv2 (M filtre)

Fiecare figura arata:
  - Un singur sample (un clip video) cu toate filtrele
  - Max-projection temporala (axa T) pt. tensorii 4D
  - Colormap 'inferno' cu normalizare per-filtru

Citeste fisierele tensor binare produse de SaveOutput (TensorWriter format),
care sunt deja generate in csnn-simulator-build/ (conv1_test, conv2_test, etc.).

Daca fisierele tensor NU exista, scriptul poate rula automat experimentul
KTH_2layer daca se specifica --run-experiment.

Exemplu de rulare:
  python src/tool/draw_feature_maps_thesis.py
  python src/tool/draw_feature_maps_thesis.py --exp-dir csnn-simulator-build --sample-idx 5
  python src/tool/draw_feature_maps_thesis.py --run-experiment --binary cmake-build-release/KTH_2layer
"""

import struct
import numpy as np
import os
import sys
import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ═══════════════════════════════════════════════════
# TensorWriter reader (identic cu draw_feature_maps.py)
# ═══════════════════════════════════════════════════
def read_tensor_file(filename, max_samples=None):
    """Citeste fisierul binar TensorWriter si returneaza lista (label, tensor)."""
    samples = []
    with open(filename, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != 0x234264FF:
            raise ValueError(f"Wrong magic: {hex(magic)}. Not a TensorWriter file.")

        flag   = struct.unpack("B", f.read(1))[0]
        sparse = bool(flag & 0x1)
        count  = struct.unpack("<I", f.read(4))[0]

        limit = count if max_samples is None else min(count, max_samples)
        for _ in range(limit):
            try:
                label_size_raw = f.read(1)
                if len(label_size_raw) < 1:
                    break  # EOF
                label_size = struct.unpack("B", label_size_raw)[0]
                label = f.read(label_size).decode("utf-8")

                dim_number = struct.unpack("B", f.read(1))[0]
                dims = []
                for _ in range(dim_number):
                    dim = struct.unpack("<H", f.read(2))[0]
                    dims.append(dim)

                total = 1
                for d in dims:
                    total *= d

                if sparse:
                    data = np.zeros(total, dtype=np.float32)
                    while True:
                        idx_bytes = f.read(4)
                        if len(idx_bytes) < 4:
                            break
                        idx = struct.unpack("<I", idx_bytes)[0]
                        if idx == 0xFFFFFFFF:
                            break
                        val = struct.unpack("<f", f.read(4))[0]
                        data[idx] = val
                else:
                    expected_bytes = total * 4
                    raw = f.read(expected_bytes)
                    if len(raw) < expected_bytes:
                        break  # truncated file, stop reading
                    data = np.frombuffer(raw, dtype=np.float32).copy()

                tensor = data.reshape(dims)
                samples.append((label, tensor))
            except (struct.error, ValueError, EOFError):
                break  # fisier trunchiat sau corupt, returnam ce avem

    return samples, count


# ═══════════════════════════════════════════════════
# Reducere temporala (max projection pe ultima axa)
# ═══════════════════════════════════════════════════
def temporal_reduce(tensor):
    """
    Daca tensor e 4D (H, W, F, T) -> max projection pe T => (H, W, F).
    Daca e 3D (H, W, F) -> return ca atare.
    Daca e 2D (H, W) -> (H, W, 1).
    """
    if tensor.ndim == 4:
        return np.max(tensor, axis=-1)
    elif tensor.ndim == 3:
        return tensor
    elif tensor.ndim == 2:
        return tensor[:, :, np.newaxis]
    elif tensor.ndim == 1:
        return tensor.reshape(-1, 1, 1)
    else:
        raise ValueError(f"Unexpected tensor ndim={tensor.ndim}")


# ═══════════════════════════════════════════════════
# Design System — tema alba, curata, pentru licenta
# ═══════════════════════════════════════════════════
COLORS = {
    "bg":        "#FFFFFF",
    "panel":     "#FFFFFF",
    "border":    "#D0D0D0",
    "text":      "#1A1A1A",
    "text_dim":  "#666666",
    "accent1":   "#2060B0",   # albastru inchis
    "accent2":   "#1A8A40",   # verde inchis
    "accent3":   "#D04A00",   # portocaliu inchis
    "accent4":   "#7B3FA0",   # mov
    "accent5":   "#C04080",   # roz inchis
}


def style_ax(ax, show_border=True):
    """Aplica stilul pe un ax."""
    ax.set_facecolor(COLORS["panel"])
    for s in ax.spines.values():
        s.set_color(COLORS["border"] if show_border else "none")
        s.set_linewidth(0.8)
    ax.tick_params(colors=COLORS["text"], labelsize=7)


# ═══════════════════════════════════════════════════
# Vizualizare Feature Maps — un singur layer
# ═══════════════════════════════════════════════════
def draw_layer_feature_maps(
    samples, layer_name, sample_idx=0, save_path="feature_maps.png",
    dpi=200, cmap="inferno", layer_label="Conv1", accent_color=None,
):
    """
    Deseneaza o figura premium cu feature maps-urile unui singur sample.
    """
    if accent_color is None:
        accent_color = COLORS["accent1"]

    if sample_idx >= len(samples):
        sample_idx = 0

    label, tensor_raw = samples[sample_idx]
    tensor_3d = temporal_reduce(tensor_raw)
    # tensor_3d: (H, W, F) — F = nr filtre

    n_filters = tensor_3d.shape[2]
    H, W = tensor_3d.shape[0], tensor_3d.shape[1]

    # Calculeaza grid optim
    n_cols = min(8, n_filters)
    n_rows = int(np.ceil(n_filters / n_cols))

    fig_w = max(2.5 * n_cols + 1.5, 10)
    fig_h = max(2.5 * n_rows + 1.0, 5)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=COLORS["bg"])

    # Layout: doar grid feature maps (fara titlu/info)
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        hspace=0.30, wspace=0.12,
        top=0.95, bottom=0.03, left=0.03, right=0.96,
    )

    # ─── Grid de feature maps ───
    for f_idx in range(n_filters):
        row = f_idx // n_cols
        col = f_idx % n_cols

        ax = fig.add_subplot(gs[row, col])
        style_ax(ax, show_border=True)

        fmap = tensor_3d[:, :, f_idx]
        vmin, vmax = fmap.min(), fmap.max()
        if vmax - vmin < 1e-8:
            vmin, vmax = 0.0, max(1.0, vmax)

        im = ax.imshow(
            fmap, cmap=cmap, interpolation="nearest",
            vmin=vmin, vmax=vmax, aspect="auto",
        )

        # Titlu filtru
        ax.set_title(
            f"F{f_idx + 1}", fontsize=8, fontweight="bold",
            color=COLORS["text_dim"], pad=3,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Activitate per filtru (text suprapus)
        mean_val = float(fmap.mean())
        max_val = float(fmap.max())
        nz_pct = 100.0 * np.count_nonzero(fmap) / max(1, fmap.size)

        # Badge cu mean val in colt
        if H <= 8 and W <= 8:
            # Pt filtre mici, afisam valoarea medie
            ax.text(
                0.5, 0.5, f"{mean_val:.3f}",
                transform=ax.transAxes, fontsize=7, color="white",
                ha="center", va="center", fontweight="bold",
                alpha=0.8,
            )
        else:
            # Pt filtre mari, afisam un badge mic
            ax.text(
                0.97, 0.03, f"μ={mean_val:.2f}",
                transform=ax.transAxes, fontsize=6, color="white",
                ha="right", va="bottom", fontweight="bold", alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5),
            )

        # Colorbar subtil — doar pe ultima coloana per rand
        if col == n_cols - 1 or f_idx == n_filters - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="6%", pad=0.04)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=5, colors=COLORS["text_dim"])
            cbar.outline.set_edgecolor(COLORS["border"])

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"  [OK] Salvat: {save_path}  ({os.path.getsize(save_path) / 1024:.0f} KB)")
    plt.close(fig)


# ═══════════════════════════════════════════════════
# Figura comparativa — Conv1 vs Conv2 (side-by-side)
# ═══════════════════════════════════════════════════
def draw_comparison_figure(
    conv1_samples, conv2_samples, sample_idx=0,
    save_path="feature_maps_comparison.png", dpi=200,
):
    """
    Deseneaza o figura comparativa cu Conv1 si Conv2 feature maps
    pentru acelasi sample (daca exista).
    """
    if sample_idx >= len(conv1_samples):
        sample_idx = 0
    if sample_idx >= len(conv2_samples):
        sample_idx = min(sample_idx, len(conv2_samples) - 1)

    label1, t1_raw = conv1_samples[sample_idx]
    label2, t2_raw = conv2_samples[sample_idx]

    t1 = temporal_reduce(t1_raw)
    t2 = temporal_reduce(t2_raw)

    nf1 = t1.shape[2]
    nf2 = t2.shape[2]

    # Selectam primele max 8 filtre per layer
    show_f1 = min(8, nf1)
    show_f2 = min(8, nf2)
    n_cols = max(show_f1, show_f2)

    fig = plt.figure(figsize=(max(2.2 * n_cols, 12), 7), facecolor=COLORS["bg"])

    gs = gridspec.GridSpec(
        2, n_cols,
        height_ratios=[1.0, 1.0],
        hspace=0.35, wspace=0.12,
        top=0.93, bottom=0.04, left=0.05, right=0.96,
    )

    # Randul 1: Conv1
    for f_idx in range(show_f1):
        ax = fig.add_subplot(gs[0, f_idx])
        style_ax(ax)
        fmap = t1[:, :, f_idx]
        vmin, vmax = fmap.min(), fmap.max()
        if vmax - vmin < 1e-8:
            vmin, vmax = 0.0, max(1.0, vmax)

        im = ax.imshow(fmap, cmap="inferno", interpolation="nearest",
                       vmin=vmin, vmax=vmax, aspect="auto")
        title_text = f"F{f_idx + 1}"
        if f_idx == 0:
            title_text = f"Conv1 - F{f_idx + 1}"
        ax.set_title(title_text, fontsize=8, fontweight="bold",
                     color=COLORS["accent1"], pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Randul 2: Conv2
    for f_idx in range(show_f2):
        ax = fig.add_subplot(gs[1, f_idx])
        style_ax(ax)
        fmap = t2[:, :, f_idx]
        vmin, vmax = fmap.min(), fmap.max()
        if vmax - vmin < 1e-8:
            vmin, vmax = 0.0, max(1.0, vmax)

        im = ax.imshow(fmap, cmap="inferno", interpolation="nearest",
                       vmin=vmin, vmax=vmax, aspect="auto")
        title_text = f"F{f_idx + 1}"
        if f_idx == 0:
            title_text = f"Conv2 - F{f_idx + 1}"
        ax.set_title(title_text, fontsize=8, fontweight="bold",
                     color=COLORS["accent2"], pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Label lateral Conv1 / Conv2
    fig.text(0.015, 0.72, "Conv1", fontsize=11, fontweight="bold",
             color=COLORS["accent1"], rotation=90, va="center")
    fig.text(0.015, 0.28, "Conv2", fontsize=11, fontweight="bold",
             color=COLORS["accent2"], rotation=90, va="center")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"  [OK] Salvat: {save_path}  ({os.path.getsize(save_path) / 1024:.0f} KB)")
    plt.close(fig)


# ═══════════════════════════════════════════════════
# Figura activitate per filtru (bar chart)
# ═══════════════════════════════════════════════════
def draw_activity_chart(
    conv1_samples, conv2_samples, sample_idx=0,
    save_path="feature_maps_activity.png", dpi=200,
):
    """
    Bar chart cu activitatea medie per filtru, Conv1 vs Conv2.
    """
    if sample_idx >= len(conv1_samples):
        sample_idx = 0
    if sample_idx >= len(conv2_samples):
        sample_idx = min(sample_idx, len(conv2_samples) - 1)

    label1, t1_raw = conv1_samples[sample_idx]
    label2, t2_raw = conv2_samples[sample_idx]

    t1 = temporal_reduce(t1_raw)
    t2 = temporal_reduce(t2_raw)

    nf1 = t1.shape[2]
    nf2 = t2.shape[2]

    mean1 = [float(t1[:, :, f].mean()) for f in range(nf1)]
    mean2 = [float(t2[:, :, f].mean()) for f in range(nf2)]
    nz1 = [100.0 * np.count_nonzero(t1[:, :, f]) / max(1, t1[:, :, f].size) for f in range(nf1)]
    nz2 = [100.0 * np.count_nonzero(t2[:, :, f]) / max(1, t2[:, :, f].size) for f in range(nf2)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=COLORS["bg"])

    # Plot 1: Mean activare Conv1
    ax = axes[0, 0]
    style_ax(ax)
    bars = ax.bar(range(1, nf1 + 1), mean1, color=COLORS["accent1"], alpha=0.85,
                  edgecolor=COLORS["border"], linewidth=0.5)
    ax.set_title("Conv1 - Activare medie per filtru", fontsize=10,
                 fontweight="bold", color=COLORS["accent1"], pad=8)
    ax.set_xlabel("Filtru", fontsize=9, color=COLORS["text_dim"])
    ax.set_ylabel("Mean activare", fontsize=9, color=COLORS["text_dim"])
    ax.grid(axis="y", alpha=0.15, color=COLORS["border"])

    # Plot 2: Mean activare Conv2
    ax = axes[0, 1]
    style_ax(ax)
    bars = ax.bar(range(1, nf2 + 1), mean2, color=COLORS["accent2"], alpha=0.85,
                  edgecolor=COLORS["border"], linewidth=0.5)
    ax.set_title("Conv2 - Activare medie per filtru", fontsize=10,
                 fontweight="bold", color=COLORS["accent2"], pad=8)
    ax.set_xlabel("Filtru", fontsize=9, color=COLORS["text_dim"])
    ax.set_ylabel("Mean activare", fontsize=9, color=COLORS["text_dim"])
    ax.grid(axis="y", alpha=0.15, color=COLORS["border"])

    # Plot 3: % non-zero Conv1
    ax = axes[1, 0]
    style_ax(ax)
    bars = ax.bar(range(1, nf1 + 1), nz1, color=COLORS["accent3"], alpha=0.85,
                  edgecolor=COLORS["border"], linewidth=0.5)
    ax.set_title("Conv1 - Activitate (% non-zero)", fontsize=10,
                 fontweight="bold", color=COLORS["accent3"], pad=8)
    ax.set_xlabel("Filtru", fontsize=9, color=COLORS["text_dim"])
    ax.set_ylabel("% pixeli non-zero", fontsize=9, color=COLORS["text_dim"])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.15, color=COLORS["border"])

    # Plot 4: % non-zero Conv2
    ax = axes[1, 1]
    style_ax(ax)
    bars = ax.bar(range(1, nf2 + 1), nz2, color=COLORS["accent5"], alpha=0.85,
                  edgecolor=COLORS["border"], linewidth=0.5)
    ax.set_title("Conv2 - Activitate (% non-zero)", fontsize=10,
                 fontweight="bold", color=COLORS["accent5"], pad=8)
    ax.set_xlabel("Filtru", fontsize=9, color=COLORS["text_dim"])
    ax.set_ylabel("% pixeli non-zero", fontsize=9, color=COLORS["text_dim"])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.15, color=COLORS["border"])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"  [OK] Salvat: {save_path}  ({os.path.getsize(save_path) / 1024:.0f} KB)")
    plt.close(fig)


# ═══════════════════════════════════════════════════
# Rulare experiment (optional)
# ═══════════════════════════════════════════════════
def run_experiment_if_needed(binary, exp_dir, args):
    """
    Ruleaza KTH_2layer cu SaveOutput daca fisierele tensor nu exista.
    Necesita ca binarul sa fie compilat cu SaveOutput activat.
    """
    conv1_test = exp_dir / "conv1_test"
    conv2_test = exp_dir / "conv2_test"

    if conv1_test.exists() and conv2_test.exists():
        print("Fisierele tensor exista deja, skip experiment.")
        return True

    if not binary:
        print("EROARE: Fisierele tensor lipsesc si nu s-a specificat --binary.")
        print("  Optiuni:")
        print("    1. Ruleaza experimentul manual cu SaveOutput activat")
        print("    2. Specifica --binary cmake-build-release/KTH_2layer")
        print("    3. Specifica --exp-dir catre un director cu conv1_test/conv2_test")
        return False

    binary_path = Path(binary).resolve()
    if not binary_path.is_file():
        print(f"EROARE: Binarul nu exista: {binary_path}")
        return False

    print(f"\nRulare experiment pentru generare feature maps...")
    print(f"  Binary: {binary_path}")
    print(f"  Output: {exp_dir}")

    env = os.environ.copy()
    env.update({
        "CSNN_SEED":       str(args.seed),
        "CSNN_T_OBJ":      str(args.t_obj1),
        "CSNN_T_OBJ2":     str(args.t_obj2),
        "CSNN_FH":         str(args.fh),
        "CSNN_FW":         str(args.fw),
        "CSNN_FT":         str(args.ft),
        "CSNN_NF":         str(args.nf1),
        "CSNN_NF2":        str(args.nf2),
        "CSNN_SAMPLER":    args.sampler,
        "CSNN_EVAL_SPLIT": "test",
        "CSNN_EPOCHS":     str(args.epochs1),
        "CSNN_EPOCHS2":    str(args.epochs2),
        "CSNN_TAG":        "featuremaps",
    })

    if args.data:
        env["CSNN_DATA"] = str(Path(args.data).resolve())
    if args.input_root:
        env["CSNN_INPUT_ROOT"] = args.input_root

    proc = subprocess.run(
        [str(binary_path)],
        env=env, cwd=str(exp_dir),
        capture_output=False, timeout=3600 * 4,
    )

    if proc.returncode != 0:
        print(f"EROARE: Experimentul a esuat (rc={proc.returncode})")
        return False

    return True


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Vizualizare Feature Maps Conv1 & Conv2 pentru licenta CSNN KTH.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemple:
  # Cu fisierele tensor existente:
  python src/tool/draw_feature_maps_thesis.py

  # Specifica directorul si sample-ul:
  python src/tool/draw_feature_maps_thesis.py --exp-dir csnn-simulator-build --sample-idx 3

  # Ruleaza experimentul daca fisierele lipsesc:
  python src/tool/draw_feature_maps_thesis.py --run-experiment --binary cmake-build-release/KTH_2layer
        """,
    )
    parser.add_argument(
        "--exp-dir", type=str, default=None,
        help="Director cu fisierele tensor conv1_test, conv2_test "
             "(default: <project>/csnn-simulator-build)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Director output pentru figuri (default: licenta/figures/feature_maps)",
    )
    parser.add_argument(
        "--sample-idx", type=int, default=0,
        help="Indexul sample-ului de vizualizat (default: 0 = primul)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=10,
        help="Cate sample-uri sa citeasca din fisier (default: 10)",
    )
    parser.add_argument(
        "--split", choices=["test", "train"], default="test",
        help="Care split sa vizualizeze: test sau train (default: test)",
    )
    parser.add_argument(
        "--cmap", type=str, default="inferno",
        help="Colormap matplotlib (default: inferno). Alte optiuni: hot, viridis, magma",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--no-comparison", action="store_true",
        help="Nu genera figura comparativa Conv1 vs Conv2",
    )
    parser.add_argument(
        "--no-activity", action="store_true",
        help="Nu genera graficul de activitate per filtru",
    )

    # Parametri pentru rulare experiment (daca e necesar)
    exp_group = parser.add_argument_group("Experiment (optional, daca fisierele lipsesc)")
    exp_group.add_argument("--run-experiment", action="store_true",
                           help="Ruleaza experimentul daca fisierele tensor lipsesc")
    exp_group.add_argument("--binary", type=str, default=None,
                           help="Calea catre binarul KTH_2layer")
    exp_group.add_argument("--seed", type=int, default=42)
    exp_group.add_argument("--t-obj1", type=float, default=0.75, dest="t_obj1")
    exp_group.add_argument("--t-obj2", type=float, default=0.75, dest="t_obj2")
    exp_group.add_argument("--fh", type=int, default=3)
    exp_group.add_argument("--fw", type=int, default=3)
    exp_group.add_argument("--ft", type=int, default=3)
    exp_group.add_argument("--nf1", type=int, default=16)
    exp_group.add_argument("--nf2", type=int, default=32)
    exp_group.add_argument("--sampler", choices=["hog", "random"], default="hog")
    exp_group.add_argument("--epochs1", type=int, default=100)
    exp_group.add_argument("--epochs2", type=int, default=80)
    exp_group.add_argument("--data", type=str, default=None)
    exp_group.add_argument("--input-root", type=str, default=None, dest="input_root")

    args = parser.parse_args()

    # ─── Resolve paths ───
    project_root = Path(__file__).resolve().parents[2]
    exp_dir = (
        Path(args.exp_dir).expanduser().resolve()
        if args.exp_dir
        else project_root / "csnn-simulator-build"
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else project_root / "licenta" / "figures" / "feature_maps"
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Feature Maps Licenta — CSNN KTH")
    print("=" * 60)
    print(f"  Experiment dir : {exp_dir}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Split          : {args.split}")
    print(f"  Sample index   : {args.sample_idx}")
    print(f"  Colormap       : {args.cmap}")
    print()

    # ─── Check / Run experiment ───
    suffix = args.split
    conv1_path = exp_dir / f"conv1_{suffix}"
    conv2_path = exp_dir / f"conv2_{suffix}"

    if not conv1_path.exists() or not conv2_path.exists():
        if args.run_experiment:
            ok = run_experiment_if_needed(args.binary, exp_dir, args)
            if not ok:
                sys.exit(1)
        else:
            missing = []
            if not conv1_path.exists():
                missing.append(str(conv1_path))
            if not conv2_path.exists():
                missing.append(str(conv2_path))
            print("EROARE: Fisierele tensor lipsesc:")
            for m in missing:
                print(f"  [X] {m}")
            print()
            print("Solutii:")
            print("  1. Ruleaza cu --run-experiment --binary <cale_KTH_2layer>")
            print("  2. Asigura-te ca experimentul a fost rulat cu SaveOutput")
            print("     (add_analysis<analysis::SaveOutput>(...) in KTH_2layer.cpp)")
            print(f"  3. Specifica alt --exp-dir daca fisierele sunt altundeva")
            sys.exit(1)

    # ─── Citire tensori ───
    print(f"Citire conv1_{suffix}...")
    conv1_samples, conv1_total = read_tensor_file(
        str(conv1_path), max_samples=args.max_samples
    )
    print(f"  >> {len(conv1_samples)} samples din {conv1_total} total")
    if conv1_samples:
        _, t0 = conv1_samples[0]
        print(f"  >> Shape tensor: {t0.shape} (ndim={t0.ndim})")

    print(f"Citire conv2_{suffix}...")
    conv2_samples, conv2_total = read_tensor_file(
        str(conv2_path), max_samples=args.max_samples
    )
    print(f"  >> {len(conv2_samples)} samples din {conv2_total} total")
    if conv2_samples:
        _, t0 = conv2_samples[0]
        print(f"  >> Shape tensor: {t0.shape} (ndim={t0.ndim})")

    print()

    # ─── Generare figuri ───
    # 1. Feature maps Conv1 (individual)
    if conv1_samples:
        print("> Generare Conv1 feature maps...")
        draw_layer_feature_maps(
            conv1_samples,
            layer_name="conv1",
            sample_idx=args.sample_idx,
            save_path=str(out_dir / f"conv1_feature_maps_{suffix}.png"),
            dpi=args.dpi,
            cmap=args.cmap,
            layer_label=f"Conv1 ({suffix})",
            accent_color=COLORS["accent1"],
        )

    # 2. Feature maps Conv2 (individual)
    if conv2_samples:
        print("> Generare Conv2 feature maps...")
        draw_layer_feature_maps(
            conv2_samples,
            layer_name="conv2",
            sample_idx=args.sample_idx,
            save_path=str(out_dir / f"conv2_feature_maps_{suffix}.png"),
            dpi=args.dpi,
            cmap=args.cmap,
            layer_label=f"Conv2 ({suffix})",
            accent_color=COLORS["accent2"],
        )

    # 3. Comparatie Conv1 vs Conv2
    if not args.no_comparison and conv1_samples and conv2_samples:
        print("> Generare comparatie Conv1 vs Conv2...")
        draw_comparison_figure(
            conv1_samples, conv2_samples,
            sample_idx=args.sample_idx,
            save_path=str(out_dir / f"conv1_vs_conv2_{suffix}.png"),
            dpi=args.dpi,
        )

    # 4. Activitate per filtru
    if not args.no_activity and conv1_samples and conv2_samples:
        print("> Generare grafic activitate per filtru...")
        draw_activity_chart(
            conv1_samples, conv2_samples,
            sample_idx=args.sample_idx,
            save_path=str(out_dir / f"activitate_per_filtru_{suffix}.png"),
            dpi=args.dpi,
        )

    print()
    print("=" * 60)
    print(f"  Toate figurile au fost salvate in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
