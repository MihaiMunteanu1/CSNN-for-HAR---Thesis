"""
Generate a 4-panel figure illustrating latency coding on a KTH frame.

Panels:
  (a) original grayscale frame
  (b) ON-filter response  (positive luminance gradient / increases)
  (c) OFF-filter response (negative luminance gradient / decreases)
  (d) spike-time map t_s — early (red) for strong gradients,
                           late  (blue) for weak gradients,
                           gray  for below-threshold pixels (no spike)

Usage:
  python visualize_latency_coding.py --input path/to/frame.jpg \
      --output figures/latency_coding_kth.jpg

Defaults use a simple Difference-of-Gaussians ON/OFF split (no dependency
on the C++ simulator), which is the classical latency-coding front end
used in Masquelier & Thorpe (2007) style networks.
"""

import argparse
import copy
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def on_off_filters(gray: np.ndarray,
                   sigma_center: float = 1.0,
                   sigma_surround: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Difference-of-Gaussians split into ON (positive) and OFF (negative)."""
    gray_f = gray.astype(np.float32) / 255.0
    center = cv2.GaussianBlur(gray_f, (0, 0), sigma_center)
    surround = cv2.GaussianBlur(gray_f, (0, 0), sigma_surround)
    dog = center - surround
    on = np.clip(dog, 0.0, None)
    off = np.clip(-dog, 0.0, None)
    return on, off


def latency_map(on: np.ndarray,
                off: np.ndarray,
                threshold_ratio: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """
    Latency coding: t_s proportional to 1 / |response|.
    Strong response  -> small t_s (early spike, warm color).
    Weak response    -> large t_s (late spike, cool color).
    Below threshold  -> no spike (masked, drawn gray).

    Returns (ts_normalized_in_[0,1], spike_mask_bool).
    """
    magnitude = np.maximum(on, off)
    max_mag = magnitude.max()
    if max_mag <= 0:
        raise ValueError("Filter response is entirely zero — check input image.")

    threshold = threshold_ratio * max_mag
    spike_mask = magnitude >= threshold

    ts = np.full_like(magnitude, np.nan, dtype=np.float32)
    ts[spike_mask] = 1.0 - (magnitude[spike_mask] / max_mag)  # high mag -> t_s ~ 0
    return ts, spike_mask


def build_figure(gray: np.ndarray,
                 on: np.ndarray,
                 off: np.ndarray,
                 ts: np.ndarray,
                 spike_mask: np.ndarray,
                 output_path: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("(a) Cadru original")

    axes[1].imshow(on, cmap="gray")
    axes[1].set_title("(b) Filtru ON")

    axes[2].imshow(off, cmap="gray")
    axes[2].set_title("(c) Filtru OFF")

    ts_display = np.ma.array(ts, mask=~spike_mask)
    cmap = copy.copy(plt.cm.jet)
    cmap.set_bad(color="#808080")  # gray for no-spike pixels
    im = axes[3].imshow(ts_display, cmap=cmap, norm=Normalize(vmin=0.0, vmax=1.0))
    axes[3].set_title(r"(d) Harta timpilor de spike $t_s$")

    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label(r"$t_s$ (devreme $\rightarrow$ târziu)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="Path to a grayscale KTH frame (.jpg).")
    parser.add_argument("--output", default="figures/latency_coding_kth.jpg",
                        help="Output figure path.")
    parser.add_argument("--sigma-center", type=float, default=1.0)
    parser.add_argument("--sigma-surround", type=float, default=2.0)
    parser.add_argument("--threshold-ratio", type=float, default=0.10,
                        help="Fraction of max response below which pixels stay silent.")
    args = parser.parse_args()

    gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    on, off = on_off_filters(gray, args.sigma_center, args.sigma_surround)
    ts, spike_mask = latency_map(on, off, args.threshold_ratio)
    build_figure(gray, on, off, ts, spike_mask, args.output)


if __name__ == "__main__":
    main()
