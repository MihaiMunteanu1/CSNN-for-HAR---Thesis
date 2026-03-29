import struct
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path

def read_tensor_file(filename, max_samples=None):
    samples = []

    with open(filename, "rb") as f:

        magic = struct.unpack("<I", f.read(4))[0]
        if magic != 0x234264FF:
            raise ValueError(f"Wrong magic number: {hex(magic)}. No TensorWriter file valid.")

        flag   = struct.unpack("B", f.read(1))[0]
        sparse = bool(flag & 0x1)
        count  = struct.unpack("<I", f.read(4))[0]

        print(f"  File: {filename}")
        print(f"  Sparse: {sparse}")
        print(f"  No tensors: {count}")

        limit = count if max_samples is None else min(count, max_samples)
        for _ in range(limit):
            label_size = struct.unpack("B", f.read(1))[0]
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
                    idx = struct.unpack("<I", idx_bytes)[0]
                    if idx == 0xFFFFFFFF:
                        break
                    val = struct.unpack("<f", f.read(4))[0]
                    data[idx] = val
            else:
                raw = f.read(total * 4)
                data = np.frombuffer(raw, dtype=np.float32).copy()

            tensor = data.reshape(dims)
            samples.append((label, tensor))

    return samples, count


def visualize_feature_maps(filename, layer_name, max_samples=5, save_dir="."):
    print(f"\nReading {filename}...")
    samples, total_count = read_tensor_file(filename, max_samples=max_samples)

    if len(samples) == 0:
        print("File is empty!")
        return

    label0, tensor0 = samples[0]
    print(f"Shape tensor: {tensor0.shape}")
    print(f"{len(samples)} samples from {total_count} total")

    n_samples = len(samples)

    if tensor0.ndim == 1:
        n_filters = tensor0.shape[0]
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2 * n_samples))
        if n_samples == 1:
            axes = [axes]
        for i in range(n_samples):
            label, tensor = samples[i]
            axes[i].bar(range(len(tensor)), tensor, color='steelblue')
            axes[i].set_title(f"Sample {i+1} | Label: {label}", fontsize=9)
            axes[i].set_ylabel("Activare")
        plt.suptitle(f"{layer_name} — Feature Maps (FC)", fontsize=13, fontweight='bold')

    elif tensor0.ndim == 2:
        fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
        if n_samples == 1:
            axes = [axes]
        for i in range(n_samples):
            label, tensor = samples[i]
            im = axes[i].imshow(tensor, cmap='hot', interpolation='nearest')
            axes[i].set_title(f"Label: {label}", fontsize=10)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        plt.suptitle(f"{layer_name} — Feature Maps", fontsize=13, fontweight='bold')

    elif tensor0.ndim == 3:
        shape = tensor0.shape
        n_filters = shape[2]
        get_filter = lambda t, f: t[:, :, f]

        fig, axes = plt.subplots(n_samples, n_filters,
                                 figsize=(3 * n_filters, 3 * n_samples))

        if n_samples == 1 and n_filters == 1:
            axes = [[axes]]
        elif n_samples == 1:
            axes = [axes]
        elif n_filters == 1:
            axes = [[ax] for ax in axes]

        for i in range(n_samples):
            label, tensor = samples[i]
            for f in range(n_filters):
                fmap = get_filter(tensor, f)
                # Normalizare per-sample
                vmin, vmax = fmap.min(), fmap.max()
                if vmax - vmin < 1e-6:
                    vmin, vmax = 0.0, 1.0
                im = axes[i][f].imshow(fmap, cmap='hot', interpolation='nearest',
                                       vmin=vmin, vmax=vmax)
                axes[i][f].axis('off')
                if i == 0:
                    axes[i][f].set_title(f"Filtru {f+1}", fontsize=10, fontweight='bold')
                if f == 0:
                    axes[i][f].set_ylabel(f"Label: {label}", fontsize=9)
                val = float(fmap.mean())
                axes[i][f].text(0.5, 0.5, f"{val:.3f}", ha='center', va='center',
                                transform=axes[i][f].transAxes,
                                color='white', fontsize=7, fontweight='bold')

        plt.suptitle(f"{layer_name} — Feature Maps ({n_filters} filtre)",
                     fontsize=13, fontweight='bold')
    elif tensor0.ndim == 4:
        shape = tensor0.shape
        n_filters = shape[2]
        temporal_depth = shape[3]
        print(f"  Tensor 4D: spatial=({shape[0]}x{shape[1]}), filters={n_filters}, temporal={temporal_depth}")
        print(f"  Agregare temporală: max projection pe axa 3")

        def get_filter_4d(tensor, f):
            return np.max(tensor[:, :, f, :], axis=-1)

        fig, axes = plt.subplots(n_samples, n_filters,
                                 figsize=(max(3 * n_filters, 6), 3 * n_samples))

        if n_samples == 1 and n_filters == 1:
            axes = [[axes]]
        elif n_samples == 1:
            axes = [axes]
        elif n_filters == 1:
            axes = [[ax] for ax in axes]

        for i in range(n_samples):
            label, tensor = samples[i]
            for f in range(n_filters):
                fmap = get_filter_4d(tensor, f)
                vmin, vmax = fmap.min(), fmap.max()
                if vmax - vmin < 1e-6:
                    vmin, vmax = 0.0, 1.0
                im = axes[i][f].imshow(fmap, cmap='hot', interpolation='nearest',
                                       aspect='auto', vmin=vmin, vmax=vmax)
                axes[i][f].axis('off')
                if i == 0:
                    axes[i][f].set_title(f"Filtru {f+1}", fontsize=10, fontweight='bold')
                if f == 0:
                    axes[i][f].set_ylabel(f"Label: {label}", fontsize=9)
                if shape[0] <= 4 and shape[1] <= 4:
                    val = float(fmap.mean())
                    axes[i][f].text(0.5, 0.5, f"{val:.3f}", ha='center', va='center',
                                    transform=axes[i][f].transAxes,
                                    color='white', fontsize=7, fontweight='bold')

        plt.suptitle(f"{layer_name} — Feature Maps ({n_filters} filtre, temporal max proj, T={temporal_depth})",
                     fontsize=13, fontweight='bold')
    else:
        print(f"Unexpected shape : {tensor0.shape} — error!.")
        return

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{layer_name}_feature_maps.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"  Saved in: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize feature maps from TensorWriter output files."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Directory that contains tensor files (default: <project>/csnn-simulator-build)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PNGs (default: <exp-dir>/FeatureMaps)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of samples to render per layer",
    )
    args = parser.parse_args()

    # Resolve paths relative to the repository root, not to a machine-specific absolute path.
    project_root = Path(__file__).resolve().parents[2]
    exp_dir = Path(args.exp_dir).expanduser().resolve() if args.exp_dir else project_root / "csnn-simulator-build"
    save_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else exp_dir / "FeatureMaps"

    print(f"Looking for files in: {exp_dir.resolve()}")
    print("-" * 50)

    layers = [
        ("conv1_test",  "Conv1_test"),
        ("conv2_test",  "Conv2_test"),
        ("fc1_test",    "FC1_test"),
        ("conv1_train", "Conv1_train"),
        ("conv2_train", "Conv2_train"),
        ("fc1_train",   "FC1_train"),
    ]

    os.makedirs(save_dir, exist_ok=True)

    found_any = False
    for filename, layer_name in layers:
        filepath = exp_dir / filename
        if filepath.exists():
            found_any = True
            visualize_feature_maps(
                str(filepath),
                layer_name,
                max_samples=args.max_samples,
                save_dir=str(save_dir)
            )
        else:
            print(f"  [skip] {filepath} — doesn`t exist")

    if not found_any:
        print("\nNo file found!")
        print("Try:")
        print("  1.Adding add_analysis<analysis::SaveOutput>(...) in main.cpp")
        print("  2.Running the experiment")
        print(f" 3.File are in: {exp_dir.resolve()}")


if __name__ == "__main__":
    main()

# python3 ../src/tool/draw_feature_maps.py