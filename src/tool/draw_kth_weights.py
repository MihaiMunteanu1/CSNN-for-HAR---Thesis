import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path


def find_latest_kth_experiment(weights_dir):
    kth_dirs = [d.name for d in weights_dir.iterdir() if d.is_dir() and d.name.startswith("kth_")]
    if not kth_dirs:
        return None
    kth_dirs.sort(
        key=lambda x: int(x.split("_")[1]) if len(x.split("_")) > 1 and x.split("_")[1].isdigit() else -1
    )
    return kth_dirs[-1]


def main():
    parser = argparse.ArgumentParser(description="Draw KTH weight maps from JSON kernels.")
    parser.add_argument(
        "exp_name",
        nargs="?",
        default=None,
        help="Experiment folder name (example: kth_12). If omitted, picks latest kth_*.",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Weights directory (default: <project>/csnn-simulator-build/Weights)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <weights-dir>/<exp_name>_images)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    base_weights_dir = (
        Path(args.weights_dir).expanduser().resolve()
        if args.weights_dir
        else (project_root / "csnn-simulator-build" / "Weights")
    )

    print(f"Using base weights directory: {base_weights_dir}")

    if not base_weights_dir.exists():
        print(f"Weights directory not found: {base_weights_dir}")
        raise SystemExit(1)

    exp_name = args.exp_name or find_latest_kth_experiment(base_weights_dir)
    if not exp_name:
        print("No kth_ directory found in Weights.")
        raise SystemExit(1)

    exp_dir = base_weights_dir / exp_name
    print(f"Reading from experiment: {exp_dir}")

    if not exp_dir.exists():
        print(f"Experiment {exp_name} not found at {exp_dir}!")
        raise SystemExit(1)

    output_base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (base_weights_dir / f"{exp_name}_images")
    )
    os.makedirs(output_base, exist_ok=True)

    subdirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    if not subdirs:
        subdirs = [exp_dir]

    for subdir in subdirs:
        json_files = [f.name for f in subdir.iterdir() if f.is_file() and f.suffix == ".json"]
        if not json_files:
            continue

        subdir_name = subdir.name
        print(f"\nFiles found in: {subdir_name}: {json_files}")

        for json_file in json_files:
            layer_name = json_file.replace(".json", "")
            json_path = subdir / json_file
            output_path = output_base / subdir_name / layer_name
            os.makedirs(output_path, exist_ok=True)

            print(f"-> Reading {json_file}...")

            try:
                with open(json_path, "r") as s1:
                    data = s1.read().strip()
                    if data.endswith(","):
                        data = data[:-1]
                    if not data.startswith("["):
                        data = "[" + data + "]"
                kernels = json.loads(data)
            except Exception as e:
                print(f"Error at {json_file}: {e}")
                continue

            straturi_unice = {}
            for kernel in kernels:
                label = kernel.get("label", "unknown")
                straturi_unice[label] = kernel

            for label, kernel in straturi_unice.items():
                na = np.array(kernel["data"])

                dim_0 = kernel.get("dim_0", 1)  # width
                dim_1 = kernel.get("dim_1", 1)  # height
                dim_2 = kernel.get("dim_2", 1)  # channels (1 or 2 - ON/OFF)
                dim_3 = kernel.get("dim_3", 1)  # filter nr
                dim_4 = kernel.get("dim_4", 1)  # time

                has_time = "dim_4" in kernel

                try:
                    if has_time:
                        draw_kernel = na.reshape((dim_3, dim_0, dim_1, dim_2, dim_4))
                    else:
                        draw_kernel = na.reshape((dim_3, dim_0, dim_1, dim_2))
                except ValueError as e:
                    print(
                        f"Reshape error for {label} ({dim_3}x{dim_0}x{dim_1}x{dim_2}"
                        + (f"x{dim_4}" if has_time else "")
                        + f") array dim={na.size}: {e}"
                    )
                    continue

                for filter_number in range(dim_3):
                    for channel in range(dim_2):
                        for time_step in range(dim_4):
                            if has_time:
                                image_array = draw_kernel[filter_number, :, :, channel, time_step]
                            else:
                                image_array = draw_kernel[filter_number, :, :, channel]

                            _min, _max = np.min(image_array), np.max(image_array)
                            if _max - _min != 0:
                                image_array = (image_array - _min) / (_max - _min)
                            elif _max != 0:
                                image_array = image_array / _max

                            final_name = f"{layer_name}_filtru_{filter_number}_ch_{channel}"
                            if has_time:
                                final_name += f"_t_{time_step}"
                            final_name += ".png"

                            # ON: blue, OFF: red, other channels: green
                            if channel == 0:
                                cm = "Blues"
                            elif channel == 1:
                                cm = "Reds"
                            else:
                                cm = "Greens"

                            plt.imshow(image_array, interpolation="none", cmap=cm)
                            plt.title(
                                f"Filter {filter_number}  |  Chanel {channel}"
                                + (f"  |  Timp {time_step}" if has_time else "")
                            )
                            plt.colorbar(label="Ponderea (Weight)")
                            plt.savefig(output_path / final_name)
                            plt.clf()

                            if not has_time:
                                break

        print(f"Saved in: {output_base / subdir_name}/")

    print("\nSucces!")
    print(f"Results in: {output_base}")


if __name__ == "__main__":
    main()

