"""
Final test protocol for the 1-layer CSNN + SVM head.
Takes one configuration (either from an OPTUNA study or from CLI),
runs N seeds with CSNN_EVAL_SPLIT=test, and reports mean ± std dev of
the SVM classification rate on test.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

ACC_RE = re.compile(r"classification rate:\s*([0-9]+(?:\.[0-9]+)?)%")


def load_best_params_from_study(db_path, study_name):
    import optuna
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    bp = study.best_params
    print(f"Loaded best params from {study_name} (trial #{study.best_trial.number}, "
          f"val acc {study.best_value:.2f}%)")

    if "filter_size" in bp:
        fh = fw = bp["filter_size"]
    else:
        fh = bp["filter_h"]
        fw = bp["filter_w"]
    return {
        "t_obj":    bp["t_obj"],
        "filter_h": fh,
        "filter_w": fw,
        "filter_t": bp["filter_t"],
        "pool_sx":  bp["pool_sx"],
        "pool_sy":  bp["pool_sy"],
    }


def run_one(args, params, seed):
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS":     str(args.threads_per_run),
        "OPENBLAS_NUM_THREADS": str(args.threads_per_run),
        "MKL_NUM_THREADS":      str(args.threads_per_run),
        "TBB_NUM_THREADS":      str(args.threads_per_run),
    })
    env.update({
        "CSNN_SEED":       str(seed),
        "CSNN_T_OBJ":      str(params["t_obj"]),
        "CSNN_FH":         str(params["filter_h"]),
        "CSNN_FW":         str(params["filter_w"]),
        "CSNN_FT":         str(params["filter_t"]),
        "CSNN_NF":         str(args.num_filters),
        "CSNN_POOL_SX":    str(params["pool_sx"]),
        "CSNN_POOL_SY":    str(params["pool_sy"]),
        "CSNN_POOL_ST":    str(args.pool_st),
        "CSNN_SAMPLER":    args.sampler,
        "CSNN_EVAL_SPLIT": "test",
        "CSNN_DATA":       args.data,
        "CSNN_INPUT_ROOT": args.input_root,
        "CSNN_EPOCHS":     str(args.epochs),
        "CSNN_TAG":        f"test_protocol_{args.sampler}",
    })

    t0 = time.time()
    proc = subprocess.run(
        [args.binary],
        env=env, cwd=args.cwd,
        capture_output=True, text=True, timeout=args.per_seed_timeout,
    )
    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"  [seed {seed}] FAILED rc={proc.returncode} in {dt:.0f}s", file=sys.stderr)
        print(proc.stderr[-1500:], file=sys.stderr)
        return None

    matches = ACC_RE.findall(proc.stdout)
    if not matches:
        print(f"  [seed {seed}] no accuracy line in stdout ({dt:.0f}s)", file=sys.stderr)
        return None

    acc = float(matches[-1])
    print(f"  [seed {seed}] acc={acc:.2f}%  ({dt:.0f}s)")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_root", default="/home/mmuntean/kth_organized_tvt/")
    parser.add_argument("--sampler", choices=["hog", "random"], default="hog")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--seeds_base", type=int, default=42)
    parser.add_argument("--num_filters", type=int, default=16)
    parser.add_argument("--pool_st", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--per_seed_timeout", type=int, default=3600 * 6)


    parser.add_argument("--params_from_study", default=None)
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--t_obj", type=float, default=None)
    parser.add_argument("--filter_h", type=int, default=None)
    parser.add_argument("--filter_w", type=int, default=None)
    parser.add_argument("--filter_t", type=int, default=None)
    parser.add_argument("--pool_sx", type=int, default=None)
    parser.add_argument("--pool_sy", type=int, default=None)

    parser.add_argument("--out_csv", default="test_protocol_results.csv")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--threads_per_run", type=int, default=1)
    args = parser.parse_args()

    args.binary = str(Path(args.binary).resolve())
    if not Path(args.binary).is_file():
        sys.exit(f"Binary not found at {args.binary}. Build it first.")
    if args.cwd is None:
        args.cwd = str(Path(args.binary).parent)

    args.data = str(Path(args.data).resolve())
    if not Path(args.data).is_file():
        sys.exit(f"Data .npy not found at {args.data}.")
    sidecar = Path(args.data).with_suffix(".json")
    if not sidecar.is_file():
        sys.exit(f"Metadata sidecar not found at {sidecar} (expected next to the .npy).")

    args.input_root = str(Path(args.input_root).resolve()) + "/"

    if args.params_from_study:
        sname = args.study_name or f"csnn_1layer_{args.sampler}"
        params = load_best_params_from_study(args.params_from_study, sname)
    else:
        params = {}

    cli_map = {
        "t_obj": args.t_obj, "filter_h": args.filter_h, "filter_w": args.filter_w,
        "filter_t": args.filter_t, "pool_sx": args.pool_sx, "pool_sy": args.pool_sy,
    }
    for k, v in cli_map.items():
        if v is not None:
            params[k] = v

    required = ["t_obj", "filter_h", "filter_w", "filter_t", "pool_sx", "pool_sy"]
    missing = [k for k in required if k not in params]
    if missing:
        sys.exit(f"Missing hyperparams: {missing}. Use --params_from_study or pass them via CLI.")

    print("=" * 60)
    print("Test protocol run")
    print(f"  sampler : {args.sampler}")
    print(f"  seeds   : {[args.seeds_base + i for i in range(args.n_seeds)]}")
    for k in required:
        print(f"  {k:8s} = {params[k]}")
    print("=" * 60)

    seeds = [args.seeds_base + i for i in range(args.n_seeds)]
    accs = []
    if args.n_jobs <= 1:
        for seed in seeds:
            acc = run_one(args, params, seed)
            if acc is not None:
                accs.append((seed, acc))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.n_jobs) as pool:
            futs = {pool.submit(run_one, args, params, s): s for s in seeds}
            for fut in as_completed(futs):
                seed = futs[fut]
                acc = fut.result()
                if acc is not None:
                    accs.append((seed, acc))
        accs.sort(key=lambda x: x[0])

    if not accs:
        sys.exit("All seeds failed.")

    vals = [a for _, a in accs]
    avg = mean(vals)
    std = pstdev(vals) if len(vals) > 1 else 0.0

    print()
    print(f"=== Final test accuracy over {len(vals)} seeds ===")
    print(f"  mean = {avg:.2f}%")
    print(f"  std  = {std:.2f}")
    print(f"  per-seed: {vals}")

    new_file = not Path(args.out_csv).exists()
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["sampler", "t_obj", "filter_h", "filter_w", "filter_t",
                        "pool_sx", "pool_sy", "n_seeds", "mean_acc", "std_acc",
                        "per_seed_accs"])
        w.writerow([
            args.sampler, params["t_obj"], params["filter_h"], params["filter_w"],
            params["filter_t"], params["pool_sx"], params["pool_sy"],
            len(vals), f"{avg:.2f}", f"{std:.2f}",
            "|".join(f"{s}:{a:.2f}" for s, a in accs),
        ])
    print(f"\nAppended to {args.out_csv}")


if __name__ == "__main__":
    main()
