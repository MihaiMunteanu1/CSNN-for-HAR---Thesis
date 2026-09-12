"""
Final test protocol for the 2-layer CSNN + SVM head.

Loads the best t_obj2 from a 2-layer OPTUNA study, applies it on top of the
frozen conv1 params (per sampler), and runs N seeds with CSNN_EVAL_SPLIT=test.
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


CONV1_PARAMS = {
    "hog":    {"filter_h": 3, "filter_w": 3, "filter_t": 3, "t_obj1": 0.75},
    "random": {"filter_h": 3, "filter_w": 3, "filter_t": 3, "t_obj1": 0.75},
}


def load_best_t_obj2(db_path, study_name):
    import optuna
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    bp = study.best_params
    print(f"Loaded best params from {study_name} (trial #{study.best_trial.number}, "
          f"val acc {study.best_value:.2f}%)")
    return float(bp["t_obj2"])


def run_one(args, t_obj2, seed):
    c1 = CONV1_PARAMS[args.sampler]
    fh = args.filter_h if args.filter_h is not None else c1["filter_h"]
    fw = args.filter_w if args.filter_w is not None else c1["filter_w"]
    ft = args.filter_t if args.filter_t is not None else c1["filter_t"]
    t1 = args.t_obj1 if args.t_obj1 is not None else c1["t_obj1"]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS":     str(args.threads_per_run),
        "OPENBLAS_NUM_THREADS": str(args.threads_per_run),
        "MKL_NUM_THREADS":      str(args.threads_per_run),
        "TBB_NUM_THREADS":      str(args.threads_per_run),
    })
    env.update({
        "CSNN_SEED":       str(seed),
        "CSNN_T_OBJ":      str(t1),
        "CSNN_T_OBJ2":     str(t_obj2),
        "CSNN_FH":         str(fh),
        "CSNN_FW":         str(fw),
        "CSNN_FT":         str(ft),
        "CSNN_NF":         str(args.num_filters1),
        "CSNN_NF2":        str(args.num_filters2),
        "CSNN_POOL_SX":    str(args.pool_sx),
        "CSNN_POOL_SY":    str(args.pool_sy),
        "CSNN_POOL_ST":    str(args.pool_st),
        "CSNN_SAMPLER":    args.sampler,
        "CSNN_EVAL_SPLIT": "test",
        "CSNN_DATA":       args.data,
        "CSNN_INPUT_ROOT": args.input_root,
        "CSNN_EPOCHS":     str(args.epochs1),
        "CSNN_EPOCHS2":    str(args.epochs2),
        "CSNN_TAG":        f"test_protocol_2layer_{args.sampler}",
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
    parser.add_argument("--num_filters1", type=int, default=16)
    parser.add_argument("--num_filters2", type=int, default=32)
    parser.add_argument("--pool_sx", type=int, default=2)
    parser.add_argument("--pool_sy", type=int, default=2)
    parser.add_argument("--pool_st", type=int, default=2)
    parser.add_argument("--epochs1", type=int, default=100)
    parser.add_argument("--epochs2", type=int, default=80)
    parser.add_argument("--per_seed_timeout", type=int, default=3600 * 8)

    parser.add_argument("--params_from_study", default=None)
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--t_obj2", type=float, default=None)
    parser.add_argument("--t_obj1", type=float, default=None)
    parser.add_argument("--filter_h", type=int, default=None)
    parser.add_argument("--filter_w", type=int, default=None)
    parser.add_argument("--filter_t", type=int, default=None)

    parser.add_argument("--out_csv", default="data/test_protocol_2layer.csv")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--threads_per_run", type=int, default=1)
    args = parser.parse_args()

    args.binary = str(Path(args.binary).resolve())
    if not Path(args.binary).is_file():
        sys.exit(f"Binary not found at {args.binary}.")
    if args.cwd is None:
        args.cwd = str(Path(args.binary).parent)
    args.data = str(Path(args.data).resolve())
    if not Path(args.data).is_file():
        sys.exit(f"Data .npy not found at {args.data}.")
    sidecar = Path(args.data).with_suffix(".json")
    if not sidecar.is_file():
        sys.exit(f"Metadata sidecar not found at {sidecar}.")
    args.input_root = str(Path(args.input_root).resolve()) + "/"

    if args.t_obj2 is not None:
        t_obj2 = args.t_obj2
        print(f"Using CLI-supplied t_obj2 = {t_obj2}")
    elif args.params_from_study:
        sname = args.study_name or f"csnn_2layer_{args.sampler}"
        t_obj2 = load_best_t_obj2(args.params_from_study, sname)
    else:
        sys.exit("Provide either --t_obj2 or --params_from_study + --study_name.")

    c1 = CONV1_PARAMS[args.sampler]
    print("=" * 60)
    print("Test protocol run (2-layer)")
    print(f"  sampler : {args.sampler}")
    print(f"  seeds   : {[args.seeds_base + i for i in range(args.n_seeds)]}")
    _fh = args.filter_h if args.filter_h is not None else c1['filter_h']
    _fw = args.filter_w if args.filter_w is not None else c1['filter_w']
    _ft = args.filter_t if args.filter_t is not None else c1['filter_t']
    _t1 = args.t_obj1 if args.t_obj1 is not None else c1['t_obj1']
    print(f"  conv1   : filter {_fh}x{_fw}x{_ft}, "
          f"t_obj1={_t1}, NF={args.num_filters1}")
    print(f"  conv2   : filter same as conv1, t_obj2={t_obj2}, NF2={args.num_filters2}")
    print(f"  pool    : {args.pool_sx}x{args.pool_sy}x{args.pool_st} (both layers)")
    print("=" * 60)

    seeds = [args.seeds_base + i for i in range(args.n_seeds)]
    accs = []
    if args.n_jobs <= 1:
        for seed in seeds:
            acc = run_one(args, t_obj2, seed)
            if acc is not None:
                accs.append((seed, acc))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.n_jobs) as pool:
            futs = {pool.submit(run_one, args, t_obj2, s): s for s in seeds}
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
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["sampler", "t_obj1", "t_obj2", "filter_h", "filter_w", "filter_t",
                        "nf1", "nf2", "pool_sx", "pool_sy", "pool_st",
                        "n_seeds", "mean_acc", "std_acc", "per_seed_accs"])
        w.writerow([
            args.sampler, _t1, t_obj2, _fh, _fw, _ft,
            args.num_filters1, args.num_filters2, args.pool_sx, args.pool_sy, args.pool_st,
            len(vals), f"{avg:.2f}", f"{std:.2f}",
            "|".join(f"{s}:{a:.2f}" for s, a in accs),
        ])
    print(f"\nAppended to {args.out_csv}")


if __name__ == "__main__":
    main()