"""
OPTUNA hyperparameter tuning driver for the 1-layer CSNN + SVM head on KTH.

Search space (per the licență protocol):
    t_obj          : 0.40, 0.45, ..., 0.80
    filter_size    : 3, 5, 7, 9
    filter_t       : 2, 3

Each trial:
    1. samples one configuration from the search space
    2. runs the C++ binary (apps/kth/KTH_1layer) for N_SEEDS_VAL seeds with
       CSNN_EVAL_SPLIT=val
    3. parses 'classification rate: XX.XX%' from each subprocess stdout
    4. returns mean across seeds as the objective (maximize)

Storage:
    SQLite at optuna_studies/csnn_kth.db. Studies are uniquely named per
    sampler kind (hog / random) so OPTUNA picks up where it left off if
    you re-launch the script.

Usage:
    python3 optuna/tune_csnn_optuna.py \\
        --binary csnn-simulator-build-roazhon4/KTH_1layer \\
        --data ../hog/kth_fullframes_tvt_g4.npy \\
        --input_root /home/mmuntean/kth_organized_tvt/ \\
        --sampler hog \\
        --n_trials 50 \\
        --n_seeds 5 \\
        --epochs 100
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

import optuna

T_OBJ_CHOICES   = [round(0.40 + 0.05 * i, 2) for i in range(9)]   # 0.40..0.80
FILTER_CHOICES  = [3, 5, 7, 9]
TEMPORAL_DEPTHS = [2, 3]
POOL_CHOICES    = [2] #, 3] #pool sx,sy

ACC_RE = re.compile(r"classification rate:\s*([0-9]+(?:\.[0-9]+)?)%")


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
        "CSNN_EVAL_SPLIT": "val",
        "CSNN_DATA":       args.data,
        "CSNN_INPUT_ROOT": args.input_root,
        "CSNN_EPOCHS":     str(args.epochs),
        "CSNN_TAG":        f"trial_{params.get('_trial_no', 'NA')}",
    })

    t0 = time.time()
    proc = subprocess.run(
        [args.binary],
        env=env, cwd=args.cwd,
        capture_output=True, text=True, timeout=args.per_seed_timeout,
    )
    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"    [seed {seed}] FAILED rc={proc.returncode} in {dt:.0f}s", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return None

    matches = ACC_RE.findall(proc.stdout)
    if not matches:
        print(f"    [seed {seed}] no accuracy line in stdout (run took {dt:.0f}s)", file=sys.stderr)
        print(proc.stdout[-1500:], file=sys.stderr)
        return None

    acc = float(matches[-1])
    print(f"    [seed {seed}] acc={acc:.2f}%  ({dt:.0f}s)")
    return acc


def make_objective(args):
    def objective(trial: optuna.Trial):
        filter_size = trial.suggest_categorical("filter_size", FILTER_CHOICES)
        params = {
            "t_obj":      trial.suggest_categorical("t_obj",      T_OBJ_CHOICES),
            "filter_h":   filter_size,
            "filter_w":   filter_size,
            "filter_t":   trial.suggest_categorical("filter_t",   TEMPORAL_DEPTHS),
            "pool_sx":    trial.suggest_categorical("pool_sx",    POOL_CHOICES),
            "pool_sy":    trial.suggest_categorical("pool_sy",    POOL_CHOICES),
            "_trial_no":  trial.number,
        }
        print(f"\n[trial {trial.number}] params={ {k:v for k,v in params.items() if not k.startswith('_')} }")

        accs = []
        for seed in args.seeds:
            acc = run_one(args, params, seed)
            if acc is None:
                raise optuna.TrialPruned()
            accs.append(acc)

            trial.report(mean(accs), step=len(accs))
            if trial.should_prune():
                raise optuna.TrialPruned()

        avg = mean(accs)
        std = pstdev(accs) if len(accs) > 1 else 0.0
        print(f"[trial {trial.number}] mean={avg:.2f}%  std={std:.2f}")

        trial.set_user_attr("seeds", args.seeds)
        trial.set_user_attr("acc_per_seed", accs)
        trial.set_user_attr("acc_std", std)
        return avg
    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_root", default="/home/mmuntean/kth_organized_tvt/")
    parser.add_argument("--sampler", choices=["hog", "random"], default="hog")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--n_seeds",  type=int, default=5)
    parser.add_argument("--seeds_base", type=int, default=42)
    parser.add_argument("--num_filters", type=int, default=16)
    parser.add_argument("--pool_st", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--per_seed_timeout", type=int, default=3600 * 6)
    parser.add_argument("--study_db", default="optuna_studies/csnn_kth.db")
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--threads_per_run", type=int, default=1)
    args = parser.parse_args()

    args.binary = str(Path(args.binary).resolve())
    if not Path(args.binary).is_file():
        sys.exit(f"Binary not found at {args.binary}. Build it first.")
    if args.cwd is None:
        args.cwd = str(Path(args.binary).parent)

    if not Path(args.data).is_file():
        sys.exit(f"Data .npy not found at {args.data}.")
    sidecar = Path(args.data).with_suffix(".json")
    if not sidecar.is_file():
        sys.exit(f"Metadata sidecar not found at {sidecar} (expected next to the .npy).")

    args.input_root = str(Path(args.input_root).resolve()) + "/"
    args.seeds = [args.seeds_base + i for i in range(args.n_seeds)]

    Path(os.path.dirname(args.study_db) or ".").mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"csnn_1layer_{args.sampler}"
    storage = f"sqlite:///{args.study_db}"

    print(f"OPTUNA study : {study_name}")
    print(f"Storage      : {storage}")
    print(f"Binary       : {args.binary}")
    print(f"Data         : {args.data}")
    print(f"Sampler      : {args.sampler}")
    print(f"Trials       : {args.n_trials}  (seeds per trial: {args.seeds})")
    print(f"Epochs/trial : {args.epochs}")
    print(f"Parallelism  : n_jobs={args.n_jobs}  threads_per_run={args.threads_per_run}")
    print(f"               -> up to {args.n_jobs * args.threads_per_run} cores in flight, "
          f"~{args.n_jobs * 6}GB RAM")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
    )

    try:

        study.optimize(
            make_objective(args),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("Interrupted — study state is persisted; rerun to resume.")

    print()
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"  value (mean val acc): {study.best_value:.2f}%")
    for k, v in study.best_params.items():
        print(f"  {k:10s} = {v}")
    print(f"  per-seed accs: {study.best_trial.user_attrs.get('acc_per_seed')}")
    print(f"  std         : {study.best_trial.user_attrs.get('acc_std')}")
    print()
    print("To run the test protocol with these params, pass them to "
          "run_test_protocol.py (e.g. --params_from_study).")


if __name__ == "__main__":
    main()
