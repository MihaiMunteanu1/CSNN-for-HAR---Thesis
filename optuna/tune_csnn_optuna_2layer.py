"""
OPTUNA hyperparameter tuning for the 2-layer CSNN + SVM head on KTH.

Conv1 is FROZEN at the per-sampler best from the 1-layer OPTUNA study
(RESULTS_KTH_1LAYER.md). Conv2 reuses the same filter shape (same FH/FW/FT)
with 32 filters; only t_obj2 is tuned.

Usage:
    python3 optuna/tune_csnn_optuna_2layer.py \\
        --binary csnn-simulator-build-roazhon4/KTH_2layer \\
        --data ../hog/kth_fullframes_tvt_19_f10_g2_80x60.npy \\
        --sampler hog \\
        --n_trials 50 \\
        --n_seeds 5
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

T_OBJ2_CHOICES = [round(0.30 + 0.05 * i, 2) for i in range(11)]


CONV1_PARAMS = {
    "hog":    {"filter_h": 3, "filter_w": 3, "filter_t": 3, "t_obj1": 0.75},
    "random": {"filter_h": 3, "filter_w": 3, "filter_t": 3, "t_obj1": 0.75},
}

ACC_RE = re.compile(r"classification rate:\s*([0-9]+(?:\.[0-9]+)?)%")


def run_one(args, t_obj2, seed):
    c1 = CONV1_PARAMS[args.sampler]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS":     str(args.threads_per_run),
        "OPENBLAS_NUM_THREADS": str(args.threads_per_run),
        "MKL_NUM_THREADS":      str(args.threads_per_run),
        "TBB_NUM_THREADS":      str(args.threads_per_run),
    })
    env.update({
        "CSNN_SEED":       str(seed),
        "CSNN_T_OBJ":      str(c1["t_obj1"]),
        "CSNN_T_OBJ2":     str(t_obj2),
        "CSNN_FH":         str(c1["filter_h"]),
        "CSNN_FW":         str(c1["filter_w"]),
        "CSNN_FT":         str(c1["filter_t"]),
        "CSNN_NF":         str(args.num_filters1),
        "CSNN_NF2":        str(args.num_filters2),
        "CSNN_POOL_SX":    str(args.pool_sx),
        "CSNN_POOL_SY":    str(args.pool_sy),
        "CSNN_POOL_ST":    str(args.pool_st),
        "CSNN_SAMPLER":    args.sampler,
        "CSNN_EVAL_SPLIT": "val",
        "CSNN_DATA":       args.data,
        "CSNN_INPUT_ROOT": args.input_root,
        "CSNN_EPOCHS":     str(args.epochs1),
        "CSNN_EPOCHS2":    str(args.epochs2),
        "CSNN_TAG":        f"2layer_trial_{env.get('_TRIAL_NO', 'NA')}",
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
    print(f"    [seed {seed}] t_obj2={t_obj2} acc={acc:.2f}%  ({dt:.0f}s)")
    return acc


def make_objective(args):
    def objective(trial: optuna.Trial):
        t_obj2 = trial.suggest_categorical("t_obj2", T_OBJ2_CHOICES)
        print(f"\n[trial {trial.number}] t_obj2={t_obj2}  "
              f"(conv1 frozen: {CONV1_PARAMS[args.sampler]})")

        accs = []
        for seed in args.seeds:
            acc = run_one(args, t_obj2, seed)
            if acc is None:
                raise optuna.TrialPruned()
            accs.append(acc)
            trial.report(mean(accs), step=len(accs))
            if trial.should_prune():
                raise optuna.TrialPruned()

        avg = mean(accs)
        std = pstdev(accs) if len(accs) > 1 else 0.0
        print(f"[trial {trial.number}] t_obj2={t_obj2}  mean={avg:.2f}%  std={std:.2f}")
        trial.set_user_attr("seeds", args.seeds)
        trial.set_user_attr("acc_per_seed", accs)
        trial.set_user_attr("acc_std", std)
        return avg
    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True,
                        help="Path to the built KTH_2layer executable.")
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_root", default="/home/mmuntean/kth_organized_tvt/")
    parser.add_argument("--sampler", choices=["hog", "random"], default="hog")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--n_seeds",  type=int, default=5)
    parser.add_argument("--seeds_base", type=int, default=42)
    parser.add_argument("--num_filters1", type=int, default=16)
    parser.add_argument("--num_filters2", type=int, default=32)
    parser.add_argument("--pool_sx", type=int, default=2)
    parser.add_argument("--pool_sy", type=int, default=2)
    parser.add_argument("--pool_st", type=int, default=2)
    parser.add_argument("--epochs1", type=int, default=100)
    parser.add_argument("--epochs2", type=int, default=80)
    parser.add_argument("--per_seed_timeout", type=int, default=3600 * 8)
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
    args.data = str(Path(args.data).resolve())
    if not Path(args.data).is_file():
        sys.exit(f"Data .npy not found at {args.data}.")
    sidecar = Path(args.data).with_suffix(".json")
    if not sidecar.is_file():
        sys.exit(f"Metadata sidecar not found at {sidecar}.")
    args.input_root = str(Path(args.input_root).resolve()) + "/"
    args.seeds = [args.seeds_base + i for i in range(args.n_seeds)]

    Path(os.path.dirname(args.study_db) or ".").mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"csnn_2layer_{args.sampler}"
    storage = f"sqlite:///{args.study_db}"

    print(f"OPTUNA study : {study_name}")
    print(f"Storage      : {storage}")
    print(f"Binary       : {args.binary}")
    print(f"Data         : {args.data}")
    print(f"Sampler      : {args.sampler}")
    print(f"Conv1 frozen : {CONV1_PARAMS[args.sampler]}")
    print(f"Conv2 fixed  : NF2={args.num_filters2}, kernel same as conv1, "
          f"pool {args.pool_sx}x{args.pool_sy}x{args.pool_st}")
    print(f"Tuning       : t_obj2 in {T_OBJ2_CHOICES}")
    print(f"Trials       : {args.n_trials}  (seeds per trial: {args.seeds})")
    print(f"Epochs       : conv1={args.epochs1}, conv2={args.epochs2}")
    print(f"Parallelism  : n_jobs={args.n_jobs}  threads_per_run={args.threads_per_run}")

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
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        n_failed = sum(1 for t in study.trials if t.state.name == "FAIL")
        n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
        print(f"NO COMPLETED TRIALS. trials={len(study.trials)} "
              f"(failed={n_failed}, pruned={n_pruned})")
        print("Check the per-seed FAILED/no-accuracy lines above to debug.")
        return
    print(f"Best trial: #{study.best_trial.number}")
    print(f"  value (mean val acc): {study.best_value:.2f}%")
    for k, v in study.best_params.items():
        print(f"  {k:10s} = {v}")
    print(f"  per-seed accs: {study.best_trial.user_attrs.get('acc_per_seed')}")
    print(f"  std         : {study.best_trial.user_attrs.get('acc_std')}")


if __name__ == "__main__":
    main()