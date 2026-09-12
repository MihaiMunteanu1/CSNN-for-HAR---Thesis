#!/bin/bash
# Test protocol for 2-layer CSNN. Reads best t_obj2 from the 2-layer OPTUNA
# study and runs 10 seeds with CSNN_EVAL_SPLIT=test.
#
# Usage:
#     ./optuna/run_test_2layer.sh hog
#     ./optuna/run_test_2layer.sh random
#
# Env overrides:
#     STUDY_NAME   default: csnn_2layer_<sampler>
#     OUT_CSV      default: data/test_protocol_2layer_<sampler>.csv

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <sampler:hog|random>"
    exit 1
fi

SAMPLER="$1"
if [ "$SAMPLER" != "hog" ] && [ "$SAMPLER" != "random" ]; then
    echo "sampler must be 'hog' or 'random'"
    exit 1
fi

STUDY_NAME="${STUDY_NAME:-csnn_2layer_${SAMPLER}}"
OUT_CSV="${OUT_CSV:-data/test_protocol_2layer_${SAMPLER}.csv}"
N_JOBS="${N_JOBS:-4}"

DATA="${DATA:-../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy}"
POOL_ST="${POOL_ST:-1}"
INPUT_ROOT="${INPUT_ROOT:-/home/mmuntean/kth_organized_tvt/}"
CSNN_BUILD_DIR="${CSNN_BUILD_DIR:-cmake-build-release}"

mkdir -p data

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TBB_NUM_THREADS=1 \
python3 -u optuna/run_test_protocol_2layer.py \
    --binary "$CSNN_BUILD_DIR/KTH_2layer" \
    --cwd . \
    --data "$DATA" \
    --input_root "$INPUT_ROOT" \
    --params_from_study optuna_studies/csnn_kth.db \
    --study_name "$STUDY_NAME" \
    --sampler "$SAMPLER" \
    --n_seeds 10 \
    --num_filters1 16 \
    --num_filters2 32 \
    --pool_sx 2 --pool_sy 2 --pool_st "$POOL_ST" \
    --epochs1 100 --epochs2 80 \
    --n_jobs "$N_JOBS" \
    --threads_per_run 1 \
    --out_csv "$OUT_CSV"