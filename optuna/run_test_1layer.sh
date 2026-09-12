#!/bin/bash
# Test protocol for the 1-layer CSNN. Reads the best config from the OPTUNA
# study and runs 10 seeds with CSNN_EVAL_SPLIT=test.
#
# Usage:
#     ./optuna/run_test_1layer.sh hog
#     ./optuna/run_test_1layer.sh random

set -euo pipefail

SAMPLER="${1:-hog}"
if [ "$SAMPLER" != "hog" ] && [ "$SAMPLER" != "random" ]; then
    echo "usage: $0 <sampler:hog|random>"
    exit 1
fi

STUDY_NAME="${STUDY_NAME:-csnn_1layer_${SAMPLER}}"
DATA="${DATA:-../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy}"
OUT_CSV="${OUT_CSV:-data/test_protocol_1layer_${SAMPLER}.csv}"
POOL_ST="${POOL_ST:-1}"
INPUT_ROOT="${INPUT_ROOT:-/home/mmuntean/kth_organized_tvt/}"
CSNN_BUILD_DIR="${CSNN_BUILD_DIR:-cmake-build-release}"
N_JOBS="${N_JOBS:-10}"

mkdir -p data

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TBB_NUM_THREADS=1 \
python3 -u optuna/run_test_protocol_1layer.py \
    --binary "$CSNN_BUILD_DIR/KTH_1layer" \
    --cwd . \
    --data "$DATA" \
    --input_root "$INPUT_ROOT" \
    --params_from_study optuna_studies/csnn_kth.db \
    --study_name "$STUDY_NAME" \
    --sampler "$SAMPLER" \
    --n_seeds 10 \
    --num_filters 16 \
    --pool_st $POOL_ST \
    --epochs 100 \
    --n_jobs "$N_JOBS" \
    --threads_per_run 1 \
    --out_csv "$OUT_CSV"
