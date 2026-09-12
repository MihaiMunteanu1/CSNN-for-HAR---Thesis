#!/bin/bash

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

STUDY_NAME="${STUDY_NAME:-csnn_3layer_${SAMPLER}}"
OUT_CSV="${OUT_CSV:-data/test_protocol_3layer_${SAMPLER}.csv}"
N_JOBS="${N_JOBS:-4}"
DATA="${DATA:-../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy}"
POOL_ST="${POOL_ST:-1}"
INPUT_ROOT="${INPUT_ROOT:-/home/mmuntean/kth_organized_tvt/}"
CSNN_BUILD_DIR="${CSNN_BUILD_DIR:-cmake-build-release}"

mkdir -p data

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TBB_NUM_THREADS=1 \
python3 -u optuna/run_test_protocol_3layer.py \
    --binary "$CSNN_BUILD_DIR/KTH_3layer" \
    --cwd . \
    --data "$DATA" \
    --input_root "$INPUT_ROOT" \
    --params_from_study optuna_studies/csnn_kth.db \
    --study_name "$STUDY_NAME" \
    --sampler "$SAMPLER" \
    --n_seeds 10 \
    --num_filters1 16 \
    --num_filters2 32 \
    --num_filters3 64 \
    --pool_sx 2 --pool_sy 2 --pool_st "$POOL_ST" \
    --epochs1 100 --epochs2 80 --epochs3 80 \
    --n_jobs "$N_JOBS" \
    --threads_per_run 1 \
    --out_csv "$OUT_CSV"