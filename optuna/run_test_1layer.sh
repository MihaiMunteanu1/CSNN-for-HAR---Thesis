#!/bin/bash
STUDY_NAME="${STUDY_NAME:-csnn_1layer_hog_19f}"
DATA="${DATA:-../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy}"
OUT_CSV="${OUT_CSV:-data/test_protocol_hog_19f.csv}"
POOL_ST="${POOL_ST:-1}"
INPUT_ROOT="${INPUT_ROOT:-/home/mmuntean/kth_organized_tvt/}"
CSNN_BUILD_DIR="${CSNN_BUILD_DIR:-cmake-build-release}"
N_JOBS="${N_JOBS:-10}"

mkdir -p data

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TBB_NUM_THREADS=1 \
python3 -u optuna/run_test_protocol.py \
    --binary "$CSNN_BUILD_DIR/KTH_1layer" \
    --cwd . \
    --data "$DATA" \
    --input_root "$INPUT_ROOT" \
    --params_from_study optuna_studies/csnn_kth.db \
    --study_name "$STUDY_NAME" \
    --sampler hog \
    --n_seeds 10 \
    --num_filters 16 \
    --pool_st $POOL_ST \
    --epochs 100 \
    --n_jobs "$N_JOBS" \
    --threads_per_run 1 \
    --out_csv "$OUT_CSV"
