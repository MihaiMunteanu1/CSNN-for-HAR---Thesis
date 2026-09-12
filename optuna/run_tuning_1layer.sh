#!/usr/bin/env bash
# Launch an OPTUNA tuning study inside a detached tmux session.
#
# Usage (from repo root):
#     ./optuna/run_tuning_1layer.sh hog       # tune skeleton-based sampling
#     ./optuna/run_tuning_1layer.sh random    # tune generic sampling
#
# Overridable env vars (set before invoking):
#     CSNN_BUILD_DIR   default: cmake-build-release
#     DATA             default: ../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy
#     INPUT_ROOT       default: /home/mmuntean/kth_organized_tvt/
#     N_TRIALS         default: 50
#     N_SEEDS          default: 5
#     N_JOBS           default: 32   (one larochette node)
#     EPOCHS           default: 100
#     POOL_ST          default: 1
#     STUDY_SUFFIX

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <sampler:hog|random>"
    exit 1
fi

SAMPLER="$1"
if [ "$SAMPLER" != "hog" ] && [ "$SAMPLER" != "random" ]; then
    echo "sampler must be 'hog' or 'random' (got '$SAMPLER')"
    exit 1
fi

CSNN_BUILD_DIR="${CSNN_BUILD_DIR:-cmake-build-release}"
DATA="${DATA:-../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy}"
INPUT_ROOT="${INPUT_ROOT:-/home/mmuntean/kth_organized_tvt/}"
N_TRIALS="${N_TRIALS:-50}"
N_SEEDS="${N_SEEDS:-5}"
N_JOBS="${N_JOBS:-32}"
EPOCHS="${EPOCHS:-100}"
STUDY_SUFFIX="${STUDY_SUFFIX:-}"
POOL_ST="${POOL_ST:-1}"


if [ -n "$STUDY_SUFFIX" ]; then
    STUDY_NAME="csnn_1layer_${SAMPLER}_${STUDY_SUFFIX}"
    SESSION_TAG="${SAMPLER}_${STUDY_SUFFIX}"
else
    STUDY_NAME="csnn_1layer_${SAMPLER}"
    SESSION_TAG="${SAMPLER}"
fi

BINARY="${CSNN_BUILD_DIR}/KTH_1layer"
if [ ! -x "$BINARY" ]; then
    echo "ERROR: binary not found at $BINARY"
    echo "  build it first: cmake --build $CSNN_BUILD_DIR --target KTH_1layer"
    exit 2
fi

SESSION="optuna_${SESSION_TAG}"


if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists."
    echo "  attach: tmux attach -t $SESSION"
    echo "  or kill: tmux kill-session -t $SESSION"
    exit 3
fi

mkdir -p optuna_studies data/logs
LOG="data/logs/log_${SESSION}_$(date +%Y%m%d_%H%M%S).txt"


CMD="python3 optuna/tune_csnn_optuna_1layer.py \
    --binary $BINARY \
    --data $DATA \
    --input_root $INPUT_ROOT \
    --sampler $SAMPLER \
    --study_name $STUDY_NAME \
    --n_trials $N_TRIALS \
    --n_seeds $N_SEEDS \
    --n_jobs $N_JOBS \
    --threads_per_run 1 \
    --pool_st $POOL_ST \
    --epochs $EPOCHS"

tmux new-session -d -s "$SESSION" "bash -lc 'cd $(pwd) && $CMD; echo \"--- DONE ---\"; exec bash'"
tmux pipe-pane -t "$SESSION" "cat > $LOG"

echo "Started tmux session '$SESSION' (log: $LOG)"
echo "  attach :  tmux attach -t $SESSION"
echo "  detach :  Ctrl+B  then  D"
echo "  list   :  tmux ls"
echo "  kill   :  tmux kill-session -t $SESSION"
echo
echo "Config:"
echo "  sampler   = $SAMPLER"
echo "  study     = $STUDY_NAME"
echo "  binary    = $BINARY"
echo "  data      = $DATA"
echo "  trials    = $N_TRIALS (seeds=$N_SEEDS, n_jobs=$N_JOBS, epochs=$EPOCHS)"
