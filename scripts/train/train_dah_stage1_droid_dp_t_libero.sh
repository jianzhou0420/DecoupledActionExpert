#!/bin/bash
#
# DAH Stage 1: Train DROID DP Transformer on LIBERO (per-suite, low-dim only)
#
# Usage:
#   ./scripts/train/train_dah_stage1_droid_dp_t_libero.sh <suite> <seed> [EXTRA_ARGS...]
#
# Suite options: libero_spatial, libero_object, libero_goal, libero_10
#
# Examples:
#   ./scripts/train/train_dah_stage1_droid_dp_t_libero.sh libero_spatial 42
#   ./scripts/train/train_dah_stage1_droid_dp_t_libero.sh libero_10 42 training.num_epochs=10

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# Input
# --------------------
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <suite> <seed> [EXTRA_ARGS...]"
    echo ""
    echo "Suite options: libero_spatial, libero_object, libero_goal, libero_10"
    echo ""
    echo "Example: $0 libero_spatial 42"
    exit 1
fi

SUITE="$1"
SEED="$2"
shift 2
EXTRA_ARGS="$@"

REPO_ID="JianZhou0420/libero_${SUITE}_alldemos_full"

echo "=========================================="
echo "DAH Stage 1 — DROID DP-T LIBERO (per-suite)"
echo "=========================================="
echo "Suite: ${SUITE}"
echo "Seed: ${SEED}"
echo "Dataset: ${REPO_ID}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=========================================="

# --------------------
# Run
# --------------------
date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
run_name="DAH_stage1_droid_dp_t_${SUITE}_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name=dah_stage1_droid_dp_t_libero \
    seed=${SEED} \
    train_mode=stage1 \
    \
    dataset.repo_id="${REPO_ID}" \
    \
    task_name="dah_stage1_droid_dp_t_${SUITE}" \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="dah_stage1_droid_dp_t_libero" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,stage1,droid-dp-t,libero,${SUITE},per-suite,transformer]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Stage 1 training completed!"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "Norm stats: ${run_dir}/norm_stats.json"
echo "=========================================="
