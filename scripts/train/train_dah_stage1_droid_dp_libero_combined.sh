#!/bin/bash
#
# DAH Stage 1: Train DROID DP UNet on LIBERO (all suites combined, low-dim only)
#
# Usage:
#   ./scripts/train/train_dah_stage1_droid_dp_libero_combined.sh <seed> [EXTRA_ARGS...]
#
# Dataset: JianZhou0420/DAH_libero_all_alldemos_lowdim (all 4 suites, low-dim only)
#
# Examples:
#   ./scripts/train/train_dah_stage1_droid_dp_libero_combined.sh 42
#   ./scripts/train/train_dah_stage1_droid_dp_libero_combined.sh 42 training.num_epochs=10

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# Input
# --------------------
if [ -z "$1" ]; then
    echo "Usage: $0 <seed> [EXTRA_ARGS...]"
    echo ""
    echo "Dataset: JianZhou0420/DAH_libero_all_alldemos_lowdim (all 4 suites, low-dim only)"
    echo ""
    echo "Example: $0 42"
    exit 1
fi

SEED="$1"
shift 1
EXTRA_ARGS="$@"

REPO_ID="JianZhou0420/DAH_libero_all_alldemos_lowdim"

echo "=========================================="
echo "DAH Stage 1 — DROID DP LIBERO (combined)"
echo "=========================================="
echo "Seed: ${SEED}"
echo "Dataset: ${REPO_ID}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=========================================="

# --------------------
# Run
# --------------------
date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
run_name="DAH_stage1_droid_dp_libero_combined_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name=dah_stage1_droid_dp_libero \
    seed=${SEED} \
    train_mode=stage1 \
    \
    dataset.repo_id="${REPO_ID}" \
    \
    task_name="dah_stage1_droid_dp_libero_combined" \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="dah_stage1_droid_dp_libero" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,stage1,droid-dp,libero,combined,all-suites]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Stage 1 training (combined) completed!"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "Norm stats: ${run_dir}/norm_stats.json"
echo "=========================================="
