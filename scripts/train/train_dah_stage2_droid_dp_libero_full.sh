#!/bin/bash
#
# DAH Stage 2: Train DROID DP vision on LIBERO (full multitask — all 4 suites)
# Loads stage 1 UNet checkpoint, freezes conv blocks, trains vision encoder.
# Trains and evaluates on all 4 LIBERO suites combined.
#
# Usage:
#   ./scripts/train/train_dah_stage2_droid_dp_libero_full.sh <seed> <stage1_ckpt> [EXTRA_ARGS...]
#
# Examples:
#   ./scripts/train/train_dah_stage2_droid_dp_libero_full.sh 42 data/outputs/.../checkpoints/last.ckpt

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# Input
# --------------------
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <seed> <stage1_ckpt> [EXTRA_ARGS...]"
    echo ""
    echo "Example: $0 42 data/outputs/.../checkpoints/last.ckpt"
    exit 1
fi

SEED="$1"
CKPT="$2"
shift 2
EXTRA_ARGS="$@"

REPO_ID="JianZhou0420/libero_openvla_LeRobotv3_0"

echo "=========================================="
echo "DAH Stage 2 — DROID DP LIBERO (full)"
echo "=========================================="
echo "Seed: ${SEED}"
echo "Stage1 ckpt: ${CKPT}"
echo "Dataset: ${REPO_ID}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=========================================="

# --------------------
# Run
# --------------------
date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
run_name="DAH_stage2_droid_dp_libero_full_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name=dah_stage2_droid_dp_libero \
    seed=${SEED} \
    train_mode=stage2_rollout \
    "ckpt_path='${CKPT}'" \
    \
    dataset.repo_id="${REPO_ID}" \
    adaptor.model.norm_stats_path="auto" \
    \
    task_name="dah_stage2_droid_dp_libero_full" \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="dah_stage2_droid_dp_libero" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,stage2,droid-dp,libero,full,language]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Stage 2 training (full) completed!"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "=========================================="
