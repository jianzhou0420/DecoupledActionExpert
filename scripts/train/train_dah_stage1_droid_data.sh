#!/bin/bash
#
# Train DAH Stage 1: MimicGen DP variants on DROID dataset (local)
#
# Reuses MimicGen stage 1 configs (dah_stage1_dp_c, etc.) with CLI overrides
# to train on DROID data instead of MimicGen data.
#
# Usage:
#   ./scripts/train/train_dah_stage1_droid_data.sh <arch> <seed> [EXTRA_ARGS...]
#
# Architecture options: dp_c, dp_t, dp_t_film, dp_mlp
#
# Examples:
#   ./scripts/train/train_dah_stage1_droid_data.sh dp_c 42
#   ./scripts/train/train_dah_stage1_droid_data.sh dp_t 42 training.max_steps=10000

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# region 1. Input
# --------------------
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <arch> <seed> [EXTRA_ARGS...]"
    echo ""
    echo "Architecture options: dp_c, dp_t, dp_t_film, dp_mlp"
    echo ""
    echo "Dataset: JianZhou0420/droid_lowdim (DROID, absolute actions)"
    echo ""
    echo "Example: $0 dp_c 42"
    exit 1
fi

ARCH="$1"
SEED="$2"
shift 2
EXTRA_ARGS="$@"

CONFIG_NAME="dah_stage1_${ARCH}"
REPO_ID="JianZhou0420/droid_lowdim"

echo "=========================================="
echo "DAH Stage 1 Training — ${ARCH} on DROID"
echo "=========================================="
echo "Architecture: ${ARCH}"
echo "Config: ${CONFIG_NAME}"
echo "Seed: ${SEED}"
echo "Dataset: ${REPO_ID}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=========================================="

# --------------------
# region 2. Run
# --------------------
date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
run_name="DAH_stage1_${ARCH}_droid_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=${SEED} \
    train_mode=stage1 \
    batch_size=256 \
    \
    dataset.repo_id="${REPO_ID}" \
    dataset.obs_keys="[observation.state]" \
    dataset.action_keys="[action.cartesian_position,action.gripper_position]" \
    adaptor.robot._target_=vlaworkspace.adaptors.robots.DroidStage1Robot \
    +adaptor.robot.cond_type=jp \
    preload=false \
    cache_in_memory=true \
    dataloader.num_workers=16 \
    dataloader.persistent_workers=true \
    training.num_epochs=-1 \
    +training.max_steps=50000 \
    training.checkpoint_every=10000 \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="DAH_stage1_${ARCH}_droid" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,stage1,${ARCH},droid]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Stage 1 training completed!"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "=========================================="
