#!/bin/bash
#
# Train DROID Diffusion Policy on LIBERO (local, single GPU)
#
# Usage:
#   ./scripts/train/train_droid_dp_libero.sh <suite> <seed> [EXTRA_ARGS...]
#
# Suite options: libero_spatial, libero_object, libero_goal, libero_10
#
# Examples:
#   ./scripts/train/train_droid_dp_libero.sh libero_spatial 42
#   ./scripts/train/train_droid_dp_libero.sh libero_10 42 batch_size=64

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

REPO_ID="JianZhou0420/libero_openvla_LeRobotv3_0"

echo "=========================================="
echo "DROID DP — ${SUITE}"
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
run_name="droid_dp_${SUITE}_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name=droid_dp_libero \
    seed=${SEED} \
    +train_mode=normal_rollout \
    \
    dataset.repo_id="${REPO_ID}" \
    adaptor.model.norm_stats_path="auto" \
    \
    task_name="droid_dp_${SUITE}" \
    libero_runner.task_suites="[${SUITE}]" \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="droid_dp_libero" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[droid-dp,libero,${SUITE},lerobot,language]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "=========================================="
