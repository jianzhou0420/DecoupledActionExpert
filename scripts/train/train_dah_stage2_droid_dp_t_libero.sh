#!/bin/bash
#
# DAH Stage 2: Train DROID DP Transformer vision on LIBERO (per-suite stage1 → per-suite stage2)
# Loads stage 1 Transformer checkpoint, trains vision encoder.
#
# Usage:
#   ./scripts/train/train_dah_stage2_droid_dp_t_libero.sh <suite> <seed> <stage1_ckpt> [EXTRA_ARGS...]
#
# Suite options: libero_spatial, libero_object, libero_goal, libero_10
#
# Examples:
#   ./scripts/train/train_dah_stage2_droid_dp_t_libero.sh libero_spatial 42 data/outputs/.../checkpoints/last.ckpt

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# Input
# --------------------
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <suite> <seed> <stage1_ckpt> [EXTRA_ARGS...]"
    echo ""
    echo "Suite options: libero_spatial, libero_object, libero_goal, libero_10"
    echo ""
    echo "Example: $0 libero_spatial 42 data/outputs/.../checkpoints/last.ckpt"
    exit 1
fi

SUITE="$1"
SEED="$2"
CKPT="$3"
shift 3
EXTRA_ARGS="$@"

REPO_ID="JianZhou0420/libero_${SUITE}_alldemos_full"

echo "=========================================="
echo "DAH Stage 2 — DROID DP-T LIBERO (per-suite)"
echo "=========================================="
echo "Suite: ${SUITE}"
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
run_name="DAH_stage2_droid_dp_t_${SUITE}_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name=dah_stage2_droid_dp_t_libero \
    seed=${SEED} \
    train_mode=stage2_rollout \
    "ckpt_path='${CKPT}'" \
    \
    dataset.repo_id="${REPO_ID}" \
    adaptor.model.norm_stats_path="auto" \
    \
    task_name="dah_stage2_droid_dp_t_${SUITE}" \
    libero_runner.task_suites="[${SUITE}]" \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="dah_stage2_droid_dp_t_libero" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,stage2,droid-dp-t,libero,${SUITE},per-suite,language,transformer]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Stage 2 training completed!"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "=========================================="
