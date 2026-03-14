#!/bin/bash
#
# DEBUG: DAH Stage 2 — DROID DP MLP on LIBERO (full vision)
# Runs a few steps to verify the pipeline works.
#
# Usage:
#   ./scripts/train/debug_dah_stage2_droid_dp_mlp_libero.sh <stage1_ckpt> [EXTRA_ARGS...]
#
# Example:
#   ./scripts/train/debug_dah_stage2_droid_dp_mlp_libero.sh data/outputs/.../checkpoints/last.ckpt

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ -z "$1" ]; then
    echo "Usage: $0 <stage1_ckpt> [EXTRA_ARGS...]"
    echo ""
    echo "Example: $0 data/outputs/.../checkpoints/last.ckpt"
    exit 1
fi

CKPT="$1"
shift
EXTRA_ARGS="$@"

RUN_DIR="data/outputs/debug/debug_dah_stage2_droid_dp_mlp_libero"
rm -rf "${RUN_DIR}"

echo "=========================================="
echo "[DEBUG] DAH Stage 2 — DROID DP-MLP LIBERO"
echo "Stage1 ckpt: ${CKPT}"
echo "=========================================="

python trainer.py \
    --config-name=dah_stage2_droid_dp_mlp_libero \
    seed=42 \
    train_mode=stage2_rollout \
    "ckpt_path='${CKPT}'" \
    \
    adaptor.model.norm_stats_path="auto" \
    \
    batch_size=4 \
    training.max_steps=5 \
    training.debug=true \
    training.use_ema=false \
    \
    dataloader.num_workers=2 \
    dataloader.persistent_workers=false \
    \
    run_dir="${RUN_DIR}" \
    run_name="debug_dah_stage2_droid_dp_mlp_libero" \
    \
    logging.mode="disabled" \
    \
    'libero_runner.task_suites=[libero_spatial]' \
    libero_runner.n_trials_per_task=10 \
    libero_runner.n_envs=10 \
    \
    ${EXTRA_ARGS}

echo "=========================================="
echo "[DEBUG] Done! Output: ${RUN_DIR}"
echo "=========================================="
