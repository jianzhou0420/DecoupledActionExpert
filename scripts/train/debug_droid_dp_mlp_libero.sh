#!/bin/bash
#
# DEBUG: DROID Diffusion Policy (MLP) on LIBERO
# Runs a few steps to verify the pipeline works.
#
# Usage:
#   ./scripts/train/debug_droid_dp_mlp_libero.sh [EXTRA_ARGS...]

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUN_DIR="data/outputs/debug/debug_droid_dp_mlp_libero"
rm -rf "${RUN_DIR}"

echo "=========================================="
echo "[DEBUG] DROID DP-MLP on LIBERO"
echo "=========================================="

python trainer.py \
    --config-name=droid_dp_mlp_libero \
    seed=42 \
    \
    run_dir="${RUN_DIR}" \
    run_name="debug_droid_dp_mlp_libero" \
    \
    batch_size=4 \
    training.max_steps=5 \
    training.debug=true \
    training.use_ema=false \
    \
    dataloader.num_workers=2 \
    dataloader.persistent_workers=false \
    \
    logging.mode="disabled" \
    \
    'libero_runner.task_suites=[libero_spatial]' \
    libero_runner.n_trials_per_task=10 \
    libero_runner.n_envs=10 \
    \
    $@

echo "=========================================="
echo "[DEBUG] Done! Output: ${RUN_DIR}"
echo "=========================================="
