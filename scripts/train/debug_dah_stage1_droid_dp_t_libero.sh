#!/bin/bash
#
# DEBUG: DAH Stage 1 — DROID DP Transformer on LIBERO (low-dim only)
# Runs 2 epochs to verify the pipeline works.
#
# Usage:
#   ./scripts/train/debug_dah_stage1_droid_dp_t_libero.sh [EXTRA_ARGS...]

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUN_DIR="data/outputs/debug/debug_dah_stage1_droid_dp_t_libero"
rm -rf "${RUN_DIR}"

echo "=========================================="
echo "[DEBUG] DAH Stage 1 — DROID DP-T LIBERO"
echo "=========================================="

python trainer.py \
    --config-name=dah_stage1_droid_dp_t_libero \
    seed=42 \
    train_mode=stage1 \
    batch_size=4 \
    \
    training.num_epochs=2 \
    training.debug=true \
    training.use_ema=false \
    \
    dataloader.num_workers=2 \
    dataloader.persistent_workers=false \
    \
    run_dir="${RUN_DIR}" \
    run_name="debug_dah_stage1_droid_dp_t_libero" \
    \
    logging.mode="disabled" \
    \
    $@

echo "=========================================="
echo "[DEBUG] Done! Output: ${RUN_DIR}"
echo "=========================================="
