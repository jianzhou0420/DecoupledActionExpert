#!/bin/bash
#
# DEBUG: DAH Stage 1 (low-dim, combined ABCDEFGH dataset)
# Runs 1 epoch to verify the pipeline works.
#
# Usage:
#   ./scripts/train/debug_dah_stage1.sh <arch> [EXTRA_ARGS...]
#
# Example:
#   ./scripts/train/debug_dah_stage1.sh dp_c

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ -z "$1" ]; then
    echo "Usage: $0 <arch> [EXTRA_ARGS...]"
    echo "Architecture options: dp_c, dp_t, dp_t_film, dp_mlp"
    exit 1
fi

ARCH="$1"
shift
EXTRA_ARGS="$@"

CONFIG_NAME="dah_stage1_${ARCH}"
RUN_DIR="data/outputs/debug/debug_dah_stage1_${ARCH}"
rm -rf "${RUN_DIR}"

echo "=========================================="
echo "[DEBUG] DAH Stage 1 — ${ARCH}"
echo "=========================================="

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=42 \
    train_mode=stage1 \
    batch_size=256 \
    \
    dataset.repo_id="JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim" \
    training.num_epochs=2 \
    \
    run_dir="${RUN_DIR}" \
    run_name="debug_dah_stage1_${ARCH}" \
    \
    logging.mode="online" \
    \
    ${EXTRA_ARGS}

echo "=========================================="
echo "[DEBUG] Done! Output: ${RUN_DIR}"
echo "=========================================="
