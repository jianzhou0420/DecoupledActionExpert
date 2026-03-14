#!/bin/bash
#
# Train DAH Stage 1: Diffusion action head with low-dim state only (no images)
# Always trains on the combined ABCDEFGH dataset (8 tasks).
#
# Usage:
#   ./scripts/train/train_dah_stage1.sh <arch> <seed> [EXTRA_ARGS...]
#
# Architecture options: dp_c, dp_t, dp_t_film, dp_mlp
#
# Examples:
#   ./scripts/train/train_dah_stage1.sh dp_c 42
#   ./scripts/train/train_dah_stage1.sh dp_t 42 batch_size=128 training.num_epochs=100

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
    echo "Dataset: DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim"
    echo "  A = stack_d1              E = stack_three_d1"
    echo "  B = square_d2             F = hammer_cleanup_d1"
    echo "  C = coffee_d2             G = three_piece_assembly_d2"
    echo "  D = threading_d2          H = mug_cleanup_d1"
    echo ""
    echo "Example: $0 dp_c 42"
    exit 1
fi

ARCH="$1"
SEED="$2"
shift 2
EXTRA_ARGS="$@"

CONFIG_NAME="dah_stage1_${ARCH}"
REPO_ID="JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim"

echo "=========================================="
echo "DAH Stage 1 Training"
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
run_name="DAH_stage1_${ARCH}_ABCDEFGH_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=${SEED} \
    train_mode=stage1 \
    batch_size=256 \
    \
    dataset.repo_id="${REPO_ID}" \
    training.num_epochs=50 \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="DAH_stage1_${ARCH}" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,stage1,${ARCH},ABCDEFGH]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Stage 1 training completed!"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "=========================================="
