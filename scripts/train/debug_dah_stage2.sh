#!/bin/bash
#
# DEBUG: DAH Stage 2 (vision encoder + frozen action head)
# Runs 1 epoch to verify the pipeline works.
#
# Usage:
#   ./scripts/train/debug_dah_stage2.sh <arch> <task_letter> <stage1_ckpt> [EXTRA_ARGS...]
#
# Example:
#   ./scripts/train/debug_dah_stage2.sh dp_c A data/outputs/stage1_run/checkpoints/epoch=005.ckpt

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <arch> <task_letter> <stage1_ckpt> [EXTRA_ARGS...]"
    echo ""
    echo "Architecture options: dp_c, dp_t, dp_t_film, dp_mlp"
    echo "Task letters: A-H"
    exit 1
fi

declare -A TASK_MAP
TASK_MAP["A"]="stack_d1"
TASK_MAP["B"]="square_d2"
TASK_MAP["C"]="coffee_d2"
TASK_MAP["D"]="threading_d2"
TASK_MAP["E"]="stack_three_d1"
TASK_MAP["F"]="hammer_cleanup_d1"
TASK_MAP["G"]="three_piece_assembly_d2"
TASK_MAP["H"]="mug_cleanup_d1"

ARCH="$1"
TASK_LETTER="$2"
STAGE1_CKPT="$3"
shift 3
EXTRA_ARGS="$@"

TASK_NAME=${TASK_MAP["$TASK_LETTER"]}
if [ -z "$TASK_NAME" ]; then
    echo "ERROR: Unknown task letter '$TASK_LETTER'. Valid: A-H."
    exit 1
fi

if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found: ${STAGE1_CKPT}"
    exit 1
fi

CONFIG_NAME="dah_stage2_or_normal_${ARCH}"
REPO_ID="JianZhou0420/DAH_mimicgen_${TASK_NAME}_alldemos"
RUN_DIR="data/outputs/debug/debug_dah_stage2_${ARCH}_${TASK_LETTER}"
rm -rf "${RUN_DIR}"

echo "=========================================="
echo "[DEBUG] DAH Stage 2 — ${ARCH} task ${TASK_LETTER}"
echo "=========================================="

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=42 \
    train_mode=stage2_rollout \
    \
    ckpt_path="${STAGE1_CKPT}" \
    dataset.repo_id="${REPO_ID}" \
    adaptor.model.norm_stats_path="auto" \
    training.num_epochs=2 \
    \
    run_dir="${RUN_DIR}" \
    run_name="debug_dah_stage2_${ARCH}_${TASK_LETTER}" \
    \
    logging.mode="online" \
    \
    ${EXTRA_ARGS}

echo "=========================================="
echo "[DEBUG] Done! Output: ${RUN_DIR}"
echo "=========================================="
