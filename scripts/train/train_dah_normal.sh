#!/bin/bash
#
# Train DAH Normal: End-to-end diffusion policy with images + state (no stage1 pretraining)
#
# Usage:
#   ./scripts/train/train_dah_normal.sh <arch> <task_letter> <seed> [EXTRA_ARGS...]
#
# Architecture options: dp_c, dp_t, dp_t_film, dp_mlp
#
# Examples:
#   ./scripts/train/train_dah_normal.sh dp_c A 42
#   ./scripts/train/train_dah_normal.sh dp_t B 42 batch_size=64

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# region 1. Input
# --------------------
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <arch> <task_letter> <seed> [EXTRA_ARGS...]"
    echo ""
    echo "Architecture options: dp_c, dp_t, dp_t_film, dp_mlp"
    echo ""
    echo "Task letters:"
    echo "  A = stack_d1              E = stack_three_d1"
    echo "  B = square_d2             F = hammer_cleanup_d1"
    echo "  C = coffee_d2             G = three_piece_assembly_d2"
    echo "  D = threading_d2          H = mug_cleanup_d1"
    echo ""
    echo "Example: $0 dp_c A 42"
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
SEED="$3"
shift 3
EXTRA_ARGS="$@"

TASK_NAME=${TASK_MAP["$TASK_LETTER"]}
if [ -z "$TASK_NAME" ]; then
    echo "ERROR: Unknown task letter '$TASK_LETTER'. Valid: A-H."
    exit 1
fi

CONFIG_NAME="dah_stage2_or_normal_${ARCH}"
REPO_ID="JianZhou0420/DAH_mimicgen_${TASK_NAME}_alldemos"

echo "=========================================="
echo "DAH Normal Training"
echo "=========================================="
echo "Architecture: ${ARCH}"
echo "Config: ${CONFIG_NAME}"
echo "Task: ${TASK_LETTER} (${TASK_NAME})"
echo "Seed: ${SEED}"
echo "Dataset: ${REPO_ID}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=========================================="

# --------------------
# region 2. Run
# --------------------
date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
run_name="DAH_normal_${ARCH}_${TASK_LETTER}_${TASK_NAME}_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=${SEED} \
    train_mode=normal_rollout \
    \
    dataset.repo_id="${REPO_ID}" \
    adaptor.model.norm_stats_path="auto" \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="DAH_normal_${ARCH}" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,normal,${ARCH},${TASK_LETTER}]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Normal training completed!"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "=========================================="
