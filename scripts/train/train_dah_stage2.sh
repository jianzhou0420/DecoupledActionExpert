#!/bin/bash
#
# Train DAH Stage 2: Vision encoder with frozen action head from Stage 1
#
# Usage:
#   ./scripts/train/train_dah_stage2.sh <arch> <task_letter> <seed> <stage1_ckpt> [EXTRA_ARGS...]
#
# Architecture options: dp_c, dp_t, dp_t_film, dp_mlp
#
# Examples:
#   # Train DP_C stage2 on task A using stage1 checkpoint
#   ./scripts/train/train_dah_stage2.sh dp_c A 42 data/outputs/stage1_run/checkpoints/epoch=005.ckpt
#
#   # Train DP_T stage2 with extra overrides
#   ./scripts/train/train_dah_stage2.sh dp_t A 42 path/to/stage1.ckpt batch_size=32

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# region 1. Input
# --------------------
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Usage: $0 <arch> <task_letter> <seed> <stage1_ckpt> [EXTRA_ARGS...]"
    echo ""
    echo "Architecture options: dp_c, dp_t, dp_t_film, dp_mlp"
    echo ""
    echo "Task letters:"
    echo "  A = stack_d1              E = stack_three_d1"
    echo "  B = square_d2             F = hammer_cleanup_d1"
    echo "  C = coffee_d2             G = three_piece_assembly_d2"
    echo "  D = threading_d2          H = mug_cleanup_d1"
    echo ""
    echo "Example: $0 dp_c A 42 data/outputs/stage1_run/checkpoints/epoch=005.ckpt"
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
STAGE1_CKPT="$4"
shift 4
EXTRA_ARGS="$@"

TASK_NAME=${TASK_MAP["$TASK_LETTER"]}
if [ -z "$TASK_NAME" ]; then
    echo "ERROR: Unknown task letter '$TASK_LETTER'"
    exit 1
fi

CONFIG_NAME="dah_stage2_or_normal_${ARCH}"
REPO_ID="JianZhou0420/DAH_mimicgen_${TASK_NAME}_alldemos"

echo "=========================================="
echo "DAH Stage 2 Training"
echo "=========================================="
echo "Architecture: ${ARCH}"
echo "Config: ${CONFIG_NAME}"
echo "Task: ${TASK_LETTER} (${TASK_NAME})"
echo "Seed: ${SEED}"
echo "Stage1 checkpoint: ${STAGE1_CKPT}"
echo "Dataset: ${REPO_ID}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=========================================="

# Verify checkpoint exists
if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found: ${STAGE1_CKPT}"
    exit 1
fi

# --------------------
# region 2. Run
# --------------------
date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
run_name="DAH_stage2_${ARCH}_${TASK_LETTER}_${TASK_NAME}_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=${SEED} \
    train_mode=stage2_rollout \
    \
    ckpt_path="${STAGE1_CKPT}" \
    dataset.repo_id="${REPO_ID}" \
    adaptor.model.norm_stats_path="auto" \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="DAH_stage2_${ARCH}" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,stage2,${ARCH},${TASK_LETTER}]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Stage 2 training completed!"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "=========================================="
