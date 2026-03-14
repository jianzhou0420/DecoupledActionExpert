#!/bin/bash
#SBATCH --job-name=dah_s2
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=g015
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# =============================================================================
# DAH Stage 2 Per-Task Training on MimicGen LeRobot Dataset (SLURM Array)
#
# Each array index maps to one task. All tasks run as parallel SLURM jobs.
#
# Usage:
#   sbatch --array=0-7 scripts/slurm/train_dah_stage2_all_stage1.sh <arch> <stage1_ckpt> [SEED] [NOTE] [EXTRA_ARGS...]
#
#   # Train all 8 tasks in parallel:
#   sbatch --array=0-7 scripts/slurm/train_dah_stage2_all_stage1.sh dp_c /path/to/stage1.ckpt
#
#   # Train specific tasks:
#   sbatch --array=0,3,6 scripts/slurm/train_dah_stage2_all_stage1.sh dp_t /path/to/stage1.ckpt 42
#
# Architecture options: dp_c, dp_t, dp_t_film, dp_mlp
#
# The stage1 checkpoint is a single shared checkpoint used by all tasks.
#
# Index mapping:
#   0=stack_d1  1=square_d2  2=coffee_d2  3=threading_d2
#   4=stack_three_d1  5=hammer_cleanup_d1  6=three_piece_assembly_d2
#   7=mug_cleanup_d1
# =============================================================================

set -e

_NOTE="${4:-}"
LOG_DIR="data/logs/$(date +'%Y.%m.%d')"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/train_dah_stage2_all_stage1_array_${1}_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}${_NOTE:+_${_NOTE}}.log" 2>&1

# --------------------
# Task mapping (index -> task)
# --------------------
TASKS=(
    "stack_d1"
    "square_d2"
    "coffee_d2"
    "threading_d2"
    "stack_three_d1"
    "hammer_cleanup_d1"
    "three_piece_assembly_d2"
    "mug_cleanup_d1"
)
LETTERS=(A B C D E F G H)

REPO_PREFIX="JianZhou0420/DAH_mimicgen"
REPO_SUFFIX="alldemos"

# --------------------
# Parse arguments
# --------------------
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch --array=0-7 $0 <arch> <stage1_ckpt> [SEED] [NOTE] [EXTRA_ARGS...]"
    echo ""
    echo "Architecture options: dp_c, dp_t, dp_t_film, dp_mlp"
    exit 1
fi

ARCH="$1"
STAGE1_CKPT="$2"
SEED="${3:-42}"
NOTE="${4:-}"
shift 4 2>/dev/null || shift 3 2>/dev/null || shift 2 2>/dev/null || true
EXTRA_ARGS="$@"

# --------------------
# Resolve current task from SLURM_ARRAY_TASK_ID
# --------------------
TASK_IDX=${SLURM_ARRAY_TASK_ID}
TASK_NAME=${TASKS[$TASK_IDX]}
LETTER=${LETTERS[$TASK_IDX]}

if [ -z "$TASK_NAME" ]; then
    echo "ERROR: Invalid array index ${TASK_IDX}. Valid range: 0-7."
    exit 1
fi

CONFIG_NAME="dah_stage2_or_normal_${ARCH}"
scontrol update JobId="$SLURM_JOB_ID" JobName="dah_s2_${ARCH}"
REPO_ID="${REPO_PREFIX}_${TASK_NAME}_${REPO_SUFFIX}"

# Verify stage 1 checkpoint exists
if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found: ${STAGE1_CKPT}"
    exit 1
fi

echo "=============================================="
echo "SLURM Array Job: DAH Stage 2 (${ARCH})"
echo "=============================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Architecture: ${ARCH}"
echo "Config: ${CONFIG_NAME}"
echo "Task: ${LETTER} = ${TASK_NAME}"
echo "Seed: $SEED"
echo "Stage1 checkpoint: ${STAGE1_CKPT}"
echo "Dataset: ${REPO_ID}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Note: $NOTE"
echo "Extra args: $EXTRA_ARGS"
echo "=============================================="

# --------------------
# Environment Setup
# --------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DecoupledActionExpert

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

export WANDB_CACHE_DIR="${WANDB_DIR:-$HOME/.wandb}/cache"
export WANDB_CONFIG_DIR="${WANDB_DIR:-$HOME/.wandb}/config"
export WANDB_DATA_DIR="${WANDB_DIR:-$HOME/.wandb}/data"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "GPU Information:"
nvidia-smi

# --------------------
# Run Training
# --------------------
DATE_PART=$(date +'%Y.%m.%d')
TIME_PART=$(date +'%H.%M.%S')
EXP_NAME="DAH_stage2_${ARCH}_seed${SEED}"

RUN_NAME="${EXP_NAME}__${LETTER}_${TASK_NAME}"
if [ -n "${NOTE}" ]; then RUN_NAME="${RUN_NAME}_${NOTE}"; fi
RUN_DIR="data/outputs/${DATE_PART}/${TIME_PART}_${RUN_NAME}"

echo ""
echo "------------------------------------------"
echo "Task ${LETTER}: ${TASK_NAME}"
echo "  repo_id:         ${REPO_ID}"
echo "  stage1_ckpt:     ${STAGE1_CKPT}"
echo "  run_dir:         ${RUN_DIR}"
echo "------------------------------------------"

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=${SEED} \
    train_mode=stage2_rollout \
    \
    "ckpt_path='${STAGE1_CKPT}'" \
    dataset.repo_id="${REPO_ID}" \
    adaptor.model.norm_stats_path="auto" \
    \
    run_dir="${RUN_DIR}" \
    run_name="${RUN_NAME}" \
    \
    dataloader.num_workers=32 \
    training.checkpoint_every=1 \
    \
    logging.project="IROS_FINAL_EXP" \
    logging.group="DAH_stage2_${ARCH}_seed${SEED}" \
    logging.name="${RUN_NAME}" \
    'logging.tags=["dah","stage2","'"${ARCH}"'","'"${LETTER}"'","slurm"]' \
    logging.mode="offline" \
    \
    ${EXTRA_ARGS}

touch "${RUN_DIR}/done.mark"

echo ""
echo "=============================================="
echo "Task ${LETTER} (${TASK_NAME}) done!"
echo "Checkpoint dir: ${RUN_DIR}/checkpoints/"
echo "=============================================="
