#!/bin/bash
#SBATCH --job-name=dah_s1
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --exclude=g015
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


# =============================================================================
# DAH Stage 1: Diffusion action head with low-dim state only (no images)
# Always trains on the combined ABCDEFGH dataset (8 tasks).
#
# Dataset: JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim
# Hardware: 1x GPU, 4 CPUs, 32GB RAM (preloaded in-memory, no dataloader workers)
#
# Usage:
#   sbatch scripts/slurm/train_dah_stage1.sh <arch> <seed> [EXTRA_ARGS...]
#   sbatch scripts/slurm/train_dah_stage1.sh dp_c 42
#   sbatch scripts/slurm/train_dah_stage1.sh dp_c 42 batch_size=128
#
# Architecture options: dp_c, dp_t, dp_t_film, dp_mlp
# =============================================================================

set -e

_NOTE="${3:-}"
LOG_DIR="data/logs/$(date +'%Y.%m.%d')"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/train_dah_stage1_all_${1}_${SLURM_JOB_ID}${_NOTE:+_${_NOTE}}.log" 2>&1

# --------------------
# Configuration
# --------------------
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch $0 <arch> <seed> [EXTRA_ARGS...]"
    echo ""
    echo "Architecture options: dp_c, dp_t, dp_t_film, dp_mlp"
    echo ""
    echo "Dataset: DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim"
    echo "  A = stack_d1              E = stack_three_d1"
    echo "  B = square_d2             F = hammer_cleanup_d1"
    echo "  C = coffee_d2             G = three_piece_assembly_d2"
    echo "  D = threading_d2          H = mug_cleanup_d1"
    echo ""
    echo "Example: sbatch $0 dp_c 42"
    exit 1
fi

ARCH="$1"
SEED="$2"
NOTE="${3:-}"
shift 3 2>/dev/null || shift 2
EXTRA_ARGS="$@"

CONFIG_NAME="dah_stage1_${ARCH}"
scontrol update JobId="$SLURM_JOB_ID" JobName="dah_s1_${ARCH}"
REPO_ID="JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim"

echo "=============================================="
echo "SLURM Job: DAH Stage 1 — ${ARCH}"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Architecture: ${ARCH}"
echo "Config: ${CONFIG_NAME}"
echo "Seed: ${SEED}"
echo "Dataset: ${REPO_ID}"
echo "Note: ${NOTE}"
echo "Extra args: ${EXTRA_ARGS}"
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

# Disable SLURM auto-detection in Lightning (single GPU training)
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

echo "GPU Information:"
nvidia-smi

# --------------------
# Run Training
# --------------------
DATE_PART=$(date +'%Y.%m.%d')
TIME_PART=$(date +'%H.%M.%S')
RUN_NAME="DAH_stage1_${ARCH}_ABCDEFGH_seed${SEED}"
if [ -n "${NOTE}" ]; then RUN_NAME="${RUN_NAME}_${NOTE}"; fi
RUN_DIR="data/outputs/${DATE_PART}/${TIME_PART}_${RUN_NAME}"

echo "Run directory: $RUN_DIR"
echo "Starting training..."

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=${SEED} \
    train_mode=stage1 \
    batch_size=256 \
    \
    dataset.repo_id="${REPO_ID}" \
    training.num_epochs=6 \
    training.checkpoint_every=1\
    \
    run_dir="${RUN_DIR}" \
    run_name="${RUN_NAME}" \
    \
    logging.project="IROS_FINAL_EXP" \
    logging.group="DAH_stage1_mimicgen_seed${SEED}" \
    logging.name="${RUN_NAME}" \
    logging.mode="offline" \
    logging.tags="[dah,stage1,${ARCH},ABCDEFGH,slurm]" \
    \
    ${EXTRA_ARGS}

touch "${RUN_DIR}/done.mark"

echo "=============================================="
echo "Stage 1 training completed!"
echo "Checkpoint dir: ${RUN_DIR}/checkpoints/"
echo "=============================================="
