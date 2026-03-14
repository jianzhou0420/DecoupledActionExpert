#!/bin/bash
#SBATCH --job-name=dah_s1_droid
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


# =============================================================================
# DAH Stage 1: MimicGen DP variants on DROID dataset (SLURM)
#
# Reuses MimicGen stage 1 configs (dah_stage1_dp_c, etc.) with CLI overrides
# to train on DROID data instead of MimicGen data.
#
# Architecture options: dp_c, dp_t, dp_t_film, dp_mlp
#
# Key overrides vs MimicGen default:
#   - Dataset: JianZhou0420/droid_lowdim (25M frames)
#   - obs_keys/action_keys: DROID format
#   - Robot adaptor: DroidStage1Robot (cond_type=jp)
#   - preload=false (dataset too large for memory)
#   - Step-based training (50K steps) instead of epoch-based
#   - IO_meta unchanged: obs=[8], action=[10] already match jp mode
#
# Usage:
#   sbatch scripts/slurm/IROS/IROS_dp_stage1_droid_data.sh <arch> <seed> [NOTE] [EXTRA_ARGS...]
#   sbatch scripts/slurm/IROS/IROS_dp_stage1_droid_data.sh dp_c 42
#   sbatch scripts/slurm/IROS/IROS_dp_stage1_droid_data.sh dp_t 42 first_run
# =============================================================================

set -e

_NOTE="${3:-}"
LOG_DIR="data/logs/$(date +'%Y.%m.%d')"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/train_dah_stage1_${1}_droid_${SLURM_JOB_ID}${_NOTE:+_${_NOTE}}.log" 2>&1

# --------------------
# Configuration
# --------------------
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch $0 <arch> <seed> [NOTE] [EXTRA_ARGS...]"
    echo ""
    echo "Architecture options: dp_c, dp_t, dp_t_film, dp_mlp"
    echo ""
    echo "Dataset: JianZhou0420/droid_lowdim (DROID)"
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
scontrol update JobId="$SLURM_JOB_ID" JobName="dah_s1_${ARCH}_droid"
REPO_ID="JianZhou0420/droid_lowdim"

echo "=============================================="
echo "SLURM Job: DAH Stage 1 — ${ARCH} on DROID"
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "GPU Information:"
nvidia-smi

# --------------------
# Run Training
# --------------------
DATE_PART=$(date +'%Y.%m.%d')
TIME_PART=$(date +'%H.%M.%S')
RUN_NAME="DAH_stage1_${ARCH}_droid_seed${SEED}"
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
    dataset.obs_keys="[observation.state]" \
    dataset.action_keys="[action.cartesian_position,action.gripper_position]" \
    adaptor.robot._target_=vlaworkspace.adaptors.robots.DroidStage1Robot \
    +adaptor.robot.cond_type=jp \
    preload=false \
    cache_in_memory=true \
    dataloader.num_workers=4 \
    dataloader.persistent_workers=true \
    training.num_epochs=-1 \
    +training.max_steps=50000 \
    training.checkpoint_every=null \
    +training.checkpoint_every_steps=10000 \
    \
    run_dir="${RUN_DIR}" \
    run_name="${RUN_NAME}" \
    \
    logging.project="IROS_FINAL_EXP" \
    logging.group="DAH_stage1_droid_seed${SEED}" \
    logging.name="${RUN_NAME}" \
    logging.mode="offline" \
    logging.tags="[dah,stage1,${ARCH},droid,slurm]" \
    \
    ${EXTRA_ARGS}

touch "${RUN_DIR}/done.mark"

echo "=============================================="
echo "Stage 1 training completed!"
echo "Checkpoint dir: ${RUN_DIR}/checkpoints/"
echo "=============================================="
