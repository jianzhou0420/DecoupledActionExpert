#!/bin/bash
#SBATCH --job-name=dah_s1_abl
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
# DAH Stage 1 Conditioning Ablation (SLURM)
#
# Tests different conditioning sources for the UNet action head during
# stage 1 pretraining:
#   jp            - joint_position[7]+gripper[1]=8D obs (baseline)
#   eepose        - eePose pos[3]+rot6d[6]+gripper[1]=10D obs
#   unconditional - zeros[8] obs
#
# Dataset: JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim
# Hardware: 1x GPU, 4 CPUs, 128GB RAM
#
# Usage:
#   sbatch scripts/slurm/DAH/train_dah_stage1_ablation_all.sh <arch> <seed> <cond_type> [NOTE] [EXTRA_ARGS...]
#   sbatch scripts/slurm/DAH/train_dah_stage1_ablation_all.sh dp_c 42 eepose
#   sbatch scripts/slurm/DAH/train_dah_stage1_ablation_all.sh dp_c 42 unconditional ablation_v1
#
# Architecture options: dp_c, dp_t, dp_t_film, dp_mlp
# =============================================================================

set -e

_NOTE="${4:-}"
LOG_DIR="data/logs/$(date +'%Y.%m.%d')"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/train_dah_stage1_ablation_all_${1}_${3}_${SLURM_JOB_ID}${_NOTE:+_${_NOTE}}.log" 2>&1

# --------------------
# Configuration
# --------------------
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: sbatch $0 <arch> <seed> <cond_type> [NOTE] [EXTRA_ARGS...]"
    echo ""
    echo "Architecture options: dp_c, dp_t, dp_t_film, dp_mlp"
    echo ""
    echo "Conditioning types:"
    echo "  jp            - joint_position[7]+gripper[1]=8D (baseline)"
    echo "  eepose        - eePose pos[3]+rot6d[6]+gripper[1]=10D"
    echo "  unconditional - zeros[8]"
    echo ""
    echo "Example: sbatch $0 dp_c 42 eepose"
    exit 1
fi

ARCH="$1"
SEED="$2"
COND_TYPE="$3"
NOTE="${4:-}"
shift 4 2>/dev/null || shift 3
EXTRA_ARGS="$@"

# Validate cond_type
if [ "$COND_TYPE" != "jp" ] && [ "$COND_TYPE" != "eepose" ] && [ "$COND_TYPE" != "unconditional" ]; then
    echo "ERROR: Invalid cond_type '$COND_TYPE'. Must be jp, eepose, or unconditional."
    exit 1
fi

CONFIG_NAME="dah_stage1_${ARCH}"
scontrol update JobId="$SLURM_JOB_ID" JobName="dah_s1_${ARCH}_${COND_TYPE}"
REPO_ID="JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim"

echo "=============================================="
echo "SLURM Job: DAH Stage 1 Ablation — ${ARCH} / ${COND_TYPE}"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Architecture: ${ARCH}"
echo "Config: ${CONFIG_NAME}"
echo "Seed: ${SEED}"
echo "Conditioning: ${COND_TYPE}"
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
# Build cond_type overrides
# --------------------
COND_OVERRIDES="+adaptor.robot.cond_type=${COND_TYPE} +adaptor.model.cond_type=${COND_TYPE}"

# eePose variant uses 10D obs (pos+rot6d+grip) — override shape_meta
if [ "$COND_TYPE" = "eepose" ]; then
    COND_OVERRIDES="${COND_OVERRIDES} IO_meta.shape_meta.obs.robot0_joint_pos.shape=[10]"
    echo "Note: eePose variant overrides obs shape to [10]"
fi

# --------------------
# Run Training
# --------------------
DATE_PART=$(date +'%Y.%m.%d')
TIME_PART=$(date +'%H.%M.%S')
RUN_NAME="DAH_stage1_${ARCH}_${COND_TYPE}_ABCDEFGH_seed${SEED}"
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
    training.checkpoint_every=1 \
    preload_path="data/cache/dah_stage1_${ARCH}_${COND_TYPE}_preloaded.pt" \
    \
    ${COND_OVERRIDES} \
    \
    run_dir="${RUN_DIR}" \
    run_name="${RUN_NAME}" \
    \
    logging.project="IROS_FINAL_EXP" \
    logging.group="DAH_stage1_mimicgen_seed${SEED}" \
    logging.name="${RUN_NAME}" \
    logging.mode="offline" \
    logging.tags="[dah,stage1,${ARCH},${COND_TYPE},ABCDEFGH,ablation,slurm]" \
    \
    ${EXTRA_ARGS}

touch "${RUN_DIR}/done.mark"

echo "=============================================="
echo "Stage 1 ablation training completed!"
echo "Conditioning: ${COND_TYPE}"
echo "Checkpoint dir: ${RUN_DIR}/checkpoints/"
echo "=============================================="
