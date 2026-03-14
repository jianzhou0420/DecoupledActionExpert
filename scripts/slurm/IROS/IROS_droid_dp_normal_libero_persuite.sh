#!/bin/bash
#SBATCH --job-name=droid_dp_libero_single
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=g015
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


# =============================================================================
# DROID Diffusion Policy variants on LIBERO — Single Suite (SLURM)
#
# Unified script for all action expert architectures:
#   dp_c     — UNet (original DROID DP)
#   dp_t     — Transformer
#   dp_mlp   — MLP+FiLM
#   dp_t_film — Transformer+FiLM
#
# Dataset: Per-suite dataset (e.g., libero_libero_spatial_alldemos_full)
# Method:  DROID DP with ResNet50 + SpatialSoftmax + action expert + DDIM
# Hardware: 1x GPU, 18 CPUs, 128GB RAM
# Training: 25,000 steps, batch_size=128
# Eval:    Rollout on the specified suite only
#
# Usage:
#   sbatch scripts/slurm/vla/train_droid_dp_libero_normal_persuite.sh <arch> <suite> <seed> [NOTE] [EXTRA_ARGS...]
#   sbatch scripts/slurm/vla/train_droid_dp_libero_normal_persuite.sh dp_c  libero_spatial 42
#   sbatch scripts/slurm/vla/train_droid_dp_libero_normal_persuite.sh dp_t libero_10 42 first_run
#
# Arch options: dp_c, dp_t, dp_mlp, dp_t_film
# Suite options: libero_spatial, libero_object, libero_goal, libero_10
#
# To launch all suites for one arch:
#   for s in libero_spatial libero_object libero_goal libero_10; do
#     sbatch scripts/slurm/vla/train_droid_dp_libero_normal_persuite.sh dp_c $s 42
#   done
# =============================================================================

set -e

# --------------------
# Architecture Lookup
# --------------------
ARCH="${1:?Usage: sbatch $0 <arch> <suite> <seed> [NOTE] [EXTRA_ARGS...]  (arch: dp_c, dp_t, dp_mlp, dp_t_film)}"
case "$ARCH" in
    dp_c)      ARCH_INFIX="droid_dp";        ARCH_LABEL="DROID DP";        ARCH_TAGS="droid-dp" ;;
    dp_t)      ARCH_INFIX="droid_dp_t";      ARCH_LABEL="DROID DP-T";      ARCH_TAGS="droid-dp-t,transformer" ;;
    dp_mlp)    ARCH_INFIX="droid_dp_mlp";    ARCH_LABEL="DROID DP-MLP";    ARCH_TAGS="droid-dp-mlp,mlp" ;;
    dp_t_film) ARCH_INFIX="droid_dp_t_film"; ARCH_LABEL="DROID DP-T-FiLM"; ARCH_TAGS="droid-dp-t-film,transformer-film" ;;
    *) echo "ERROR: Invalid arch '$ARCH'. Must be: dp_c, dp_t, dp_mlp, dp_t_film"; exit 1 ;;
esac

# --------------------
# Configuration
# --------------------
SUITE="${2:?Usage: sbatch $0 <arch> <suite> <seed> [NOTE] [EXTRA_ARGS...]}"
SEED="${3:?Usage: sbatch $0 <arch> <suite> <seed> [NOTE] [EXTRA_ARGS...]}"
NOTE="${4:-}"
EXTRA_ARGS="${@:5}"

LOG_DIR="data/logs/$(date +'%Y.%m.%d')"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/${ARCH_INFIX}_${SUITE}_${SLURM_JOB_ID}${NOTE:+_${NOTE}}.log" 2>&1

scontrol update JobId="$SLURM_JOB_ID" JobName="${ARCH_INFIX}_${SUITE}"

REPO_ID="JianZhou0420/libero_${SUITE}_alldemos_full"

echo "=============================================="
echo "SLURM Job: ${ARCH_LABEL} — ${SUITE} (single suite)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Arch: ${ARCH} (${ARCH_LABEL})"
echo "Suite: ${SUITE}"
echo "Seed: $SEED"
echo "Note: $NOTE"
echo "Dataset: $REPO_ID"
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
RUN_NAME="${ARCH_INFIX}_${SUITE}_seed${SEED}"
if [ -n "${NOTE}" ]; then RUN_NAME="${RUN_NAME}_${NOTE}"; fi
RUN_DIR="data/outputs/${DATE_PART}/${TIME_PART}_${RUN_NAME}"

echo "Run directory: $RUN_DIR"
echo "Starting training..."

python trainer.py \
    --config-name=${ARCH_INFIX}_libero \
    seed=${SEED} \
    dataset.repo_id="${REPO_ID}" \
    adaptor.model.norm_stats_path="auto" \
    \
    task_name="${ARCH_INFIX}_${SUITE}" \
    libero_runner.task_suites="[${SUITE}]" \
    'libero_runner.seed=[0,42,420]' \
    \
    run_dir="${RUN_DIR}" \
    run_name="${RUN_NAME}" \
    \
    dataloader.num_workers=16 \
    logging.project="IROS_FINAL_EXP" \
    logging.group="DAH_normal_${ARCH_INFIX}_seed${SEED}" \
    logging.name="${RUN_NAME}" \
    logging.mode="offline" \
    logging.tags="[${ARCH_TAGS},libero,${SUITE},single-suite,lerobot,language,slurm]" \
    \
    ${EXTRA_ARGS}

touch "${RUN_DIR}/done.mark"

echo "=============================================="
echo "Training completed!"
echo "Checkpoint dir: ${RUN_DIR}/checkpoints/"
echo "=============================================="
