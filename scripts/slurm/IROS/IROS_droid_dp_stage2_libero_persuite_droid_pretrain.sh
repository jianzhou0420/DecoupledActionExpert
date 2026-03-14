#!/bin/bash
#SBATCH --job-name=dah_s2_dp_lib
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
# DAH Stage 2: DROID DP variants on LIBERO — DROID Pretrained (SLURM)
#
# Same as IROS_droid_dp_stage2_libero_persuite.sh, but uses a Stage 1
# checkpoint pretrained on DROID data (from IROS_droid_dp_stage1_droid_data.sh).
#
# The DROID Stage 1 used cond_type=jp and action_mode=delta (instead of the
# LIBERO default eepose). This script applies matching overrides so the Stage 2
# model architecture is compatible with the DROID-pretrained checkpoint:
#   - adaptor.model.cond_type=jp           (8D joint positions, not 10D eepose)
#   - +adaptor.model.action_mode=delta     (velocity commands)
#   - IO_meta obs shape = [8]              (matching jp dimensions)
#
# Usage:
#   sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite_droid_pretrain.sh <arch> <suite> <seed> <stage1_ckpt> [NOTE] [EXTRA_ARGS...]
#
# Arch options: dp_c, dp_t, dp_mlp, dp_t_film
# Suite options: libero_spatial, libero_object, libero_goal, libero_10
#
# Example:
#   sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite_droid_pretrain.sh dp_c libero_spatial 42 /path/to/stage1.ckpt
#
# To launch all suites for one arch:
#   for s in libero_spatial libero_object libero_goal libero_10; do
#     sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite_droid_pretrain.sh dp_c $s 42 $CKPT
#   done
# =============================================================================

set -e

# --------------------
# Architecture Lookup
# --------------------
ARCH="${1:?Usage: sbatch $0 <arch> <suite> <seed> <stage1_ckpt> [NOTE] [EXTRA_ARGS...]  (arch: dp_c, dp_t, dp_mlp, dp_t_film)}"
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
SUITE="${2:?Usage: sbatch $0 <arch> <suite> <seed> <stage1_ckpt> [NOTE] [EXTRA_ARGS...]}"
SEED="${3:?Usage: sbatch $0 <arch> <suite> <seed> <stage1_ckpt> [NOTE] [EXTRA_ARGS...]}"
CKPT="${4:?Usage: sbatch $0 <arch> <suite> <seed> <stage1_ckpt> [NOTE] [EXTRA_ARGS...]}"
NOTE="${5:-}"
EXTRA_ARGS="${@:6}"

LOG_DIR="data/logs/$(date +'%Y.%m.%d')"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/dah_stage2_${ARCH_INFIX}_${SUITE}_droid_pretrain_${SLURM_JOB_ID}${NOTE:+_${NOTE}}.log" 2>&1

scontrol update JobId="$SLURM_JOB_ID" JobName="dah_s2_dp_${ARCH_INFIX}_${SUITE}"

REPO_ID="JianZhou0420/libero_${SUITE}_alldemos_full"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found: ${CKPT}"
    exit 1
fi

echo "=============================================="
echo "SLURM Job: DAH Stage 2 — ${ARCH_LABEL} LIBERO (${SUITE}) — DROID Pretrained"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Arch: ${ARCH} (${ARCH_LABEL})"
echo "Suite: ${SUITE}"
echo "Seed: $SEED"
echo "Stage1 ckpt: ${CKPT}"
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
RUN_NAME="DAH_stage2_${ARCH_INFIX}_${SUITE}_droid_pretrain_seed${SEED}"
if [ -n "${NOTE}" ]; then RUN_NAME="${RUN_NAME}_${NOTE}"; fi
RUN_DIR="data/outputs/${DATE_PART}/${TIME_PART}_${RUN_NAME}"

echo "Run directory: $RUN_DIR"
echo "Starting training..."

python trainer.py \
    --config-name=dah_stage2_${ARCH_INFIX}_libero \
    seed=${SEED} \
    train_mode=stage2_rollout \
    "ckpt_path='${CKPT}'" \
    \
    dataset.repo_id="${REPO_ID}" \
    adaptor.model.norm_stats_path="auto" \
    adaptor.model.cond_type=jp \
    +adaptor.model.action_mode=delta \
    IO_meta.shape_meta.obs.robot0_joint_pos.shape="[8]" \
    \
    task_name="dah_stage2_${ARCH_INFIX}_${SUITE}_droid_pretrain" \
    libero_runner.task_suites="[${SUITE}]" \
    'libero_runner.seed=[0,42,420]' \
    \
    run_dir="${RUN_DIR}" \
    run_name="${RUN_NAME}" \
    \
    dataloader.num_workers=16 \
    logging.project="IROS_FINAL_EXP" \
    logging.group="DAH_stage2_${ARCH_INFIX}_droid_pretrain_seed${SEED}" \
    logging.name="${RUN_NAME}" \
    logging.mode="offline" \
    logging.tags="[dah,stage2,${ARCH_TAGS},libero,${SUITE},per-suite,droid-pretrain,slurm]" \
    \
    ${EXTRA_ARGS}

touch "${RUN_DIR}/done.mark"

echo "=============================================="
echo "Stage 2 training completed!"
echo "Checkpoint dir: ${RUN_DIR}/checkpoints/"
echo "=============================================="
