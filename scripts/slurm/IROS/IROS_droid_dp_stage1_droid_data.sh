#!/bin/bash
#SBATCH --job-name=dah_s1_drdp_droid
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
# DAH Stage 1: DROID DP variants on DROID dataset (SLURM)
#
# Reuses LIBERO DROID-DP stage 1 configs (dah_stage1_droid_dp_libero, etc.)
# with CLI overrides to train on DROID data instead of LIBERO data.
#
# Architecture options: dp_c, dp_t, dp_mlp, dp_t_film
#
# Key overrides vs LIBERO default:
#   - Dataset: JianZhou0420/droid_lowdim (25M frames)
#   - obs_keys/action_keys: DROID velocity format (delta actions)
#   - Robot adaptor: DroidStage1Robot (cond_type=jp, action_mode=delta)
#   - Model adaptor: cond_type=jp (from eepose)
#   - IO_meta obs shape: [8] (jp) instead of [10] (eepose→rot6d)
#   - preload=false (dataset too large for memory)
#   - Step-based training (50K steps) instead of epoch-based
#
# Usage:
#   sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_droid_data.sh <arch> <seed> [NOTE] [EXTRA_ARGS...]
#   sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_droid_data.sh dp_c 42
#   sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_droid_data.sh dp_t 42 first_run
# =============================================================================

set -e

# --------------------
# Architecture Lookup
# --------------------
ARCH="${1:?Usage: sbatch $0 <arch> <seed> [NOTE] [EXTRA_ARGS...]  (arch: dp_c, dp_t, dp_mlp, dp_t_film)}"
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
SEED="${2:?Usage: sbatch $0 <arch> <seed> [NOTE] [EXTRA_ARGS...]}"
NOTE="${3:-}"
EXTRA_ARGS="${@:4}"

LOG_DIR="data/logs/$(date +'%Y.%m.%d')"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/dah_stage1_${ARCH_INFIX}_droid_${SLURM_JOB_ID}${NOTE:+_${NOTE}}.log" 2>&1

scontrol update JobId="$SLURM_JOB_ID" JobName="dah_s1_${ARCH_INFIX}_droid"

REPO_ID="JianZhou0420/droid_lowdim"

echo "=============================================="
echo "SLURM Job: DAH Stage 1 — ${ARCH_LABEL} on DROID"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Arch: ${ARCH} (${ARCH_LABEL})"
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
RUN_NAME="DAH_stage1_${ARCH_INFIX}_droid_seed${SEED}"
if [ -n "${NOTE}" ]; then RUN_NAME="${RUN_NAME}_${NOTE}"; fi
RUN_DIR="data/outputs/${DATE_PART}/${TIME_PART}_${RUN_NAME}"

echo "Run directory: $RUN_DIR"
echo "Starting training..."

python trainer.py \
    --config-name=dah_stage1_${ARCH_INFIX}_libero \
    seed=${SEED} \
    train_mode=stage1 \
    \
    dataset.repo_id="${REPO_ID}" \
    dataset.obs_keys="[observation.state]" \
    dataset.action_keys="[action.cartesian_velocity,action.gripper_velocity]" \
    adaptor.robot._target_=vlaworkspace.adaptors.robots.DroidStage1Robot \
    adaptor.robot.cond_type=jp \
    +adaptor.robot.action_mode=delta \
    adaptor.model.cond_type=jp \
    +adaptor.model.action_mode=delta \
    IO_meta.shape_meta.obs.robot0_joint_pos.shape="[8]" \
    preload=false \
    cache_in_memory=true \
    dataloader.num_workers=4 \
    dataloader.persistent_workers=true \
    training.num_epochs=-1 \
    training.max_steps=25000 \
    training.checkpoint_every=null \
    +training.checkpoint_every_steps=5000 \
    \
    task_name="dah_stage1_${ARCH_INFIX}_droid" \
    \
    run_dir="${RUN_DIR}" \
    run_name="${RUN_NAME}" \
    \
    logging.project="IROS_FINAL_EXP" \
    logging.group="DAH_stage1_droid_seed${SEED}" \
    logging.name="${RUN_NAME}" \
    logging.mode="offline" \
    logging.tags="[dah,stage1,${ARCH_TAGS},droid,slurm]" \
    \
    ${EXTRA_ARGS}

touch "${RUN_DIR}/done.mark"

echo "=============================================="
echo "Stage 1 training completed!"
echo "Checkpoint dir: ${RUN_DIR}/checkpoints/"
echo "Norm stats: ${RUN_DIR}/norm_stats.json"
echo "=============================================="
