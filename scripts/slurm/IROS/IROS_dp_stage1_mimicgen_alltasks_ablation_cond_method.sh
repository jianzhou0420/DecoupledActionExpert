#!/bin/bash
#SBATCH --job-name=dah_s1_cond
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
# DAH Stage 1 Conditioning Method Ablation (SLURM)
#
# Tests 7 conditioning mechanisms for the Diffusion Policy Transformer:
#   cross_attn  - Encoder-decoder cross-attention
#   prefix      - Prefix tokens, self-attention
#   film        - FiLM affine modulation
#   adaln_zero  - Adaptive LayerNorm + gating (DiT-style)
#   adaln       - Adaptive LayerNorm (no zero-init gating)
#   lora_cond   - Low-rank conditioned Q/V bias
#   additive    - Projected bias addition
#
# Dataset: JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim
# Hardware: 1x GPU, 4 CPUs, 128GB RAM
#
# Usage:
#   sbatch scripts/slurm/DAH/train_dah_stage1_cond_method_ablation.sh <cond_method> <seed> [NOTE] [EXTRA_ARGS...]
#
# Examples:
#   sbatch scripts/slurm/DAH/train_dah_stage1_cond_method_ablation.sh cross_attn 42
#   sbatch scripts/slurm/DAH/train_dah_stage1_cond_method_ablation.sh adaln_zero 42 v1
#
# Launch all methods:
#   for m in cross_attn film adaln_zero adaln lora_cond prefix additive lora_cond_uncond; do
#       sbatch scripts/slurm/DAH/train_dah_stage1_cond_method_ablation.sh $m 42
#   done
# =============================================================================

set -e

_NOTE="${3:-}"
LOG_DIR="data/logs/$(date +'%Y.%m.%d')"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/train_dah_stage1_cond_method_${1}_${SLURM_JOB_ID}${_NOTE:+_${_NOTE}}.log" 2>&1

# --------------------
# Configuration
# --------------------
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch $0 <cond_method> <seed> [NOTE] [EXTRA_ARGS...]"
    echo ""
    echo "Conditioning methods:"
    echo "  cross_attn  - Encoder-decoder cross-attention"
    echo "  prefix      - Prefix tokens, self-attention"
    echo "  film        - FiLM affine modulation"
    echo "  adaln_zero  - Adaptive LayerNorm + gating"
    echo "  adaln       - Adaptive LayerNorm (no zero-init gating)"
    echo "  ada_rms_norm - Adaptive RMSNorm (no zero-init gating)"
    echo "  lora_cond   - Low-rank conditioned Q/V bias"
    echo "  additive    - Projected bias addition"
    echo "  lora_cond_uncond - LoRA with unconditional obs (zeros)"
    echo ""
    echo "Example: sbatch $0 cross_attn 42"
    exit 1
fi

COND_METHOD="$1"
SEED="$2"
NOTE="${3:-}"
shift 3 2>/dev/null || shift 2
EXTRA_ARGS="$@"

# Validate cond_method
VALID_METHODS="cross_attn prefix film adaln_zero adaln ada_rms_norm lora_cond additive lora_cond_uncond"
if ! echo "$VALID_METHODS" | grep -qw "$COND_METHOD"; then
    echo "ERROR: Invalid cond_method '$COND_METHOD'."
    echo "Must be one of: $VALID_METHODS"
    exit 1
fi

# Translate compound methods to policy cond_method + extra trainer args
POLICY_COND_METHOD="$COND_METHOD"
COND_EXTRA_ARGS=""
if [ "$COND_METHOD" = "lora_cond_uncond" ]; then
    POLICY_COND_METHOD="lora_cond"
    COND_EXTRA_ARGS="+adaptor.robot.cond_type=unconditional +adaptor.model.cond_type=unconditional"
fi

CONFIG_NAME="dah_stage1_dp_t_unified"
scontrol update JobId="$SLURM_JOB_ID" JobName="dah_s1_${COND_METHOD}"
REPO_ID="JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim"

echo "=============================================="
echo "SLURM Job: DAH Stage 1 Cond Method — ${COND_METHOD}"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: ${CONFIG_NAME}"
echo "Cond method: ${COND_METHOD}"
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
RUN_NAME="DAH_stage1_dp_t_${COND_METHOD}_ABCDEFGH_seed${SEED}"
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
    policy.cond_method="${POLICY_COND_METHOD}" \
    \
    dataset.repo_id="${REPO_ID}" \
    training.num_epochs=6 \
    training.checkpoint_every=1 \
    \
    run_dir="${RUN_DIR}" \
    run_name="${RUN_NAME}" \
    \
    logging.project="IROS_FINAL_EXP" \
    logging.group="DAH_stage1_mimicgen_seed${SEED}" \
    logging.name="${RUN_NAME}" \
    logging.mode="offline" \
    'logging.tags=["dah","stage1","dp_t_unified","'"${COND_METHOD}"'","ABCDEFGH","cond_ablation","slurm"]' \
    \
    ${COND_EXTRA_ARGS} \
    ${EXTRA_ARGS}

touch "${RUN_DIR}/done.mark"

echo "=============================================="
echo "Stage 1 cond method ablation completed!"
echo "Method: ${COND_METHOD}"
echo "Checkpoint dir: ${RUN_DIR}/checkpoints/"
echo "=============================================="
