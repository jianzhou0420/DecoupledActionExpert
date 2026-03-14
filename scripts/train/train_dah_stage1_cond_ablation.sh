#!/bin/bash
#
# Train DAH Stage 1 Conditioning Method Ablation (Unified DP-T)
# Tests different conditioning mechanisms for the transformer action head.
# Always trains on the combined ABCDEFGH dataset (8 tasks).
#
# Usage:
#   ./scripts/train/train_dah_stage1_cond_ablation.sh <cond_method> <seed> [EXTRA_ARGS...]
#
# Conditioning methods:
#   cross_attn  - Encoder-decoder cross-attention (Q=actions, KV=obs)
#   prefix      - Obs tokens prepended to action sequence, self-attend
#   film        - FiLM affine modulation (gamma*x + beta)
#   adaln_zero  - DiT-style adaptive LayerNorm + zero-initialized gating
#   adaln       - Adaptive LayerNorm (no zero-init gating)
#   lora_cond   - Low-rank conditioned Q/V bias injection
#   additive    - Simple projected bias addition after sublayers
#
# Examples:
#   ./scripts/train/train_dah_stage1_cond_ablation.sh cross_attn 42
#   ./scripts/train/train_dah_stage1_cond_ablation.sh adaln_zero 42
#   ./scripts/train/train_dah_stage1_cond_ablation.sh lora_cond 42 training.num_epochs=100

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# region 1. Input
# --------------------
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <cond_method> <seed> [EXTRA_ARGS...]"
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
    echo ""
    echo "Dataset: DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim"
    echo "  A = stack_d1              E = stack_three_d1"
    echo "  B = square_d2             F = hammer_cleanup_d1"
    echo "  C = coffee_d2             G = three_piece_assembly_d2"
    echo "  D = threading_d2          H = mug_cleanup_d1"
    echo ""
    echo "Example: $0 cross_attn 42"
    exit 1
fi

COND_METHOD="$1"
SEED="$2"
shift 2
EXTRA_ARGS="$@"

# Validate cond_method
VALID_METHODS="cross_attn prefix film adaln_zero adaln ada_rms_norm lora_cond additive"
if ! echo "$VALID_METHODS" | grep -qw "$COND_METHOD"; then
    echo "ERROR: Invalid cond_method '$COND_METHOD'."
    echo "Must be one of: $VALID_METHODS"
    exit 1
fi

CONFIG_NAME="dah_stage1_dp_t_unified"
REPO_ID="JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim"

echo "=========================================="
echo "DAH Stage 1 Conditioning Method Ablation"
echo "=========================================="
echo "Cond method: ${COND_METHOD}"
echo "Config: ${CONFIG_NAME}"
echo "Seed: ${SEED}"
echo "Dataset: ${REPO_ID}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=========================================="

# --------------------
# region 2. Run
# --------------------
date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
run_name="DAH_stage1_dp_t_${COND_METHOD}_ABCDEFGH_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=${SEED} \
    train_mode=stage1 \
    batch_size=256 \
    \
    policy.cond_method="${COND_METHOD}" \
    \
    dataset.repo_id="${REPO_ID}" \
    training.num_epochs=50 \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="DAH_stage1_cond_ablation" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,stage1,dp_t_unified,${COND_METHOD},ABCDEFGH]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Stage 1 conditioning ablation completed!"
echo "Method: ${COND_METHOD}"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "=========================================="
