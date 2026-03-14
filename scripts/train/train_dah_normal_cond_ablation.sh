#!/bin/bash
#
# Train DAH Normal (end-to-end) with Unified DP-T conditioning method ablation.
# No stage1 pretraining — trains images + state from scratch.
#
# Usage:
#   ./scripts/train/train_dah_normal_cond_ablation.sh <cond_method> <task_letter> <seed> [EXTRA_ARGS...]
#
# Conditioning methods:
#   cross_attn  - Encoder-decoder cross-attention
#   prefix      - Prefix tokens, self-attention
#   film        - FiLM affine modulation
#   adaln_zero  - Adaptive LayerNorm + gating
#   adaln       - Adaptive LayerNorm (no zero-init gating)
#   lora_cond   - Low-rank conditioned Q/V bias
#   additive    - Projected bias addition
#
# Examples:
#   ./scripts/train/train_dah_normal_cond_ablation.sh cross_attn A 42
#   ./scripts/train/train_dah_normal_cond_ablation.sh adaln_zero B 42 batch_size=64

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# region 1. Input
# --------------------
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <cond_method> <task_letter> <seed> [EXTRA_ARGS...]"
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
    echo "Task letters:"
    echo "  A = stack_d1              E = stack_three_d1"
    echo "  B = square_d2             F = hammer_cleanup_d1"
    echo "  C = coffee_d2             G = three_piece_assembly_d2"
    echo "  D = threading_d2          H = mug_cleanup_d1"
    echo ""
    echo "Example: $0 cross_attn A 42"
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

COND_METHOD="$1"
TASK_LETTER="$2"
SEED="$3"
shift 3
EXTRA_ARGS="$@"

# Validate cond_method
VALID_METHODS="cross_attn prefix film adaln_zero adaln ada_rms_norm lora_cond additive"
if ! echo "$VALID_METHODS" | grep -qw "$COND_METHOD"; then
    echo "ERROR: Invalid cond_method '$COND_METHOD'."
    echo "Must be one of: $VALID_METHODS"
    exit 1
fi

TASK_NAME=${TASK_MAP["$TASK_LETTER"]}
if [ -z "$TASK_NAME" ]; then
    echo "ERROR: Unknown task letter '$TASK_LETTER'. Valid: A-H."
    exit 1
fi

CONFIG_NAME="dah_stage2_or_normal_dp_t_unified"
REPO_ID="JianZhou0420/DAH_mimicgen_${TASK_NAME}_alldemos"

echo "=========================================="
echo "DAH Normal Training — Cond Method Ablation"
echo "=========================================="
echo "Cond method: ${COND_METHOD}"
echo "Config: ${CONFIG_NAME}"
echo "Task: ${TASK_LETTER} (${TASK_NAME})"
echo "Seed: ${SEED}"
echo "Dataset: ${REPO_ID}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=========================================="

# --------------------
# region 2. Run
# --------------------
date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
run_name="DAH_normal_dp_t_${COND_METHOD}_${TASK_LETTER}_${TASK_NAME}_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=${SEED} \
    train_mode=normal_rollout \
    \
    policy.cond_method="${COND_METHOD}" \
    \
    dataset.repo_id="${REPO_ID}" \
    adaptor.model.norm_stats_path="auto" \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="DAH_normal_cond_ablation" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,normal,dp_t_unified,${COND_METHOD},${TASK_LETTER}]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Normal training completed!"
echo "Method: ${COND_METHOD}"
echo "Task: ${TASK_LETTER} (${TASK_NAME})"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "=========================================="
