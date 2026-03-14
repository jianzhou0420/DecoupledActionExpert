#!/bin/bash
#
# Train DAH Stage 1 Conditioning Ablation: test different conditioning sources
# Always trains on the combined ABCDEFGH dataset (8 tasks).
#
# Usage:
#   ./scripts/train/train_dah_stage1_ablation.sh <arch> <seed> <cond_type> [EXTRA_ARGS...]
#
# Architecture options: dp_c, dp_t, dp_t_film, dp_mlp
# Conditioning types:
#   jp            - joint_position[7]+gripper[1]=8D obs (baseline, same as train_dah_stage1.sh)
#   eepose        - eePose pos[3]+rot6d[6]+gripper[1]=10D obs
#   unconditional - zeros[8] obs
#
# Examples:
#   ./scripts/train/train_dah_stage1_ablation.sh dp_c 42 jp
#   ./scripts/train/train_dah_stage1_ablation.sh dp_c 42 eepose
#   ./scripts/train/train_dah_stage1_ablation.sh dp_c 42 unconditional
#   ./scripts/train/train_dah_stage1_ablation.sh dp_c 42 eepose training.num_epochs=2

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# region 1. Input
# --------------------
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <arch> <seed> <cond_type> [EXTRA_ARGS...]"
    echo ""
    echo "Architecture options: dp_c, dp_t, dp_t_film, dp_mlp"
    echo ""
    echo "Conditioning types:"
    echo "  jp            - joint_position[7]+gripper[1]=8D (baseline)"
    echo "  eepose        - eePose pos[3]+rot6d[6]+gripper[1]=10D"
    echo "  unconditional - zeros[8]"
    echo ""
    echo "Dataset: DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim"
    echo "  A = stack_d1              E = stack_three_d1"
    echo "  B = square_d2             F = hammer_cleanup_d1"
    echo "  C = coffee_d2             G = three_piece_assembly_d2"
    echo "  D = threading_d2          H = mug_cleanup_d1"
    echo ""
    echo "Example: $0 dp_c 42 eepose"
    exit 1
fi

ARCH="$1"
SEED="$2"
COND_TYPE="$3"
shift 3
EXTRA_ARGS="$@"

# Validate cond_type
if [ "$COND_TYPE" != "jp" ] && [ "$COND_TYPE" != "eepose" ] && [ "$COND_TYPE" != "unconditional" ]; then
    echo "ERROR: Invalid cond_type '$COND_TYPE'. Must be jp, eepose, or unconditional."
    exit 1
fi

CONFIG_NAME="dah_stage1_${ARCH}"
REPO_ID="JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim"

echo "=========================================="
echo "DAH Stage 1 Conditioning Ablation"
echo "=========================================="
echo "Architecture: ${ARCH}"
echo "Config: ${CONFIG_NAME}"
echo "Seed: ${SEED}"
echo "Conditioning: ${COND_TYPE}"
echo "Dataset: ${REPO_ID}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=========================================="

# --------------------
# region 2. Build cond_type overrides
# --------------------
COND_OVERRIDES="+adaptor.robot.cond_type=${COND_TYPE} +adaptor.model.cond_type=${COND_TYPE}"

# eePose variant uses 10D obs (pos+rot6d+grip) — override shape_meta
if [ "$COND_TYPE" = "eepose" ]; then
    COND_OVERRIDES="${COND_OVERRIDES} IO_meta.shape_meta.obs.robot0_joint_pos.shape=[10]"
    echo "Note: eePose variant overrides obs shape to [10]"
fi

# --------------------
# region 3. Run
# --------------------
date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
run_name="DAH_stage1_${ARCH}_${COND_TYPE}_ABCDEFGH_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

python trainer.py \
    --config-name="${CONFIG_NAME}" \
    seed=${SEED} \
    train_mode=stage1 \
    batch_size=256 \
    \
    dataset.repo_id="${REPO_ID}" \
    training.num_epochs=50 \
    preload_path="data/cache/dah_stage1_${ARCH}_${COND_TYPE}_preloaded.pt" \
    \
    ${COND_OVERRIDES} \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="DAH_stage1_${ARCH}" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,stage1,${ARCH},${COND_TYPE},ABCDEFGH]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Stage 1 ablation training completed!"
echo "Conditioning: ${COND_TYPE}"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "=========================================="
