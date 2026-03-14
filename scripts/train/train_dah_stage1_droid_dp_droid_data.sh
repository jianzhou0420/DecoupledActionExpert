#!/bin/bash
#
# Train DAH Stage 1: DROID DP variants on DROID dataset (local)
#
# Reuses LIBERO DROID-DP stage 1 configs (dah_stage1_droid_dp_libero, etc.)
# with CLI overrides to train on DROID data instead of LIBERO data.
#
# Usage:
#   ./scripts/train/train_dah_stage1_droid_dp_droid_data.sh <arch> <seed> [EXTRA_ARGS...]
#
# Architecture options: dp_c, dp_t, dp_mlp, dp_t_film
#
# Key overrides vs LIBERO default:
#   - Dataset: JianZhou0420/droid_lowdim (25M frames, delta actions)
#   - Robot adaptor: DroidStage1Robot (cond_type=jp, action_mode=delta)
#   - Model adaptor: cond_type=jp (from eepose), action_mode=delta
#   - IO_meta obs shape: [8] (jp) instead of [10] (eepose->rot6d)
#   - preload=false (dataset too large for memory)
#   - Step-based training (50K steps) instead of epoch-based
#
# Examples:
#   ./scripts/train/train_dah_stage1_droid_dp_droid_data.sh dp_c 42
#   ./scripts/train/train_dah_stage1_droid_dp_droid_data.sh dp_t 42 training.max_steps=10000

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --------------------
# region 1. Input
# --------------------
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <arch> <seed> [EXTRA_ARGS...]"
    echo ""
    echo "Architecture options: dp_c, dp_t, dp_mlp, dp_t_film"
    echo ""
    echo "Dataset: JianZhou0420/droid_lowdim (DROID, delta actions)"
    echo ""
    echo "Example: $0 dp_c 42"
    exit 1
fi

ARCH="$1"
SEED="$2"
shift 2
EXTRA_ARGS="$@"

# --------------------
# Architecture Lookup
# --------------------
case "$ARCH" in
    dp_c)      ARCH_INFIX="droid_dp";        ARCH_LABEL="DROID DP";        ARCH_TAGS="droid-dp" ;;
    dp_t)      ARCH_INFIX="droid_dp_t";      ARCH_LABEL="DROID DP-T";      ARCH_TAGS="droid-dp-t,transformer" ;;
    dp_mlp)    ARCH_INFIX="droid_dp_mlp";    ARCH_LABEL="DROID DP-MLP";    ARCH_TAGS="droid-dp-mlp,mlp" ;;
    dp_t_film) ARCH_INFIX="droid_dp_t_film"; ARCH_LABEL="DROID DP-T-FiLM"; ARCH_TAGS="droid-dp-t-film,transformer-film" ;;
    *) echo "ERROR: Invalid arch '$ARCH'. Must be: dp_c, dp_t, dp_mlp, dp_t_film"; exit 1 ;;
esac

REPO_ID="JianZhou0420/droid_lowdim"

echo "=========================================="
echo "DAH Stage 1 Training — ${ARCH_LABEL} on DROID"
echo "=========================================="
echo "Architecture: ${ARCH} (${ARCH_LABEL})"
echo "Config: dah_stage1_${ARCH_INFIX}_libero"
echo "Seed: ${SEED}"
echo "Dataset: ${REPO_ID}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=========================================="

# --------------------
# region 2. Run
# --------------------
date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
run_name="DAH_stage1_${ARCH_INFIX}_droid_seed${SEED}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

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
    dataloader.num_workers=16 \
    dataloader.persistent_workers=true \
    training.num_epochs=-1 \
    training.max_steps=50000 \
    training.checkpoint_every=10000 \
    \
    task_name="dah_stage1_${ARCH_INFIX}_droid" \
    \
    run_dir="${run_dir}" \
    run_name="${run_name}" \
    \
    logging.project="DAH_stage1_${ARCH_INFIX}_droid" \
    logging.name="${run_name}" \
    logging.mode="online" \
    logging.tags="[dah,stage1,${ARCH_TAGS},droid]" \
    \
    ${EXTRA_ARGS}

touch "${run_dir}/done.mark"
echo ""
echo "=========================================="
echo "Stage 1 training completed!"
echo "Checkpoint dir: ${run_dir}/checkpoints/"
echo "Norm stats: ${run_dir}/norm_stats.json"
echo "=========================================="
