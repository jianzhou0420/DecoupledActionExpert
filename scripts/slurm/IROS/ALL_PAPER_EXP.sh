# =============================================================================
# IROS Experiment Commands Reference
#
# Copy-paste container — NOT a runnable script.
# Organized by paper section (§V-A through §V-E).
#
# Two experimental domains: MimicGen (DAH) and LIBERO (DROID DP).
#
# MimicGen tasks (SLURM array index → task):
#   0=stack_d1  1=square_d2  2=coffee_d2  3=threading_d2
#   4=stack_three_d1  5=hammer_cleanup_d1  6=three_piece_assembly_d2
#   7=mug_cleanup_d1
#
# LIBERO suites: libero_spatial, libero_object, libero_goal, libero_10
#
# Architecture options:
#   MimicGen: dp_c, dp_t, dp_t_film, dp_mlp
#   LIBERO:   dp_c, dp_t, dp_mlp, dp_t_film
#
# Conditioning methods (§V-E ablation):
#   cross_attn, prefix, film, adaln_zero, adaln, ada_rms_norm, lora_cond, additive, lora_cond_uncond
#
# Conditioning sources (§V-D ablation):
#   jp (baseline), eepose, unconditional
#
# Seeds: 0, 42, 420
#
# Layout format:
#   # --- <Domain>: <Mode> × <count> ---
#   #     Script: <script.sh> <args...>
#   # DONE <arch>                    ← per-arch status, or just # DONE if all done
#   # <arch>                         ← arch label
#   export VAR="..."                 ← inline checkpoint export (Decoupled only)
#   sbatch ...                       ← one line per seed or suite
#   sbatch ...
#   sbatch ...
# =============================================================================


# #############################################################################
# region                    STAGE 1 PREREQUISITES (run first)
#
# Stage 1 checkpoints are shared across multiple tables.
# Each group below is a prerequisite for specific downstream experiments.
# #############################################################################

# --- Stage 1: Architecture variants (MimicGen) ---
#     Prerequisite for: Table I (DP-C), Table III (DP-MLP, DP-T, DP-T-FiLM)
#     Script: IROS_dp_stage1_mimicgen_alltasks.sh <arch> <seed> [EXTRA_ARGS...]
#     Config: dah_stage1_<arch>
#     Dataset: DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim
#     Training: 6 epochs, batch_size=256

sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks.sh dp_c      42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks.sh dp_t      42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks.sh dp_mlp    42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks.sh dp_t_film 42  $mynote


# --- Stage 1: Conditioning method ablation (MimicGen) ---
#     Prerequisite for: Table V (§V-E, Decoupled)
#     Script: IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh <cond_method> <seed> [NOTE] [EXTRA_ARGS...]
#     Config: dah_stage1_dp_t_unified
#     Dataset: DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim
#     Training: 6 epochs, batch_size=256

sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh cross_attn        42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh film              42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh adaln_zero        42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh adaln             42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh ada_rms_norm      42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh lora_cond         42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh prefix            42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh additive          42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh lora_cond_uncond  42  $mynote


# --- Stage 1: Conditioning source ablation (MimicGen) ---
#     Prerequisite for: Table IV (§V-D)
#     Script: IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh <arch> <seed> <cond_type> [NOTE] [EXTRA_ARGS...]
#     Config: dah_stage1_<arch>
#     Dataset: DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim
#     Training: 6 epochs, batch_size=256

# dp_c
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_c      42  jp            $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_c      42  eepose        $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_c      42  unconditional $mynote


# --- Stage 1: LIBERO ---
#     Prerequisite for: Table I LIBERO (§V-A, Decoupled)
#     Script: IROS_droid_dp_stage1_libero_allsuites.sh <arch> <seed> [NOTE] [EXTRA_ARGS...]
#     Config: dah_stage1_<arch_infix>_libero
#     Dataset: DAH_libero_all_alldemos_lowdim
#     Training: 10 epochs, batch_size=128

sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_libero_allsuites.sh dp_c      42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_libero_allsuites.sh dp_t      42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_libero_allsuites.sh dp_mlp    42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_libero_allsuites.sh dp_t_film 42  $mynote

# endregion
# #############################################################################
# region               V-A: Does Decoupling Preserve Task Performance?
#
# DP-C: Normal vs Decoupled × {MimicGen (3 seeds), LIBERO (1 seed)}
# Requires: Stage 1 architecture variants (dp_c) from prerequisites
# #############################################################################
# --- MimicGen: Normal × 3 seeds ---
#     Script: IROS_dp_normal_mimicgen_pertask.sh <arch> [seed] [note] [EXTRA_ARGS...]
# DONE
# dp_c
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_c      0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_c      42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_c      420 $mynote


# --- MimicGen: Decoupled (Stage 2) × 3 seeds ---
#     Script: IROS_dp_stage2_mimicgen_pertask.sh <arch> <stage1_ckpt> [SEED] [NOTE] [EXTRA_ARGS...]
# DONE
# dp_c
export STAGE1_CKPT_dp_c="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_c      $STAGE1_CKPT_dp_c      0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_c      $STAGE1_CKPT_dp_c      42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_c      $STAGE1_CKPT_dp_c      420 $mynote


# --- LIBERO: Normal × 4 suites ---
#     Script: IROS_droid_dp_normal_libero_persuite.sh <arch> <suite> <seed> [NOTE] [EXTRA_ARGS...]
# dp_c
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_c      libero_spatial 42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_c      libero_object  42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_c      libero_goal    42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_c      libero_10      42  $mynote


# --- LIBERO: Decoupled (Stage 2) × 4 suites ---
#     Script: IROS_droid_dp_stage2_libero_persuite.sh <arch> <suite> <seed> <stage1_ckpt> [NOTE] [EXTRA_ARGS...]
# dp_c
export LIBERO_STAGE1_CKPT_dp_c="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_c      libero_spatial 42  $LIBERO_STAGE1_CKPT_dp_c      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_c      libero_object  42  $LIBERO_STAGE1_CKPT_dp_c      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_c      libero_goal    42  $LIBERO_STAGE1_CKPT_dp_c      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_c      libero_10      42  $LIBERO_STAGE1_CKPT_dp_c      $mynote


# endregion
# #############################################################################
# region               V-B: 

# stage1 
sbatch scripts/slurm/IROS/IROS_dp_stage1_droid_data.sh dp_c 42
sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_droid_data.sh dp_c 42

export STAGE1_CKPT_DROID_DATA_dp_c="<YOUR_STAGE1_CHECKPOINT_PATH>"
export LIBERO_STAGE1_CKPT_DROID_DATA_dp_c="<YOUR_STAGE1_CHECKPOINT_PATH>"

# stage2 — MimicGen downstream (DROID-pretrained stage1 → MimicGen per-task)
#     Script: IROS_dp_stage2_mimicgen_pertask_droid_pretrain.sh <arch> <stage1_ckpt> [SEED] [NOTE] [EXTRA_ARGS...]
# dp_c
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_droid_pretrain.sh dp_c      $STAGE1_CKPT_DROID_DATA_dp_c      0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_droid_pretrain.sh dp_c      $STAGE1_CKPT_DROID_DATA_dp_c      42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_droid_pretrain.sh dp_c      $STAGE1_CKPT_DROID_DATA_dp_c      420 $mynote

# stage2 — LIBERO downstream (DROID-pretrained stage1 → LIBERO per-suite)
#     Script: IROS_droid_dp_stage2_libero_persuite_droid_pretrain.sh <arch> <suite> <seed> <stage1_ckpt> [NOTE] [EXTRA_ARGS...]
# dp_c
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite_droid_pretrain.sh dp_c      libero_spatial 42  $LIBERO_STAGE1_CKPT_DROID_DATA_dp_c      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite_droid_pretrain.sh dp_c      libero_object  42  $LIBERO_STAGE1_CKPT_DROID_DATA_dp_c      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite_droid_pretrain.sh dp_c      libero_goal    42  $LIBERO_STAGE1_CKPT_DROID_DATA_dp_c      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite_droid_pretrain.sh dp_c      libero_10      42  $LIBERO_STAGE1_CKPT_DROID_DATA_dp_c      $mynote


# #############################################################################
# region               V-C: Lightweight Action Backbone
#
# DP-C (244M U-Net) vs DP-MLP (4M) × {Normal, Decoupled} × 3 seeds
# DP-C results shared with Table I above.
# Also includes DP-T, DP-T-FiLM for extended comparison.
# Requires: Stage 1 architecture variants from prerequisites
# #############################################################################

# --- MimicGen: DP-MLP Normal × 3 seeds ---
# DONE dp_c, conducted in V-A
sbatch ...
# DONE dp_mlp
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_mlp    0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_mlp    42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_mlp    420 $mynote

# --- MimicGen: DP-MLP Decoupled (Stage 2) × 3 seeds ---
# DONE dp_c, conducted in V-A
sbatch ...
# DONE dp_mlp
export STAGE1_CKPT_dp_mlp="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_mlp    $STAGE1_CKPT_dp_mlp    0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_mlp    $STAGE1_CKPT_dp_mlp    42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_mlp    $STAGE1_CKPT_dp_mlp    420 $mynote

# --- LIBERO: DP-C/DP-MLP Normal × 4 suites (extended comparison) ---
# DONE dp_c, conducted in V-A
sbatch ...
# dp_mlp
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_mlp    libero_spatial 42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_mlp    libero_object  42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_mlp    libero_goal    42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_mlp    libero_10      42  $mynote

# --- LIBERO: DP-C/DP-MLP Decoupled × 4 suites (extended comparison) ---
# DONE dp_c, conducted in V-A
sbatch ...
# dp_mlp
export LIBERO_STAGE1_CKPT_dp_mlp="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_mlp    libero_spatial 42  $LIBERO_STAGE1_CKPT_dp_mlp    $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_mlp    libero_object  42  $LIBERO_STAGE1_CKPT_dp_mlp    $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_mlp    libero_goal    42  $LIBERO_STAGE1_CKPT_dp_mlp    $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_mlp    libero_10      42  $LIBERO_STAGE1_CKPT_dp_mlp    $mynote




# endregion
# #############################################################################
# region               V-D: Ablation — Stage 1 Conditioning Source
#
# DP-C: JP vs eePose vs Unconditional vs Random frozen
# Requires: Stage 1 conditioning source ablation checkpoints from prerequisites
# #############################################################################

# --- MimicGen: Decoupled (Stage 2) × 3 seeds ---
#     Script: IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh <arch> <stage1_ckpt> <cond_type> [SEED] [NOTE] [EXTRA_ARGS...]
#     Note: Same Stage 2 training (no cond overrides). cond_type is for WandB naming only.
# DONE jp (same as V-A Decoupled dp_c)
export STAGE1_CKPT_dp_c_jp="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_jp            jp            0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_jp            jp            42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_jp            jp            420 $mynote
# eepose, 0:0-7;42:2,6,7; 420:6;
export STAGE1_CKPT_dp_c_eepose="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_eepose        eepose        0   $mynote
sbatch --array=2,6,7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_eepose        eepose        42  $mynote
sbatch --array=6 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_eepose        eepose        420 $mynote
# unconditional, 42:0,2,6,7 
export STAGE1_CKPT_dp_c_unconditional="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_unconditional unconditional 0   $mynote
sbatch --array=0,2,6,7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_unconditional unconditional 42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_unconditional unconditional 420 $mynote
# random_frozen
#     Script: train_dah_stage2_random_frozen_array.sh <arch> <ref_ckpt> [SEED] [NOTE] [EXTRA_ARGS...]
#     Note: ref_ckpt is only used to identify action head layer names/shapes — no weights loaded.
#     Create ref ckpt: python scripts/create_random_ckpt.py --config-name dah_stage1_dp_c
export REF_CKPT_dp_c="<YOUR_RANDOM_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/DAH/train_dah_stage2_random_frozen_array.sh dp_c      $REF_CKPT_dp_c      0   $mynote
sbatch --array=0-7 scripts/slurm/DAH/train_dah_stage2_random_frozen_array.sh dp_c      $REF_CKPT_dp_c      42  $mynote
sbatch --array=0-7 scripts/slurm/DAH/train_dah_stage2_random_frozen_array.sh dp_c      $REF_CKPT_dp_c      420 $mynote

# endregion
# #############################################################################
# region               V-E: Ablation — Conditioning Mechanism
#
# DP-T backbone with 9 conditioning mechanisms: cross_attn, film, adaln_zero,
# adaln, ada_rms_norm, lora_cond, prefix, additive, lora_cond_uncond
# Normal and Decoupled on MimicGen
# Requires: Stage 1 conditioning method ablation checkpoints from prerequisites
# #############################################################################

# --- MimicGen: Normal × 3 seeds ---
#     Script: IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh <cond_method> [seed] [note] [EXTRA_ARGS...]
# DONE cross_attn
# cross_attn, DP-T
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh cross_attn        42  $mynote
# DONE film
# film, DP-T-FILM
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh film              42  $mynote
# adaln_zero
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh adaln_zero        42  $mynote
# adaln
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh adaln             42  $mynote
# ada_rms_norm
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh ada_rms_norm      42  $mynote
# lora_cond
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh lora_cond         42  $mynote
# prefix
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh prefix            42  $mynote
# additive
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh additive          42  $mynote
# lora_cond_uncond
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh lora_cond_uncond  42  $mynote

# --- MimicGen: Decoupled (Stage 2) × 3 seeds ---
#     Script: IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh <cond_method> <stage1_ckpt> [SEED] [NOTE] [EXTRA_ARGS...]
# cross_attn
# export STAGE1_CKPT_cross_attn="<YOUR_STAGE1_CHECKPOINT_PATH>"
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh cross_attn        $STAGE1_CKPT_cross_attn        42  $mynote
# film
# export STAGE1_CKPT_film="<YOUR_STAGE1_CHECKPOINT_PATH>"
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh film              $STAGE1_CKPT_film              42  $mynote
# adaln_zero
export STAGE1_CKPT_adaln_zero="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh adaln_zero        $STAGE1_CKPT_adaln_zero        42  $mynote
# adaln
export STAGE1_CKPT_adaln="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh adaln             $STAGE1_CKPT_adaln             42  $mynote
# ada_rms_norm
export STAGE1_CKPT_ada_rms_norm="TODO_FILL_AFTER_STAGE1"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh ada_rms_norm      $STAGE1_CKPT_ada_rms_norm      42  $mynote
# lora_cond
export STAGE1_CKPT_lora_cond="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh lora_cond         $STAGE1_CKPT_lora_cond         42  $mynote
# prefix
export STAGE1_CKPT_prefix="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh prefix            $STAGE1_CKPT_prefix            42  $mynote
# additive
export STAGE1_CKPT_additive="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh additive          $STAGE1_CKPT_additive          42  $mynote
# lora_cond_uncond
export STAGE1_CKPT_lora_cond_uncond="<YOUR_STAGE1_CHECKPOINT_PATH>"
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh lora_cond_uncond  $STAGE1_CKPT_lora_cond_uncond  42  $mynote

# endregion
