# =============================================================================
# IROS Experiment Commands Reference
#
# Copy-paste container — NOT a runnable script.
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
# Conditioning methods (ablation scripts):
#   cross_attn, prefix, film, adaln_zero, adaln, lora_cond, additive, lora_cond_uncond
#
# Conditioning sources (ablation scripts):
#   jp (baseline), eepose, unconditional
#
# Seeds: 0, 42, 420
# =============================================================================


# =============================================================================
# MIMICGEN — Stage 1 (low-dim only, combined 8-task dataset)
# =============================================================================

# --- 1. DAH Stage 1 — Architecture comparison ---
#     Script: IROS_dp_stage1_mimicgen_alltasks.sh <arch> <seed> [EXTRA_ARGS...]
#     Config: dah_stage1_<arch>
#     Dataset: DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim
#     Training: 6 epochs, batch_size=256

done
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks.sh dp_c      42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks.sh dp_t      42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks.sh dp_mlp    42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks.sh dp_t_film 42  $mynote


# --- 2. DAH Stage 1 — Conditioning method ablation (dp_t_unified) ---
#     Script: IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh <cond_method> <seed> [NOTE] [EXTRA_ARGS...]
#     Config: dah_stage1_dp_t_unified
#     Dataset: DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim
#     Training: 6 epochs, batch_size=256
done
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh cross_attn        42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh film              42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh adaln_zero        42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh adaln             42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh lora_cond         42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh prefix            42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh additive          42  $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_method.sh lora_cond_uncond  42  $mynote


# --- 3. DAH Stage 1 — Conditioning source ablation ---
#     Script: IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh <arch> <seed> <cond_type> [NOTE] [EXTRA_ARGS...]
#     Config: dah_stage1_<arch>
#     Dataset: DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim
#     Training: 6 epochs, batch_size=256

done
# dp_c
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_c      42  jp            $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_c      42  eepose        $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_c      42  unconditional $mynote
# dp_t
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_t      42  jp            $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_t      42  eepose        $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_t      42  unconditional $mynote
# dp_mlp
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_mlp    42  jp            $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_mlp    42  eepose        $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_mlp    42  unconditional $mynote
# dp_t_film
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_t_film 42  jp            $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_t_film 42  eepose        $mynote
sbatch scripts/slurm/IROS/IROS_dp_stage1_mimicgen_alltasks_ablation_cond_source.sh dp_t_film 42  unconditional $mynote


# =============================================================================
# MIMICGEN — Stage 2 (per-task, loads stage1 checkpoint)
# =============================================================================

# --- 4. DAH Stage 2 — Architecture comparison ---
#     Script: IROS_dp_stage2_mimicgen_pertask.sh <arch> <stage1_ckpt> [SEED] [NOTE] [EXTRA_ARGS...]
#     Config: dah_stage2_or_normal_<arch>
#     Dataset: DAH_mimicgen_<task>_alldemos (per task via SLURM array)
#     Training: step-controlled, batch_size=128
running
export STAGE1_CKPT_dp_c="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_dp_t="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_dp_mlp="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_dp_t_film="<YOUR_STAGE1_CHECKPOINT_PATH>"

sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_c      $STAGE1_CKPT_dp_c      0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_c      $STAGE1_CKPT_dp_c      42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_c      $STAGE1_CKPT_dp_c      420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_t      $STAGE1_CKPT_dp_t      0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_t      $STAGE1_CKPT_dp_t      42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_t      $STAGE1_CKPT_dp_t      420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_mlp    $STAGE1_CKPT_dp_mlp    0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_mlp    $STAGE1_CKPT_dp_mlp    42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_mlp    $STAGE1_CKPT_dp_mlp    420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_t_film $STAGE1_CKPT_dp_t_film 0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_t_film $STAGE1_CKPT_dp_t_film 42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask.sh dp_t_film $STAGE1_CKPT_dp_t_film 420 $mynote


# --- 5. DAH Stage 2 — Conditioning method ablation (dp_t_unified) ---
#     Script: IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh <cond_method> <stage1_ckpt> [SEED] [NOTE] [EXTRA_ARGS...]
#     Config: dah_stage2_or_normal_dp_t_unified
#     Dataset: DAH_mimicgen_<task>_alldemos (per task via SLURM array)

export STAGE1_CKPT_cross_attn="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_film="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_adaln_zero="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_adaln="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_lora_cond="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_prefix="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_additive="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_lora_cond_uncond="<YOUR_STAGE1_CHECKPOINT_PATH>"

sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh cross_attn        $STAGE1_CKPT_cross_attn        0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh cross_attn        $STAGE1_CKPT_cross_attn        42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh cross_attn        $STAGE1_CKPT_cross_attn        420 $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh film              $STAGE1_CKPT_film              0   $mynote dp-t-film
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh film              $STAGE1_CKPT_film              42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh film              $STAGE1_CKPT_film              420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh adaln_zero        $STAGE1_CKPT_adaln_zero        0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh adaln_zero        $STAGE1_CKPT_adaln_zero        42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh adaln_zero        $STAGE1_CKPT_adaln_zero        420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh adaln             $STAGE1_CKPT_adaln             0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh adaln             $STAGE1_CKPT_adaln             42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh adaln             $STAGE1_CKPT_adaln             420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh lora_cond         $STAGE1_CKPT_lora_cond         0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh lora_cond         $STAGE1_CKPT_lora_cond         42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh lora_cond         $STAGE1_CKPT_lora_cond         420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh prefix            $STAGE1_CKPT_prefix            0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh prefix            $STAGE1_CKPT_prefix            42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh prefix            $STAGE1_CKPT_prefix            420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh additive          $STAGE1_CKPT_additive          0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh additive          $STAGE1_CKPT_additive          42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh additive          $STAGE1_CKPT_additive          420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh lora_cond_uncond  $STAGE1_CKPT_lora_cond_uncond  0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh lora_cond_uncond  $STAGE1_CKPT_lora_cond_uncond  42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh lora_cond_uncond  $STAGE1_CKPT_lora_cond_uncond  420 $mynote


# --- 5b. DAH Stage 2 — Conditioning source ablation ---
#     Script: IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh <arch> <stage1_ckpt> <cond_type> [SEED] [NOTE] [EXTRA_ARGS...]
#     Config: dah_stage2_or_normal_<arch>
#     Dataset: DAH_mimicgen_<task>_alldemos (per task via SLURM array)
#     Note: Same training as regular stage2 (no cond overrides). cond_type is for WandB naming only.
done
# dp_c
export STAGE1_CKPT_dp_c_jp="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_dp_c_eepose="<YOUR_STAGE1_CHECKPOINT_PATH>"
export STAGE1_CKPT_dp_c_unconditional="<YOUR_STAGE1_CHECKPOINT_PATH>"

# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_jp            jp            0   $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_jp            jp            42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_jp            jp            420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_eepose        eepose        0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_eepose        eepose        42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_eepose        eepose        420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_unconditional unconditional 0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_unconditional unconditional 42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_c      $STAGE1_CKPT_dp_c_unconditional unconditional 420 $mynote
# # dp_t
# export STAGE1_CKPT_dp_t_jp="<YOUR_STAGE1_CHECKPOINT_PATH>"
# export STAGE1_CKPT_dp_t_eepose="<YOUR_STAGE1_CHECKPOINT_PATH>"
# export STAGE1_CKPT_dp_t_unconditional="<YOUR_STAGE1_CHECKPOINT_PATH>"

# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t      $STAGE1_CKPT_dp_t_jp            jp            0   $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t      $STAGE1_CKPT_dp_t_jp            jp            42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t      $STAGE1_CKPT_dp_t_jp            jp            420 $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t      $STAGE1_CKPT_dp_t_eepose        eepose        0   $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t      $STAGE1_CKPT_dp_t_eepose        eepose        42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t      $STAGE1_CKPT_dp_t_eepose        eepose        420 $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t      $STAGE1_CKPT_dp_t_unconditional unconditional 0   $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t      $STAGE1_CKPT_dp_t_unconditional unconditional 42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t      $STAGE1_CKPT_dp_t_unconditional unconditional 420 $mynote
# # dp_mlp
# export STAGE1_CKPT_dp_mlp_jp="<YOUR_STAGE1_CHECKPOINT_PATH>"
# export STAGE1_CKPT_dp_mlp_eepose="<YOUR_STAGE1_CHECKPOINT_PATH>"
# export STAGE1_CKPT_dp_mlp_unconditional="<YOUR_STAGE1_CHECKPOINT_PATH>"

# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_mlp    $STAGE1_CKPT_dp_mlp_jp            jp            0   $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_mlp    $STAGE1_CKPT_dp_mlp_jp            jp            42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_mlp    $STAGE1_CKPT_dp_mlp_jp            jp            420 $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_mlp    $STAGE1_CKPT_dp_mlp_eepose        eepose        0   $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_mlp    $STAGE1_CKPT_dp_mlp_eepose        eepose        42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_mlp    $STAGE1_CKPT_dp_mlp_eepose        eepose        420 $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_mlp    $STAGE1_CKPT_dp_mlp_unconditional unconditional 0   $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_mlp    $STAGE1_CKPT_dp_mlp_unconditional unconditional 42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_mlp    $STAGE1_CKPT_dp_mlp_unconditional unconditional 420 $mynote
# # dp_t_film
# export STAGE1_CKPT_dp_t_film_jp="<YOUR_STAGE1_CHECKPOINT_PATH>"
# export STAGE1_CKPT_dp_t_film_eepose="<YOUR_STAGE1_CHECKPOINT_PATH>"
# export STAGE1_CKPT_dp_t_film_unconditional="<YOUR_STAGE1_CHECKPOINT_PATH>"

# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t_film $STAGE1_CKPT_dp_t_film_jp            jp            0   $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t_film $STAGE1_CKPT_dp_t_film_jp            jp            42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t_film $STAGE1_CKPT_dp_t_film_jp            jp            420 $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t_film $STAGE1_CKPT_dp_t_film_eepose        eepose        0   $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t_film $STAGE1_CKPT_dp_t_film_eepose        eepose        42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t_film $STAGE1_CKPT_dp_t_film_eepose        eepose        420 $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t_film $STAGE1_CKPT_dp_t_film_unconditional unconditional 0   $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t_film $STAGE1_CKPT_dp_t_film_unconditional unconditional 42  $mynote
# sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh dp_t_film $STAGE1_CKPT_dp_t_film_unconditional unconditional 420 $mynote

# =============================================================================
# MIMICGEN — Normal (end-to-end, no stage1 pretraining)
# =============================================================================

# --- 6. DAH Normal — Architecture comparison ---
#     Script: IROS_dp_normal_mimicgen_pertask.sh <arch> [seed] [note] [EXTRA_ARGS...]
#     Config: dah_stage2_or_normal_<arch>
#     Dataset: DAH_mimicgen_<task>_alldemos (per task via SLURM array)
done
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_c      0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_c      42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_c      420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_t      0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_t      42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_t      420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_mlp    0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_mlp    42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_mlp    420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_t_film 0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_t_film 42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask.sh dp_t_film 420 $mynote


# --- 7. DAH Normal — Conditioning method ablation (dp_t_unified) ---
#     Script: IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh <cond_method> [seed] [note] [EXTRA_ARGS...]
#     Config: dah_stage2_or_normal_dp_t_unified
#     Dataset: DAH_mimicgen_<task>_alldemos (per task via SLURM array)

sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh cross_attn        0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh cross_attn        42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh cross_attn        420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh film              0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh film              42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh film              420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh adaln_zero        0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh adaln_zero        42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh adaln_zero        420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh adaln             0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh adaln             42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh adaln             420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh lora_cond         0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh lora_cond         42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh lora_cond         420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh prefix            0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh prefix            42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh prefix            420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh additive          0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh additive          42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh additive          420 $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh lora_cond_uncond  0   $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh lora_cond_uncond  42  $mynote
sbatch --array=0-7 scripts/slurm/IROS/IROS_dp_normal_mimicgen_pertask_ablation_cond_method.sh lora_cond_uncond  420 $mynote


# =============================================================================
# LIBERO — Stage 1 (low-dim only, all suites combined)
# =============================================================================

# --- 8. DAH Stage 1 — DROID DP variants on LIBERO ---
#     Script: IROS_droid_dp_stage1_libero_allsuites.sh <arch> <seed> [NOTE] [EXTRA_ARGS...]
#     Config: dah_stage1_<arch_infix>_libero
#     Dataset: DAH_libero_all_alldemos_lowdim
#     Training: 10 epochs, batch_size=128

done
sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_libero_allsuites.sh dp_c      42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_libero_allsuites.sh dp_t      42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_libero_allsuites.sh dp_mlp    42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage1_libero_allsuites.sh dp_t_film 42  $mynote


# =============================================================================
# LIBERO — Stage 2 (per-suite, loads stage1 checkpoint)
# =============================================================================

# --- 9. DAH Stage 2 — DROID DP variants on LIBERO ---
#     Script: IROS_droid_dp_stage2_libero_persuite.sh <arch> <suite> <seed> <stage1_ckpt> [NOTE] [EXTRA_ARGS...]
#     Config: dah_stage2_<arch_infix>_libero
#     Dataset: libero_<suite>_alldemos_full
#     Training: 25,000 steps, batch_size=128
running
export LIBERO_STAGE1_CKPT_dp_c="<YOUR_STAGE1_CHECKPOINT_PATH>"
export LIBERO_STAGE1_CKPT_dp_t="<YOUR_STAGE1_CHECKPOINT_PATH>"
export LIBERO_STAGE1_CKPT_dp_mlp="<YOUR_STAGE1_CHECKPOINT_PATH>"
export LIBERO_STAGE1_CKPT_dp_t_film="<YOUR_STAGE1_CHECKPOINT_PATH>"

# dp_c
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_c      libero_spatial 42  $LIBERO_STAGE1_CKPT_dp_c      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_c      libero_object  42  $LIBERO_STAGE1_CKPT_dp_c      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_c      libero_goal    42  $LIBERO_STAGE1_CKPT_dp_c      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_c      libero_10      42  $LIBERO_STAGE1_CKPT_dp_c      $mynote
# dp_t
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_t      libero_spatial 42  $LIBERO_STAGE1_CKPT_dp_t      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_t      libero_object  42  $LIBERO_STAGE1_CKPT_dp_t      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_t      libero_goal    42  $LIBERO_STAGE1_CKPT_dp_t      $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_t      libero_10      42  $LIBERO_STAGE1_CKPT_dp_t      $mynote
# dp_mlp
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_mlp    libero_spatial 42  $LIBERO_STAGE1_CKPT_dp_mlp    $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_mlp    libero_object  42  $LIBERO_STAGE1_CKPT_dp_mlp    $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_mlp    libero_goal    42  $LIBERO_STAGE1_CKPT_dp_mlp    $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_mlp    libero_10      42  $LIBERO_STAGE1_CKPT_dp_mlp    $mynote
# dp_t_film
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_t_film libero_spatial 42  $LIBERO_STAGE1_CKPT_dp_t_film $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_t_film libero_object  42  $LIBERO_STAGE1_CKPT_dp_t_film $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_t_film libero_goal    42  $LIBERO_STAGE1_CKPT_dp_t_film $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_stage2_libero_persuite.sh dp_t_film libero_10      42  $LIBERO_STAGE1_CKPT_dp_t_film $mynote


# =============================================================================
# LIBERO — Normal (end-to-end, no stage1 pretraining)
# =============================================================================

# --- 10. DROID DP variants — Normal per-suite on LIBERO ---
#     Script: IROS_droid_dp_normal_libero_persuite.sh <arch> <suite> <seed> [NOTE] [EXTRA_ARGS...]
#     Config: <arch_infix>_libero
#     Dataset: libero_<suite>_alldemos_full
#     Training: 25,000 steps, batch_size=128
running
# dp_c
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_c      libero_spatial 42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_c      libero_object  42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_c      libero_goal    42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_c      libero_10      42  $mynote
# dp_t
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_t      libero_spatial 42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_t      libero_object  42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_t      libero_goal    42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_t      libero_10      42  $mynote
# dp_mlp
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_mlp    libero_spatial 42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_mlp    libero_object  42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_mlp    libero_goal    42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_mlp    libero_10      42  $mynote
# dp_t_film
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_t_film libero_spatial 42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_t_film libero_object  42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_t_film libero_goal    42  $mynote
sbatch scripts/slurm/IROS/IROS_droid_dp_normal_libero_persuite.sh dp_t_film libero_10      42  $mynote
