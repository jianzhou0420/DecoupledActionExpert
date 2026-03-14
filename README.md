# Decoupled Action Head

Official codebase for the paper [*"Decoupled Action Head: Confining Task Knowledge to Conditioning Layers"*](https://arxiv.org/abs/2511.12101) (IROS 2026).

## Overview

Diffusion Policy (DP) and other Behavior Cloning methods remain constrained by scarce paired training data, and the internal mechanisms underlying DP's effectiveness are insufficiently understood. We observe that while observation-action pairs are expensive, continuous action sequences can be generated at nearly zero cost using forward kinematics (Joint Position -> End-Effector Pose).

We propose a **decoupled training recipe** that leverages this observation-free data to pretrain a general action head, then freezes it and adapts to new tasks through conditioning layers only. This serves as both a practical training method and an analysis tool to understand where task knowledge resides in Diffusion Policy.

### Key Findings

1. **Task knowledge is confined to conditioning layers.** The near-identical performance of DP-C under normal (64.0%) vs decoupled (63.4%) training shows that the action generation backbone plays a limited role — task-specific knowledge lives in the conditioning modules.

2. **Feature modulation is the effective conditioning method under decoupling.** FiLM conditioning works well because Stage 2 can learn new (gamma, beta) parameters for the frozen backbone. Cross-attention fails (-21.1%) because frozen Q/K/V weights force new features to align with the pretrained JP representation.

3. **The action head is less critical than expected.** A 4M-parameter MLP (DP-MLP) can replace the 244M-parameter U-Net (DP-C) while preserving performance, achieving 83.9% faster training under normal training and 89.1% under decoupling.

### Method

- **Stage 1**: Pretrain the action head on observation-free data (Joint Position -> End-Effector Pose) using forward kinematics. The conditioning modules and action backbone are trained together.
- **Stage 2**: Freeze the action backbone, replace conditioning layers, and train only the observation encoders and new conditioning modules on task-specific image data.

### Evaluation Environments

- **MimicGen** (8 tasks): stack, square, coffee, threading, stack_three, hammer_cleanup, three_piece_assembly, mug_cleanup
- **LIBERO** (4 suites): libero_spatial, libero_object, libero_goal, libero_10

## Installation

### Prerequisites

- Linux (tested on Ubuntu 20.04/22.04)
- NVIDIA GPU with CUDA 12.4
- [Miniforge](https://github.com/conda-forge/miniforge) or Miniconda

### Setup

```bash
git clone --recurse-submodules https://github.com/jianzhou0420/DecoupledActionExpert.git
cd DecoupledActionExpert
bash setup.sh                  # Init submodules + apply third-party patches
bash env_install.sh            # Create conda env + install packages
conda activate DecoupledActionExpert
```

## Quick Start

### Debug (Quick Verification)

```bash
# Verify pipeline works (2 epochs)
bash scripts/train/debug_dah_stage1.sh dp_c                    # Stage 1
bash scripts/train/debug_dah_stage2.sh dp_c A <stage1-ckpt>    # Stage 2
bash scripts/train/debug_dah_normal.sh dp_c A                  # Normal (end-to-end)
```

### Training

```bash
# Stage 1: Pretrain action head on observation-free data
bash scripts/train/train_dah_stage1.sh dp_c                    # MimicGen, DP-C
bash scripts/train/train_dah_stage1.sh dp_mlp                  # MimicGen, DP-MLP

# Stage 2: Train vision encoder with frozen action head
bash scripts/train/train_dah_stage2.sh dp_c A <stage1-ckpt>    # MimicGen, per-task
bash scripts/train/train_dah_stage2_droid_dp_libero.sh <stage1-ckpt>  # LIBERO

# Normal (end-to-end baseline)
bash scripts/train/train_dah_normal.sh dp_c A                  # MimicGen, per-task
bash scripts/train/train_droid_dp_libero.sh                    # LIBERO
```

Architecture options: `dp_c` (U-Net, 244M), `dp_t` (Transformer), `dp_t_film` (Transformer+FiLM), `dp_mlp` (MLP, 4M). Task letters: A-H (see `scripts/slurm/IROS/ALL_PAPER_EXP.sh` for mapping).

## Reproducing Paper Experiments

All experiments from the paper are documented in `scripts/slurm/IROS/ALL_PAPER_EXP.sh`. This file is organized by paper section:

| Section | Experiment | Script |
|---------|-----------|--------|
| V-A | Normal vs Decoupled (DP-C) | `IROS_dp_normal_mimicgen_pertask.sh`, `IROS_dp_stage2_mimicgen_pertask.sh` |
| V-B | DROID pretraining transfer | `IROS_dp_stage2_mimicgen_pertask_droid_pretrain.sh` |
| V-C | Lightweight backbones (DP-MLP vs DP-C) | Same scripts with `dp_mlp` argument |
| V-D | Conditioning source ablation (JP vs eePose vs unconditional vs random frozen) | `IROS_dp_stage2_mimicgen_pertask_ablation_cond_source.sh` |
| V-E | Conditioning method ablation (8 methods) | `IROS_dp_stage2_mimicgen_pertask_ablation_cond_method.sh` |

Before running SLURM scripts, set `#SBATCH --account=<YOUR_ACCOUNT>` in each script.

## Project Structure

```
DecoupledActionExpert/
├── trainer.py                          # Main training entry point (Hydra + PyTorch Lightning)
├── src/vlaworkspace/
│   ├── policy/                         # Policy implementations (DAH + DROID variants)
│   ├── adaptors/                       # Data transformation (Robot + Model adaptors)
│   ├── dataset/                        # LeRobot dataset loading
│   ├── model/
│   │   ├── DecoupledActionHead/        # Core model components
│   │   │   ├── diffusion/              # Diffusion models (UNet, Transformer, MLP, unified)
│   │   │   ├── vision/                 # Vision encoders (ResNet)
│   │   │   └── common/                 # Normalizers, rotation, utilities
│   │   ├── droid/                      # DROID observation encoder
│   │   └── action_expert/              # Action head implementations
│   ├── env_runner/                     # Evaluation runners (MimicGen, LIBERO)
│   ├── config/                         # Hydra YAML configurations
│   └── z_utils/                        # Utility modules
├── scripts/
│   ├── train/                          # Local training scripts
│   └── slurm/IROS/                     # SLURM scripts for paper experiments
├── assets/                             # Normalization statistics
├── patches/                            # Third-party compatibility patches
├── third_party/                        # Git submodules (lerobot, libero, mimicgen, etc.)
├── environment.yaml                    # Conda environment specification
├── env_install.sh                      # One-command installation
└── setup.sh                            # Submodule init + patch application
```

## Datasets

Training data is hosted on HuggingFace and downloaded automatically:

- **MimicGen**: `JianZhou0420/DAH_mimicgen_<task>_alldemos` (per-task, with images)
- **MimicGen (Stage 1)**: `JianZhou0420/DAH_mimicgen_ABCDEFGH_8tasks_alldemos_lowdim` (all tasks, low-dim only)
- **LIBERO**: `JianZhou0420/libero_<suite>_alldemos_full` (per-suite, with images)
- **LIBERO (Stage 1)**: `JianZhou0420/DAH_libero_all_alldemos_lowdim` (all suites, low-dim only)

## Citation

```bibtex
@article{zhou2025decoupled,
  title={Decoupled Action Head: Confining Task Knowledge to Conditioning Layers},
  author={Zhou, Jian and Lin, Sihao and Fu, Shuai and Wu, Qi},
  journal={arXiv preprint arXiv:2511.12101},
  year={2025}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

This codebase builds on [LeRobot](https://github.com/huggingface/lerobot), [MimicGen](https://github.com/NVlabs/mimicgen), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [robomimic](https://github.com/ARISE-Initiative/robomimic), and [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
