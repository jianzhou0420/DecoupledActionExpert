# DecoupledActionExpert - Project Overview

## Description

Official codebase for the IROS 2026 paper "Decoupled Action Head: Confining Task Knowledge to Conditioning Layers" ([arXiv:2511.12101](https://arxiv.org/abs/2511.12101)).

The project proposes a **decoupled training recipe** for Diffusion Policy in robot manipulation:
- **Stage 1**: Pretrain an action head on observation-free data (Joint Position → End-Effector Pose) using forward kinematics
- **Stage 2**: Freeze the action backbone, replace conditioning layers, and train only vision encoders + conditioning modules on task-specific image data

Key finding: task knowledge is confined to conditioning layers, not the diffusion backbone.

## Tech Stack

- **Language**: Python 3.10
- **Framework**: PyTorch 2.6 + PyTorch Lightning 2.6
- **Config**: Hydra + OmegaConf (YAML configs)
- **Diffusion**: Custom implementations (UNet, Transformer, MLP backbones)
- **Vision**: ResNet-based encoders, spatial softmax
- **Data**: HuggingFace LeRobot format, auto-downloaded from HuggingFace Hub
- **Simulation**: MuJoCo 2.3.2, robosuite, MimicGen, LIBERO
- **Logging**: Weights & Biases (wandb)
- **Package**: setuptools (`pip install -e .`), conda environment

## Directory Structure

```
DecoupledActionExpert/
├── trainer.py                     # Main entry point (Hydra + PyTorch Lightning trainer, ~1600 lines)
├── src/vlaworkspace/              # Core Python package
│   ├── policy/                    # Policy implementations (BasePolicy interface)
│   │   ├── base_policy.py         # Abstract base: compute_loss() + predict_action()
│   │   ├── dah_dp_c.py            # Diffusion Policy with UNet (DP-C, 244M params)
│   │   ├── dah_dp_t.py            # Diffusion Policy with Transformer (DP-T)
│   │   ├── dah_dp_t_film.py       # Transformer + FiLM conditioning (DP-T-FiLM)
│   │   ├── dah_dp_t_unified.py    # Unified Transformer variant
│   │   ├── dah_dp_mlp.py          # Lightweight MLP backbone (DP-MLP, 4M params)
│   │   ├── droid_dp.py            # DROID observation encoder + DP
│   │   ├── droid_dp_t.py          # DROID + Transformer DP
│   │   ├── droid_dp_t_film.py     # DROID + Transformer + FiLM
│   │   └── droid_dp_mlp.py        # DROID + MLP DP
│   ├── adaptors/                  # Data transformation layer
│   │   ├── adaptor.py             # Main adaptor orchestrator
│   │   ├── canonical.py           # Canonical data format
│   │   ├── models/                # Model-specific adaptors (dp_model, dp_stage1_model)
│   │   └── robots/                # Robot-specific adaptors (mimicgen, libero, droid)
│   ├── dataset/                   # Dataset loading
│   │   └── lerobot_dataset.py     # HuggingFace LeRobot dataset wrapper
│   ├── model/                     # Neural network components
│   │   ├── DecoupledActionHead/   # Core DAH model
│   │   │   ├── diffusion/         # Diffusion backbones (UNet, Transformer, MLP, FiLM)
│   │   │   ├── vision/            # Vision encoders (ResNet, spatial softmax, crop randomizer)
│   │   │   └── common/            # Normalizers, rotation transforms, utilities
│   │   ├── droid/                 # DROID observation encoder
│   │   ├── action_expert/         # Action head implementations (CNN1D, MLP, Transformer, FiLM)
│   │   └── ema_model.py           # Exponential Moving Average model wrapper
│   ├── env_runner/                # Evaluation environment runners
│   │   ├── robomimic_runner.py    # MimicGen evaluation (robosuite environments)
│   │   ├── libero_runner.py       # LIBERO evaluation
│   │   ├── base_runner.py         # Abstract base runner
│   │   ├── env/                   # Environment wrappers (robomimic, libero)
│   │   └── gym_util/              # Gym utilities (async/sync vector envs, video recording)
│   ├── config/                    # Hydra YAML configurations
│   │   ├── dah_stage1_*.yaml      # Stage 1 configs (per architecture)
│   │   ├── dah_stage2_or_normal_*.yaml  # Stage 2 / Normal training configs
│   │   ├── dah_stage1_droid_dp_*.yaml   # DROID Stage 1 configs
│   │   ├── dah_stage2_droid_dp_*.yaml   # DROID Stage 2 configs
│   │   ├── droid_dp_*.yaml        # DROID normal training configs
│   │   └── dah_config_hint.py     # Config documentation / hints
│   ├── serving/                   # Model serving infrastructure
│   │   ├── serve.py               # Serving entry point
│   │   ├── policy_server.py       # HTTP policy server
│   │   └── websocket_policy_server.py  # WebSocket policy server
│   ├── z_utils/                   # Utility modules
│   │   ├── JianFrankaPandaFK.py   # Franka Panda forward kinematics (NumPy)
│   │   ├── JianFrankaPandaFKTorch.py  # FK (PyTorch, differentiable)
│   │   ├── JianRotation.py        # Rotation utilities (NumPy)
│   │   ├── JianRotationTorch.py   # Rotation utilities (PyTorch)
│   │   ├── normalizer_action.py   # Action normalization
│   │   └── pytorch_util.py        # PyTorch helpers
│   └── normalizer.py              # Global normalizer
├── scripts/
│   ├── train/                     # Local training shell scripts (~50 scripts)
│   │   ├── debug_*.sh             # Quick debug runs (2 epochs)
│   │   ├── train_dah_stage1*.sh   # Stage 1 pretraining scripts
│   │   ├── train_dah_stage2*.sh   # Stage 2 fine-tuning scripts
│   │   ├── train_dah_normal*.sh   # Normal (end-to-end) training
│   │   └── train_droid_dp_*.sh    # DROID-based training scripts
│   ├── slurm/IROS/               # SLURM cluster scripts for paper experiments
│   │   ├── ALL_PAPER_EXP.sh       # Master index of all paper experiments
│   │   └── IROS_*.sh              # Individual experiment scripts
│   └── create_random_ckpt.py     # Utility to create random checkpoint
├── assets/                        # Pre-computed normalization statistics
│   ├── DAH_normalizers_*/         # Normalizer pickle files + norm_stats.json
│   └── JianZhou0420/*/            # Per-dataset norm_stats.json
├── patches/                       # Compatibility patches for third-party deps
│   ├── lerobot.patch
│   ├── libero.patch
│   └── mimicgen_*.patch
├── third_party/                   # Git submodules
│   ├── lerobot/                   # HuggingFace LeRobot
│   ├── libero/                    # LIBERO benchmark
│   ├── mimicgen/                  # MimicGen (+ robosuite, robomimic, task-zoo)
│   └── droid_policy_learning/     # DROID policy learning
├── data/outputs/                  # Training outputs (checkpoints, logs, media)
├── environment.yaml               # Conda environment spec
├── env_install.sh                 # One-command installation script
├── setup.sh                       # Submodule init + patch application
└── pyproject.toml                 # Python package config (vlaworkspace)
```

## Architecture Options

| Name | Architecture | Params | Config prefix |
|------|-------------|--------|---------------|
| `dp_c` | UNet (Conditional) | 244M | `dah_*_dp_c` |
| `dp_t` | Transformer | - | `dah_*_dp_t` |
| `dp_t_film` | Transformer + FiLM | - | `dah_*_dp_t_film` |
| `dp_t_unified` | Unified Transformer | - | `dah_*_dp_t_unified` |
| `dp_mlp` | MLP | 4M | `dah_*_dp_mlp` |

## Task Mapping (MimicGen)

| Letter | Task |
|--------|------|
| A | stack |
| B | square |
| C | coffee |
| D | threading |
| E | stack_three |
| F | hammer_cleanup |
| G | three_piece_assembly |
| H | mug_cleanup |

## Evaluation Environments

- **MimicGen**: 8 robosuite tasks (letters A-H above)
- **LIBERO**: 4 suites — libero_spatial, libero_object, libero_goal, libero_10

## Data Flow

```
LeRobot Dataset (HuggingFace)
    ↓
Robot Adaptor (mimicgen_robot, libero_robot, etc.)
    ↓
Model Adaptor (dp_model, dp_stage1_model)
    ↓
Canonical Format
    ↓
Policy.compute_loss() / Policy.predict_action()
```

## Key Entry Points

- **Training**: `python trainer.py --config-name=<config> [overrides...]`
- **Debug**: `bash scripts/train/debug_dah_stage1.sh dp_c`
- **Serving**: `src/vlaworkspace/serving/serve.py`

## Quick Start

```bash
# 1. Setup
git clone --recurse-submodules <repo-url>
cd DecoupledActionExpert
bash setup.sh && bash env_install.sh
conda activate DecoupledActionExpert

# 2. Debug (verify pipeline)
bash scripts/train/debug_dah_stage1.sh dp_c
bash scripts/train/debug_dah_stage2.sh dp_c A <stage1-ckpt>

# 3. Full training
bash scripts/train/train_dah_stage1.sh dp_c
bash scripts/train/train_dah_stage2.sh dp_c A <stage1-ckpt>
```
