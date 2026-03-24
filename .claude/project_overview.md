# Project Overview

**Project**: DecoupledActionExpert
**Paper**: "Decoupled Action Expert: Confining Task Knowledge to the Conditioning Pathway" (IROS 2026, arXiv:2511.12101)
**Author**: Jian Zhou, Sihao Lin, Shuai Fu, Zerui Li, Gengze Zhou, Qi Wu

## Purpose
Decoupled training recipe for Diffusion Policy in robot manipulation. Stage 1 pretrains a task-agnostic action expert on observation-free FK data; Stage 2 freezes the backbone and trains only the conditioning pathway on task-specific image data.

## Tech Stack
Python 3.10, PyTorch 2.6, PyTorch Lightning 2.6, Hydra, MuJoCo 2.3.2, HuggingFace LeRobot, wandb

## Key Entry Points
- `trainer.py` — Main Hydra + Lightning trainer
- `src/vlaworkspace/` — Core package (policy, model, adaptors, dataset, env_runner, serving, config)
- `scripts/train/` — Local training scripts (debug + full)
- `scripts/slurm/IROS/` — SLURM cluster scripts for paper experiments

## Architecture Variants
dp_c (UNet 244M), dp_t (Transformer), dp_t_film (Transformer+FiLM), dp_t_unified, dp_mlp (MLP 5M)

## Evaluation
MimicGen (8 tasks A-H), LIBERO (4 suites)

## Known Problems
None currently tracked.

---
*Last updated: 2026-03-24 — Renamed legacy TaskFusion references to DecoupledActionExpert*
