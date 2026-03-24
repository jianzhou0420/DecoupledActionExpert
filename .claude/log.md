# Development Log

This file tracks all code modifications, bug fixes, and feature additions.

---

## [2026-03-24 13:41] Add Figures to README and Improve Overview

### Summary
Added paper figures (teaser, method) to README and rewrote the overview paragraph to match the paper's motivation (image diffusion vs action generation dimensionality mismatch).

### Files Modified
| File | Change Type | Description |
|------|-------------|-------------|
| `assets/figures/teaser.png` | Created | Teaser figure (image diffusion vs action generation) |
| `assets/figures/method.png` | Created | Method overview (Stage 1 & 2 pipeline) |
| `assets/figures/cond_ablation.png` | Created | Conditioning mechanism ablation chart |
| `README.md` | Modified | Added teaser and method figures, rewrote overview to discuss dimensionality mismatch |

### Features/Improvements
- Added teaser figure under Overview section
- Added method figure under Method section
- Rewrote overview paragraph to follow paper's motivation: 16,384 latent values vs 160 physically correlated action values, replanning after partial execution

---

## [2026-03-24 13:28] Rename TaskFusion to DecoupledActionExpert

### Summary
Replaced all legacy "TaskFusion" references with "DecoupledActionExpert" across 4 files (7 occurrences).

### Files Modified
| File | Change Type | Description |
|------|-------------|-------------|
| `pyproject.toml` | Modified | Updated project description |
| `trainer.py` | Modified | Updated module docstring |
| `src/vlaworkspace/policy/base_policy.py` | Modified | Updated docstring (2 occurrences) |
| `src/vlaworkspace/serving/websocket_policy_server.py` | Modified | Updated docstring and comments (3 occurrences) |

### Features/Improvements
- Removed all legacy "TaskFusion" naming from codebase

---

## [2026-03-24 13:22] Explore Project & Update README for v2 Paper

### Summary
Set up `.claude/` project understanding system and updated README.md to reflect the v2 paper (arXiv:2511.12101v2) — new title, authors, question-driven framing, and corrected experiment section IDs.

### Files Modified
| File | Change Type | Description |
|------|-------------|-------------|
| `README.md` | Modified | Updated title, authors, overview, key findings, citation, and experiment section IDs to match v2 paper |
| `.claude/PROJECT_OVERVIEW.md` | Created | Full project documentation (structure, architecture, data flow, quick start) |
| `.claude/cmds.sh` | Created | All repository commands organized by category |
| `.claude/commands/collect-commands.md` | Created | Slash command to scan repo and write commands to cmds.sh |
| `.claude/commands/understand.md` | Created | Slash command to read PROJECT_OVERVIEW.md |
| `.claude/commands/update-project-overview.md` | Created | Slash command to refresh PROJECT_OVERVIEW.md |

### Features/Improvements
- **README title**: "Decoupled Action Head" → "Decoupled Action Expert: Confining Task Knowledge to the Conditioning Pathway"
- **Authors**: Added Zerui Li and Gengze Zhou (4 → 6 authors)
- **Key Findings → Questions We Ask**: Reframed as research questions with softened language, ordered to match paper sections IV-A through IV-E
- **Conditioning ablation**: Expanded from FiLM/Cross-Attention only to full 7-method table from Section IV-E
- **Experiment table**: Updated section IDs from V-A/B/C/D/E to IV-A/B/C/D/E, reordered to match paper, fixed "8 methods" → "7 methods"
- **Citation**: Updated title and author list

### Notes
- Numbers updated to v2: MimicGen DP-C 63.6±1.8% (normal) vs 62.2±2.3% (decoupled), LIBERO 79.3±0.6% vs 76.8±0.9%
- MLP parameter count updated from 4M to 5M per v2

---
