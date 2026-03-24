# Development Log

This file tracks all code modifications, bug fixes, and feature additions.

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
