Scan the repository for all executable commands and scripts, then write them to `.claude/cmds.sh`.

## Steps:
1. Find all `.sh` scripts in the repo (scripts/train/, scripts/slurm/, setup.sh, env_install.sh)
2. Find the main Python entry point (trainer.py) and its usage patterns
3. Extract key commands from README.md
4. Write all discovered commands to `.claude/cmds.sh` with descriptions

Output format in cmds.sh:
```bash
# === Setup ===
# command description
command

# === Training ===
# command description
command
```
