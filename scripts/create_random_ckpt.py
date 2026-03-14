"""Create a random-weight checkpoint for the 'random frozen' ablation.

This instantiates a stage 1 policy from a Hydra config and saves its
randomly-initialized state_dict in PyTorch Lightning checkpoint format.
The resulting .ckpt file can be used as the reference_ckpt for
`freeze_random_action_head()` — it only needs correct layer names/shapes.

Usage:
    conda activate DecoupledActionExpert
    python scripts/create_random_ckpt.py --config-name dah_stage1_dp_c
    python scripts/create_random_ckpt.py --config-name dah_stage1_dp_t
    python scripts/create_random_ckpt.py --config-name dah_stage1_dp_mlp
    python scripts/create_random_ckpt.py --config-name dah_stage1_dp_t_film

Output: data/random_ckpt/<config_name>_random.ckpt
"""

import sys
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_path="../src/vlaworkspace/config",
)
def main(cfg: DictConfig):
    # Instantiate policy with random weights
    print(f"Instantiating policy from config...")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.eval()

    state_dict = policy.state_dict()
    print(f"Policy has {len(state_dict)} parameters")
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameter count: {total_params:,}")

    # Save in Lightning checkpoint format (state_dict key)
    ckpt = {"state_dict": state_dict}

    out_dir = Path(os.environ.get("PROJECT_ROOT", ".")) / "data" / "random_ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_name = cfg.get("task_name", "unknown")
    out_path = out_dir / f"{config_name}_random.ckpt"
    torch.save(ckpt, out_path)
    print(f"Saved random checkpoint to: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    # Override run dir to avoid Hydra creating output dirs
    sys.argv.extend(["hydra.run.dir=/tmp/hydra_random_ckpt", "hydra.output_subdir=null"])
    main()
