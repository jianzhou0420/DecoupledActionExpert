"""
Dataset module for VLAWorkspace.

Uses the official lerobot library's LeRobotDataset with a convenience
factory function for configuring observation history and action horizon.

Usage:
    from vlaworkspace.dataset import create_lerobot_dataset

    dataset = create_lerobot_dataset(
        data_dir="data/libero/libero_Lerobot",
        n_horizon=50,
        n_obs=1,
    )
"""

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vlaworkspace.dataset.lerobot_dataset import (
    InMemoryTransformedDataset,
    TransformedDataset,
    create_lerobot_dataset,
)


def lerobot_collate_fn(batch):
    """Collate function for LeRobot datasets with nested dicts."""

    def stack_nested(items):
        first = items[0]
        if isinstance(first, dict):
            return {k: stack_nested([item[k] for item in items]) for k in first}
        elif isinstance(first, np.ndarray):
            return torch.from_numpy(np.stack(items, axis=0))
        elif isinstance(first, torch.Tensor):
            return torch.stack(items, dim=0)
        elif isinstance(first, (np.bool_, bool)):
            return torch.tensor(items, dtype=torch.bool)
        elif isinstance(first, (int, float)):
            return torch.tensor(items)
        elif isinstance(first, str):
            return items  # Keep strings as list
        else:
            return items

    return stack_nested(batch)


__all__ = [
    "InMemoryTransformedDataset",
    "LeRobotDataset",
    "TransformedDataset",
    "create_lerobot_dataset",
    "lerobot_collate_fn",
]
