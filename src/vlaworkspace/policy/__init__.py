"""
Policy modules for VLAWorkspace.

All policies implement the BasePolicy interface with two required methods:
    - compute_loss(batch) -> Tensor: Training forward pass
    - predict_action(batch) -> Tensor: Inference forward pass

Usage:
    from vlaworkspace.policy import BasePolicy, DiffusionUnetHybridImagePolicy

    # Create policy
    policy = DiffusionUnetHybridImagePolicy(config)

    # Training
    loss = policy.compute_loss(batch)

    # Inference
    actions = policy.predict_action(obs_batch)
"""

# Base classes
from vlaworkspace.policy.base_policy import (
    BasePolicy,
    PolicyOutput,
)

# DecoupledActionHead Policies
from vlaworkspace.policy.dah_dp_c import DiffusionUnetHybridImagePolicy
from vlaworkspace.policy.dah_dp_t import DiffusionTransformerHybridImagePolicy
from vlaworkspace.policy.dah_dp_mlp import DiffusionMLPHybridImagePolicy

__all__ = [
    # Base classes
    "BasePolicy",
    "PolicyOutput",
    # DecoupledActionHead Policies
    "DiffusionUnetHybridImagePolicy",
    "DiffusionTransformerHybridImagePolicy",
    "DiffusionMLPHybridImagePolicy",
]
