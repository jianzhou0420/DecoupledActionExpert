"""LIBERO Stage 1 robot adaptor for FK learning.

Reads observation.state as observation and actions as action target
from the LIBERO LeRobot dataset.  Both are fetched at the same timesteps
when n_obs == n_horizon and n_latency == 0.

Supports multiple conditioning types via ``cond_type``:
  - ``"eepose"`` (default): pos[3] + quat[4] + gripper[1] = 8D
  - ``"unconditional"``: zeros[8]

LIBERO state layout: [pos(3), quat(4), gripper(1)] = 8D
LIBERO actions layout: [delta_pos(3), delta_axis_angle(3), abs_gripper(1)] = 7D

State (obs):    varies by cond_type (see above), always 8D in canonical
Action target:  delta_pos[3] + delta_axis_angle[3] + abs_gripper[1]
"""

from __future__ import annotations

import numpy as np

from vlaworkspace.adaptors.canonical import CanonicalDict, CanonicalInfo, make_canonical_obs


def _to_numpy(x):
    """Convert PyTorch Tensor to numpy array if needed."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


class LiberoStage1Robot:
    """Robot adaptor for LIBERO Stage 1 FK learning.

    Observation varies by ``cond_type`` (always 8D in canonical):
        - ``"eepose"``: pos(3) + quat(4) + gripper(1) = 8D
        - ``"unconditional"``: zeros(8)
    Action target: delta_pos(3) + delta_axis_angle(3) + abs_gripper(1)

    This adaptor is training-only — env_to_canonical and canonical_to_env
    are not implemented since Stage 1 does not run in a simulator.
    """

    VALID_COND_TYPES = ("eepose", "unconditional")

    def __init__(self, *, tasks: dict[int, str] | None = None, cond_type: str = "eepose") -> None:
        # tasks accepted for config compatibility but unused in stage1
        self.tasks = tasks or {}
        if cond_type not in self.VALID_COND_TYPES:
            raise ValueError(f"Invalid cond_type '{cond_type}'. Must be one of {self.VALID_COND_TYPES}")
        self.cond_type = cond_type

    def get_canonical_info(self) -> CanonicalInfo:
        return CanonicalInfo(
            state_type={"joint_position": "absolute"},
            state_rot_repr="none",
            action_type={"pos": "delta", "rot": "delta", "gripper": "absolute"},
            action_rot_repr="axis_angle",
            state_dims={"joint_position": 8},  # pos[3] + quat[4] + gripper[1]
            action_dims={"pos": 3, "rot": 3, "gripper": 1},
        )

    def dataset_to_canonical(self, data: dict) -> CanonicalDict:
        """Convert LeRobot LIBERO sample to canonical format for Stage 1 FK.

        Input keys:
            observation.state   [n_obs, 8]      = pos[3] + quat[4] + gripper[1]
            actions             [n_horizon, 7]   = dpos[3] + daa[3] + grip[1]

        Observation construction depends on ``cond_type``:
        - ``"eepose"``: state pos(3) + quat(4) + gripper(1) = 8D
        - ``"unconditional"``: zeros(8)
        """
        # Parse state
        state_raw = np.asarray(_to_numpy(data["observation.state"]), dtype=np.float32)

        # Parse actions: [n, 7] = dpos[3] + daa[3] + grip[1]
        actions_raw = np.asarray(_to_numpy(data["actions"]), dtype=np.float32)
        action_pos = actions_raw[..., :3]
        action_rot = actions_raw[..., 3:6]
        action_gripper = actions_raw[..., 6:7]

        # Construct 8D obs based on cond_type
        if self.cond_type == "eepose":
            n_obs = state_raw.shape[0]
            # state is already pos[3]+quat[4]+grip[1] = 8D
            state_obs = state_raw[:n_obs].copy()
        elif self.cond_type == "unconditional":
            n_obs = state_raw.shape[0]
            state_obs = np.zeros((n_obs, 8), dtype=np.float32)

        return make_canonical_obs(
            images={},
            state={"joint_position": state_obs},
            actions={"pos": action_pos, "rot": action_rot, "gripper": action_gripper},
            info=self.get_canonical_info(),
        )

    def env_to_canonical(self, data: dict) -> CanonicalDict:
        raise NotImplementedError("Stage 1 is training-only, no env inference")

    def canonical_to_env(self, canonical_action: CanonicalDict, state: dict | None = None) -> dict:
        raise NotImplementedError("Stage 1 is training-only, no env inference")

    def get_state_dim(self) -> int:
        return 8  # pos[3] + quat[4] + gripper[1]

    def get_action_dim(self) -> int:
        return 7  # dpos[3] + daa[3] + grip[1]

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return ("state/joint_position", "actions/pos", "actions/rot", "actions/gripper")

    def env_obs(self) -> dict:
        raise NotImplementedError("Stage 1 is training-only")

    def env_action(self) -> dict:
        raise NotImplementedError("Stage 1 is training-only")

    def datasets(self) -> dict:
        return {
            "observation.state": "[n_obs, 8] float32 (pos[3]+quat[4]+grip[1])",
            "actions": "[n_horizon, 7] float32 (dpos[3]+daa[3]+grip[1])",
        }
