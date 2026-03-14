"""MimicGen Stage 1 robot adaptor for FK learning.

Reads joint_position + gripper_command as observation and eePose as action
target from the standard MimicGen LeRobot dataset.  Both are fetched at
the same timesteps when n_obs == n_horizon and n_latency == 0.

Supports multiple conditioning types via ``cond_type``:
  - ``"jp"`` (default): joint_position[7] + gripper_cmd[1] = 8D
  - ``"eepose"``: eePose pos[3] + quat[4] + gripper_cmd[1] = 8D
  - ``"unconditional"``: zeros[8]

The gripper command comes from the raw ``actions`` field (last dim),
which is a binary signal (-1=close, +1=open).  This matches the original
DecoupledActionHead where JPOpen[7] = actions[:, -1].

State (obs):    varies by cond_type (see above), always 8D in canonical
Action target:  observation.eePose pos[3]+quat[4], actions gripper_cmd[1] → pos[3] + quat[4] + gripper_cmd[1]
"""

from __future__ import annotations

import numpy as np

from vlaworkspace.adaptors.canonical import CanonicalDict, CanonicalInfo, make_canonical_obs


def _to_numpy(x):
    """Convert PyTorch Tensor to numpy array if needed."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


class MimicGenStage1Robot:
    """Robot adaptor for MimicGen Stage 1 FK learning.

    Observation varies by ``cond_type`` (always 8D in canonical):
        - ``"jp"``: joint_position(7) + gripper_command(1) = 8D
        - ``"eepose"``: eePose pos(3) + quat(4) + gripper_command(1) = 8D
        - ``"unconditional"``: zeros(8)
    Action target: eePose pos(3) + quat(4), gripper_command(1) from raw actions

    The gripper_command (both obs and action) comes from ``actions[:, -1]``
    (binary -1/+1), NOT from ``eePose[:, 7:8]`` (actual finger joint position).

    This adaptor is training-only — env_to_canonical and canonical_to_env
    are not implemented since Stage 1 does not run in a simulator.
    """

    VALID_COND_TYPES = ("jp", "eepose", "unconditional")

    def __init__(self, *, tasks: dict[int, str] | None = None, cond_type: str = "jp") -> None:
        # tasks accepted for config compatibility but unused in stage1
        self.tasks = tasks or {}
        if cond_type not in self.VALID_COND_TYPES:
            raise ValueError(f"Invalid cond_type '{cond_type}'. Must be one of {self.VALID_COND_TYPES}")
        self.cond_type = cond_type

    def get_canonical_info(self) -> CanonicalInfo:
        return CanonicalInfo(
            state_type={"joint_position": "absolute"},
            state_rot_repr="none",
            action_type={"pos": "absolute", "rot": "absolute", "gripper": "absolute"},
            action_rot_repr="quaternion",
            state_dims={"joint_position": 8},  # 7 joints + 1 gripper_cmd
            action_dims={"pos": 3, "rot": 4, "gripper": 1},  # gripper from raw actions (binary cmd)
        )

    def dataset_to_canonical(self, data: dict) -> CanonicalDict:
        """Convert LeRobot sample to canonical format for Stage 1 FK.

        Input keys:
            observation.joint_position  [n_obs, 7]
            observation.eePose          [n_horizon, 9]  (used as action target)
            actions                     [n_obs, 7]      (raw actions; gripper_cmd = last dim)

        Observation construction depends on ``cond_type``:
        - ``"jp"``: joint_position(7) + gripper_cmd(1) = 8D
        - ``"eepose"``: eePose pos(3) + quat(4) + gripper_cmd(1) = 8D
        - ``"unconditional"``: zeros(8)
        """
        # Action target: pos+rot from eePose, gripper from raw actions
        ee_pose = np.asarray(_to_numpy(data["observation.eePose"]), dtype=np.float32)
        pos = ee_pose[..., :3]
        rot = ee_pose[..., 3:7]

        # Raw actions: [n, 7] = dpos[3] + axis_angle[3] + gripper_cmd[1]
        # Gripper command (last dim) is binary -1/+1, matching original DAH
        raw_actions = np.asarray(_to_numpy(data["actions"]), dtype=np.float32)
        gripper_cmd = raw_actions[..., -1:]  # [n, 1]

        # Construct 8D obs based on cond_type
        if self.cond_type == "jp":
            jp = np.asarray(_to_numpy(data["observation.joint_position"]), dtype=np.float32)
            n_obs = jp.shape[0]
            state_obs = np.concatenate([jp, gripper_cmd[:n_obs]], axis=-1)  # [n_obs, 8]
        elif self.cond_type == "eepose":
            n_obs = ee_pose.shape[0]
            # eePose obs: pos(3) + quat(4) + gripper_cmd(1) = 8D in canonical
            # The model adaptor converts quat→rot6d to produce 10D
            ee_obs = ee_pose[:n_obs, :7]  # pos[3] + quat[4]
            state_obs = np.concatenate([ee_obs, gripper_cmd[:n_obs]], axis=-1)  # [n_obs, 8]
        elif self.cond_type == "unconditional":
            jp = np.asarray(_to_numpy(data["observation.joint_position"]), dtype=np.float32)
            n_obs = jp.shape[0]
            state_obs = np.zeros((n_obs, 8), dtype=np.float32)

        return make_canonical_obs(
            images={},
            state={"joint_position": state_obs},
            actions={"pos": pos, "rot": rot, "gripper": gripper_cmd},
            info=self.get_canonical_info(),
        )

    def env_to_canonical(self, data: dict) -> CanonicalDict:
        raise NotImplementedError("Stage 1 is training-only, no env inference")

    def canonical_to_env(self, canonical_action: CanonicalDict, state: dict | None = None) -> dict:
        raise NotImplementedError("Stage 1 is training-only, no env inference")

    def get_state_dim(self) -> int:
        return 8  # 7 joints + 1 gripper_cmd

    def get_action_dim(self) -> int:
        return 8  # pos(3) + quat(4) + gripper_cmd(1)

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return ("state/joint_position", "actions/pos", "actions/rot", "actions/gripper")

    def env_obs(self) -> dict:
        raise NotImplementedError("Stage 1 is training-only")

    def env_action(self) -> dict:
        raise NotImplementedError("Stage 1 is training-only")

    def datasets(self) -> dict:
        return {
            "observation.joint_position": "[n_obs, 7] float32 (concatenated with gripper_cmd → 8D)",
            "observation.eePose": "[n_horizon, 9] float32 (action target: pos+quat from eePose)",
            "actions": "[n_obs, 7] float32 (gripper_cmd from last dim, used in both obs and action)",
        }
