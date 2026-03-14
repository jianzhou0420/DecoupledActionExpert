"""DROID Stage 1 robot adaptor for FK learning.

Reads observation.state (joints+gripper) as observation and
action targets from the DROID LeRobot dataset.

Supports multiple conditioning types via ``cond_type``:
  - ``"jp"`` (default): joints[7] + binarized_gripper[1] = 8D
  - ``"unconditional"``: zeros[8]

Supports multiple action modes via ``action_mode``:
  - ``"absolute"`` (default): action.cartesian_position + action.gripper_position
      pos[3] + euler[3] → pos[3] + quat[4] + binarized_gripper[1]
  - ``"delta"``: action.cartesian_velocity + action.gripper_velocity
      dpos[3] + deuler[3] + gripper_vel[1] (no rotation conversion, passthrough)

DROID gripper values are continuous [0,1] (position) or [-1,1] (velocity)
and are binarized to {-1, +1} to match the DAH stage1 convention.

DROID uses euler angles (roll/pitch/yaw = XYZ extrinsic) for EE pose.
In absolute mode, these are converted to quaternions via scipy.
In delta mode, euler velocities are passed through as axis_angle (3D).

State (obs):    varies by cond_type (see above), always 8D in canonical
Action target:  varies by action_mode (see above)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from vlaworkspace.adaptors.canonical import CanonicalDict, CanonicalInfo, make_canonical_obs


def _to_numpy(x):
    """Convert PyTorch Tensor to numpy array if needed."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _binarize_gripper(gripper, threshold=0.5):
    """Binarize continuous [0,1] gripper to {-1, +1}."""
    return np.where(gripper > threshold, 1.0, -1.0).astype(np.float32)


class DroidStage1Robot:
    """Robot adaptor for DROID Stage 1 FK learning.

    Observation varies by ``cond_type`` (always 8D in canonical):
        - ``"jp"``: joints(7) + binarized_gripper(1) = 8D
        - ``"unconditional"``: zeros(8)

    Action target varies by ``action_mode``:
        - ``"absolute"``: pos(3) + quat(4) + binarized_gripper(1) = 8D
        - ``"delta"``: dpos(3) + deuler(3) + gripper_vel(1) = 7D

    This adaptor is training-only — env_to_canonical and canonical_to_env
    are not implemented since Stage 1 does not run in a simulator.
    """

    VALID_COND_TYPES = ("jp", "unconditional")
    VALID_ACTION_MODES = ("absolute", "delta")

    def __init__(self, *, tasks: dict[int, str] | None = None, cond_type: str = "jp", action_mode: str = "absolute") -> None:
        self.tasks = tasks or {}
        if cond_type not in self.VALID_COND_TYPES:
            raise ValueError(f"Invalid cond_type '{cond_type}'. Must be one of {self.VALID_COND_TYPES}")
        if action_mode not in self.VALID_ACTION_MODES:
            raise ValueError(f"Invalid action_mode '{action_mode}'. Must be one of {self.VALID_ACTION_MODES}")
        self.cond_type = cond_type
        self.action_mode = action_mode

    def get_canonical_info(self) -> CanonicalInfo:
        if self.action_mode == "absolute":
            return CanonicalInfo(
                state_type={"joint_position": "absolute"},
                state_rot_repr="none",
                action_type={"pos": "absolute", "rot": "absolute", "gripper": "absolute"},
                action_rot_repr="quaternion",
                state_dims={"joint_position": 8},
                action_dims={"pos": 3, "rot": 4, "gripper": 1},
            )
        else:  # delta
            return CanonicalInfo(
                state_type={"joint_position": "absolute"},
                state_rot_repr="none",
                action_type={"pos": "delta", "rot": "delta", "gripper": "absolute"},
                action_rot_repr="axis_angle",
                state_dims={"joint_position": 8},
                action_dims={"pos": 3, "rot": 3, "gripper": 1},
            )

    def dataset_to_canonical(self, data: dict) -> CanonicalDict:
        """Convert DROID LeRobot sample to canonical format for Stage 1 FK.

        Input keys depend on ``action_mode``:
          absolute:
            action.cartesian_position   [n_horizon, 6]  = pos[3] + euler[3]
            action.gripper_position     [n_horizon, 1]  = continuous [0,1]
          delta:
            action.cartesian_velocity   [n_horizon, 6]  = dpos[3] + deuler[3]
            action.gripper_velocity     [n_horizon, 1]  = [-1, 1]

        Observation (both modes):
            observation.state           [n_obs, 8]      = joints[7] + gripper[1]
        """
        # --- Action target ---
        if self.action_mode == "absolute":
            cart_pos = np.atleast_2d(np.asarray(_to_numpy(data["action.cartesian_position"]), dtype=np.float32))
            pos = cart_pos[..., :3]
            euler = cart_pos[..., 3:6]

            # Convert euler (XYZ extrinsic) → quaternion (scalar-last, scipy default)
            orig_shape = euler.shape[:-1]
            euler_flat = euler.reshape(-1, 3)
            quat_flat = R.from_euler("XYZ", euler_flat).as_quat().astype(np.float32)
            rot = quat_flat.reshape(*orig_shape, 4)

            # Gripper: binarize continuous [0,1] → {-1, +1}
            grip_action = np.atleast_1d(np.asarray(_to_numpy(data["action.gripper_position"]), dtype=np.float32))
            if grip_action.ndim == 1:
                grip_action = grip_action[..., np.newaxis]
            gripper = _binarize_gripper(grip_action)
        else:  # delta
            cart_vel = np.atleast_2d(np.asarray(_to_numpy(data["action.cartesian_velocity"]), dtype=np.float32))
            pos = cart_vel[..., :3]
            rot = cart_vel[..., 3:6]  # delta euler passthrough as axis_angle

            # Gripper velocity: [-1, 1] → binarize to {-1, +1}
            grip_vel = np.atleast_1d(np.asarray(_to_numpy(data["action.gripper_velocity"]), dtype=np.float32))
            if grip_vel.ndim == 1:
                grip_vel = grip_vel[..., np.newaxis]
            gripper = _binarize_gripper(grip_vel, threshold=0.0)

        # --- Observation ---
        obs_state = np.atleast_2d(np.asarray(_to_numpy(data["observation.state"]), dtype=np.float32))

        if self.cond_type == "jp":
            n_obs = obs_state.shape[0]
            joints = obs_state[..., :7]
            obs_gripper = _binarize_gripper(obs_state[..., 7:8])
            state_obs = np.concatenate([joints, obs_gripper], axis=-1)  # [n_obs, 8]
        elif self.cond_type == "unconditional":
            n_obs = obs_state.shape[0]
            state_obs = np.zeros((n_obs, 8), dtype=np.float32)

        return make_canonical_obs(
            images={},
            state={"joint_position": state_obs},
            actions={"pos": pos, "rot": rot, "gripper": gripper},
            info=self.get_canonical_info(),
        )

    def env_to_canonical(self, data: dict) -> CanonicalDict:
        raise NotImplementedError("Stage 1 is training-only, no env inference")

    def canonical_to_env(self, canonical_action: CanonicalDict, state: dict | None = None) -> dict:
        raise NotImplementedError("Stage 1 is training-only, no env inference")

    def get_state_dim(self) -> int:
        return 8  # 7 joints + 1 binarized gripper

    def get_action_dim(self) -> int:
        if self.action_mode == "absolute":
            return 8  # pos(3) + quat(4) + gripper(1)
        else:  # delta
            return 7  # dpos(3) + deuler(3) + gripper(1)

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return ("state/joint_position", "actions/pos", "actions/rot", "actions/gripper")

    def env_obs(self) -> dict:
        raise NotImplementedError("Stage 1 is training-only")

    def env_action(self) -> dict:
        raise NotImplementedError("Stage 1 is training-only")

    def datasets(self) -> dict:
        base = {"observation.state": "[n_obs, 8] float32 (joints[7] + gripper[1], gripper binarized → 8D)"}
        if self.action_mode == "absolute":
            base["action.cartesian_position"] = "[n_horizon, 6] float32 (pos[3] + euler[3] → pos+quat in canonical)"
            base["action.gripper_position"] = "[n_horizon, 1] float32 (continuous [0,1], binarized → {-1,+1})"
        else:  # delta
            base["action.cartesian_velocity"] = "[n_horizon, 6] float32 (dpos[3] + deuler[3], passthrough as axis_angle)"
            base["action.gripper_velocity"] = "[n_horizon, 1] float32 ([-1,1], binarized → {-1,+1})"
        return base
