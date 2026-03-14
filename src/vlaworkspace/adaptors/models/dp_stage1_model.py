"""Diffusion Policy model adaptor for Stage 1 FK learning.

Subclass of DPModel that maps joint_position + gripper_qpos as the
observation state instead of eePose components.  Action handling
(quat→rot6d, gripper truncation, limits normalization) is reused from
the parent.

Model format:
    Input:
        - obs.robot0_joint_pos: [T, 8] float32  (joint_pos[7] + gripper_qpos[1])
        - action: [horizon, 10] float32  (pos[3] + rotation_6d[6] + gripper[1])
    Output:
        - action: [horizon, 10] float32
"""

from __future__ import annotations

import logging

import numpy as np

from vlaworkspace.adaptors.canonical import CanonicalDict, CanonicalInfo, make_canonical_action
from vlaworkspace.adaptors.models.dp_model import DPModel, _convert_rotation

logger = logging.getLogger(__name__)


class DPStage1Model(DPModel):
    """Model adaptor for Stage 1 FK learning with Diffusion Policy.

    Overrides the parent DPModel to:
    - Map joint_position state → obs/robot0_joint_pos
    - Normalize using data-derived stats
    - Keep action handling identical (quat→rot6d, gripper truncation, limits norm)

    Supports multiple conditioning types via ``cond_type``:
    - ``"jp"`` (default): 8D obs passthrough
    - ``"eepose"``: 8D canonical (pos+quat+grip) → 10D model (pos+rot6d+grip)
    - ``"unconditional"``: 8D zeros passthrough with dummy norm stats
    """

    VALID_COND_TYPES = ("jp", "eepose", "unconditional")
    VALID_ACTION_MODES = ("absolute", "delta")

    def __init__(self, *, cond_type: str = "jp", action_mode: str = "absolute", **kwargs):
        super().__init__(**kwargs)
        if cond_type not in self.VALID_COND_TYPES:
            raise ValueError(f"Invalid cond_type '{cond_type}'. Must be one of {self.VALID_COND_TYPES}")
        if action_mode not in self.VALID_ACTION_MODES:
            raise ValueError(f"Invalid action_mode '{action_mode}'. Must be one of {self.VALID_ACTION_MODES}")
        self.cond_type = cond_type
        self.action_mode = action_mode

    def canonical_to_model(self, canonical: CanonicalDict) -> dict:
        """Convert canonical observation to DP model input for Stage 1.

        Maps joint_position to obs, reuses parent's action conversion logic.
        No images in Stage 1.

        For ``cond_type="eepose"``, converts canonical 8D (pos+quat+grip) to
        10D (pos+rot6d+grip) to match the action representation.
        """
        data = canonical["data"]
        info = canonical["info"]

        obs = {}

        # Map state: joint_position → robot0_joint_pos
        state_data = data.get("state", {})
        if "joint_position" in state_data and state_data["joint_position"] is not None:
            jp = np.asarray(state_data["joint_position"], dtype=np.float32)
            if self.cond_type == "eepose":
                # canonical 8D: pos[3]+quat[4]+grip[1] → model 10D: pos[3]+rot6d[6]+grip[1]
                pos = jp[..., :3]
                quat = jp[..., 3:7]
                grip = jp[..., 7:8]
                rot6d = _convert_rotation(quat, "quaternion", "rotation_6d")
                jp = np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)
            obs["robot0_joint_pos"] = jp

        result = {"obs": obs}

        # Reuse parent's action conversion: quat→rot6d, gripper truncation
        action_data = data.get("actions", {})
        if action_data:
            action_components = []

            if "pos" in action_data and action_data["pos"] is not None:
                action_components.append(np.asarray(action_data["pos"], dtype=np.float32))

            if "rot" in action_data and action_data["rot"] is not None:
                rot = np.asarray(action_data["rot"], dtype=np.float32)
                rot6d = _convert_rotation(rot, info.action_rot_repr, "rotation_6d")
                action_components.append(rot6d)

            if "gripper" in action_data and action_data["gripper"] is not None:
                gripper = np.asarray(action_data["gripper"], dtype=np.float32)
                if gripper.shape[-1] > 1:
                    gripper = gripper[..., :1]
                action_components.append(gripper)

            if action_components:
                result["action"] = np.concatenate(action_components, axis=-1).astype(np.float32)

        # Normalize
        if self._norm_stats is not None:
            self._normalize_dp(result)

        return result

    def model_to_canonical(self, model_output: dict, info: CanonicalInfo) -> CanonicalDict:
        """Convert DP model output back to canonical — identical to parent."""
        return super().model_to_canonical(model_output, info)

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return ("obs/robot0_joint_pos", "action")

    def model_input(self) -> dict:
        return {
            "obs": {
                "robot0_joint_pos": "[T, 8] float32 (joint_pos[7] + gripper_qpos[1])",
            },
            "action": f"[horizon, {self.model_action_dim}] float32",
        }

    def model_output(self) -> dict:
        return {"action": f"[horizon, {self.model_action_dim}] float32"}

    def canonical_to_norm_stats_format(self, canonical: CanonicalDict) -> dict:
        """Convert canonical data to DP norm stats format for Stage 1.

        Maps joint_position to obs/robot0_joint_pos and actions via rot6d conversion.
        For ``cond_type="eepose"``, converts 8D canonical → 10D model format.
        """
        data = canonical["data"]
        info = canonical["info"]
        obs = {}

        # State: joint_position → obs/robot0_joint_pos
        state_data = data.get("state", {})
        if "joint_position" in state_data and state_data["joint_position"] is not None:
            jp = np.asarray(state_data["joint_position"], dtype=np.float32)
            if self.cond_type == "eepose":
                pos = jp[..., :3]
                quat = jp[..., 3:7]
                grip = jp[..., 7:8]
                rot6d = _convert_rotation(quat, "quaternion", "rotation_6d")
                jp = np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)
            obs["robot0_joint_pos"] = jp

        result = {}
        if obs:
            result["obs"] = obs

        # Actions: convert to rotation_6d, compose flat vector
        action_data = data.get("actions", {})
        if action_data:
            action_components = []
            if "pos" in action_data and action_data["pos"] is not None:
                action_components.append(np.asarray(action_data["pos"], dtype=np.float32))
            if "rot" in action_data and action_data["rot"] is not None:
                rot = np.asarray(action_data["rot"], dtype=np.float32)
                rot6d = _convert_rotation(rot, info.action_rot_repr, "rotation_6d")
                action_components.append(rot6d)
            if "gripper" in action_data and action_data["gripper"] is not None:
                gripper = np.asarray(action_data["gripper"], dtype=np.float32)
                if gripper.shape[-1] > 1:
                    gripper = gripper[..., :1]
                action_components.append(gripper)
            if action_components:
                result["action"] = np.concatenate(action_components, axis=-1).astype(np.float32)

        return result

    def auto_compute_norm_stats(self, hf_dataset) -> None:
        """Compute norm stats for Stage 1 obs + action-derived targets.

        Detects dataset type by column names:
        - DROID: has ``action.cartesian_position`` column
        - MimicGen: has ``observation.joint_position`` column

        Branches on ``cond_type``:
        - ``"jp"``: data-derived stats from joints + gripper = 8D
        - ``"eepose"`` (MimicGen only): eePose 8D → 10D (quat→rot6d)
        - ``"unconditional"``: dummy stats (min=-1, max=1, mean=0, std=1) for 8D

        For MimicGen, action stats use ABCDEFGH_ACTION_STATS (precomputed).
        For DROID, both obs and action stats are computed from data.
        """
        columns = hf_dataset.column_names

        if "action.cartesian_position" in columns:
            # DROID path
            self._auto_compute_norm_stats_droid(hf_dataset)
        elif "observation.joint_position" in columns:
            # MimicGen path (original)
            self._auto_compute_norm_stats_mimicgen(hf_dataset)
        elif "observation.state" in columns and "actions" in columns:
            # LIBERO path
            self._auto_compute_norm_stats_libero(hf_dataset)
        else:
            raise ValueError(
                f"Cannot detect dataset type for auto_compute_norm_stats. "
                f"Expected 'action.cartesian_position' (DROID) or "
                f"'observation.joint_position' (MimicGen) or "
                f"'observation.state'+'actions' (LIBERO). Got columns: {columns}"
            )

    def _auto_compute_norm_stats_droid(self, hf_dataset) -> None:
        """Load precomputed norm stats for DROID Stage 1.

        Uses hardcoded stats from dp_defaults.py (computed via
        scripts/compute_droid_norm_stats.py from ~1M subsampled frames).

        Obs stats: DROID_OBS_STATS (8D = joints[7] + binarized_gripper[1])
        Action stats:
          absolute: DROID_ABS_ACTION_STATS (10D = pos[3] + rot6d[6] + grip[1])
          delta: DROID_DELTA_ACTION_STATS (10D = dpos[3] + rot6d[6] + grip[1])
        """
        from vlaworkspace.adaptors.models.dp_defaults import (
            DROID_OBS_STATS,
            DROID_ABS_ACTION_STATS,
            DROID_DELTA_ACTION_STATS,
        )

        if self.cond_type == "unconditional":
            dim = 8
            obs_stats = {
                "min": np.full(dim, -1.0, dtype=np.float32),
                "max": np.full(dim, 1.0, dtype=np.float32),
                "mean": np.zeros(dim, dtype=np.float32),
                "std": np.ones(dim, dtype=np.float32),
            }
        else:
            obs_stats = DROID_OBS_STATS

        action_stats = DROID_ABS_ACTION_STATS if self.action_mode == "absolute" else DROID_DELTA_ACTION_STATS

        self._norm_stats = {
            "obs/robot0_joint_pos": obs_stats,
            "action": action_stats,
        }
        self._auto_norm_stats = False
        logger.info(
            f"Stage1 DROID norm stats (cond_type={self.cond_type}, action_mode={self.action_mode}): "
            f"obs dim={len(obs_stats['min'])}, action dim={len(action_stats['min'])} [hardcoded]"
        )

    def _auto_compute_norm_stats_mimicgen(self, hf_dataset) -> None:
        """Compute norm stats for MimicGen Stage 1 (original logic).

        Action stats use precomputed ABCDEFGH_ACTION_STATS.
        """
        from vlaworkspace.adaptors.models.dp_defaults import ABCDEFGH_ACTION_STATS

        if self.cond_type == "jp":
            # Build JPOpen[8] = joint_position(7) + gripper_cmd(1) from dataset
            jp = hf_dataset["observation.joint_position"]
            if hasattr(jp, "numpy"):
                jp = jp.numpy()
            else:
                jp = np.array(jp)
            jp = jp.astype(np.float32)

            raw_actions = hf_dataset["actions"]
            if hasattr(raw_actions, "numpy"):
                raw_actions = raw_actions.numpy()
            else:
                raw_actions = np.array(raw_actions)
            gripper_cmd = raw_actions[:, -1:].astype(np.float32)

            obs_data = np.concatenate([jp, gripper_cmd], axis=-1)  # [N, 8]

        elif self.cond_type == "eepose":
            # Build eePose obs: pos(3)+quat(4)+grip(1)=8D → convert to 10D (pos+rot6d+grip)
            ee_pose = hf_dataset["observation.eePose"]
            if hasattr(ee_pose, "numpy"):
                ee_pose = ee_pose.numpy()
            else:
                ee_pose = np.array(ee_pose)
            ee_pose = ee_pose.astype(np.float32)

            raw_actions = hf_dataset["actions"]
            if hasattr(raw_actions, "numpy"):
                raw_actions = raw_actions.numpy()
            else:
                raw_actions = np.array(raw_actions)
            gripper_cmd = raw_actions[:, -1:].astype(np.float32)

            pos = ee_pose[:, :3]
            quat = ee_pose[:, 3:7]
            rot6d = _convert_rotation(quat, "quaternion", "rotation_6d")
            obs_data = np.concatenate([pos, rot6d, gripper_cmd], axis=-1).astype(np.float32)  # [N, 10]

        elif self.cond_type == "unconditional":
            # Dummy stats: zeros map to 0 under limits normalization
            dim = 8
            obs_stats = {
                "min": np.full(dim, -1.0, dtype=np.float32),
                "max": np.full(dim, 1.0, dtype=np.float32),
                "mean": np.zeros(dim, dtype=np.float32),
                "std": np.ones(dim, dtype=np.float32),
            }
            self._norm_stats = {
                "obs/robot0_joint_pos": obs_stats,
                "action": ABCDEFGH_ACTION_STATS,
            }
            self._auto_norm_stats = False
            logger.info(
                f"Stage1 norm stats (unconditional): dummy stats "
                f"min={obs_stats['min']}, max={obs_stats['max']}"
            )
            return

        # Data-derived stats (for jp and eepose)
        obs_stats = {
            "min": obs_data.min(axis=0).astype(np.float32),
            "max": obs_data.max(axis=0).astype(np.float32),
            "mean": obs_data.mean(axis=0).astype(np.float32),
            "std": obs_data.std(axis=0).astype(np.float32),
        }

        self._norm_stats = {
            "obs/robot0_joint_pos": obs_stats,
            "action": ABCDEFGH_ACTION_STATS,
        }
        self._auto_norm_stats = False
        logger.info(
            f"Stage1 norm stats (cond_type={self.cond_type}): "
            f"obs dim={len(obs_stats['min'])}, "
            f"min={obs_stats['min']}, max={obs_stats['max']}"
        )

    def _auto_compute_norm_stats_libero(self, hf_dataset) -> None:
        """Compute norm stats for LIBERO Stage 1.

        Obs: computed from data (matches MimicGen stage1 pattern).
            eepose: 8D (pos+quat+grip) → 10D (pos+rot6d+grip)
            unconditional: dummy stats
        Action: LIBERO_ACTION_STATS (hardcoded, same across normal/stage1/stage2).
        """
        from vlaworkspace.adaptors.models.dp_defaults import LIBERO_ACTION_STATS

        if self.cond_type == "unconditional":
            dim = 8
            obs_stats = {
                "min": np.full(dim, -1.0, dtype=np.float32),
                "max": np.full(dim, 1.0, dtype=np.float32),
                "mean": np.zeros(dim, dtype=np.float32),
                "std": np.ones(dim, dtype=np.float32),
            }
        elif self.cond_type == "eepose":
            # Obs: observation.state [N, 8] = pos[3]+quat[4]+grip[1]
            # Convert to 10D: pos[3]+rot6d[6]+grip[1]
            obs_state = hf_dataset["observation.state"]
            if hasattr(obs_state, "numpy"):
                obs_state = obs_state.numpy()
            else:
                obs_state = np.array(obs_state)
            obs_state = obs_state.astype(np.float32)

            pos = obs_state[:, :3]
            quat = obs_state[:, 3:7]
            grip = obs_state[:, 7:8]
            rot6d = _convert_rotation(quat, "quaternion", "rotation_6d")
            obs_data = np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)  # [N, 10]

            obs_stats = {
                "min": obs_data.min(axis=0).astype(np.float32),
                "max": obs_data.max(axis=0).astype(np.float32),
                "mean": obs_data.mean(axis=0).astype(np.float32),
                "std": obs_data.std(axis=0).astype(np.float32),
            }
        else:
            raise ValueError(f"LIBERO stage1 does not support cond_type='{self.cond_type}'")

        self._norm_stats = {
            "obs/robot0_joint_pos": obs_stats,
            "action": LIBERO_ACTION_STATS,
        }
        self._auto_norm_stats = False
        logger.info(
            f"Stage1 LIBERO norm stats (cond_type={self.cond_type}): "
            f"obs computed from data (dim={len(obs_stats['min'])}), action hardcoded (dim={len(LIBERO_ACTION_STATS['min'])})"
        )

    def _normalize_dp(self, data: dict) -> None:
        """Apply DP normalization in-place for Stage 1.

        Normalizes robot0_joint_pos (limits mode) and action position dims.
        """
        # Normalize obs: robot0_joint_pos
        if "obs" in data:
            key = "robot0_joint_pos"
            stats_key = f"obs/{key}"
            if key in data["obs"] and stats_key in self._norm_stats:
                stats = self._norm_stats[stats_key]
                x = np.asarray(data["obs"][key], dtype=np.float32)
                data["obs"][key] = self._normalize_limits(x, stats)

        # Normalize action: only first 3 dims (position) — same as parent
        if "action" in data and "action" in self._norm_stats:
            stats = self._norm_stats["action"]
            x = np.asarray(data["action"], dtype=np.float32)
            action_dim = 3
            sliced_stats = {k: np.asarray(v, dtype=np.float32)[..., :action_dim] for k, v in stats.items()}
            normalized_pos = self._normalize_limits(x[..., :action_dim], sliced_stats)
            data["action"] = np.concatenate([normalized_pos, x[..., action_dim:]], axis=-1).astype(np.float32)
