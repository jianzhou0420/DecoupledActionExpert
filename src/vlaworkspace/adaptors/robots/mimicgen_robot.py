"""MimicGen robot adaptor.

Converts between MimicGen robot format and canonical intermediate format.

MimicGen state (eePose): pos[3] + quat[4] + gripper_qpos[2] = 9D
MimicGen actions: pos[3] + axis_angle[3] + gripper[1] = 7D (absolute)
Cameras: front=observation.image (84x84), wrist=observation.wrist_image (84x84)
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from vlaworkspace.adaptors.canonical import CanonicalDict, CanonicalInfo, make_canonical_action, make_canonical_obs

logger = logging.getLogger(__name__)


def _to_numpy(x):
    """Convert PyTorch Tensor to numpy array if needed."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _parse_image_to_chw_float(image) -> np.ndarray:
    """Parse image to CHW float32 [0,1] format."""
    image = np.asarray(_to_numpy(image), dtype=None)

    if image.dtype == np.uint8:
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        elif image.ndim == 4 and image.shape[-1] in (1, 3, 4):
            image = np.transpose(image, (0, 3, 1, 2))  # THWC -> TCHW
        return image.astype(np.float32) / 255.0

    image = image.astype(np.float32)
    if image.ndim == 3 and image.shape[-1] in (1, 3, 4) and image.shape[0] not in (1, 3, 4):
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    elif image.ndim == 4 and image.shape[-1] in (1, 3, 4) and image.shape[1] not in (1, 3, 4):
        image = np.transpose(image, (0, 3, 1, 2))  # THWC -> TCHW

    if image.max() > 1.0:
        image = image / 255.0

    return image


class MimicGenRobot:
    """Robot adaptor for MimicGen environment.

    State layout (eePose): [pos(3), quat(4), gripper_qpos(2)] = 9D
    Action layout: [pos(3), axis_angle(3), gripper(1)] = 7D (absolute)

    Args:
        tasks: Task index to prompt mapping.
    """

    # 12 MimicGen task instructions (fallback if tasks not provided)
    TASK_INSTRUCTIONS: ClassVar[dict[int, str]] = {
        0: "pick up the red cube and stack it on the green cube",
        1: "pick up the square nut and place it on the square peg",
        2: "pick up the coffee pod, place it in the coffee machine pod holder, and close the lid",
        3: "pick up the needle and thread it through the ring on the tripod",
        4: "stack the three cubes on top of each other in order",
        5: "pick up the hammer, place it in the drawer, and close the drawer",
        6: "assemble the three pieces by placing piece 1 and piece 2 onto the base",
        7: "pick up the mug, place it in the drawer, and close the drawer",
        8: "pick up the round nut and square nut and place them on their corresponding pegs",
        9: "put the bread in the pot, place the pot on the stove, turn on the stove, then turn off the stove and move the pot to the serving region",
        10: "pick up the milk, cereal, bread and can from the bin and place them in the correct target bins",
        11: "open the drawer, take out the mug and coffee pod, place the mug under the coffee machine, place the pod in the pod holder, and close the lid",
    }

    # Key remap for env observations
    ENV_KEY_REMAP: ClassVar[dict[str, str]] = {
        "agentview_image": "image",
        "robot0_eye_in_hand_image": "wrist_image",
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "robot0_eef_pos": "eef_pos",
        "robot0_eef_quat": "eef_quat",
        "robot0_gripper_qpos": "gripper_qpos",
    }

    def __init__(
        self,
        *,
        tasks: dict[int, str] | None = None,
    ) -> None:
        self.tasks = tasks if tasks is not None else self.TASK_INSTRUCTIONS.copy()

    def get_canonical_info(self) -> CanonicalInfo:
        return CanonicalInfo(
            state_type={"pos": "absolute", "rot": "absolute", "gripper": "absolute"},
            state_rot_repr="quat",
            action_type={"pos": "absolute", "rot": "absolute", "gripper": "absolute"},
            action_rot_repr="axis_angle",
            state_dims={"pos": 3, "rot": 4, "gripper": 2},
            action_dims={"pos": 3, "rot": 3, "gripper": 1},
        )

    def dataset_to_canonical(self, data: dict) -> CanonicalDict:
        """Convert LeRobot MimicGen sample to canonical format.

        Input keys: observation.image, observation.wrist_image, observation.eePose,
                    actions, task_index
        """
        # Extract prompt
        prompt = ""
        if self.tasks and "task_index" in data:
            task_idx = int(data["task_index"])
            prompt = self.tasks.get(task_idx, "")

        # Parse images: squeeze obs dim, convert to CHW float [0,1]
        front_img = None
        if "observation.image" in data:
            img = _to_numpy(data["observation.image"])
            if isinstance(img, np.ndarray) and img.ndim >= 2 and img.shape[0] == 1:
                img = np.squeeze(img, axis=0)
            front_img = _parse_image_to_chw_float(img)

        wrist_img = None
        if "observation.wrist_image" in data:
            img = _to_numpy(data["observation.wrist_image"])
            if isinstance(img, np.ndarray) and img.ndim >= 2 and img.shape[0] == 1:
                img = np.squeeze(img, axis=0)
            wrist_img = _parse_image_to_chw_float(img)

        # Parse eePose: squeeze, split into components
        ee_pose = np.asarray(_to_numpy(data["observation.eePose"]), dtype=np.float32)
        if ee_pose.ndim >= 2 and ee_pose.shape[0] == 1:
            ee_pose = ee_pose.squeeze(0)

        state_pos = ee_pose[..., :3]
        state_rot = ee_pose[..., 3:7]
        state_gripper = ee_pose[..., 7:9]

        # Parse actions: squeeze
        actions_raw = np.asarray(_to_numpy(data["actions"]), dtype=np.float32)
        if actions_raw.ndim >= 2 and actions_raw.shape[0] == 1:
            actions_raw = actions_raw.squeeze(0)

        action_pos = actions_raw[..., :3]
        action_rot = actions_raw[..., 3:6]
        action_gripper = actions_raw[..., 6:7]

        return make_canonical_obs(
            images={"front": front_img, "wrist": wrist_img},
            state={"pos": state_pos, "rot": state_rot, "gripper": state_gripper},
            actions={"pos": action_pos, "rot": action_rot, "gripper": action_gripper},
            prompt=prompt,
            info=self.get_canonical_info(),
        )

    def env_to_canonical(self, data: dict) -> CanonicalDict:
        """Convert MimicGen env observation to canonical format.

        Handles multiple input formats:
        - Raw MimicGen env (agentview_image, robot0_eef_pos, etc.)
        - WebSocket client format (observation/*, state)
        - Pre-split state format (eef_pos, eef_quat, gripper_qpos)
        """
        # Convert lists to arrays first
        processed = {}
        for key, value in data.items():
            if isinstance(value, list):
                processed[key] = np.asarray(value, dtype=np.float32)
            else:
                processed[key] = value

        # Remap keys
        remapped = {}
        for key, value in processed.items():
            new_key = self.ENV_KEY_REMAP.get(key, key)
            remapped[new_key] = value

        # Parse images
        front_img = None
        if "image" in remapped:
            front_img = _parse_image_to_chw_float(remapped["image"])

        wrist_img = None
        if "wrist_image" in remapped:
            wrist_img = _parse_image_to_chw_float(remapped["wrist_image"])

        # Parse state: either concatenated 'state' or pre-split components
        if "state" in remapped:
            state_raw = np.asarray(_to_numpy(remapped["state"]), dtype=np.float32)
            state_pos = state_raw[..., :3]
            state_rot = state_raw[..., 3:7]
            state_gripper = state_raw[..., 7:9]
        else:
            state_pos = np.asarray(_to_numpy(remapped.get("eef_pos", np.zeros(3))), dtype=np.float32)
            state_rot = np.asarray(_to_numpy(remapped.get("eef_quat", np.zeros(4))), dtype=np.float32)
            state_gripper = np.asarray(_to_numpy(remapped.get("gripper_qpos", np.zeros(2))), dtype=np.float32)

        prompt = remapped.get("prompt", "")

        return make_canonical_obs(
            images={"front": front_img, "wrist": wrist_img},
            state={"pos": state_pos, "rot": state_rot, "gripper": state_gripper},
            actions={},
            prompt=prompt,
            info=self.get_canonical_info(),
        )

    def canonical_to_env(self, canonical_action: CanonicalDict, state: dict | None = None) -> dict:
        """Convert canonical actions back to MimicGen env format.

        Composes action components back into flat [horizon, 7] array.
        """
        actions_data = canonical_action["data"]["actions"]

        action_pos = actions_data["pos"]       # [horizon, 3]
        action_rot = actions_data["rot"]       # [horizon, 3] axis_angle
        action_gripper = actions_data["gripper"]   # [horizon, 1]

        # Compose flat actions: [pos(3), rot(3), gripper(1)] = 7D
        actions = np.concatenate([action_pos, action_rot, action_gripper], axis=-1).astype(np.float32)

        return {"actions": actions}

    def get_state_dim(self) -> int:
        return 9

    def get_action_dim(self) -> int:
        return 7

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return ("state/pos", "state/rot", "state/gripper", "actions/pos", "actions/rot", "actions/gripper")

    def env_obs(self) -> dict:
        return {
            "agentview_image": "[T, 84, 84, 3] uint8 [0, 255]",
            "robot0_eye_in_hand_image": "[T, 84, 84, 3] uint8 [0, 255]",
            "robot0_eef_pos": "[T, 3] float32",
            "robot0_eef_quat": "[T, 4] float32",
            "robot0_gripper_qpos": "[T, 2] float32",
            "prompt": "str",
        }

    def env_action(self) -> dict:
        return {"actions": "[horizon, 7] float32"}

    def datasets(self) -> dict:
        return {
            "observation.image": "[n_obs, 3, 84, 84] float32 [0, 1]",
            "observation.wrist_image": "[n_obs, 3, 84, 84] float32 [0, 1]",
            "observation.eePose": "[n_obs, 9] float32",
            "actions": "[horizon, 7] float32",
            "task_index": "int",
        }
