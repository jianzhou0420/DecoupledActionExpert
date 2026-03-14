"""Unified Policy Server for VLA Policy Serving.

A single PolicyServer class that works with any policy + Adaptor combination.
All environment-specific key remapping is handled by the adaptor's transforms.

Usage:
    from vlaworkspace.adaptors import Adaptor
    from vlaworkspace.adaptors.robots import LiberoRobot
    from vlaworkspace.adaptors.models import Pi0Model
    from vlaworkspace.serving import PolicyServer, WebsocketPolicyServer

    adaptor = Adaptor(robot=LiberoRobot(), model=Pi0Model(norm_stats_path=...))
    server_policy = PolicyServer(policy, adaptor, device="cuda:0")
    server = WebsocketPolicyServer(server_policy, port=8000)
    server.serve_forever()
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from vlaworkspace.adaptors.adaptor import Adaptor
from vlaworkspace.policy.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class PolicyServer:
    """Unified wrapper for serving VLA policies via WebSocket.

    This class provides a clean interface between the WebSocket server
    and the policy + adaptor combination. It handles:

    1. Adding batch dimension to single observations
    2. Applying adaptor transforms (env_input_transforms)
    3. Converting numpy arrays to torch tensors
    4. Running policy inference
    5. Converting outputs back to numpy
    6. Applying output transforms (unnormalization, trimming)

    All environment-specific key remapping is handled by the adaptor,
    not by this class. This keeps the PolicyServer generic.

    Args:
        policy: VLA policy implementing BasePolicy interface.
        adaptor: Adaptor for input/output transforms.
        device: PyTorch device for inference.
    """

    def __init__(
        self,
        policy: BasePolicy,
        adaptor: Adaptor,
        device: str = "cuda:0",
    ):
        self._policy = policy
        self._adaptor = adaptor
        self._device = torch.device(device)
        self._first_inference = True  # Flag for debug logging

        # Move policy to device and set eval mode
        self._policy.to(self._device)
        self._policy.eval()
        self._adaptor.eval()

        # Log adaptor configuration for debugging
        logger.info(f"PolicyServer initialized on {device}")
        logger.info(f"  Adaptor: {type(adaptor).__name__}")
        logger.info(f"  Robot: {type(adaptor.robot).__name__}")
        logger.info(f"  Model: {type(adaptor.model).__name__}")

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run policy inference on observation.

        Args:
            obs: Observation dict from environment. Keys can be in any
                environment-specific format (e.g., "observation/image" for LIBERO).
                The adaptor's env_input_transforms will handle key remapping.

        Returns:
            Dict with "actions" key containing unnormalized actions as numpy array.
            Shape is [action_horizon, action_dim] (e.g., [50, 7] for LIBERO).
        """
        # DEBUG: Log raw observation
        if self._first_inference:
            raw_state = obs.get("observation/state", obs.get("state"))
            if raw_state is not None:
                logger.info(f"Raw state from env: {raw_state}")
            prompt = obs.get("prompt")
            if prompt:
                logger.info(f"Prompt: {prompt}")

        # 1. Apply adaptor's env input transforms (on single sample)
        #    This handles key remapping, image parsing, normalization, tokenization
        sample = self._adaptor.env_input_transforms(obs)

        # DEBUG: Log after transforms
        if self._first_inference:
            if "state" in sample:
                logger.info(f"Normalized state: {sample['state'][:8]}")
            if "tokenized_prompt" in sample:
                logger.info(f"Tokenized prompt shape: {sample['tokenized_prompt'].shape}")

        # 2. Add batch dimension
        batch = self._add_batch_dim(sample)

        # 3. Convert numpy arrays to torch tensors
        batch = self._to_torch(batch)

        # 4. Run policy inference
        with torch.no_grad():
            actions = self._policy.predict_action(batch)

        # 5. Extract actions and convert to numpy
        if isinstance(actions, torch.Tensor):
            actions_np = actions[0].cpu().numpy()  # Remove batch dim: [T, D]
        else:
            # Some policies return dict with 'action' key
            actions_np = actions["action"][0].cpu().numpy()

        # 6. Apply output transforms (unnormalize, trim to robot action dim)
        # Include state for DeltaActionTransform (converts delta actions to absolute)
        # Note: output_transforms expects "action" key (singular), not "actions"
        raw_state = obs.get("observation/state", obs.get("state"))
        output_data = {"action": actions_np[np.newaxis, ...]}  # Add batch dim for transforms
        if raw_state is not None:
            output_data["state"] = np.asarray(raw_state)[np.newaxis, ...]  # Add batch dim
        output = self._adaptor.output_transforms(output_data)

        final_actions = output["actions"][0].astype(np.float32)

        # Log first inference at INFO level for debugging
        if self._first_inference:
            self._first_inference = False
            logger.info("=== First Inference Debug ===")
            logger.info(
                f"Raw model output: shape={actions_np.shape}, "
                f"range=[{actions_np.min():.4f}, {actions_np.max():.4f}]"
            )
            logger.info(f"  First timestep (7 dims): {actions_np[0, :7]}")
            logger.info(
                f"After unnorm+trim: shape={final_actions.shape}, "
                f"range=[{final_actions.min():.4f}, {final_actions.max():.4f}]"
            )
            logger.info(f"  First timestep: {final_actions[0]}")
            logger.info("==============================")

        # Remove batch dim and ensure float32
        return {"actions": final_actions}

    def _add_batch_dim(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Add batch dimension to single observation.

        Args:
            obs: Single observation dict.

        Returns:
            Batched observation dict (batch_size=1).
        """
        result = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                result[key] = value[np.newaxis, ...]  # [1, ...]
            elif isinstance(value, np.generic):
                # Handle numpy scalars (np.bool_, np.float32, etc.)
                result[key] = np.array([value])  # [1]
            elif isinstance(value, str):
                result[key] = [value]  # List of strings for batch
            elif isinstance(value, dict):
                result[key] = self._add_batch_dim(value)
            else:
                result[key] = value
        return result

    def _to_torch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Convert numpy arrays to torch tensors recursively.

        Args:
            batch: Dict that may contain nested dicts and numpy arrays.

        Returns:
            Dict with all numpy arrays converted to torch tensors on device.
        """
        result = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                result[key] = self._to_torch(value)
            elif isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value).to(self._device)
            else:
                result[key] = value
        return result

    def reset(self) -> None:
        """Reset policy state (if applicable)."""
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def infer_batch(self, obs_list: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
        """Run batched policy inference on multiple observations.

        Args:
            obs_list: List of observation dicts from environments.

        Returns:
            List of dicts, each with "actions" key containing unnormalized actions.
        """
        if not obs_list:
            return []

        batch_size = len(obs_list)

        # 1. Apply adaptor transforms per sample (produces no-batch-dim data)
        samples = []
        raw_states = []

        # DEBUG: Log raw input from first observation BEFORE transforms
        if self._first_inference and obs_list:
            first_obs = obs_list[0]
            logger.info("=== Raw Input BEFORE Transforms ===")
            for key, value in first_obs.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                               f"range=[{value.min():.4f}, {value.max():.4f}]")
                    # Show actual values for state keys
                    if "image" not in key.lower() and value.size < 20:
                        logger.info(f"    values: {value}")
                else:
                    logger.info(f"  {key}: type={type(value).__name__}")

        for obs in obs_list:
            sample = self._adaptor.env_input_transforms(obs)
            samples.append(sample)
            raw_state = obs.get("observation/state", obs.get("state"))
            # Take last timestep if state has a time dimension (n_obs_steps > 1)
            if raw_state is not None and np.ndim(raw_state) >= 2:
                raw_state = raw_state[-1]
            raw_states.append(raw_state)

        # DEBUG: Log first sample shapes AND VALUES on first inference
        if self._first_inference and samples:
            logger.info("=== First Batch Inference Debug ===")
            logger.info(f"Batch size: {batch_size}")
            logger.info("Sample keys after adaptor transforms (with value ranges):")
            for key, value in samples[0].items():
                if isinstance(value, np.ndarray):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                               f"range=[{value.min():.4f}, {value.max():.4f}]")
                    # Show actual values for state keys (not images)
                    if "image" not in key and value.size < 20:
                        logger.info(f"    values: {value}")
                else:
                    logger.info(f"  {key}: type={type(value).__name__}")

        # 2. Stack samples into batch [B, ...]
        batch = self._stack_samples(samples)

        # DEBUG: Log batch shapes on first inference
        if self._first_inference:
            logger.info("Batch shapes after stacking:")
            for key, value in batch.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            self._first_inference = False

        # 3. Convert to torch tensors
        batch = self._to_torch(batch)

        # 4. Run policy inference (already supports [B, ...])
        with torch.no_grad():
            actions = self._policy.predict_action(batch)

        # 5. Convert to numpy [B, T, D]
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions["action"].cpu().numpy()

        # DEBUG: Log model output on first batch
        log_first_batch = batch_size > 0 and hasattr(self, '_logged_first_batch') is False
        if log_first_batch:
            self._logged_first_batch = True
            logger.info("=== First Batch Model Output Debug ===")
            logger.info(f"Raw model output: shape={actions_np.shape}, "
                       f"range=[{actions_np.min():.4f}, {actions_np.max():.4f}]")
            logger.info(f"  First env, first timestep (10 dims): {actions_np[0, 0]}")

        # 6. Apply output transforms per sample
        # Note: output_transforms expects "action" key (singular), not "actions"
        results = []
        for i in range(batch_size):
            output_data = {"action": actions_np[i : i + 1, ...]}  # [1, T, D]
            if raw_states[i] is not None:
                output_data["state"] = np.asarray(raw_states[i])[np.newaxis, ...]
            output = self._adaptor.output_transforms(output_data)
            results.append({"actions": output["actions"][0].astype(np.float32)})

        # DEBUG: Log transformed output on first batch
        if log_first_batch and results:
            final = results[0]["actions"]
            logger.info(f"After unnorm+trim: shape={final.shape}, "
                       f"range=[{final.min():.4f}, {final.max():.4f}]")
            logger.info(f"  First timestep (7 dims): {final[0]}")
            logger.info("=" * 40)

        return results

    def _stack_samples(self, samples: list[dict]) -> dict:
        """Stack list of transformed samples into a batch.

        Handles:
        - numpy arrays → np.stack(..., axis=0)
        - nested dicts (images) → recursive stacking
        - strings → keep as list
        """
        if not samples:
            return {}

        batch = {}
        for key in samples[0].keys():
            values = [s[key] for s in samples]
            if isinstance(values[0], dict):
                # Nested dict (e.g., images): recurse
                batch[key] = self._stack_samples(values)
            elif isinstance(values[0], np.ndarray):
                batch[key] = np.stack(values, axis=0)
            elif isinstance(values[0], str):
                batch[key] = values  # Keep as list
            else:
                batch[key] = np.array(values)
        return batch
