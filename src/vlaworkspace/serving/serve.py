"""Policy Server Entry Point.

Start a WebSocket server for VLA policy inference using either:
1. VLAWorkspace training run directory
2. OpenPi experiment directory (PyTorch safetensors only)

Usage:
    # Mode 1: VLAWorkspace run directory (auto-discovers config, checkpoint, norm_stats)
    python -m vlaworkspace.serving.serve --run-dir /path/to/run_dir

    # With specific checkpoint
    python -m vlaworkspace.serving.serve \\
        --run-dir /path/to/run_dir \\
        --checkpoint-name step=010000.ckpt

    # Mode 2: OpenPi experiment directory (auto-selects latest step)
    python -m vlaworkspace.serving.serve \\
        --openpi-dir /path/to/openpi/experiment

    # OpenPi with specific step
    python -m vlaworkspace.serving.serve \\
        --openpi-dir /path/to/openpi/experiment \\
        --openpi-step 10000

    # Override device
    python -m vlaworkspace.serving.serve \\
        --run-dir /path/to/run_dir \\
        --device cuda:1

Checkpoint formats:
    - PyTorch Lightning .ckpt file (from VLAWorkspace training)
    - SafeTensors .safetensors file (from OpenPi PyTorch training)

VLAWorkspace run directory structure:
    run_dir/
    ├── config.yaml          # Training configuration (required)
    ├── norm_stats.json      # Normalization statistics (required)
    └── checkpoints/
        └── step=XXXXX.ckpt  # Checkpoint files

OpenPi experiment directory structure:
    experiment_dir/
    ├── 5000/
    ├── 10000/
    │   ├── model.safetensors     # PyTorch weights (required)
    │   ├── metadata.pt           # Full config (required)
    │   └── assets/
    │       └── physical-intelligence/
    │           └── libero/
    │               └── norm_stats.json
    └── ... more steps
"""

from __future__ import annotations

import dataclasses
import glob
import logging
import os
import re

import hydra
import torch
import tyro
from omegaconf import OmegaConf, open_dict

# Disable torch.compile for serving (must be before other torch imports)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ServerConfig:
    """WebSocket server for VLA policy inference."""

    # Mode 1: VLAWorkspace run directory
    run_dir: str | None = None
    """Training run directory. Auto-discovers config, checkpoint, and norm stats."""

    checkpoint_name: str | None = None
    """Which checkpoint to load. If None, auto-selects the largest step checkpoint."""

    # Mode 2: OpenPi experiment directory
    openpi_dir: str | None = None
    """OpenPi experiment directory containing step folders (e.g., 5000/, 10000/)."""

    openpi_step: int | None = None
    """Which step to load. If None, auto-selects the largest step."""

    # Server config
    host: str = "0.0.0.0"
    """Host address to bind to."""

    port: int = 8000
    """Port number to listen on."""

    device: str = "cuda:0"
    """PyTorch device for inference."""

    num_inference_steps: int = 10
    """Number of diffusion denoising steps during inference."""

    cache_dir: str = "data/models"
    """Directory to cache downloaded model weights."""


# =============================================================================
# Checkpoint / path helpers
# =============================================================================


def find_latest_checkpoint(checkpoints_dir: str) -> str:
    """Find the checkpoint with the largest step number.

    Looks for files matching 'step=XXXXX.ckpt' or 'epoch=XXX.ckpt'.
    """
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    step_pattern = re.compile(r"step=(\d+)\.ckpt$")
    epoch_pattern = re.compile(r"epoch=(\d+)\.ckpt$")
    checkpoints = []

    for filename in os.listdir(checkpoints_dir):
        match = step_pattern.match(filename)
        if match:
            checkpoints.append((int(match.group(1)), filename))
            continue
        match = epoch_pattern.match(filename)
        if match:
            checkpoints.append((int(match.group(1)), filename))

    if not checkpoints:
        # Fall back to last.ckpt if no numbered checkpoints found
        last_ckpt = os.path.join(checkpoints_dir, "last.ckpt")
        if os.path.isfile(last_ckpt):
            logger.info("No numbered checkpoints found, using last.ckpt")
            return "last.ckpt"
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoints_dir}. "
            f"Expected files like 'step=010000.ckpt', 'epoch=049.ckpt', or 'last.ckpt'"
        )

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_num, latest_filename = checkpoints[0]
    logger.info(f"Found {len(checkpoints)} checkpoints, selecting {latest_filename}")

    return latest_filename


def resolve_checkpoint_path(config: ServerConfig) -> str:
    """Resolve checkpoint path from run_dir."""
    checkpoints_dir = os.path.join(config.run_dir, "checkpoints")
    if config.checkpoint_name is None:
        checkpoint_name = find_latest_checkpoint(checkpoints_dir)
    else:
        checkpoint_name = config.checkpoint_name

    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_norm_stats_path(run_dir: str) -> str:
    """Resolve norm_stats.json path from run_dir."""
    for name in ["norm_stats.json", "norm_stats_smolvla.json"]:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Norm stats not found: checked {run_dir}/norm_stats.json. "
        f"This file is required for proper action unnormalization."
    )


# =============================================================================
# VLAWorkspace run-dir mode (Hydra-based)
# =============================================================================


def serve_from_run_dir(config: ServerConfig) -> None:
    """Start server from VLAWorkspace run directory using Hydra instantiate."""
    from vlaworkspace.serving.policy_server import PolicyServer
    from vlaworkspace.serving.websocket_policy_server import WebsocketPolicyServer

    if not os.path.isdir(config.run_dir):
        raise FileNotFoundError(f"Run directory not found: {config.run_dir}")

    # Load saved training config
    config_path = os.path.join(config.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    # Resolve paths
    checkpoint_path = resolve_checkpoint_path(config)
    norm_stats_path = resolve_norm_stats_path(config.run_dir)

    if not OmegaConf.select(cfg, "adaptor"):
        raise ValueError("No adaptor config found in config.yaml")

    # Apply inference overrides before instantiation
    with open_dict(cfg):
        # Disable gradient checkpointing for inference
        if OmegaConf.select(cfg, "policy.gradient_checkpointing") is not None:
            cfg.policy.gradient_checkpointing = False

        # Override norm_stats_path to resolved absolute path
        cfg.adaptor.model.norm_stats_path = norm_stats_path

    # Instantiate policy via Hydra (handles Pi0, Pi0LoRA, DP, SmolVLA, etc.)
    logger.info(f"Instantiating policy: {cfg.policy._target_}")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_checkpoint(checkpoint_path)

    # Instantiate adaptor via Hydra (handles legacy and composable formats)
    logger.info(f"Instantiating adaptor: {cfg.adaptor._target_}")
    adaptor = hydra.utils.instantiate(cfg.adaptor)

    # Create PolicyServer
    policy_server = PolicyServer(
        policy=policy,
        adaptor=adaptor,
        device=config.device,
    )

    # Build metadata from config
    metadata = {
        "policy_target": str(cfg.policy._target_),
        "action_dim": adaptor.get_action_dim(),
        "run_dir": config.run_dir,
        "checkpoint": checkpoint_path,
    }

    # Create and start WebSocket server
    server = WebsocketPolicyServer(
        policy=policy_server,
        host=config.host,
        port=config.port,
        metadata=metadata,
    )

    logger.info("=" * 60)
    logger.info(f"Policy Server starting on ws://{config.host}:{config.port}")
    logger.info(f"  Mode: VLAWorkspace")
    logger.info(f"  Run dir: {config.run_dir}")
    logger.info(f"  Policy: {cfg.policy._target_}")
    logger.info(f"  Adaptor: {cfg.adaptor._target_}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Device: {config.device}")
    logger.info("=" * 60)

    server.serve_forever()


# =============================================================================
# OpenPi mode (manual construction — no config.yaml available)
# =============================================================================


def find_latest_openpi_step(openpi_dir: str) -> int:
    """Find the largest step number in OpenPi experiment directory."""
    steps = []

    for entry in os.listdir(openpi_dir):
        entry_path = os.path.join(openpi_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            step = int(entry)
        except ValueError:
            continue
        model_path = os.path.join(entry_path, "model.safetensors")
        if os.path.exists(model_path):
            steps.append(step)

    if not steps:
        raise FileNotFoundError(
            f"No valid OpenPi step directories found in {openpi_dir}. "
            f"Expected directories like '5000/', '10000/' containing model.safetensors"
        )

    steps.sort(reverse=True)
    logger.info(f"Found {len(steps)} OpenPi checkpoints, selecting step={steps[0]}")
    return steps[0]


def find_openpi_norm_stats(step_dir: str) -> str:
    """Find norm_stats.json in assets/ subdirectory."""
    pattern = os.path.join(step_dir, "assets", "*", "*", "norm_stats.json")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"norm_stats.json not found in {step_dir}/assets/*/*/. "
            f"Expected pattern: assets/<org>/<dataset>/norm_stats.json"
        )

    if len(matches) > 1:
        logger.warning(f"Found multiple norm_stats.json files, using first: {matches[0]}")

    return matches[0]


def serve_from_openpi_dir(config: ServerConfig) -> None:
    """Start server from OpenPi experiment directory.

    This path requires manual policy/adaptor construction because OpenPi
    directories don't have a Hydra config.yaml.
    """
    from vlaworkspace.adaptors.adaptor import Adaptor
    from vlaworkspace.adaptors.models.pi0_model import Pi0Model
    from vlaworkspace.adaptors.robots.libero_robot import LiberoRobot
    from vlaworkspace.policy.pi0_policy import Pi0Config, Pi0Policy
    from vlaworkspace.serving.policy_server import PolicyServer
    from vlaworkspace.serving.websocket_policy_server import WebsocketPolicyServer

    if not os.path.isdir(config.openpi_dir):
        raise FileNotFoundError(f"OpenPi directory not found: {config.openpi_dir}")

    # Find step directory
    if config.openpi_step is None:
        step = find_latest_openpi_step(config.openpi_dir)
    else:
        step = config.openpi_step
    step_dir = os.path.join(config.openpi_dir, str(step))

    if not os.path.isdir(step_dir):
        raise FileNotFoundError(f"Step directory not found: {step_dir}")

    # Verify checkpoint exists
    checkpoint_path = os.path.join(step_dir, "model.safetensors")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"model.safetensors not found in {step_dir}. "
            f"Only PyTorch OpenPi checkpoints are supported."
        )

    # Load config from metadata.pt
    metadata_path = os.path.join(step_dir, "metadata.pt")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.pt not found in {step_dir}")
    try:
        metadata = torch.load(metadata_path, weights_only=False)
        openpi_config = metadata["config"]
    except ModuleNotFoundError as e:
        logger.warning(f"Could not load metadata.pt due to missing module: {e}")
        logger.warning("Using default OpenPi config (Pi0 full model, delta actions enabled)")
        openpi_config = {
            "model": {"pi05": False, "max_token_len": 48},
            "data": {"extra_delta_transform": True},
        }

    # Find norm_stats.json
    norm_stats_path = find_openpi_norm_stats(step_dir)

    # Extract settings from openpi_config
    model_config = openpi_config.get("model", {})
    data_config = openpi_config.get("data", {})
    use_delta_actions = data_config.get("extra_delta_transform", False)
    pi05 = model_config.get("pi05", False)
    max_token_len = model_config.get("max_token_len", 48)

    # Create Pi0 config
    pi0_config = Pi0Config(pi05=pi05, max_token_len=max_token_len)

    # Create policy
    policy = Pi0Policy(
        config=pi0_config,
        num_inference_steps=config.num_inference_steps,
        use_pretrained_weight=True,
        cache_dir=config.cache_dir,
        gradient_checkpointing=False,
    )
    policy_type = "pi0"

    # Load checkpoint (safetensors)
    policy.load_checkpoint(checkpoint_path)

    # Create adaptor (OpenPi is always LIBERO)
    delta_action_mask = (True, True, True, True, True, True, False) if use_delta_actions else None
    robot = LiberoRobot(
        use_delta_actions=use_delta_actions,
        delta_action_mask=delta_action_mask,
    )
    model = Pi0Model(
        norm_stats_path=norm_stats_path,
        max_token_len=max_token_len,
    )
    adaptor = Adaptor(robot=robot, model=model)

    # Create PolicyServer
    policy_server = PolicyServer(
        policy=policy,
        adaptor=adaptor,
        device=config.device,
    )

    metadata = {
        "policy_type": policy_type,
        "environment": "libero",
        "pi05": pi05,
        "action_dim": adaptor.get_action_dim(),
    }

    # Create and start WebSocket server
    server = WebsocketPolicyServer(
        policy=policy_server,
        host=config.host,
        port=config.port,
        metadata=metadata,
    )

    logger.info("=" * 60)
    logger.info(f"Policy Server starting on ws://{config.host}:{config.port}")
    logger.info(f"  Mode: OpenPi")
    logger.info(f"  Directory: {config.openpi_dir}")
    logger.info(f"  Step: {step}")
    logger.info(f"  Policy type: {policy_type}")
    logger.info(f"  Pi0.5: {pi05}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Device: {config.device}")
    logger.info("=" * 60)

    server.serve_forever()


# =============================================================================
# Entry point
# =============================================================================


def main(config: ServerConfig) -> None:
    """Start the WebSocket policy server."""
    if config.run_dir and config.openpi_dir:
        raise ValueError("Specify either --run-dir or --openpi-dir, not both")
    if not config.run_dir and not config.openpi_dir:
        raise ValueError("Must specify either --run-dir or --openpi-dir")

    if config.run_dir:
        serve_from_run_dir(config)
    else:
        serve_from_openpi_dir(config)


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
