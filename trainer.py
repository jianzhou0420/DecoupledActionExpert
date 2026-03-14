"""
TaskFusion Trainer.

A clean PyTorch Lightning trainer for VLA models.

Data Flow:
    LeRobot Dataset
        ↓
    Adaptor (transforms data for specific VLA)
        ↓
    Policy.compute_loss() / Policy.predict_action()

Usage:
    python trainer.py --config-name=pi0_libero seed=42 batch_size=32

    # Or with custom config path
    python trainer.py --config-path=taskfusion/config --config-name=pi0_libero
"""

import collections
import copy
import json
import logging
import os
import pathlib
import random
import shutil
import sys
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
from typing import Any

import hydra

logger = logging.getLogger(__name__)
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import mimicgen
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# import base class
from vlaworkspace.policy.base_policy import BasePolicy

# Line buffering for stdout/stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

# Environment setup
os.environ["HYDRA_FULL_ERROR"] = "1"
torch.set_float32_matmul_precision("medium")


# =============================================================================
# region Utilities
# =============================================================================


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def seed_worker(worker_id: int) -> None:
    """Seed worker for reproducible data loading."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    worker_seed = (torch.initial_seed() + rank * 1000 + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class SlurmProgressBar(Callback):
    """Progress bar that prints on new lines, suitable for SLURM log files."""

    def __init__(self, refresh_rate: int = 100, speed_window: int = 200):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.speed_window = speed_window
        self.start_time = None
        self.start_step = 0
        self._step_times: collections.deque[float] = collections.deque(maxlen=speed_window)
        self._eval_time: float = 0.0  # cumulative time spent in eval/rollout
        self._eval_count: int = 0  # number of epoch boundaries with eval time measured
        self._epoch_end_time: float | None = None  # timestamp when training batches ended

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.start_step = trainer.global_step
        self._step_times.clear()
        self._eval_time = 0.0
        self._eval_count = 0
        self._epoch_end_time = None
        max_epochs = trainer.max_epochs if trainer.max_epochs and trainer.max_epochs > 0 else None
        max_steps = trainer.max_steps if trainer.max_steps > 0 else None
        num_batches = trainer.num_training_batches if hasattr(trainer, "num_training_batches") else "?"
        if max_epochs:
            logger.info(f"Training started | max_epochs: {max_epochs} | batches/epoch: {num_batches}")
        elif max_steps:
            logger.info(f"Training started | max_steps: {max_steps}")
        else:
            logger.info("Training started")

    def on_train_epoch_start(self, trainer, pl_module):
        # Account for eval/rollout time between epochs
        if self._epoch_end_time is not None:
            gap = time.time() - self._epoch_end_time
            self._eval_time += gap
            self._eval_count += 1
            self._epoch_end_time = None
        # Clear rolling window so it doesn't span the eval gap
        self._step_times.clear()

        max_epochs = trainer.max_epochs if trainer.max_epochs and trainer.max_epochs > 0 else None
        epoch_str = f"{trainer.current_epoch + 1}/{max_epochs}" if max_epochs else f"{trainer.current_epoch + 1}"
        logger.info(f"Epoch {epoch_str} started")

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"

    def _get_rolling_speed(self) -> float | None:
        """Compute rolling it/s from recent step timestamps."""
        if len(self._step_times) < 2:
            return None
        dt = self._step_times[-1] - self._step_times[0]
        if dt <= 0:
            return None
        return (len(self._step_times) - 1) / dt

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        self._step_times.append(now)
        self._epoch_end_time = now  # updated every batch; final value = last batch of epoch

        if trainer.global_step % self.refresh_rate == 0 and trainer.global_step > 0:
            # Loss and LR
            loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
            lr = trainer.optimizers[0].param_groups[0]["lr"]

            # Progress — epoch-based or step-based
            max_epochs = trainer.max_epochs if trainer.max_epochs and trainer.max_epochs > 0 else None
            max_steps = trainer.max_steps if trainer.max_steps > 0 else None
            current_epoch = trainer.current_epoch

            if max_epochs:
                # Epoch-based: show "Epoch X/Y | Step Z/Total"
                total_steps = max_epochs * trainer.num_training_batches if trainer.num_training_batches else None
                epoch_frac = current_epoch + (batch_idx + 1) / trainer.num_training_batches if trainer.num_training_batches else current_epoch
                progress = 100 * epoch_frac / max_epochs
                step_str = f"{trainer.global_step}/{total_steps}" if total_steps else f"{trainer.global_step}"
                progress_str = f"Epoch {current_epoch + 1}/{max_epochs} | Step {step_str} ({progress:.1f}%)"
            elif max_steps:
                progress = 100 * trainer.global_step / max_steps
                progress_str = f"Step {trainer.global_step}/{max_steps} ({progress:.1f}%)"
            else:
                progress_str = f"Epoch {current_epoch + 1} | Step {trainer.global_step}"

            # Time calculations (exclude eval/rollout time from speed)
            elapsed = time.time() - self.start_time
            train_elapsed = elapsed - self._eval_time
            elapsed_str = self._format_time(elapsed)
            steps_done = trainer.global_step - self.start_step
            overall_speed = steps_done / train_elapsed if train_elapsed > 0 else 0
            rolling_speed = self._get_rolling_speed()

            if rolling_speed and rolling_speed > 0:
                # Compute ETA from epoch progress or step progress
                remaining_epochs = 0
                if max_epochs and trainer.num_training_batches:
                    total_steps = max_epochs * trainer.num_training_batches
                    remaining_steps = total_steps - (current_epoch * trainer.num_training_batches + batch_idx + 1)
                    remaining_epochs = max_epochs - current_epoch - 1
                elif max_steps:
                    remaining_steps = max_steps - trainer.global_step
                else:
                    remaining_steps = None

                if remaining_steps is not None and remaining_steps > 0:
                    eta_train = remaining_steps / rolling_speed
                    # Add estimated eval time for remaining epoch boundaries
                    avg_eval = self._eval_time / self._eval_count if self._eval_count > 0 else 0
                    eta_eval = avg_eval * remaining_epochs
                    eta_seconds = eta_train + eta_eval
                    eta_str = self._format_time(eta_seconds)
                    time_str = f"elapsed: {elapsed_str} | ETA: {eta_str} | {rolling_speed:.2f} it/s ({overall_speed:.2f} avg)"
                else:
                    time_str = f"elapsed: {elapsed_str} | {rolling_speed:.2f} it/s ({overall_speed:.2f} avg)"
            else:
                time_str = f"elapsed: {elapsed_str} | {overall_speed:.2f} avg it/s"

            # GPU memory (if available)
            try:
                mem_used = torch.cuda.max_memory_allocated() / 1024**3
                mem_str = f" | GPU: {mem_used:.1f}GB"
            except:
                mem_str = ""

            logger.info(f"{progress_str} | loss: {loss:.4f} | lr: {lr:.2e} | {time_str}{mem_str}")

    def on_train_end(self, trainer, pl_module):
        if self.start_time:
            total_time = time.time() - self.start_time
            train_time = total_time - self._eval_time
            total_steps = trainer.global_step - self.start_step
            avg_speed = total_steps / train_time if train_time > 0 else 0
            epochs_str = f" | epochs: {trainer.current_epoch + 1}" if trainer.current_epoch > 0 else ""
            eval_str = f" | eval: {self._format_time(self._eval_time)}" if self._eval_time > 1.0 else ""
            logger.info(
                f"Training finished | total: {self._format_time(total_time)}{eval_str} | steps: {total_steps}{epochs_str} | avg: {avg_speed:.2f} it/s"
            )

# endregion
# =============================================================================
# region Action MSE Callback
# =============================================================================


def _deep_clone_batch(batch):
    """Recursively clone all tensors in a batch dict."""
    if isinstance(batch, torch.Tensor):
        return batch.clone()
    elif isinstance(batch, dict):
        return {k: _deep_clone_batch(v) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(_deep_clone_batch(v) for v in batch)
    else:
        return batch


def _batch_to_device(batch, device):
    """Recursively move all tensors in a batch dict to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(_batch_to_device(v, device) for v in batch)
    else:
        return batch


class ActionMseLoss(Callback):
    """Compute MSE between predicted and ground-truth actions using full inference.

    For diffusion policies, training loss is noise prediction MSE — not action MSE.
    This callback runs the full denoising pipeline on a fixed reference batch and
    logs the actual action prediction error, giving a more interpretable metric.

    Fires evenly spaced throughout training (default: 100 times total).
    Works with all policy types (DAH diffusion, Pi0, SmolVLA).

    Args:
        n_evaluations: How many times to evaluate during the entire training run.
            The interval is computed as total_steps // n_evaluations.
    """

    def __init__(self, n_evaluations: int = 100):
        super().__init__()
        self._n_evaluations = n_evaluations
        self._eval_interval = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Capture first batch as fixed reference
        if not hasattr(pl_module, "_action_mse_ref_batch"):
            pl_module._action_mse_ref_batch = _deep_clone_batch(batch)

        # Compute interval lazily (estimated_stepping_batches is ready by first step)
        if self._eval_interval is None:
            total_steps = trainer.estimated_stepping_batches
            self._eval_interval = max(1, int(total_steps) // self._n_evaluations)
            logger.info(
                f"ActionMseLoss: will evaluate every {self._eval_interval} steps "
                f"({self._n_evaluations} times over {int(total_steps)} total steps)"
            )

        # Only evaluate at the interval
        step = pl_module.global_step
        if step == 0 or step % self._eval_interval != 0:
            return

        self._evaluate(trainer, pl_module)

    def _evaluate(self, trainer, pl_module):
        ref_batch = getattr(pl_module, "_action_mse_ref_batch", None)
        if ref_batch is None:
            return

        policy = pl_module.policy_ema if pl_module.policy_ema else pl_module.policy
        batch = _batch_to_device(ref_batch, pl_module.device)

        # Find ground truth actions ('action' for DAH, 'actions' for Pi0/SmolVLA)
        gt_action = batch.get("action", batch.get("actions"))
        if gt_action is None:
            return

        # Run full inference pipeline (no grad — this is just a probe)
        with torch.no_grad():
            result = policy.predict_action(batch)

        # Extract predicted actions (dict for DAH, tensor for Pi0/SmolVLA)
        if isinstance(result, dict):
            pred_action = result.get("action_pred", result.get("action"))
        else:
            pred_action = result

        if pred_action is None:
            return

        # Align shapes — truncate to the smaller time horizon and action dim
        if pred_action.dim() >= 2 and gt_action.dim() >= 2:
            min_T = min(pred_action.shape[1], gt_action.shape[1])
            pred_action = pred_action[:, :min_T]
            gt_action = gt_action[:, :min_T]
        if pred_action.dim() >= 3 and gt_action.dim() >= 3:
            min_D = min(pred_action.shape[2], gt_action.shape[2])
            pred_action = pred_action[..., :min_D]
            gt_action = gt_action[..., :min_D]

        pred = pred_action.float()
        gt = gt_action.float()

        # Overall action MSE
        mse = torch.nn.functional.mse_loss(pred, gt)

        log_dict = {"train/action_mse_loss": mse.item()}

        # --- #1: Per-component action MSE (position / orientation / gripper) ---
        D = pred.shape[-1]
        if D >= 4:
            log_dict["train/action_mse_position"] = (
                torch.nn.functional.mse_loss(pred[..., :3], gt[..., :3]).item()
            )
            log_dict["train/action_mse_orientation"] = (
                torch.nn.functional.mse_loss(pred[..., 3:-1], gt[..., 3:-1]).item()
            )
            log_dict["train/action_mse_gripper"] = (
                torch.nn.functional.mse_loss(pred[..., -1:], gt[..., -1:]).item()
            )

        # --- #2: Predicted action statistics ---
        log_dict["train/pred_action_mean"] = pred.mean().item()
        log_dict["train/pred_action_std"] = pred.std().item()
        log_dict["train/pred_action_min"] = pred.min().item()
        log_dict["train/pred_action_max"] = pred.max().item()
        log_dict["train/gt_action_mean"] = gt.mean().item()
        log_dict["train/gt_action_std"] = gt.std().item()

        # Log to wandb
        trainer.logger.experiment.log(log_dict, step=pl_module.global_step)


# endregion
# =============================================================================
# region Rollout Callback
# =============================================================================


class MimicgenRolloutCallback(Callback):
    """Run policy rollout in robomimic simulation environment periodically.

    Ported from DecoupledActionHead/trainer_pl_all.py.
    Creates a RobomimicRunner from config and runs evaluation at the
    end of every N training epochs, logging success rates and videos to WandB.
    """

    # Max episode steps per task (avg demo length * 2.5), matching docker/mimicgen eval
    TASK_MAX_STEPS = {
        "stack": 270,
        "stack_three": 638,
        "square": 383,
        "threading": 568,
        "coffee": 560,
        "coffee_preparation": 1718,
        "three_piece_assembly": 838,
        "hammer_cleanup": 715,
        "mug_cleanup": 845,
        "kitchen": 1548,
        "nut_assembly": 895,
        "pick_place": 1693,
    }

    def __init__(self, env_runner_cfg: DictConfig, rollout_every_n_epochs: int = 1, repo_id: str = ""):
        super().__init__()
        from vlaworkspace.env_runner.base_runner import BaseRunner

        # Auto-resolve dataset_path and max_steps from repo_id if not set
        if repo_id:
            dataset_name = repo_id.split("/")[-1]  # e.g. DAH_mimicgen_stack_d1_alldemos
            prefix = "DAH_mimicgen_"
            suffix = "_alldemos"
            if dataset_name.startswith(prefix) and dataset_name.endswith(suffix):
                task = dataset_name[len(prefix):-len(suffix)]  # e.g. stack_d1
                # Strip difficulty suffix (_d0, _d1, _d2) to get base task name
                import re
                base_task = re.sub(r"_d\d+$", "", task)  # e.g. stack

                # Auto-resolve dataset_path (HDF5 file is optional — runner falls back to hardcoded data)
                if not env_runner_cfg.get("dataset_path"):
                    resolved_path = f"data/robomimic/datasets/{task}/{task}_abs_traj_eePose.hdf5"
                    env_runner_cfg.dataset_path = resolved_path
                    if os.path.exists(resolved_path):
                        logger.info(f"Auto-resolved env_runner.dataset_path: {resolved_path}")
                    else:
                        logger.info(f"HDF5 not found at {resolved_path}, will use hardcoded env data")

                # Auto-resolve max_steps
                if env_runner_cfg.get("max_steps") is None or env_runner_cfg.max_steps == 400:
                    if base_task in self.TASK_MAX_STEPS:
                        env_runner_cfg.max_steps = self.TASK_MAX_STEPS[base_task]
                        logger.info(f"Auto-resolved env_runner.max_steps: {env_runner_cfg.max_steps} (task={base_task})")
            else:
                if not env_runner_cfg.get("dataset_path"):
                    raise ValueError(f"Cannot auto-resolve from repo_id '{repo_id}'. "
                                     f"Expected format: .../DAH_mimicgen_{{task}}_alldemos")

        env_runner = hydra.utils.instantiate(env_runner_cfg, output_dir='data/outputs')
        assert isinstance(env_runner, BaseRunner)
        self.env_runner = env_runner
        self.rollout_every_n_epochs = rollout_every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: "VLATrainer"):
        if (trainer.current_epoch + 1) % self.rollout_every_n_epochs != 0:
            return
        if pl_module.global_step <= 0:
            return

        policy = pl_module.policy_ema if pl_module.policy_ema else pl_module.policy

        # Get adaptor from datamodule for obs/action transforms
        adaptor = None
        if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'dataset'):
            adaptor = getattr(trainer.datamodule.dataset, 'adaptor', None)

        runner_log = self.env_runner.run(policy, adaptor=adaptor)
        trainer.logger.experiment.log(runner_log, step=trainer.global_step)

    def teardown(self, trainer: pl.Trainer, pl_module: "VLATrainer", stage: str) -> None:
        """Close env runner to avoid BrokenPipeError and leaked semaphores at shutdown."""
        if hasattr(self.env_runner, 'env') and hasattr(self.env_runner.env, 'close'):
            try:
                self.env_runner.env.close()
            except Exception:
                pass


class LiberoRolloutCallback(Callback):
    """Run policy rollout in LIBERO simulation environment at the end of training.

    Creates a LiberoRunner from config and runs evaluation at the last training step,
    logging success rates and videos to WandB.

    Runs in on_train_batch_end (not on_train_end) so the WandB logger is still active.
    """

    def __init__(self, libero_runner_cfg: DictConfig, rollout_on_start: bool = False,
                 rollout_on_end: bool = False, rollout_every_n_epochs: int | None = None):
        super().__init__()
        if rollout_on_end or rollout_every_n_epochs is not None:
            raise NotImplementedError(
                "Epoch-wise rollout (rollout_on_end, rollout_every_n_epochs) is no longer supported. "
                "LiberoRolloutCallback now always runs at the last training step. "
                "Remove 'rollout_on_end' and 'rollout_every' from your config."
            )
        from vlaworkspace.env_runner.libero_runner import LiberoRunner

        runner = hydra.utils.instantiate(libero_runner_cfg, output_dir='data/outputs')
        assert isinstance(runner, LiberoRunner)
        self.runner = runner
        self.rollout_on_start = rollout_on_start
        self._rollout_done = False
        self._zero_logged = False

    def _run_rollout(self, trainer: pl.Trainer, pl_module: "VLATrainer"):
        policy = pl_module.policy_ema if pl_module.policy_ema else pl_module.policy

        # Get adaptor from datamodule for obs/action transforms
        adaptor = None
        if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'dataset'):
            adaptor = getattr(trainer.datamodule.dataset, 'adaptor', None)

        seeds = self.runner.seeds
        multi_seed = len(seeds) > 1
        combined_log = {"trainer/global_step": trainer.global_step}

        all_seed_mean_scores = []
        all_seed_max_scores = []

        for seed in seeds:
            logger.info(f"LiberoRolloutCallback: running rollout with seed={seed}")
            runner_log = self.runner.run(policy, adaptor=adaptor, seed=seed)

            all_seed_mean_scores.append(runner_log.get("test/mean_score", 0.0))
            all_seed_max_scores.append(runner_log.get("test/max_score", 0.0))

            if multi_seed:
                # Prefix per-seed keys: test/X -> test/seed_{s}/X
                for key, value in runner_log.items():
                    if key.startswith("test/"):
                        new_key = f"test/seed_{seed}/{key[len('test/'):]}"
                        combined_log[new_key] = value
                    else:
                        combined_log[key] = value
            else:
                combined_log.update(runner_log)

        # Cross-seed aggregates (only when multi-seed)
        if multi_seed:
            combined_log["test/mean_score"] = float(np.mean(all_seed_mean_scores))
            combined_log["test/max_score"] = float(np.mean(all_seed_max_scores))
            # Per-suite aggregates across seeds
            for suite_name in self.runner.task_suites:
                suite_scores = [
                    combined_log.get(f"test/seed_{s}/{suite_name}/mean_score", 0.0)
                    for s in seeds
                ]
                combined_log[f"test/{suite_name}/mean_score"] = float(np.mean(suite_scores))

        trainer.logger.experiment.log(combined_log, step=trainer.global_step)

    def on_train_start(self, trainer: pl.Trainer, pl_module: "VLATrainer"):
        if self.rollout_on_start:  # just for debug
            logger.info("LiberoRolloutCallback: running initial rollout before training")
            self._run_rollout(trainer, pl_module)

    def _log_zero_baseline(self, trainer: pl.Trainer):
        """Log all-zero rollout metrics at step 1 so WandB charts have a baseline."""
        zero_log = {"trainer/global_step": trainer.global_step}
        seeds = self.runner.seeds
        multi_seed = len(seeds) > 1

        for suite_name in self.runner.task_suites:
            zero_log[f"test/{suite_name}/mean_score"] = 0.0
            if multi_seed:
                for seed in seeds:
                    zero_log[f"test/seed_{seed}/{suite_name}/mean_score"] = 0.0
        zero_log["test/mean_score"] = 0.0
        zero_log["test/max_score"] = 0.0
        if multi_seed:
            for seed in seeds:
                zero_log[f"test/seed_{seed}/mean_score"] = 0.0
                zero_log[f"test/seed_{seed}/max_score"] = 0.0
        trainer.logger.experiment.log(zero_log, step=trainer.global_step)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: "VLATrainer", outputs, batch, batch_idx):
        if self._rollout_done:
            return
        # Log zero baseline at step 1
        if trainer.global_step == 1 and not self._zero_logged:
            self._log_zero_baseline(trainer)
            self._zero_logged = True
        if trainer.max_steps > 0 and trainer.global_step >= trainer.max_steps:
            logger.info("LiberoRolloutCallback: running final rollout at last step")
            self._run_rollout(trainer, pl_module)
            self._rollout_done = True


# endregion
# =============================================================================
# region Pretrained Weight Loading (DAH Two-Stage Training)
# =============================================================================


def load_pretrained_weights(model, ckpt_path):
    """Load pretrained weights and freeze loaded parameters.

    Used for DAH Stage 2 training with Conv1D (DP_C) models.
    Ported from DecoupledActionHead/trainer_pl_all.py.

    Steps:
    1. Record initially frozen parameters
    2. Load checkpoint state_dict
    3. Match keys by name AND shape
    4. Load compatible weights
    5. Freeze all loaded parameters
    6. Keep unloaded parameters (new vision encoder) trainable

    Args:
        model: The policy model to load weights into.
        ckpt_path: Path to stage 1 checkpoint (.ckpt).

    Returns:
        The model with loaded and frozen weights.
    """
    # 0. Record initially frozen parameters
    initially_frozen_keys = {
        name for name, param in model.named_parameters() if not param.requires_grad
    }
    if initially_frozen_keys:
        logger.info(
            f"Detected {len(initially_frozen_keys)} parameters frozen before weight loading."
        )

    if not ckpt_path:
        logger.info("No checkpoint path provided, skipping weight loading.")
        return model

    # 1. Load and filter compatible weights
    logger.info(f"Loading stage1 weights from '{ckpt_path}'...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pretrained_dict = ckpt["state_dict"]

    new_model_dict = model.state_dict()
    loadable_keys = set()
    filtered_dict = {}

    for k, v in pretrained_dict.items():
        if k in new_model_dict and new_model_dict[k].shape == v.shape:
            filtered_dict[k] = v
            loadable_keys.add(k)
    logger.info(f"Found {len(loadable_keys)} compatible parameters out of {len(pretrained_dict)}.")

    # 2. Load filtered weights
    new_model_dict.update(filtered_dict)
    model.load_state_dict(new_model_dict)
    logger.info("Successfully loaded compatible weights.")

    # 3. Freeze/unfreeze parameters
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        is_initially_frozen = name in initially_frozen_keys
        is_loaded_from_ckpt = name in loadable_keys

        # Smart bias logic: keep bias trainable if its weight wasn't loaded
        if name.endswith(".bias") and not is_initially_frozen:
            weight_name = name.replace(".bias", ".weight")
            if weight_name not in loadable_keys:
                is_loaded_from_ckpt = False

        if is_initially_frozen or is_loaded_from_ckpt:
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True
            trainable_params += 1

    logger.info(f"Stage2 weight loading: {frozen_params} frozen, {trainable_params} trainable.")

    # 4. Verification summary
    logger.info("--- Parameter status after stage2 weight loading ---")
    for name, param in model.named_parameters():
        status = "FROZEN" if not param.requires_grad else "TRAINABLE"
        reason = ""
        if not param.requires_grad:
            if name in initially_frozen_keys:
                reason = "(initially frozen)"
            elif name in loadable_keys:
                reason = "(loaded from ckpt)"
        logger.info(f"  [{status}] {name} {reason}")

    return model


def load_pretrained_weights_DP_T(model, ckpt_path):
    """Load pretrained weights for Transformer architectures (DP_T, DP_T_FILM, DP_MLP).

    Same as load_pretrained_weights but with manually unfrozen keys for
    transformer-specific layers that need to remain trainable even when loaded.

    Args:
        model: The policy model to load weights into.
        ckpt_path: Path to stage 1 checkpoint (.ckpt).

    Returns:
        The model with loaded and frozen weights.
    """
    # 0. Record initially frozen parameters
    initially_frozen_keys = {
        name for name, param in model.named_parameters() if not param.requires_grad
    }
    if initially_frozen_keys:
        logger.info(
            f"Detected {len(initially_frozen_keys)} parameters frozen before weight loading."
        )

    if not ckpt_path:
        logger.info("No checkpoint path provided, skipping weight loading.")
        return model

    # Transformer layers that should remain trainable even when loaded
    manually_unfrozen_keys = [
        "model.pos_emb",
        "model.encoder.0.weight",
        "model.encoder.0.bias",
        "model.encoder.2.weight",
        "model.encoder.2.bias",
    ]

    # 1. Load and filter compatible weights
    logger.info(f"Loading stage1 weights from '{ckpt_path}'...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pretrained_dict = ckpt["state_dict"]

    new_model_dict = model.state_dict()
    loadable_keys = set()
    filtered_dict = {}

    for k, v in pretrained_dict.items():
        if k in new_model_dict and new_model_dict[k].shape == v.shape:
            filtered_dict[k] = v
            loadable_keys.add(k)
    logger.info(f"Found {len(loadable_keys)} compatible parameters out of {len(pretrained_dict)}.")

    # 2. Load filtered weights
    new_model_dict.update(filtered_dict)
    model.load_state_dict(new_model_dict)
    logger.info("Successfully loaded compatible weights.")

    # 3. Freeze/unfreeze parameters
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        is_initially_frozen = name in initially_frozen_keys
        is_loaded_from_ckpt = name in loadable_keys
        is_manually_unfrozen = name in manually_unfrozen_keys

        # Smart bias logic
        if name.endswith(".bias") and not is_initially_frozen:
            weight_name = name.replace(".bias", ".weight")
            if weight_name not in loadable_keys:
                is_loaded_from_ckpt = False

        if (is_initially_frozen or is_loaded_from_ckpt) and not is_manually_unfrozen:
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True
            trainable_params += 1

    logger.info(f"Stage2 weight loading (transformer): {frozen_params} frozen, {trainable_params} trainable.")

    # 4. Verification summary
    logger.info("--- Parameter status after stage2 weight loading ---")
    for name, param in model.named_parameters():
        status = "FROZEN" if not param.requires_grad else "TRAINABLE"
        reason = ""
        if not param.requires_grad:
            if name in initially_frozen_keys:
                reason = "(initially frozen)"
            elif name in loadable_keys:
                reason = "(loaded from ckpt)"
        elif name in manually_unfrozen_keys:
            reason = "(manually unfrozen)"
        logger.info(f"  [{status}] {name} {reason}")

    return model


def freeze_random_action_head(model, reference_ckpt_path):
    """Freeze randomly-initialized action head layers (no weight loading).

    Uses a reference stage 1 checkpoint to identify which layers constitute
    the action head (by name + shape matching), but does NOT load any weights.
    The matched layers are frozen at their random init values; everything else
    remains trainable.

    This creates a "random frozen" baseline: the action head has the same
    architecture as a stage 1 pretrained head, but with random weights.

    Args:
        model: The freshly instantiated policy model.
        reference_ckpt_path: Path to a stage 1 checkpoint used only to
            identify action head layer names/shapes.

    Returns:
        The model with action head layers frozen at random init.
    """
    if not reference_ckpt_path:
        raise ValueError("freeze_random_action_head requires a reference_ckpt_path")

    logger.info(f"Identifying action head layers from reference ckpt: '{reference_ckpt_path}'...")
    ckpt = torch.load(reference_ckpt_path, map_location="cpu", weights_only=False)
    ref_dict = ckpt["state_dict"]

    # Match by name AND shape to identify action head layers
    model_dict = model.state_dict()
    action_head_keys = set()
    for k, v in ref_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            action_head_keys.add(k)
    logger.info(f"Matched {len(action_head_keys)} action head layers from reference ckpt.")

    # Freeze matched layers (at random init), keep everything else trainable
    frozen_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if name in action_head_keys:
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True
            trainable_params += 1

    logger.info(f"Random frozen: {frozen_params} frozen (random init), {trainable_params} trainable.")

    # Verification summary
    logger.info("--- Parameter status after random frozen ---")
    for name, param in model.named_parameters():
        status = "FROZEN (random)" if not param.requires_grad else "TRAINABLE"
        logger.info(f"  [{status}] {name}")

    return model


# endregion
# =============================================================================
# region Lightning Module
# =============================================================================


class VLATrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training VLA policies.

    Supports:
        - Any policy implementing BasePolicy interface
        - EMA (Exponential Moving Average) model
        - LoRA checkpoint saving
        - DAH two-stage training (train_mode: stage1, stage2, normal)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Instantiate policy
        self.policy:BasePolicy = hydra.utils.instantiate(cfg.policy)

        # Handle train_mode for DAH two-stage training
        train_mode = cfg.get("train_mode", "normal")
        if train_mode in ("stage2", "stage2_rollout"):
            ckpt_path = cfg.get("ckpt_path")
            using_transformers = cfg.get("using_transformers", False)
            if using_transformers:
                self.policy = load_pretrained_weights_DP_T(self.policy, ckpt_path)
            else:
                self.policy = load_pretrained_weights(self.policy, ckpt_path)

            # Optionally unfreeze + reinit conditioning params for LoRA stage 2
            if cfg.get("unfreeze_cond_params", False):
                unfrozen = 0
                for name, param in self.policy.named_parameters():
                    if '.lora_' in name:
                        param.requires_grad = True
                        unfrozen += 1
                # Reinit LoRA for clean stage 2 start:
                # - lora_down: random init (stage 1 learned on zeros, useless for vision)
                # - lora_up_q/v: zero-init (identity at init, output = pretrained backbone)
                for layer in self.policy.model.decoder.layers:
                    if hasattr(layer, 'lora_down'):
                        torch.nn.init.normal_(layer.lora_down.weight, 0.0, 0.02)
                        torch.nn.init.zeros_(layer.lora_down.bias)
                        torch.nn.init.zeros_(layer.lora_up_q.weight)
                        torch.nn.init.zeros_(layer.lora_up_q.bias)
                        torch.nn.init.zeros_(layer.lora_up_v.weight)
                        torch.nn.init.zeros_(layer.lora_up_v.bias)
                logger.info(f"Unfroze and reinitialized {unfrozen} LoRA conditioning params for stage 2.")

        elif train_mode in ("random_frozen", "random_frozen_rollout"):
            reference_ckpt = cfg.get("ckpt_path")
            self.policy = freeze_random_action_head(self.policy, reference_ckpt)

        # EMA model (optional)
        self.ema_handler = None
        self.policy_ema = None
        if cfg.training.get("use_ema", False):
            self.policy_ema = copy.deepcopy(self.policy)
            self.ema_handler = hydra.utils.instantiate(cfg.ema, model=self.policy_ema)

        # Timing stats
        self._step_times = []
        self._last_step_time = None

        # Metrics buffer for averaging (like OpenPi)
        self._metrics_buffer = []
        self._log_every_n_steps = cfg.training.get("log_every_n_steps", 100)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Timing
        now = time.time()
        if self._last_step_time is not None and self.global_step > 0:
            self._step_times.append(now - self._last_step_time)
        self._last_step_time = now

        # Log timing every 25 steps
        if len(self._step_times) >= 25:
            times = torch.tensor(self._step_times)
            logger.debug(f"Step timing: mean={times.mean() * 1000:.0f}ms, std={times.std() * 1000:.0f}ms")
            self._step_times = []

        # Compute loss (adaptor transform already applied in TransformedDataset)
        loss = self.policy.compute_loss(batch)

        # Accumulate metrics for averaging (like OpenPi)
        # grad_norm will be added in configure_gradient_clipping
        self._metrics_buffer.append({
            "loss": loss.detach().item(),
            "lr": self.optimizers().param_groups[0]["lr"],
        })

        # Log averaged metrics every N steps (like OpenPi)
        if (self.global_step + 1) % self._log_every_n_steps == 0 and len(self._metrics_buffer) > 0:
            avg_loss = sum(m["loss"] for m in self._metrics_buffer) / len(self._metrics_buffer)
            avg_lr = sum(m["lr"] for m in self._metrics_buffer) / len(self._metrics_buffer)

            log_dict = {
                "train/loss": avg_loss,
                "train/lr": avg_lr,
            }

            # Average grad_norm if available
            grad_norms = [m["grad_norm"] for m in self._metrics_buffer if "grad_norm" in m]
            if grad_norms:
                log_dict["train/grad_norm"] = sum(grad_norms) / len(grad_norms)

            self.log_dict(log_dict, on_step=True, prog_bar=True)
            self._metrics_buffer = []  # Reset buffer

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each training step."""
        if self.ema_handler is not None:
            self.ema_handler.step(self.policy)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        policy = self.policy_ema if self.policy_ema else self.policy
        loss = policy.compute_loss(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ):
        """
        Clip gradients and accumulate grad_norm for averaged logging (matches OpenPi behavior).

        OpenPi accumulates grad_norm over log_interval steps and logs the average.
        This helps monitor gradient health with reduced noise.
        """
        if gradient_clip_val is None:
            return

        # clip_grad_norm_ returns the total norm BEFORE clipping
        params = [p for group in optimizer.param_groups for p in group["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=gradient_clip_val,
        )

        # Add grad_norm to current step's metrics (will be averaged with loss)
        if self._metrics_buffer:
            self._metrics_buffer[-1]["grad_norm"] = grad_norm.item()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Both optimizer and scheduler are provided by the policy.
        """
        cfg = self.cfg
        num_training_steps = self.trainer.estimated_stepping_batches

        optimizer = self.policy.get_optimizer(**cfg.optimizer)
        lr_scheduler = self.policy.get_scheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            last_epoch=self.global_step - 1,
            warmup_steps=cfg.training.get("lr_warmup_steps", 1000),
            decay_lr_ratio=cfg.training.get("lr_decay_ratio", 0.1),
            decay_steps=cfg.training.get("lr_decay_steps", None),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        """Save checkpoint - handle LoRA policies specially."""
        policy = self.policy_ema if self.cfg.training.get("use_ema") else self.policy

        # Check for LoRA policy
        if hasattr(policy, "save_lora_weights"):
            # Save only trainable parameters
            lora_state = {n: p.data for n, p in policy.named_parameters() if p.requires_grad}
            checkpoint["state_dict"] = lora_state
            checkpoint["is_lora_checkpoint"] = True
        else:
            checkpoint["state_dict"] = policy.state_dict()
            checkpoint["is_lora_checkpoint"] = False

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        """Load checkpoint - handle LoRA checkpoints."""
        if checkpoint.get("is_lora_checkpoint", False):
            policy = self.policy_ema if self.cfg.training.get("use_ema") else self.policy
            model_state = policy.state_dict()

            for name, param in checkpoint["state_dict"].items():
                if name in model_state:
                    model_state[name] = param

            policy.load_state_dict(model_state)
            checkpoint["state_dict"] = {}  # Prevent Lightning from loading again

# endregion
# =============================================================================
# region Data Module
# =============================================================================

from vlaworkspace.dataset import InMemoryTransformedDataset, TransformedDataset


class VLADataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for VLA training.

    Supports:
        - LeRobot dataset format
        - Config-driven adaptor via TransformedDataset
        - Automatic train/val split
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_generator = None
        self.val_generator = None
        self.normalizer = None

    def setup(self, stage: str):
        """Setup datasets."""
        if stage == "fit":
            # Seed generators
            base_seed = self.cfg.training.seed
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            seed = base_seed + rank * 1000

            self.train_generator = torch.Generator().manual_seed(seed)
            self.val_generator = torch.Generator().manual_seed(seed)

            # Instantiate base dataset
            base_dataset = hydra.utils.instantiate(self.cfg.dataset)

            # Get adaptor config (may be None) - check root config first, then task config
            adaptor_cfg = self.cfg.get("adaptor")

            # Wrap with TransformedDataset (adaptor handles norm_stats and tokenizer internally)
            if self.cfg.get("preload", False):
                cache_path = self.cfg.get("preload_path", None)
                self.dataset = InMemoryTransformedDataset(base_dataset, adaptor_cfg, cache_path=cache_path)
            else:
                self.dataset = TransformedDataset(base_dataset, adaptor_cfg)

            # Cache HF columns in memory for fast random access (after adaptor init / norm stats)
            if self.cfg.get("cache_in_memory", False):
                from vlaworkspace.dataset.lerobot_dataset import cache_dataset_in_memory
                cache_dataset_in_memory(base_dataset)

            # Get validation dataset (skip if validation is disabled)
            if self.cfg.training.get("val_every", None) is not None:
                if hasattr(base_dataset, "get_validation_dataset"):
                    val_base = base_dataset.get_validation_dataset()
                    self.val_dataset = TransformedDataset(val_base, adaptor_cfg)
                elif hasattr(base_dataset, "get_validation_split"):
                    _, val_base = base_dataset.get_validation_split(
                        val_ratio=self.cfg.get("val_ratio", 0.02),
                        seed=seed,
                    )
                    self.val_dataset = TransformedDataset(val_base, adaptor_cfg)
                else:
                    self.val_dataset = self.dataset  # Use same dataset for val

            # Save auto-computed norm stats to run_dir for inference
            self._save_auto_norm_stats()

            # Get normalizer if available (legacy compatibility)
            if hasattr(base_dataset, "get_normalizer"):
                self.normalizer = base_dataset.get_normalizer()

    def _save_auto_norm_stats(self) -> None:
        """Save auto-computed norm stats to run_dir so serve.py can load them."""
        adaptor = getattr(self.dataset, "adaptor", None)
        if adaptor is None:
            return
        model = getattr(adaptor, "model", None)
        if model is None:
            return
        norm_stats = model.get_norm_stats()
        if norm_stats is None:
            return

        # Only save if norm_stats were auto-computed (check config)
        norm_stats_path = (
            OmegaConf.select(self.cfg, "adaptor.model.norm_stats_path")
            or OmegaConf.select(self.cfg, "adaptor.norm_stats_path")
        )
        if norm_stats_path != "auto":
            return

        run_dir = self.cfg.run_dir
        dst_path = os.path.join(run_dir, "norm_stats.json")
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for key, stats in norm_stats.items():
            serializable[key] = {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in stats.items()
            }
        with open(dst_path, "w") as f:
            json.dump({"norm_stats": serializable}, f, indent=2)
        logger.info(f"Saved auto-computed norm_stats to {dst_path}")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        from vlaworkspace.dataset import lerobot_collate_fn

        collate_fn = getattr(self.dataset, "collate_fn", None) or lerobot_collate_fn
        num_workers = self.cfg.dataloader.get("num_workers", 4)
        # persistent_workers and prefetch_factor require num_workers > 0
        persistent_workers = self.cfg.dataloader.get("persistent_workers", True) and num_workers > 0
        prefetch_factor = self.cfg.dataloader.get("prefetch_factor", 2) if num_workers > 0 else None

        # Use EpisodeAwareSampler to drop frames near episode boundaries,
        # matching DAH's SequenceSampler behavior:
        #   drop_n_last = horizon - n_obs_steps - n_action_steps - n_latency_steps + 1
        use_episode_aware = self.cfg.dataloader.get("use_episode_aware_sampler", False)
        sampler = None
        shuffle = self.cfg.dataloader.get("shuffle", True)
        if use_episode_aware:
            horizon = self.cfg.get("horizon", 0)
            n_obs = self.cfg.get("n_obs_steps", 0)
            n_action = self.cfg.get("n_action_steps", 0)
            n_latency = self.cfg.get("n_latency_steps", 0)
            drop_n_last = max(0, horizon - n_obs - n_action - n_latency + 1)
        else:
            drop_n_last = 0
        if drop_n_last > 0 and hasattr(self.dataset, "meta"):
            from lerobot.datasets.sampler import EpisodeAwareSampler

            sampler = EpisodeAwareSampler(
                dataset_from_indices=self.dataset.meta.episodes["dataset_from_index"],
                dataset_to_indices=self.dataset.meta.episodes["dataset_to_index"],
                drop_n_last_frames=drop_n_last,
                shuffle=shuffle,
            )
            shuffle = False  # DataLoader doesn't allow both sampler and shuffle

        return DataLoader(
            self.dataset,
            batch_size=self.cfg.dataloader.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.cfg.dataloader.get("pin_memory", True),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=self.train_generator,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Create validation dataloader. Returns None if validation is disabled."""
        if self.cfg.training.get("val_every", None) is None:
            return None

        from vlaworkspace.dataset import lerobot_collate_fn

        collate_fn = getattr(self.val_dataset, "collate_fn", None) or lerobot_collate_fn
        val_cfg = self.cfg.get("val_dataloader", self.cfg.dataloader)
        num_workers = val_cfg.get("num_workers", 4)
        # persistent_workers and prefetch_factor require num_workers > 0
        persistent_workers = val_cfg.get("persistent_workers", True) and num_workers > 0
        prefetch_factor = val_cfg.get("prefetch_factor", 2) if num_workers > 0 else None

        # Use EpisodeAwareSampler to drop frames near episode boundaries
        # (same logic as train_dataloader)
        use_episode_aware = val_cfg.get("use_episode_aware_sampler", False)
        sampler = None
        if use_episode_aware:
            horizon = self.cfg.get("horizon", 0)
            n_obs = self.cfg.get("n_obs_steps", 0)
            n_action = self.cfg.get("n_action_steps", 0)
            n_latency = self.cfg.get("n_latency_steps", 0)
            drop_n_last = max(0, horizon - n_obs - n_action - n_latency + 1)
        else:
            drop_n_last = 0
        if drop_n_last > 0 and hasattr(self.val_dataset, "meta"):
            from lerobot.datasets.sampler import EpisodeAwareSampler

            sampler = EpisodeAwareSampler(
                dataset_from_indices=self.val_dataset.meta.episodes["dataset_from_index"],
                dataset_to_indices=self.val_dataset.meta.episodes["dataset_to_index"],
                drop_n_last_frames=drop_n_last,
                shuffle=False,
            )

        return DataLoader(
            self.val_dataset,
            batch_size=val_cfg.get("batch_size", self.cfg.dataloader.batch_size),
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=val_cfg.get("pin_memory", True),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=self.val_generator,
            collate_fn=collate_fn,
        )

# endregion
# =============================================================================
# Training Function
# =============================================================================


def train(cfg: DictConfig):
    """Main training function."""
    set_all_seeds(cfg.training.seed)

    # Create run directory
    run_dir = cfg.run_dir
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

    # Save norm_stats.json alongside config.yaml
    norm_stats_path = OmegaConf.select(cfg, "adaptor.model.norm_stats_path")
    if norm_stats_path and norm_stats_path != "auto" and os.path.exists(norm_stats_path):
        dst_path = os.path.join(run_dir, "norm_stats.json")
        shutil.copy2(norm_stats_path, dst_path)
        logger.info(f"Saved norm_stats.json to {dst_path}")

    # Checkpoint callback — epoch-based (primary) or step-based (fallback)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    checkpoint_epochs = cfg.training.get("checkpoint_every")

    if checkpoint_epochs:
        # Epoch-based checkpointing (primary)
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:03d}",
            every_n_epochs=checkpoint_epochs,
            save_top_k=-1,
            save_last=True,
            save_weights_only=True,
            save_on_train_epoch_end=True,
        )
    else:
        # Step-based checkpointing (fallback)
        checkpoint_steps = cfg.training.get("checkpoint_every_steps")
        if checkpoint_steps:
            checkpoint_callback = ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="{step:06d}",
                every_n_train_steps=checkpoint_steps,
                save_top_k=-1,
                save_last=True,
                save_weights_only=True,
            )
        else:
            # Default: every 5 epochs
            checkpoint_callback = ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="{epoch:03d}",
                every_n_epochs=5,
                save_top_k=-1,
                save_last=True,
                save_weights_only=True,
                save_on_train_epoch_end=True,
            )

    # WandB logger
    os.makedirs(os.path.join(run_dir, "wandb"), exist_ok=True)
    wandb_logger = WandbLogger(
        save_dir=run_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.logging,
    )

    # Save WandB run ID
    try:
        run_id = wandb_logger.experiment.id
        if callable(run_id):
            run_id = run_id()
        with open(os.path.join(run_dir, "wandb_run_id.txt"), "w") as f:
            f.write(str(run_id))
    except Exception as e:
        logger.warning(f"Could not save WandB run ID: {e}")

    # Create trainer
    slurm_progress = SlurmProgressBar(refresh_rate=100)
    n_action_mse_evals = cfg.training.get("n_action_mse_evaluations", 100)
    action_mse = ActionMseLoss(n_evaluations=n_action_mse_evals)
    callbacks = [checkpoint_callback,
                 slurm_progress,
                 action_mse,
                 ]

    # Add mimicgen rollout callback when train_mode ends with _rollout
    train_mode = cfg.get("train_mode", "normal")
    if train_mode in ("stage2_rollout", "normal_rollout", "random_frozen_rollout"):
        env_runner_cfg = cfg.get("env_runner")
        if env_runner_cfg is not None:
            rollout_every = cfg.training.get("rollout_every", 1)
            repo_id = cfg.dataset.get("repo_id", "")
            rollout_cb = MimicgenRolloutCallback(env_runner_cfg, rollout_every_n_epochs=rollout_every, repo_id=repo_id)
            callbacks.append(rollout_cb)
            logger.info(f"MimicgenRolloutCallback enabled: rollout every {rollout_every} epochs")
        else:
            logger.warning("train_mode='{}' but no env_runner config found".format(train_mode))

    # Add LIBERO rollout callback when train_mode ends with _rollout
    if train_mode in ("stage2_rollout", "normal_rollout", "random_frozen_rollout"):
        libero_runner_cfg = cfg.get("libero_runner")
        if libero_runner_cfg is not None:
            rollout_on_start = cfg.training.get("rollout_on_start", False)
            libero_cb = LiberoRolloutCallback(
                libero_runner_cfg,
                rollout_on_start=rollout_on_start,
                rollout_on_end=cfg.training.get("rollout_on_end", False),
                rollout_every_n_epochs=cfg.training.get("rollout_every", None),
            )
            callbacks.append(libero_cb)
            logger.info("LiberoRolloutCallback enabled: rollout at last step"
                         f"{', rollout_on_start=True' if rollout_on_start else ''}")

    trainer = pl.Trainer(
        callbacks=callbacks,
        enable_progress_bar=False,  # Always enable Lightning's default tqdm bar
        accelerator=cfg.training.get("accelerator", "gpu"),
        devices="auto",
        strategy=cfg.training.get("strategy", "auto"),
        precision=cfg.training.get("precision", "32"),
        max_steps=cfg.training.get("max_steps", -1),
        max_epochs=cfg.training.get("num_epochs", -1),
        accumulate_grad_batches=cfg.training.get("gradient_accumulate_every", 1),
        gradient_clip_val=cfg.training.get("gradient_clip_val", None),  # OpenPI default: 1.0
        log_every_n_steps=cfg.training.get("log_every_n_steps", 100),  # OpenPI default: 100
        logger=[wandb_logger],
        use_distributed_sampler=True,
        check_val_every_n_epoch=cfg.training.get("val_every", None),
        limit_val_batches=0 if cfg.training.get("val_every", None) is None else 1.0,
    )

    # Train
    model = VLATrainer(cfg)
    datamodule = VLADataModule(cfg)
    trainer.fit(model, datamodule=datamodule)


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent / "src" / "vlaworkspace" / "config"),
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    train(cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    main()
