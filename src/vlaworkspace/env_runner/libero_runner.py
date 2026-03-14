"""
LIBERO environment runner for in-training rollout evaluation.

Follows the same pattern as RobomimicRunner but adapted for LIBERO:
- Per-task env lifecycle (each LIBERO task has a different BDDL file)
- Uses AsyncVectorEnv with spawn context for safe multiprocessing
- Collects per-task success rates and videos for WandB logging
"""

import gc
import logging
import math
import os
import pathlib
import random
import time

import dill
import imageio
import numpy as np
import torch
import wandb

from vlaworkspace.z_utils.pytorch_util import dict_apply
from vlaworkspace.env_runner.env.libero.async_vector_env import AsyncVectorEnv
from vlaworkspace.env_runner.env.libero.libero_wrapper import LiberoWrapper
from vlaworkspace.env_runner.env.libero.multistep_wrapper import MultiStepWrapper
from vlaworkspace.env_runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)

# Max episode steps per LIBERO task suite
SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class LiberoRunner(BaseRunner):
    """LIBERO environment runner for in-training rollout evaluation.

    Creates AsyncVectorEnv per task (each LIBERO task has a different BDDL file),
    runs chunked rollouts, and returns WandB log dict with success rates and videos.
    """

    def __init__(
        self,
        output_dir,
        task_suites=None,
        n_trials_per_task=10,
        n_video_per_task=1,
        n_envs=8,
        replan_steps=8,
        n_obs_steps=1,
        max_steps=None,
        num_steps_wait=10,
        fps=10,
        seed=42,
        resolution=256,
    ):
        super().__init__(output_dir)

        # Accept single string or list of suites
        if task_suites is None:
            task_suites = ["libero_spatial"]
        elif isinstance(task_suites, str):
            task_suites = [task_suites]
        self.task_suites = list(task_suites)

        self.n_trials_per_task = n_trials_per_task
        self.n_video_per_task = n_video_per_task
        self.n_envs = n_envs
        self.replan_steps = replan_steps
        self.n_obs_steps = n_obs_steps
        self.max_steps_override = max_steps  # None = auto per suite
        self.num_steps_wait = num_steps_wait
        self.fps = fps
        # Normalize seed to list for multi-seed support (includes Hydra ListConfig)
        if isinstance(seed, (list, tuple)) or hasattr(seed, '__iter__') and not isinstance(seed, str):
            self.seeds = [int(s) for s in seed]
        else:
            self.seeds = [int(seed)]
        self.seed = self.seeds[0]  # backward compat default
        self.resolution = resolution

        # Validate suite names when max_steps is not overridden
        if max_steps is None:
            for suite in self.task_suites:
                if suite not in SUITE_MAX_STEPS:
                    raise ValueError(
                        f"Unknown task suite '{suite}'. "
                        f"Known suites: {list(SUITE_MAX_STEPS.keys())}. "
                        f"Set max_steps explicitly."
                    )

        # Track best score across evaluations (per-seed for multi-seed isolation)
        self.max_mean_score = 0.0
        self._per_seed_max = {s: 0.0 for s in self.seeds}

        logger.info(
            f"LiberoRunner: suites={self.task_suites}, "
            f"n_trials={n_trials_per_task}, n_envs={n_envs}, "
            f"replan_steps={replan_steps}"
        )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Seed all RNG sources for reproducible rollouts."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self, policy, adaptor=None, seed=None):
        """Run LIBERO rollout evaluation across all tasks in all suites.

        Args:
            policy: Policy with predict_action() method.
            adaptor: Optional Adaptor for obs/action transforms.
            seed: Optional seed override. Falls back to self.seed.

        Returns:
            Dict of WandB log entries (success rates + videos).
        """
        device = policy.device
        active_seed = seed if seed is not None else self.seed
        self._seed_everything(active_seed)

        if adaptor is not None:
            adaptor.eval()

        # Lazy import LIBERO (heavy deps)
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()

        log_data = {}
        all_suite_mean_scores = []

        for suite_name in self.task_suites:
            max_steps = self.max_steps_override or SUITE_MAX_STEPS[suite_name]

            task_suite = benchmark_dict[suite_name]()
            n_tasks = task_suite.n_tasks

            logger.info(
                f"=== Suite: {suite_name} ({n_tasks} tasks, max_steps={max_steps}) ==="
            )

            task_success_rates = []

            for task_id in range(n_tasks):
                task = task_suite.get_task(task_id)
                task_description = task.language
                task_name = task_description.replace(" ", "_")
                initial_states = task_suite.get_task_init_states(task_id)

                logger.info(
                    f"[{suite_name} Task {task_id + 1}/{n_tasks}] {task_description}"
                )

                # Create env functions for this task
                env_fn, dummy_env_fn = self._make_env_fns(
                    task, get_libero_path, OffScreenRenderEnv, max_steps
                )

                # Create init functions for each trial
                env_init_fn_dills = []
                for ep_idx in range(self.n_trials_per_task):
                    init_state = initial_states[ep_idx % len(initial_states)]

                    def make_init_fn(init_state):
                        def init_fn(wrapper):
                            # wrapper is MultiStepWrapper, wrapper.env is LiberoWrapper
                            wrapper.env.init_state = init_state
                        return init_fn

                    env_init_fn_dills.append(dill.dumps(make_init_fn(init_state)))

                env = AsyncVectorEnv(
                    [env_fn] * self.n_envs,
                    dummy_env_fn=dummy_env_fn,
                    shared_memory=False,
                    context="spawn",
                )
                env.seed(active_seed)

                try:
                    task_successes, task_videos = self._run_task_rollouts(
                        env, env_init_fn_dills, policy, adaptor, device,
                        task_description, task_name, max_steps,
                    )
                finally:
                    env.close()
                    gc.collect()

                # Compute task success rate
                task_rate = task_successes / self.n_trials_per_task
                task_success_rates.append(task_rate)
                log_data[f"test/{suite_name}/task_{task_name}_success_rate"] = task_rate

                logger.info(
                    f"[{suite_name} Task {task_id + 1}/{n_tasks}] {task_description}: "
                    f"{task_successes}/{self.n_trials_per_task} ({100 * task_rate:.1f}%)"
                )

                # Add videos
                for vid_key, vid_obj in task_videos.items():
                    log_data[vid_key] = vid_obj

            # Per-suite aggregate
            suite_mean = np.mean(task_success_rates) if task_success_rates else 0.0
            all_suite_mean_scores.append(suite_mean)
            log_data[f"test/{suite_name}/mean_score"] = suite_mean

            logger.info(
                f"Suite {suite_name} done: mean_score={100 * suite_mean:.1f}%"
            )

        # Global aggregate across all suites
        mean_score = np.mean(all_suite_mean_scores) if all_suite_mean_scores else 0.0
        # Track per-seed historical best (isolated); fall back to shared tracker
        if active_seed in self._per_seed_max:
            self._per_seed_max[active_seed] = max(self._per_seed_max[active_seed], mean_score)
            max_score = self._per_seed_max[active_seed]
        else:
            self.max_mean_score = max(self.max_mean_score, mean_score)
            max_score = self.max_mean_score
        log_data["test/mean_score"] = mean_score
        log_data["test/max_score"] = max_score

        logger.info(
            f"LIBERO rollout done (suites={self.task_suites}): "
            f"mean_score={100 * mean_score:.1f}%, "
            f"max_score={100 * self.max_mean_score:.1f}%"
        )

        return log_data

    def _make_env_fns(self, task, get_libero_path, OffScreenRenderEnv, max_steps):
        """Create env_fn and dummy_env_fn for a LIBERO task."""
        task_bddl_file = str(
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        resolution = self.resolution
        replan_steps = self.replan_steps

        def env_fn():
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=resolution,
                camera_widths=resolution,
            )
            return MultiStepWrapper(
                LiberoWrapper(env, render_hw=(resolution, resolution)),
                n_action_steps=replan_steps,
                max_episode_steps=max_steps,
            )

        def dummy_env_fn():
            return _DummyEnv(resolution)

        return env_fn, dummy_env_fn

    def _run_task_rollouts(
        self, env, env_init_fn_dills, policy, adaptor, device,
        task_description, task_name, max_steps,
    ):
        """Run chunked rollouts for a single task, return (n_successes, video_dict)."""
        n_inits = len(env_init_fn_dills)
        n_chunks = math.ceil(n_inits / self.n_envs)
        task_successes = 0
        task_videos = {}

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.n_envs
            end = min(n_inits, start + self.n_envs)
            this_n_active = end - start

            # Pad init functions if needed
            this_init_fns = list(env_init_fn_dills[start:end])
            if len(this_init_fns) < self.n_envs:
                this_init_fns.extend(
                    [env_init_fn_dills[0]] * (self.n_envs - len(this_init_fns))
                )

            # Init environments
            env.call_each(
                "run_dill_function",
                args_list=[(fn,) for fn in this_init_fns],
            )

            # Reset
            obs = env.reset()

            # Wait for physics to stabilize
            dummy_action = np.zeros(
                (self.n_envs, 1, 7), dtype=np.float32
            )
            dummy_action[..., 6] = -1.0  # Keep gripper closed
            for _ in range(self.num_steps_wait):
                obs, _, _, _ = env.step(dummy_action)

            # Reset step count so wait doesn't count against budget
            env.call_each("reset_step_count")

            # Main rollout loop
            rollout_tag = f"Rollout {task_name} chunk {chunk_idx+1}/{n_chunks}"
            logger.info(f"{rollout_tag} | starting | max_steps={max_steps}")

            policy.reset()
            step_count = 0
            dones = np.zeros(self.n_envs, dtype=bool)
            successes = np.zeros(self.n_envs, dtype=bool)
            chunk_frames = {i: [] for i in range(this_n_active)}

            t_start = time.time()

            while step_count < max_steps and not np.all(dones[:this_n_active]):
                # Collect video frames
                for i in range(this_n_active):
                    ep_idx = start + i
                    if not dones[i] and ep_idx < self.n_video_per_task:
                        chunk_frames[i].append(
                            obs["agentview_image"][i].copy()
                        )

                # Build observation dict for policy
                np_obs_dict = self._build_obs_dict(obs, task_description)

                # Save raw state before adaptor transforms (needed for
                # delta→absolute conversion in postprocessing)
                raw_states = np_obs_dict.get("state")

                # Preprocess through adaptor
                if adaptor is not None:
                    np_obs_dict = BaseRunner._adaptor_preprocess_obs(
                        np_obs_dict, adaptor
                    )

                # Transfer to device (skip non-numeric arrays like prompt)
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device)
                    if isinstance(x, np.ndarray) and x.dtype.kind != 'U'
                    else x,
                )

                # Run policy
                with torch.no_grad():
                    action_out = policy.predict_action(obs_dict)

                # Handle both dict (DP) and tensor (SmolVLA/Pi0) returns
                if isinstance(action_out, dict):
                    np_action_dict = dict_apply(
                        action_out,
                        lambda x: x.detach().to("cpu").numpy(),
                    )
                    action = np_action_dict["action"]
                else:
                    action = action_out.detach().to("cpu").numpy()
                if not np.all(np.isfinite(action)):
                    logger.warning("Non-finite action detected, clipping")
                    action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

                # Postprocess through adaptor
                if adaptor is not None:
                    env_action = BaseRunner._adaptor_postprocess_actions(
                        action, adaptor, raw_states=raw_states
                    )
                else:
                    env_action = action

                # Slice to replan_steps (model may output more than needed)
                env_action = env_action[:, :self.replan_steps]

                # Step
                obs, _, step_dones, infos = env.step(env_action)
                dones = dones | np.array(step_dones)
                for i, info in enumerate(infos):
                    if info.get("success", False):
                        successes[i] = True

                step_count += self.replan_steps

                # Log every replan step
                elapsed = time.time() - t_start
                pct = min(100.0, 100.0 * step_count / max_steps)
                speed = step_count / elapsed if elapsed > 0 else 0
                n_succ = int(sum(successes[:this_n_active]))
                logger.info(f"{rollout_tag} | {step_count}/{max_steps} ({pct:.0f}%) | {speed:.1f} steps/s | {elapsed:.1f}s | success {n_succ}/{this_n_active}")

            elapsed = time.time() - t_start
            logger.info(f"{rollout_tag} | done | {step_count} steps in {elapsed:.1f}s | success {int(sum(successes[:this_n_active]))}/{this_n_active}")

            # Collect results
            for i in range(this_n_active):
                ep_idx = start + i
                if successes[i]:
                    task_successes += 1

                # Save video
                if ep_idx < self.n_video_per_task and chunk_frames.get(i):
                    frames = chunk_frames[i]
                    if frames:
                        video_path = (
                            pathlib.Path(self.output_dir)
                            / "media"
                            / f"{task_name}_ep{ep_idx}.mp4"
                        )
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        imageio.mimwrite(str(video_path), frames, fps=self.fps)
                        task_videos[f"test/sim_video_{task_name}_{ep_idx}"] = (
                            wandb.Video(str(video_path), format="mp4")
                        )

        return task_successes, task_videos

    def _build_obs_dict(self, obs, task_description):
        """Build observation dict from vectorized env obs.

        Remaps LIBERO keys to adaptor-expected keys:
        - agentview_image -> image
        - adds prompt

        Returns dict with values of shape [n_envs, ...].
        """
        n_envs = self.n_envs

        # Extract per-env observations and remap keys
        images = []
        wrist_images = []
        states = []

        for i in range(n_envs):
            images.append(obs["agentview_image"][i])
            wrist_images.append(obs["wrist_image"][i])
            states.append(obs["state"][i])

        result = {
            "image": np.stack(images, axis=0),
            "wrist_image": np.stack(wrist_images, axis=0),
            "state": np.stack(states, axis=0),
            "prompt": np.array([task_description] * n_envs),
        }

        return result


class _DummyEnv:
    """Minimal dummy env for space inference without OpenGL init."""

    def __init__(self, resolution=256):
        from gymnasium import spaces

        self.observation_space = spaces.Dict(
            {
                "agentview_image": spaces.Box(
                    0, 255, (resolution, resolution, 3), np.uint8
                ),
                "wrist_image": spaces.Box(
                    0, 255, (resolution, resolution, 3), np.uint8
                ),
                "state": spaces.Box(-np.inf, np.inf, (8,), np.float32),
            }
        )
        self.action_space = spaces.Box(-1, 1, (7,), np.float32)
        self.metadata = {}

    def close(self):
        pass
