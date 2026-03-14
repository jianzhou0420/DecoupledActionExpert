"""
LIBERO Environment Wrapper for multiprocess evaluation.

Wraps LIBERO's OffScreenRenderEnv to match the diffusion policy interface,
supporting initialization via dill functions for AsyncVectorEnv compatibility.

Note: This is a standalone wrapper that doesn't inherit from gymnasium.Wrapper
because LIBERO's OffScreenRenderEnv is not a gymnasium.Env.
"""

import math

from gymnasium import spaces
import numpy as np
import dill

# Monkey-patch robosuite's buggy MjRenderContext.__del__ to prevent AttributeError spam
# during multiprocess garbage collection. The original __del__ assumes self.con exists.
def _patch_robosuite_render_context():
    try:
        from robosuite.utils import binding_utils
        def safe_del(self):
            if hasattr(self, 'con') and self.con is not None:
                try:
                    self.con.free()
                except Exception:
                    pass
        binding_utils.MjRenderContext.__del__ = safe_del
    except Exception:
        pass

_patch_robosuite_render_context()


class LiberoWrapper:
    """Wrapper for LIBERO env to match diffusion policy interface.

    This is a standalone wrapper (not inheriting from gymnasium.Wrapper)
    because LIBERO's OffScreenRenderEnv is not a gymnasium.Env.
    """

    def __init__(self, env, render_hw=(256, 256)):
        self.env = env
        self.render_hw = render_hw
        self.init_state = None  # Set via run_dill_function
        self._rewards = []
        self._successes = []

        # Define observation space (images + state)
        self.observation_space = spaces.Dict({
            'agentview_image': spaces.Box(0, 255, (*render_hw, 3), np.uint8),
            'wrist_image': spaces.Box(0, 255, (*render_hw, 3), np.uint8),
            'state': spaces.Box(-np.inf, np.inf, (8,), np.float32),
        })
        self.action_space = spaces.Box(-1, 1, (7,), np.float32)

        # Copy metadata from wrapped env if available
        self.metadata = getattr(env, 'metadata', {})

    def reset(self):
        self.env.reset()
        if self.init_state is not None:
            obs = self.env.set_init_state(self.init_state)
        else:
            obs = self.env._get_observations()
        self._rewards = []
        self._successes = []
        return self._process_obs(obs)

    def step(self, action):
        # Convert numpy array to list for LIBERO
        if isinstance(action, np.ndarray):
            action = action.tolist()
        obs, reward, done, info = self.env.step(action)
        self._rewards.append(reward)
        self._successes.append(float(done))  # LIBERO: done=True means success
        # Add success to info for proper tracking
        info["success"] = done  # LIBERO returns done=True only on task success
        return self._process_obs(obs), reward, done, info

    def _process_obs(self, obs):
        """Convert LIBERO obs format to dict format."""
        return {
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            'agentview_image': np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]),
            'wrist_image': np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]),
            'state': np.concatenate([
                obs['robot0_eef_pos'],
                self._quat2axisangle(obs['robot0_eef_quat']),
                obs['robot0_gripper_qpos'],
            ]).astype(np.float32),
        }

    def _quat2axisangle(self, quat):
        """
        Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
        """
        # clip quaternion
        quat = quat.copy()
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            # This is (close to) a zero degree rotation, immediately return
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)

    @property
    def reward(self):
        return self._rewards

    @property
    def success(self):
        return max(self._successes) if self._successes else 0.0

    def close(self):
        """Close the wrapped environment."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def seed(self, seed=None):
        """Set the random seed for the wrapped environment."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def render(self, *args, **kwargs):
        """Render the wrapped environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(*args, **kwargs)
