"""
MultiStepWrapper for executing multiple action steps per inference.

Simplified version that only handles action chunking (n_action_steps).
Observation history (n_obs_steps) is not supported by the server/client system.
"""

from gymnasium import spaces
import numpy as np
import dill


def repeated_box(box_space, n):
    """Create a repeated Box space for action chunking."""
    low = np.repeat(np.expand_dims(box_space.low, axis=0), n, axis=0)
    high = np.repeat(np.expand_dims(box_space.high, axis=0), n, axis=0)
    return spaces.Box(
        low=low,
        high=high,
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )


class MultiStepWrapper:
    """Wrapper for executing multiple action steps per inference.

    This wrapper:
    - Takes chunked actions of shape (n_action_steps, action_dim)
    - Executes them sequentially in the environment
    - Returns the final observation (no history stacking)

    This is a standalone wrapper (not inheriting from gymnasium.Wrapper)
    because the wrapped env (LiberoWrapper) is not a gymnasium.Env.
    """

    def __init__(self, env, n_action_steps, max_episode_steps=None):
        self.env = env
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps

        # Action space is repeated for chunking
        self.action_space = repeated_box(env.action_space, n_action_steps)
        # Observation space is unchanged (no history)
        self.observation_space = env.observation_space

        # Copy metadata from wrapped env if available
        self.metadata = getattr(env, 'metadata', {})

        self._step_count = 0
        self._rewards = []
        self._dones = []
        self._last_obs = None

    def reset(self):
        """Reset the environment."""
        obs = self.env.reset()
        self._step_count = 0
        self._rewards = []
        self._dones = []
        self._last_obs = obs
        return obs

    def step(self, action):
        """Execute chunked actions sequentially.

        Args:
            action: Array of shape (n_action_steps, action_dim)

        Returns:
            obs: Final observation after all actions
            reward: Max reward across all steps
            done: True if any step was done
            info: Info with aggregated success status
        """
        info = {}
        step_successes = []
        truncated = False

        for act in action:
            if self._dones and self._dones[-1]:
                # Already terminated, skip remaining actions
                break

            obs, reward, done, step_info = self.env.step(act)
            self._last_obs = obs
            self._step_count += 1
            self._rewards.append(reward)

            # Track success from underlying env (before we potentially override done)
            step_successes.append(step_info.get("success", False))

            # Check for truncation (timeout)
            if self.max_episode_steps and self._step_count >= self.max_episode_steps:
                truncated = True
                done = True

            self._dones.append(done)
            info = step_info

        # Aggregate results
        final_reward = max(self._rewards) if self._rewards else 0.0
        final_done = any(self._dones) if self._dones else False

        # Success is True if any step achieved success (not just timeout)
        info["success"] = any(step_successes)
        info["truncated"] = truncated

        return self._last_obs, final_reward, final_done, info

    def get_rewards(self):
        return self._rewards

    def reset_step_count(self):
        """Reset step count (call after wait period so wait doesn't count against budget)."""
        self._step_count = 0

    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)

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
