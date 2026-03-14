import numpy as np
from typing import Dict
from vlaworkspace.policy.base_policy import BasePolicy


class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def _adaptor_preprocess_obs(np_obs_dict, adaptor):
        """Preprocess batched env obs through adaptor (per-env loop).

        Args:
            np_obs_dict: {key: [n_envs, T, ...]} raw env observations.
            adaptor: Adaptor with env_input_transforms().

        Returns:
            Preprocessed obs dict {key: [n_envs, T, ...]}.
            Handles both flat dicts and nested dicts (e.g., images, image_masks
            from composable Pi0Model adaptor).
        """
        n_envs = next(iter(np_obs_dict.values())).shape[0]
        processed_list = []
        for i in range(n_envs):
            single_obs = {k: v[i] for k, v in np_obs_dict.items()}
            processed = adaptor.env_input_transforms(single_obs)
            processed_list.append(processed)

        # Re-batch: stack per-env results
        return BaseRunner._stack_processed(processed_list)

    @staticmethod
    def _stack_processed(processed_list):
        """Stack list of per-env processed dicts into a batched dict.

        Handles:
        - numpy arrays -> np.stack(..., axis=0)
        - nested dicts (e.g., images, image_masks) -> recursive stacking
        - numpy scalars -> np.array(values)
        - strings -> keep as list
        """
        sample = processed_list[0]
        result = {}
        for key in sample:
            val = sample[key]
            if isinstance(val, dict):
                # Nested dict (e.g., images, image_masks): recurse
                sub_dicts = [p[key] for p in processed_list]
                result[key] = BaseRunner._stack_processed(sub_dicts)
            elif isinstance(val, np.ndarray):
                result[key] = np.stack([p[key] for p in processed_list], axis=0)
            elif isinstance(val, str):
                result[key] = [p[key] for p in processed_list]
            else:
                result[key] = np.array([p[key] for p in processed_list])
        return result

    @staticmethod
    def _adaptor_postprocess_actions(actions, adaptor, raw_states=None):
        """Postprocess batched model actions through adaptor (per-env loop).

        Args:
            actions: [n_envs, n_action_steps, action_dim] model output.
            adaptor: Adaptor with output_transforms().
            raw_states: Optional [n_envs, state_dim] raw env states.
                If provided, passed to output_transforms for delta→absolute
                conversion.

        Returns:
            [n_envs, n_action_steps, env_action_dim] env actions.
        """
        n_envs = actions.shape[0]
        env_actions = []
        for i in range(n_envs):
            model_output = {"action": actions[i]}
            if raw_states is not None:
                model_output["state"] = raw_states[i]
            result = adaptor.output_transforms(model_output)
            env_actions.append(result["actions"])
        return np.stack(env_actions, axis=0)
