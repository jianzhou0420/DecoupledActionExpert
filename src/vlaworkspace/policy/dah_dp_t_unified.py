"""
Unified Diffusion Transformer policy supporting 6 conditioning methods.

Thin wrapper around TransformerForDiffusionUnified. The only difference from
dah_dp_t.py is the cond_method and lora_rank constructor params, and the import.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vlaworkspace.policy.base_policy import BasePolicy
from vlaworkspace.model.DecoupledActionHead.diffusion.transformer_for_diffusion_unified import TransformerForDiffusionUnified
from vlaworkspace.model.DecoupledActionHead.diffusion.mask_generator import LowdimMaskGenerator
from vlaworkspace.model.DecoupledActionHead.vision import create_obs_encoder
from vlaworkspace.z_utils.pytorch_util import dict_apply


class DiffusionTransformerUnifiedPolicy(BasePolicy):
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 # task params
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 # image
                 crop_shape=(76, 76),
                 obs_encoder_group_norm=False,
                 # arch
                 n_layer=8,
                 n_cond_layers=0,
                 n_head=4,
                 n_emb=256,
                 p_drop_emb=0.0,
                 p_drop_attn=0.3,
                 causal_attn=True,
                 time_as_cond=True,
                 obs_as_cond=True,
                 pred_action_steps_only=False,
                 # conditioning method
                 cond_method='cross_attn',
                 lora_rank=16,
                 skip_normalization=False,  # Deprecated, unused
                 # parameters passed to step
                 **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # Apply crop_shape to RGB observations
        if crop_shape is not None:
            crop_h, crop_w = crop_shape
            for key in obs_config['rgb']:
                if key in obs_key_shapes:
                    c = obs_key_shapes[key][0]
                    obs_key_shapes[key] = [c, crop_h, crop_w]

        # Create observation encoder
        obs_encoder = create_obs_encoder(
            obs_key_shapes=obs_key_shapes,
            obs_config=obs_config,
            feature_dimension=64,
            obs_encoder_group_norm=obs_encoder_group_norm,
        )

        # Create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusionUnified(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
            cond_method=cond_method,
            lora_rank=lora_rank,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data, condition_mask,
                           cond=None, generator=None,
                           **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, cond)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict

        if 'obs' in obs_dict and isinstance(obs_dict['obs'], dict):
            raw_obs = obs_dict['obs']
        else:
            raw_obs = obs_dict

        nobs = raw_obs
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(
            cond_data, cond_mask, cond=cond, **self.kwargs)

        action_pred = nsample[..., :Da]

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def get_scheduler(self, optimizer, num_training_steps, last_epoch=-1, **kwargs):
        from vlaworkspace.model.DecoupledActionHead.common.lr_scheduler import get_scheduler
        return get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

    def get_optimizer(
        self,
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = batch['obs']
        nactions = batch['action']
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs,
                                   lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:, start:end]
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape, device=trajectory.device)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
