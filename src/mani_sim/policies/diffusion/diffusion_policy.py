"""Diffusion Policy — low-dim(상태 기반) 버전.

이미지 obs·vision encoder는 M3에서 추가한다 (docs/plan.md M3). 이 파일은 low_dim obs만
다룬다: obs_horizon(To)개 프레임의 obs를 그대로 이어붙여(flatten) global conditioning
벡터로 쓴다 (Chi et al. state-based 변형과 동일한 방식).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mani_sim.networks.conditional_unet1d import ConditionalUnet1d


def _flatten_obs(obs, obs_keys):
    """obs: {key: Tensor[B, To, D_k]} → Tensor[B, To * sum(D_k)] (obs_keys 순서로 concat 후 flatten)."""
    per_step = torch.cat([obs[k] for k in obs_keys], dim=-1)  # (B, To, sum(D_k))
    return per_step.flatten(start_dim=1)  # (B, To*sum(D_k))


class DiffusionPolicyLowDim(nn.Module):
    def __init__(
        self,
        obs_keys,
        obs_dims,
        obs_horizon,
        action_dim,
        pred_horizon,
        down_dims=(256, 512),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=256,
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        num_inference_steps=10,
    ):
        super().__init__()
        self.obs_keys = list(obs_keys)
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.num_inference_steps = num_inference_steps

        global_cond_dim = obs_horizon * sum(obs_dims[k] for k in self.obs_keys)
        self.unet = ConditionalUnet1d(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
        )

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=True,
            prediction_type="epsilon",
        )

    def get_global_cond(self, obs):
        """obs conditioning 계산 — APODiffusionPolicy 등 외부 래퍼가 LowDim/Image를
        구분 없이 쓸 수 있게 하는 공통 인터페이스(DiffusionPolicyImage._encode와 대응)."""
        return _flatten_obs(obs, self.obs_keys)

    def compute_loss(self, batch):
        """batch: {'obs': {key: (B,To,Dk)}, 'action': (B,Tp,Da), 'action_mask': (B,Tp)} (전부 정규화된 값)."""
        obs = batch["obs"]
        action = batch["action"]
        action_mask = batch["action_mask"]

        global_cond = self.get_global_cond(obs)

        batch_size = action.shape[0]
        device = action.device
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps, (batch_size,), device=device
        ).long()
        noisy_action = self.train_scheduler.add_noise(action, noise, timesteps)

        pred_noise = self.unet(noisy_action, timesteps, global_cond)

        loss = F.mse_loss(pred_noise, noise, reduction="none")  # (B, Tp, Da)
        mask = action_mask.unsqueeze(-1).float()  # (B, Tp, 1)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0) / self.action_dim

        return loss

    @torch.no_grad()
    def predict_action_chunk(self, obs):
        """obs: {key: (B,To,Dk)} (정규화된 값) → 정규화된 action chunk (B,Tp,Da). DDIM 샘플링."""
        global_cond = _flatten_obs(obs, self.obs_keys)
        batch_size = global_cond.shape[0]
        device = global_cond.device

        self.inference_scheduler.set_timesteps(self.num_inference_steps, device=device)
        sample = torch.randn((batch_size, self.pred_horizon, self.action_dim), device=device)

        for t in self.inference_scheduler.timesteps:
            noise_pred = self.unet(sample, t, global_cond)
            sample = self.inference_scheduler.step(noise_pred, t, sample).prev_sample

        return sample
