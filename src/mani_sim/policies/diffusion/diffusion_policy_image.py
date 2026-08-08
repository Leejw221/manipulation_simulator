"""Diffusion Policy — 이미지(픽셀) obs 버전.

low-dim 버전과 U-Net·스케줄러는 동일하고, obs conditioning만 다르다:
  - rgb 키: 카메라별 vision encoder(ResNet-18)로 (C,H,W)→feature 벡터
  - lowdim 키(proprio·stage): 그대로
  두 종류를 obs_horizon 프레임마다 이어붙여(concat) global conditioning 벡터로 쓴다.

이미지는 SequenceDataset(ObsUtils rgb 모달리티)이 (C,H,W) float[0,1]로 넘겨준다고 가정한다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mani_sim.networks.conditional_unet1d import ConditionalUnet1d


def _replace_bn_with_gn(module, num_groups=16):
    """BatchNorm2d → GroupNorm (작은 배치·per-camera 인코더에서 BN 통계 불안정 회피, DP 관례)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups=min(num_groups, child.num_features),
                                               num_channels=child.num_features))
        else:
            _replace_bn_with_gn(child, num_groups)


class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax (Finn et al.). (B,C,H,W) → (B,K,2) 키포인트. manipulation_pipeline 참고."""

    def __init__(self, input_shape, num_kp=32):
        super().__init__()
        c, h, w = input_shape
        self.nets = nn.Conv2d(c, num_kp, kernel_size=1)
        self._out_c, self._h, self._w = num_kp, h, w
        pos_x, pos_y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        pos = np.concatenate([pos_x.reshape(-1, 1), pos_y.reshape(-1, 1)], axis=1)
        self.register_buffer("pos_grid", torch.from_numpy(pos).float())

    def forward(self, x):
        x = self.nets(x).reshape(-1, self._h * self._w)
        att = F.softmax(x, dim=-1)
        return (att @ self.pos_grid).view(-1, self._out_c, 2)


class VisionEncoder(nn.Module):
    """ResNet-18 backbone(spatial 유지) + SpatialSoftmax → 키포인트 feature. 입력 (B,3,H,W) float[0,1].

    crop_hw가 주어지면 학습 중엔 RandomCrop, eval 중엔 CenterCrop(DP 논문 Table7 CropRes, lerobot의
    train/eval crop 분기 방식을 따름 — self.training으로 분기, policy.eval()은 rollout.py/eval.py에서
    이미 호출됨)."""

    def __init__(self, num_kp=32, input_hw=(84, 84), crop_hw=None):
        super().__init__()
        self.crop_hw = tuple(crop_hw) if crop_hw is not None else None
        if self.crop_hw is not None:
            self.random_crop = torchvision.transforms.RandomCrop(self.crop_hw)
            self.center_crop = torchvision.transforms.CenterCrop(self.crop_hw)
        probe_hw = self.crop_hw if self.crop_hw is not None else input_hw
        resnet = torchvision.models.resnet18(weights=None)
        _replace_bn_with_gn(resnet)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # avgpool·fc 제거 → (B,512,h,w)
        with torch.no_grad():
            c, h, w = self.backbone(torch.zeros(1, 3, *probe_hw)).shape[1:]
        self.pool = SpatialSoftmax((c, h, w), num_kp=num_kp)
        self.out_dim = num_kp * 2

    def forward(self, x):
        if self.crop_hw is not None:
            x = self.random_crop(x) if self.training else self.center_crop(x)
        return self.pool(self.backbone(x)).flatten(1)  # (B, num_kp*2)


class DiffusionPolicyImage(nn.Module):
    def __init__(
        self,
        rgb_keys,
        lowdim_keys,
        obs_dims,
        obs_horizon,
        action_dim,
        pred_horizon,
        num_kp=32,
        image_hw=(84, 84),
        crop_hw=None,
        down_dims=(256, 512),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=256,
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        num_inference_steps=10,
        num_stages=None,
        stage_embed_dim=32,
    ):
        super().__init__()
        self.rgb_keys = list(rgb_keys)
        self.lowdim_keys = list(lowdim_keys)
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.num_inference_steps = num_inference_steps

        self.encoders = nn.ModuleDict(
            {k: VisionEncoder(num_kp=num_kp, input_hw=tuple(image_hw), crop_hw=crop_hw) for k in self.rgb_keys}
        )
        feat = next(iter(self.encoders.values())).out_dim if self.rgb_keys else 0

        # stage를 raw 정수로 concat하면 이미지(카메라당 64차원) 대비 무시될 만큼 작음
        # (2026-07-24 교수님 미팅 지적) — 학습되는 임베딩 공간으로 보낸 뒤 concat.
        # robocasa/research_design.md "Stage conditioning 주입 방법" 참고.
        self.num_stages = num_stages
        self.stage_embed = nn.Embedding(num_stages, stage_embed_dim) if num_stages else None
        stage_dim = stage_embed_dim if num_stages else 0

        per_step_dim = len(self.rgb_keys) * feat + sum(obs_dims[k] for k in self.lowdim_keys) + stage_dim
        global_cond_dim = obs_horizon * per_step_dim
        self.unet = ConditionalUnet1d(
            input_dim=action_dim, global_cond_dim=global_cond_dim, down_dims=down_dims,
            kernel_size=kernel_size, n_groups=n_groups, diffusion_step_embed_dim=diffusion_step_embed_dim,
        )

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule,
            clip_sample=True, prediction_type="epsilon",
        )
        # DP 논문: sim에서는 train/inference 모두 DDPM 100 step(가속 없음) — DDIM은
        # real-world 전용(16 step). 공식코드·lerobot 모두 inference도 DDPM 그대로 씀.
        self.inference_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule,
            clip_sample=True, prediction_type="epsilon",
        )

    def _encode(self, obs):
        """obs: rgb (B,To,C,H,W) / lowdim (B,To,D) / stage (B,To) int64 → global_cond (B, To*per_step)."""
        feats = []
        for k in self.rgb_keys:
            img = obs[k]                       # (B, To, C, H, W)
            b, to = img.shape[:2]
            f = self.encoders[k](img.reshape(b * to, *img.shape[2:]))  # (B*To, feat)
            feats.append(f.reshape(b, to, -1))
        for k in self.lowdim_keys:
            feats.append(obs[k])               # (B, To, D)
        if self.stage_embed is not None:
            # ZarrSequenceDataset은 rgb가 아닌 모든 obs_key를 float32로 캐스팅해서 돌려줌
            # (stage도 예외 없음) - nn.Embedding엔 정수 인덱스가 필요해 여기서 되돌림.
            feats.append(self.stage_embed(obs["stage"].long()))  # (B, To, stage_embed_dim)
        per_step = torch.cat(feats, dim=-1)    # (B, To, per_step)
        return per_step.flatten(start_dim=1)

    def get_global_cond(self, obs):
        """DiffusionPolicyLowDim.get_global_cond과 대응 — APODiffusionPolicy 등 외부
        래퍼가 LowDim/Image를 구분 없이 쓸 수 있게 하는 공통 인터페이스."""
        return self._encode(obs)

    def compute_loss(self, batch, reduction="mean", timesteps=None):
        """reduction="mean" → 스칼라. reduction="none" → 샘플별 스칼라 (B,)
        (BCRNNPolicyLowDim.compute_loss와 동일 관례 — SIRIUS류 외부 가중치 결합용).

        timesteps: (B,) 또는 None(기본, 표준 DDPM대로 샘플별 랜덤 t). 값을 주면 그 t를
        그대로 쓴다 — adaptive weighting이 샘플 간 오차를 비교할 때 t를 고정하기 위한 용도
        (weighting.ref_timestep, diffusion_trainer.py 참고)."""
        obs, action, action_mask = batch["obs"], batch["action"], batch["action_mask"]
        global_cond = self.get_global_cond(obs)
        noise = torch.randn_like(action)
        if timesteps is None:
            timesteps = torch.randint(
                0, self.train_scheduler.config.num_train_timesteps, (action.shape[0],), device=action.device
            ).long()
        noisy = self.train_scheduler.add_noise(action, noise, timesteps)
        pred = self.unet(noisy, timesteps, global_cond)
        loss = F.mse_loss(pred, noise, reduction="none")
        mask = action_mask.unsqueeze(-1).float()
        masked = loss * mask

        if reduction == "none":
            return masked.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0) / self.action_dim  # (B,)
        return masked.sum() / mask.sum().clamp(min=1.0) / self.action_dim

    @torch.no_grad()
    def predict_action_chunk(self, obs):
        global_cond = self._encode(obs)
        b = global_cond.shape[0]
        device = global_cond.device
        self.inference_scheduler.set_timesteps(self.num_inference_steps, device=device)
        sample = torch.randn((b, self.pred_horizon, self.action_dim), device=device)
        for t in self.inference_scheduler.timesteps:
            sample = self.inference_scheduler.step(self.unet(sample, t, global_cond), t, sample).prev_sample
        return sample
