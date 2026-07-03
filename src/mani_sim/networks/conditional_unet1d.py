"""1D conditional U-Net (FiLM conditioning) — Diffusion Policy(Chi et al., 2023)의 표준 아키텍처.

이 구조 자체는 Chi et al. 공개 구현·LeRobot 포팅에서 널리 쓰이는 공개된(표준) 아키텍처이며,
manipulation_pipeline(flare)의 `networks/diffusion/conditional_unet1d.py`도 같은 구조를 그대로
차용하고 있다. docs/plan.md 원칙(§2, §5)에 따라 flare/lerobot 코드는 import하지 않고 이 구조를
참고해 자체 구현한다.
"""

import math

import einops
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1d(nn.Module):
    """FiLM(scale, bias)로 조건(diffusion timestep + global obs cond)을 주입하는 residual conv block."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=5, n_groups=8, use_film_scale=True):
        super().__init__()
        self.use_film_scale = use_film_scale
        self.out_channels = out_channels

        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        cond_channels = out_channels * 2 if use_film_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale:
            scale, bias = cond_embed.chunk(2, dim=1)
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class ConditionalUnet1d(nn.Module):
    """action 시퀀스(B, T, action_dim)에 노이즈를 예측하는 1D U-Net.

    Args:
        input_dim (int): action_dim.
        global_cond_dim (int): diffusion timestep 임베딩을 제외한, obs 조건 벡터 차원.
        down_dims (list[int]): 인코더 각 단계의 채널 수.
    """

    def __init__(
        self,
        input_dim,
        global_cond_dim,
        down_dims=(256, 512),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=256,
        use_film_scale=True,
    ):
        super().__init__()

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        down_dims = list(down_dims)
        in_out = [(input_dim, down_dims[0])] + list(zip(down_dims[:-1], down_dims[1:]))
        common = dict(cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, use_film_scale=use_film_scale)

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, **common),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(mid_dim, mid_dim, **common),
                ConditionalResidualBlock1d(mid_dim, mid_dim, **common),
            ]
        )

        self.up_modules = nn.ModuleList()
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in * 2, dim_out, **common),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond=None):
        """
        Args:
            sample (Tensor[B, T, input_dim]): 노이즈가 섞인 action 시퀀스.
            timestep (Tensor[B] | Tensor[] | int): diffusion timestep.
            global_cond (Tensor[B, global_cond_dim] | None): obs 조건 벡터.
        """
        x = einops.rearrange(sample, "b t d -> b d t")

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=x.device, dtype=torch.long)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] != x.shape[0]:
            timestep = timestep.expand(x.shape[0])

        timestep_embed = self.diffusion_step_encoder(timestep.float())
        global_feature = torch.cat([timestep_embed, global_cond], dim=-1) if global_cond is not None else timestep_embed

        skips = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            skips.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for res1, res2, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b d t -> b t d")
