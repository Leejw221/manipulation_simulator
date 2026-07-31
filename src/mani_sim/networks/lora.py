"""LoRA(r=32, APO 원문 방식) — ConditionalUnet1d(Conv1d/Linear 기반)에 적용.

원문 대조: APO(GeWu-Lab/Action-Preference-Optimization)는 항상 LoRA로 파인튜닝한다
(peft.LoraConfig, r=32) — 우리 UNet은 diffusers 표준 UNet2DConditionModel이 아니라
Chi et al. Diffusion Policy의 Conv1d 기반 UNet이라 peft가 기본 지원하는 레이어
타입(Linear/Conv2d)과 안 맞는다. Conv1d/Linear 둘 다 수동으로 LoRA 래퍼를 씌운다.

EXP-10.md "6차 시도" 절 참고 — 5번의 시도(EMA reference, grad_clip, weight_decay 등)가
전부 실패한 뒤 "전체 파라미터를 자유롭게 학습 가능하게 둔 것" 자체가 근본 원인으로
지목되어 도입.

**Conv2d(2026-08-01 추가)**: policy.encoders(ResNet-18 비전 인코더)에 적용하기 위함.
round1에서 encoder_lr_scale로 인코더 전체를 풀어(full fine-tune) 열었더니, 이동량을
키울수록(scale 0.1->0.3->1.0) 오히려 성공률이 떨어지는 패턴이 관측됐다(base 36% > 이전
34%(약하게 열기) > 29%/28%(세게 열기), EXP-10.md 2026-08-01 절). UNet은 LoRA(저랭크
제약)로 열려 있어 망각에 안전한데 인코더는 무제약 전체 파인튜닝이었다는 게 차이 —
같은 "이동 거리"라도 자유도가 다르면 망각 위험이 다를 수 있다는 가설[산출, 미검증]을
검증하기 위해 인코더도 LoRA로 제약해서 재시도한다.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=32, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


class LoRAConv1d(nn.Module):
    def __init__(self, base: nn.Conv1d, rank=32, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_down = nn.Conv1d(
            base.in_channels, rank, kernel_size=base.kernel_size, stride=base.stride,
            padding=base.padding, bias=False, device=device, dtype=dtype,
        )
        self.lora_up = nn.Conv1d(rank, base.out_channels, kernel_size=1, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank=32, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_down = nn.Conv2d(
            base.in_channels, rank, kernel_size=base.kernel_size, stride=base.stride,
            padding=base.padding, bias=False, device=device, dtype=dtype,
        )
        self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale


def apply_lora(module, rank=32, alpha=32):
    """module 안의 모든 nn.Linear/nn.Conv1d/nn.Conv2d를 LoRA 래퍼로 in-place 교체(재귀).
    base 가중치는 래퍼 생성 시점에 자동으로 freeze됨 — LoRA 파라미터만 requires_grad=True로
    남는다."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
        elif isinstance(child, nn.Conv1d):
            setattr(module, name, LoRAConv1d(child, rank, alpha))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank, alpha))
        else:
            apply_lora(child, rank, alpha)
    return module


@torch.no_grad()
def merge_lora(module):
    """LoRALinear/LoRAConv1d를 학습된 LoRA 기여분이 합쳐진 일반 nn.Linear/nn.Conv1d로
    되돌린다(재귀, in-place) — eval.py 등 LoRA를 모르는 코드가 체크포인트를 그대로
    불러올 수 있게 하기 위한 표준 LoRA 배포 방식(merge-then-save)."""
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            merged = nn.Linear(
                child.base.in_features, child.base.out_features,
                bias=child.base.bias is not None,
                device=child.base.weight.device, dtype=child.base.weight.dtype,
            )
            merged.weight.copy_(child.base.weight + (child.lora_B @ child.lora_A) * child.scale)
            if child.base.bias is not None:
                merged.bias.copy_(child.base.bias)
            setattr(module, name, merged)
        elif isinstance(child, LoRAConv1d):
            merged = nn.Conv1d(
                child.base.in_channels, child.base.out_channels, kernel_size=child.base.kernel_size,
                stride=child.base.stride, padding=child.base.padding, bias=child.base.bias is not None,
                device=child.base.weight.device, dtype=child.base.weight.dtype,
            )
            # lora_up은 kernel_size=1(채널만 섞는 pointwise)이라, lora_down(k)과의 합성이
            # 커널 하나로 접힌다: combined[o,i,dt] = sum_r lora_up[o,r,0] * lora_down[r,i,dt].
            up = child.lora_up.weight.squeeze(-1)  # (out, rank)
            delta = torch.einsum("or,rik->oik", up, child.lora_down.weight) * child.scale
            merged.weight.copy_(child.base.weight + delta)
            if child.base.bias is not None:
                merged.bias.copy_(child.base.bias)
            setattr(module, name, merged)
        elif isinstance(child, LoRAConv2d):
            merged = nn.Conv2d(
                child.base.in_channels, child.base.out_channels, kernel_size=child.base.kernel_size,
                stride=child.base.stride, padding=child.base.padding, bias=child.base.bias is not None,
                device=child.base.weight.device, dtype=child.base.weight.dtype,
            )
            # Conv1d 케이스와 동일한 논리(모듈 docstring 참고) — lora_up이 1x1 pointwise라
            # lora_down(kh,kw)과의 합성이 커널 하나로 접힌다.
            up = child.lora_up.weight.squeeze(-1).squeeze(-1)  # (out, rank)
            delta = torch.einsum("or,rikl->oikl", up, child.lora_down.weight) * child.scale
            merged.weight.copy_(child.base.weight + delta)
            if child.base.bias is not None:
                merged.bias.copy_(child.base.bias)
            setattr(module, name, merged)
        else:
            merge_lora(child)
    return module
