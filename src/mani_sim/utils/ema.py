"""정책 가중치의 지수이동평균(EMA) — Diffusion Policy 공식 학습 레시피의 일부.

원문(real-stanford/diffusion_policy)은 `use_ema: True`가 기본이고, **rollout/평가를 EMA
가중치로 한다**(`train_diffusion_unet_hybrid_workspace.py`: `policy = self.ema_model`).
우리 코드는 lerobot 컨벤션을 따라 만들었는데 lerobot의 diffusion policy에는 EMA 필드
자체가 없어서 이 항목이 누락돼 있었다(2026-08-10 발견, EXP-10 (111)절).

감쇠 스케줄은 원문 `EMAModel`과 동일하다:

    decay = 1 - (1 + step / inv_gamma) ** (-power),  [min_value, max_value]로 clamp

기본값도 원문 config 그대로: inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999.
초반에는 decay가 작아 새 가중치를 빠르게 따라가고, 학습이 진행될수록 0.9999에 수렴해
후반 궤적을 길게 평균한다.
"""

import copy

import torch


class EMAModel:
    def __init__(self, model, update_after_step=0, inv_gamma=1.0, power=0.75,
                 min_value=0.0, max_value=0.9999):
        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.optimization_step = 0
        self.decay = 0.0

    def get_decay(self, step):
        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** (-self.power)
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, model):
        self.decay = self.get_decay(self.optimization_step)
        src, dst = model.state_dict(), self.averaged_model.state_dict()
        for k, v in src.items():
            if dst[k].dtype.is_floating_point:
                dst[k].mul_(self.decay).add_(v.detach().to(dst[k].dtype), alpha=1.0 - self.decay)
            else:
                dst[k].copy_(v)  # 정수 버퍼는 평균이 무의미하므로 그대로 복사
        self.optimization_step += 1

    def state_dict(self):
        return {"averaged_model": self.averaged_model.state_dict(),
                "optimization_step": self.optimization_step}

    def load_state_dict(self, sd):
        self.averaged_model.load_state_dict(sd["averaged_model"])
        self.optimization_step = sd["optimization_step"]
