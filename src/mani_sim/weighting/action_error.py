"""가중치 축 — action_error (APO 원문의 adaptive reweight 이식).

원문 대조: GeWu-Lab/Action-Preference-Optimization trainer/ddp_robotic_trainer.py
get_batch_metrics() L324-339. [검증-원문, 2026-07-16]

원문 공식: 그룹(desirable/undesirable)별로 오차를 정규화한 뒤
  desirable(chosen):   λ = (1 − e^(−βD·w))^γ   (오차 클수록 weight↑ — "잘 못 따라했으니 더 배워라")
  undesirable(rejected): λ = (e^(−βU·w))^γ      (오차 작을수록 weight↑ — "실패행동 따라하려 하니 떼어내라")
  w = (error + eps) / (Σgroup error + eps)

**정책(policy) 종류와 무관** — "오차"의 정의만 정책이 넘겨준다: BC(GMM)는 점추정 L1,
Diffusion은 노이즈예측 MSE, OpenVLA는 토큰 cross-entropy 등. 이 클래스는 이미 계산된
error(B,) 텐서만 받는다.

desirable/undesirable 판정은 `datasets/labels.desirable_mask`(앞 8프레임 다수결, apo_loss.py와
공유하는 단일 기준)로 통일한다.

반환은 (B, T)로 브로드캐스트한다 — class_based(원래 (B,T), 프레임마다 다른 클래스 가중)와
인터페이스를 맞춰 두 가중치 축을 어느 loss 구조에나 갈아 끼울 수 있게 한다(원문은 샘플당
가중치 1개뿐이라 T축에 상수로 반복되는 형태 = 원문과 동일 의미, 표현만 맞춤).
"""

import torch

from mani_sim.datasets.labels import desirable_mask as _desirable_mask


class ActionErrorWeight:
    def __init__(self, beta_d=1.0, beta_u=1.0, gamma=1.0, eps=1e-4, preference_frames=8):
        self.beta_d = beta_d
        self.beta_u = beta_u
        self.gamma = gamma
        self.eps = eps
        self.preference_frames = preference_frames

    def compute_weights(self, action_mode, error, **_unused):
        """action_mode: (B, T). error: (B,) — 정책이 계산한 per-sample 오차.
        **_unused: class_based과 인터페이스를 맞추기 위한 자리."""
        mask = _desirable_mask(action_mode, preference_frames=self.preference_frames)
        weights = torch.zeros_like(error)

        d_error = error[mask]
        w_d = (d_error + self.eps) / (d_error.sum() + self.eps)
        weights[mask] = (1 - torch.exp(-self.beta_d * w_d)) ** self.gamma

        u_error = error[~mask]
        w_u = (u_error + self.eps) / (u_error.sum() + self.eps)
        weights[~mask] = torch.exp(-self.beta_u * w_u) ** self.gamma

        T = action_mode.shape[1]
        return weights.unsqueeze(1).expand(-1, T)
