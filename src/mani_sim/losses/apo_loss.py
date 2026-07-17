"""시스템 축 — apo (reference 모델 + KTO 앵커 필요).

원문 대조: GeWu-Lab/Action-Preference-Optimization trainer/ddp_robotic_trainer.py
APOTrainer.loss()/get_batch_metrics() (L173-364). [검증-원문, 2026-07-16]

SIRIUS와의 결정적 차이: **reference 모델(정책의 이전 상태, 고정)이 필요**하다 — reward를
"GT action을 policy가 reference보다 얼마나 더 그럴듯하다고 보는가"(log-ratio)로 정의하고,
z0(=배치 내 무관 pair로 추정한 KL 앵커)를 기준점 삼아 KTO의 sigmoid 손실을 쓴다.

**가중치는 이 클래스가 계산하지 않는다** — 외부(weighting/class_based.py 또는
weighting/action_error.py)에서 계산한 (B,T) weight를 받아 곱하기만 한다. 원문 APO는
adaptive weight를 자체 내장하지만(=action_error 축), 우리는 이걸 분리해서 class_based
가중치를 KTO 구조에 꽂는 조합도 실험할 수 있게 했다(2026-07-16(4) 사용자 지시 — 세 축
독립 조합).

preference(chosen/rejected) 판정은 `datasets/labels.desirable_mask`로 weighting/action_error.py
와 동일 기준 공유.
"""

from collections import OrderedDict

import torch

from mani_sim.datasets.labels import desirable_mask as _desirable_mask


class APOKTOLoss:
    def __init__(
        self,
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        kl_min=-5.0,
        kl_max=5.0,
        preference_frames=8,
    ):
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.kl_min = kl_min
        self.kl_max = kl_max
        self.preference_frames = preference_frames

    def compute(
        self,
        log_probs,
        ref_log_probs,
        mismatch_log_probs,
        ref_mismatch_log_probs,
        weight,
        action_mode_window,
    ):
        """log_probs 계열: (B, T). weight: (B, T) — 외부(weighting/*.py)에서 계산해 넘김.
        action_mode_window: (B, W), W>=preference_frames.

        반환: (scalar loss, dict 진단 지표).
        """
        reward = log_probs.sum(dim=1) - ref_log_probs.sum(dim=1)  # (B,)
        kl_reward = mismatch_log_probs.sum(dim=1) - ref_mismatch_log_probs.sum(dim=1)  # (B,)
        z0 = kl_reward.detach().mean().clamp(self.kl_min, self.kl_max)

        mask = _desirable_mask(action_mode_window, preference_frames=self.preference_frames)
        sample_weight = weight.mean(dim=1)  # (B,T) -> (B,) — class_based처럼 T별로 다르면 평균

        chosen_reward = reward[mask]
        chosen_losses = self.desirable_weight * (1 - torch.sigmoid(self.beta * (chosen_reward - z0)))
        chosen_weight = sample_weight[mask]

        rejected_reward = reward[~mask]
        rejected_losses = self.undesirable_weight * (1 - torch.sigmoid(self.beta * (z0 - rejected_reward)))
        rejected_weight = sample_weight[~mask]

        losses = torch.cat([chosen_losses, rejected_losses])
        weights = torch.cat([chosen_weight, rejected_weight])
        total_loss = (losses * weights).sum()

        n_chosen = int(mask.sum().item())
        n_rejected = int((~mask).sum().item())
        metrics = OrderedDict(
            num_chosen=n_chosen,
            num_rejected=n_rejected,
            reward_chosen_mean=float(chosen_reward.detach().mean().item()) if n_chosen else float("nan"),
            reward_rejected_mean=float(rejected_reward.detach().mean().item()) if n_rejected else float("nan"),
            z0=float(z0.item()),
        )
        return total_loss, metrics
