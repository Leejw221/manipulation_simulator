"""시스템 축 — apo (reference 모델 + KTO 앵커 필요).

원문 대조: GeWu-Lab/Action-Preference-Optimization trainer/ddp_robotic_trainer.py
APOTrainer.loss()/get_batch_metrics() (L173-364) [검증-원문, 2026-07-16]. z0(anchor) 추정은
mismatch pair 대신 batch reward 평균을 쓴다(우리 reward가 노이즈예측 MSE 차이라 가능,
Diffusion-KTO와 동일 방식). clamp 범위는 APO 원문 방식(대칭, 음수 허용) — 상세 근거는
EXP-10.md "z0 clamp" 절 참고.

SIRIUS와의 결정적 차이: **reference 모델(정책의 이전 상태, 고정)이 필요**하다 — reward를
"GT action을 policy가 reference보다 얼마나 더 그럴듯하다고 보는가"(log-ratio)로 정의하고,
z0를 기준점 삼아 KTO의 sigmoid 손실을 쓴다.

**가중치는 이 클래스가 계산하지 않는다** — 외부(weighting/class_based.py 또는
weighting/action_error.py)에서 계산한 (B,T) weight를 받아 곱하기만 한다. 원문 APO는
adaptive weight를 자체 내장하지만(=action_error 축), 우리는 이걸 분리해서 class_based
가중치를 KTO 구조에 꽂는 조합도 실험할 수 있게 했다(2026-07-16(4) 사용자 지시 — 세 축
독립 조합).

preference(chosen/rejected) 판정은 `datasets/labels.desirable_mask`로 weighting/action_error.py
와 동일 기준 공유.

**beta는 확정 전 기본값**(Diffusion-KTO 공식값 그대로) — APO(0.1)는 토큰 log확률 합
스케일용이라 우리(MSE 차이 스케일)엔 안 맞는다. 배포 데이터로 실측 후
`|beta*(reward-z0)|`의 중앙값이 0.5~2에 오도록 재조정할 것.
"""

from collections import OrderedDict

import torch

from mani_sim.datasets.labels import desirable_mask as _desirable_mask


class APOKTOLoss:
    def __init__(
        self,
        beta=1000.0,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        preference_frames=8,
        z0_clamp_min=-50.0,
        z0_clamp_max=50.0,
    ):
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.preference_frames = preference_frames
        self.z0_clamp_min = z0_clamp_min
        self.z0_clamp_max = z0_clamp_max

    def compute(self, log_probs, ref_log_probs, weight, action_mode_window, mismatch_reward=None):
        """log_probs, ref_log_probs, weight: (B, T) — 패딩 프레임은 호출부가 미리 0으로
        마스킹해서 넘긴다(합산에서 자동 제외). action_mode_window: (B, W), W>=preference_frames.
        mismatch_reward: (B,) 또는 None — KTO/APO 원문 방식의 mismatched-pair reward(호출부가
        계산해서 넘김). 주어지면 z0를 여기서 추정(원문 그대로), None이면 구버전 호환으로
        matched reward의 배치 평균을 씀(EXP-10.md "z0 mismatched pair" 절 — 이게 버그였음).

        반환: (scalar loss, dict 진단 지표).
        """
        reward = log_probs.sum(dim=1) - ref_log_probs.sum(dim=1)  # (B,)
        z0_source = mismatch_reward if mismatch_reward is not None else reward
        # 대칭 clamp(APO 원문 방식) — 근거: EXP-10.md "z0 clamp" 절.
        z0 = z0_source.detach().mean().clamp(min=self.z0_clamp_min, max=self.z0_clamp_max)

        mask = _desirable_mask(action_mode_window, preference_frames=self.preference_frames)
        sample_weight = weight.mean(dim=1)  # (B,T) -> (B,) — class_based처럼 T별로 다르면 평균

        n_chosen = int(mask.sum().item())
        n_rejected = int((~mask).sum().item())

        chosen_reward = reward[mask]
        chosen_losses = self.desirable_weight * (1 - torch.sigmoid(self.beta * (chosen_reward - z0)))
        chosen_weight_raw = sample_weight[mask]
        chosen_weight_sum = float(chosen_weight_raw.sum().item()) if n_chosen else 0.0
        # 그룹 내 평균을 1.0으로 재정규화 — action_error weight의 그룹합이 exp(-x) 곡률 때문에
        # 그룹 크기에 비대칭적으로 반응해(desirable은 beta_d 근처로 포화, undesirable은 n에
        # 비례) 의도치 않게 그룹 간 loss 기여도를 왜곡시킨다(EXP-10.md "7차 시도" 절 참고).
        # 재정규화 후엔 그룹별 총 기여도가 desirable_weight/undesirable_weight·n_chosen/
        # n_rejected로만 결정된다.
        chosen_weight = chosen_weight_raw / max(chosen_weight_sum, 1e-8) * n_chosen if n_chosen else chosen_weight_raw

        rejected_reward = reward[~mask]
        rejected_losses = self.undesirable_weight * (1 - torch.sigmoid(self.beta * (z0 - rejected_reward)))
        rejected_weight_raw = sample_weight[~mask]
        rejected_weight_sum = float(rejected_weight_raw.sum().item()) if n_rejected else 0.0
        rejected_weight = (
            rejected_weight_raw / max(rejected_weight_sum, 1e-8) * n_rejected if n_rejected else rejected_weight_raw
        )

        losses = torch.cat([chosen_losses, rejected_losses])
        weights = torch.cat([chosen_weight, rejected_weight])
        total_loss = (losses * weights).sum()

        metrics = OrderedDict(
            num_chosen=n_chosen,
            num_rejected=n_rejected,
            reward_chosen_mean=float(chosen_reward.detach().mean().item()) if n_chosen else float("nan"),
            reward_rejected_mean=float(rejected_reward.detach().mean().item()) if n_rejected else float("nan"),
            z0=float(z0.item()),
            reward_min=float(reward.detach().min().item()),
            reward_max=float(reward.detach().max().item()),
            reward_std=float(reward.detach().std().item()) if reward.numel() > 1 else 0.0,
            raw_weight_sum_chosen=chosen_weight_sum,
            raw_weight_sum_rejected=rejected_weight_sum,
        )
        return total_loss, metrics
