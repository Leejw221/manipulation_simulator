"""가중치 축 — class_based (SIRIUS 원문 이식).

원문 대조: UT-Austin-RPL/sirius robomimic/utils/dataset.py
WeightedDataset._sirius_reweight()(L967-979). [검증-원문, 2026-07-16]

4-class(demo/rollout/intervention/pre-intv) 각각에 **dataset-level 고정** 가중치를 부여한다
— 샘플·시점과 무관하게 클래스 안에서는 항상 동일 가중치(action_error 축과 다른 점).

공식(원문 그대로): P*(intv)=target_intv(기본 0.5), P*(preintv)=target_preintv(기본 0.002),
P*(demo)=P(demo)(=w_demos=1, 비가중), P*(rollout)=나머지. w_c = P*(c)/P(c).

정책(policy) 종류와 무관 — action_mode 라벨만 있으면 BC/Diffusion/OpenVLA 어디든 동일하게 적용.

이 클래스는 hdf5/zarr 등 저장 포맷을 전혀 모른다 - 전체 데이터셋의 action_mode 배열을
그대로 받는다(2026-07-27 밤 - 원래 hdf5_path를 직접 열었으나, Zarr task에서도 SIRIUS 조건이
돌아가게 하려고 파일 I/O를 utils.task_utils.read_all_action_modes로 분리).
"""

import numpy as np
import torch

from mani_sim.datasets.labels import LABEL_DEMO, LABEL_INTV, LABEL_PREINTV, LABEL_ROLLOUT


class ClassBasedWeight:
    def __init__(
        self,
        modes,
        target_intv=0.5,
        target_preintv=0.002,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        modes = np.asarray(modes)
        ratio_demos = float(np.mean(modes == LABEL_DEMO))
        ratio_intv = float(np.mean(modes == LABEL_INTV))
        ratio_rollouts = float(np.mean(modes == LABEL_ROLLOUT))
        ratio_preintv = float(np.mean(modes == LABEL_PREINTV))

        self.w_demos = 1.0
        self.w_intvs = target_intv / ratio_intv if ratio_intv > 0 else 0.0
        self.w_rollouts = (
            (1 - target_intv - ratio_demos - target_preintv) / ratio_rollouts
            if ratio_rollouts > 0
            else 0.0
        )
        self.w_pre_intvs = target_preintv / ratio_preintv if ratio_preintv > 0 else 0.0

        self.weight_by_label = {
            LABEL_DEMO: self.w_demos,
            LABEL_ROLLOUT: self.w_rollouts,
            LABEL_INTV: self.w_intvs,
            LABEL_PREINTV: self.w_pre_intvs,
        }
        print(
            f"class_based weight: ratio(demo={ratio_demos:.3f} rollout={ratio_rollouts:.3f} "
            f"intv={ratio_intv:.3f} preintv={ratio_preintv:.3f}) -> "
            f"weight(demo={self.w_demos:.3f} rollout={self.w_rollouts:.3f} "
            f"intv={self.w_intvs:.3f} preintv={self.w_pre_intvs:.3f})",
            flush=True,
        )

    def compute_weights(self, action_mode, **_unused):
        """action_mode: (B, T) 라벨 텐서 -> 같은 shape의 가중치 텐서.
        **_unused: action_error 축과 인터페이스를 맞추기 위한 자리(error 등 무시)."""
        weights = torch.zeros_like(action_mode, dtype=torch.float32)
        for label, w in self.weight_by_label.items():
            mask = torch.isclose(
                action_mode.float(), torch.full_like(action_mode.float(), float(label))
            )
            weights = torch.where(mask, torch.full_like(weights, w), weights)
        return weights.to(self.device)
