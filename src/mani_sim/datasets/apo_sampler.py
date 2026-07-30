"""APO의 50/25/25(Correct/Interaction/Incorrect) balanced sampling — sampler.py 이식.

원문: GeWu-Lab/Action-Preference-Optimization dataset/sampler.py
`BalancedInteractionDistributedSampler`(에폭을 "interaction 소진 시점"으로 정의하는 커스텀
배치 구성)를 그대로 옮기는 대신, **기댓값으로 같은 비율**을 만드는
`torch.utils.data.WeightedRandomSampler`로 단순화했다(단일 프로세스·robomimic
DataLoader와 자연 결합 — robomimic SequenceDataset.get_dataset_sampler() 훅이
Sampler 객체 하나를 기대하는 구조와 맞물린다). 종료조건(interaction 소진)의 엄밀한 재현은
포기 — [적응, 원문 배치구성 아님].
"""

import numpy as np
from torch.utils.data import WeightedRandomSampler

from mani_sim.datasets.labels import (
    DESIRABLE_LABELS,
    LABEL_DEMO,
    LABEL_INTV,
    LABEL_PREINTV,
    LABEL_ROLLOUT,
)


def build_balanced_sampler(
    action_mode_first_frame, target_correct=0.5, target_interaction=0.25, target_incorrect=0.25,
    action_mode_window=None, preference_frames=8,
):
    """action_mode_first_frame: (N,) 각 샘플(윈도우)의 대표 라벨(SIRIUS 관례와 동일하게 index 0).

    action_mode_window: (N, W) 또는 None. 주어지면 **incorrect(undesirable) 판정을 loss와
    동일한 기준**(`datasets/labels.desirable_mask`: 앞 preference_frames 프레임의 다수결)으로
    한다. None이면 기존처럼 첫 프레임 라벨만 본다.

    두 기준이 어긋나면 샘플러가 목표한 배치 구성이 loss에서 그대로 재현되지 않는다 —
    실측(EXP-10.md 2026-07-30~31, square_merged 전수조사): preintv_length(15) >
    preference_frames(8)라서 PREINTV 구간 뒤쪽 4/15 지점에서 시작한 윈도우는 앞 8프레임 중
    과반이 이미 INTV라 loss에선 desirable로 뒤집힌다. PREINTV로 뽑힌 405개 중 108개
    (정확히 26.67%=4/15)가 뒤집혀 목표 25%가 실제로는 18.3%만 undesirable로 반영됐다.

    반환: WeightedRandomSampler(replacement=True, num_samples=N) — 그룹별 개수와 무관하게
    기댓값 target_* 비율로 뽑힌다(그룹 내부는 균등, RA-BC/SIRIUS와 달리 순수 sampling 축).
    """
    labels = np.asarray(action_mode_first_frame)
    if action_mode_window is not None:
        window = np.asarray(action_mode_window)[:, :preference_frames]
        desirable_frac = np.isin(window, DESIRABLE_LABELS).mean(axis=1)
        is_incorrect = desirable_frac < 0.5  # desirable_mask(>=0.5)의 여집합
        # 나머지(=loss가 desirable로 보는 것)를 correct/interaction으로 나눈다.
        is_interaction = (~is_incorrect) & (labels == LABEL_INTV)
        is_correct = (~is_incorrect) & ~is_interaction
    else:
        is_correct = (labels == LABEL_DEMO) | (labels == LABEL_ROLLOUT)
        is_interaction = labels == LABEL_INTV
        is_incorrect = labels == LABEL_PREINTV

    weights = np.zeros(len(labels), dtype=np.float64)
    n_correct, n_interaction, n_incorrect = is_correct.sum(), is_interaction.sum(), is_incorrect.sum()
    if n_correct > 0:
        weights[is_correct] = target_correct / n_correct
    if n_interaction > 0:
        weights[is_interaction] = target_interaction / n_interaction
    if n_incorrect > 0:
        weights[is_incorrect] = target_incorrect / n_incorrect

    if weights.sum() == 0:
        raise ValueError("balanced sampler: 라벨이 전부 비어있음(action_mode 확인 필요)")

    return WeightedRandomSampler(weights.tolist(), num_samples=len(labels), replacement=True)
