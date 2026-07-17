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

from mani_sim.datasets.labels import LABEL_DEMO, LABEL_INTV, LABEL_PREINTV, LABEL_ROLLOUT


def build_balanced_sampler(
    action_mode_first_frame, target_correct=0.5, target_interaction=0.25, target_incorrect=0.25
):
    """action_mode_first_frame: (N,) 각 샘플(윈도우)의 대표 라벨(SIRIUS 관례와 동일하게 index 0).

    반환: WeightedRandomSampler(replacement=True, num_samples=N) — 그룹별 개수와 무관하게
    기댓값 target_* 비율로 뽑힌다(그룹 내부는 균등, RA-BC/SIRIUS와 달리 순수 sampling 축).
    """
    labels = np.asarray(action_mode_first_frame)
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
