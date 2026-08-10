"""APO의 50/25/25(Correct/Interaction/Incorrect) balanced sampling — sampler.py 이식.

원문: GeWu-Lab/Action-Preference-Optimization dataset/sampler.py
`BalancedInteractionDistributedSampler`(에폭을 "interaction 소진 시점"으로 정의하는 커스텀
배치 구성). 두 가지 구현을 제공한다:

1. `build_balanced_sampler`(기본, 2026-07 도입) — **기댓값으로 같은 비율**을 만드는
   `torch.utils.data.WeightedRandomSampler`로 단순화. "1 epoch = len(dataset)" 고정,
   종료조건(interaction 소진)의 엄밀한 재현은 포기 — [적응, 원문 배치구성 아님].
2. `build_interaction_exhaustion_sampler`(2026-08-05 추가) — 원문처럼 **1 epoch =
   interaction 인덱스 소진 시점**으로 정의(`InteractionExhaustionSampler`). round2가
   round1보다 intervention 수가 적은데도 "전체 데이터 기준"으로 학습량을 늘렸던 게
   맞는 비교였는지 의문이 제기돼(EXP-10.md 2026-08-04/05 절) 추가함 — [검증 대기,
   아직 실제 학습에 안 써봄].

둘 다 robomimic SequenceDataset.get_dataset_sampler() 훅(Sampler 객체 하나를 기대)과
자연 결합되는 단일 프로세스 구조.
"""

import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler

from mani_sim.datasets.labels import (
    DESIRABLE_LABELS,
    LABEL_DEMO,
    LABEL_INTV,
    LABEL_PREINTV,
    LABEL_ROLLOUT,
)


def _classify(action_mode_first_frame, action_mode_window=None, preference_frames=8):
    """build_balanced_sampler와 동일한 correct/interaction/incorrect 판정 로직(공유).

    APO 공식 코드(`dataset/balance_apo_dataset.py`, 2026-08-11 gh로 직접 대조)의 버킷 정의:
      correct     <- is_expert=True(전문가 데모)  +  is_human==2
      interaction <- is_human==1
      incorrect   <- is_human==0(개입 직전 K스텝, 소급 재라벨)

    **주의 — 논문 본문과 코드가 반대다.** 논문 3.1절은 "c_t = 2 represents the action is
    corrected by human intervention while c_t = 1 denotes the action is executed by policy"라고
    쓰지만, 코드 로직은 그 반대여야 자기정합적이다:
      `if is_first_human and is_human == 1:` -> 직전 K스텝을 incorrect로 옮김 (= 1이 개입 시작 신호)
      `if is_human == 2: is_first_human = True` -> 탐지기 재무장 (= 2가 정책 실행 복귀)
    즉 코드 기준 **is_human==1 = 사람 개입 · is_human==2 = 정책 실행**이고, 배치 구성은
    "correct(데모+로봇 액션) 50% / interaction(사람 개입) 25% / incorrect(개입 직전) 25%"가 된다.
    우리 라벨 대응: DEMO+ROLLOUT -> correct · INTV -> interaction · PREINTV -> incorrect.
    """
    labels = np.asarray(action_mode_first_frame)
    if action_mode_window is not None:
        window = np.asarray(action_mode_window)[:, :preference_frames]
        desirable_frac = np.isin(window, DESIRABLE_LABELS).mean(axis=1)
        is_incorrect = desirable_frac < 0.5
        is_interaction = (~is_incorrect) & (labels == LABEL_INTV)
        is_correct = (~is_incorrect) & ~is_interaction
    else:
        is_correct = (labels == LABEL_DEMO) | (labels == LABEL_ROLLOUT)
        is_interaction = labels == LABEL_INTV
        is_incorrect = labels == LABEL_PREINTV
    return is_correct, is_interaction, is_incorrect


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
    is_correct, is_interaction, is_incorrect = _classify(
        action_mode_first_frame, action_mode_window, preference_frames
    )

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


class InteractionExhaustionSampler(Sampler):
    """APO 원문 `sampler.py::BalancedInteractionDistributedSampler`의 종료조건을 그대로
    재현한 버전(단일 프로세스용). 2026-08-05 추가 — 위 `build_balanced_sampler`(기댓값
    방식, WeightedRandomSampler)는 "1 epoch = len(dataset)"으로 고정돼 있어 interaction이
    적은 라운드는 소수 interaction 샘플이 한 epoch 안에서도 여러 번 반복되는 문제가 있음
    (round2 실측: epoch당 interaction 기대 노출 1.34회, EXP-10.md 2026-08-04/05 절 참고).

    이 클래스는 원문처럼 **1 epoch = interaction 인덱스가 정확히 한 번씩(중복 없이) 다
    소진되는 시점**으로 정의한다. correct/incorrect는 부족하면 순환 재사용(원문의
    `.copy()` 재순환과 동일 효과, 여기선 무한 셔플-순환 제너레이터로 구현).
    """

    def __init__(
        self, action_mode_first_frame, batch_size,
        target_correct=0.5, target_interaction=0.25, target_incorrect=0.25,
        action_mode_window=None, preference_frames=8, seed=None,
    ):
        is_correct, is_interaction, is_incorrect = _classify(
            action_mode_first_frame, action_mode_window, preference_frames
        )
        self.correct_idx = np.nonzero(is_correct)[0]
        self.interaction_idx = np.nonzero(is_interaction)[0]
        self.incorrect_idx = np.nonzero(is_incorrect)[0]
        if len(self.interaction_idx) == 0:
            raise ValueError(
                "InteractionExhaustionSampler: interaction(INTV) 샘플이 0개 — "
                "개입 데이터가 없는 라운드(예: base)엔 이 sampler를 쓸 수 없음"
            )
        if len(self.correct_idx) == 0 or len(self.incorrect_idx) == 0:
            raise ValueError(
                "InteractionExhaustionSampler: correct 또는 incorrect 샘플이 0개 — "
                "재순환할 풀이 비어있어 배치 구성을 못 채움"
            )

        self.batch_size = batch_size
        self.n_interaction = max(1, round(batch_size * target_interaction))
        self.n_correct = max(1, round(batch_size * target_correct))
        self.n_incorrect = max(0, batch_size - self.n_correct - self.n_interaction)
        # epoch 길이 = interaction을 딱 한 번씩 다 쓰는 데 필요한 batch 수(ceil, 원문과 동일 정의)
        self.num_batches = -(-len(self.interaction_idx) // self.n_interaction)
        self.seed = seed
        self._epoch = 0

    def __len__(self):
        return self.num_batches * self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(None if self.seed is None else self.seed + self._epoch)
        self._epoch += 1
        interaction_pool = rng.permutation(self.interaction_idx).tolist()

        def cycle(idx_array):
            while True:
                for i in rng.permutation(idx_array):
                    yield int(i)

        correct_cycle = cycle(self.correct_idx)
        incorrect_cycle = cycle(self.incorrect_idx)

        indices = []
        for _ in range(self.num_batches):
            n_int_this = min(self.n_interaction, len(interaction_pool))
            batch = [interaction_pool.pop() for _ in range(n_int_this)]
            # interaction이 마지막 배치에서 모자라면 그 부족분은 correct로 메움(원문과 동일 취급)
            n_correct_this = self.n_correct + (self.n_interaction - n_int_this)
            n_correct_this = min(n_correct_this, self.batch_size - len(batch))
            batch += [next(correct_cycle) for _ in range(n_correct_this)]
            n_incorrect_this = self.batch_size - len(batch)
            batch += [next(incorrect_cycle) for _ in range(n_incorrect_this)]
            rng.shuffle(batch)
            indices.extend(batch)
        return iter(indices)


def build_interaction_exhaustion_sampler(
    action_mode_first_frame, batch_size,
    target_correct=0.5, target_interaction=0.25, target_incorrect=0.25,
    action_mode_window=None, preference_frames=8, seed=None,
):
    """InteractionExhaustionSampler 생성 헬퍼(build_balanced_sampler와 같은 호출부 형태)."""
    return InteractionExhaustionSampler(
        action_mode_first_frame, batch_size,
        target_correct=target_correct, target_interaction=target_interaction,
        target_incorrect=target_incorrect, action_mode_window=action_mode_window,
        preference_frames=preference_frames, seed=seed,
    )
