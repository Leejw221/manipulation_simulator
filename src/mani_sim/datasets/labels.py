"""개입 데이터의 프레임별 action_mode 라벨 + pre-intervention 재라벨 + preference(선호) 판정.

SIRIUS/APO와 동일한 4-class 스킴을 쓴다:
  -1  demo         : 순수 시연 (round 0, 사람만)
   0  rollout      : 정책이 실행한 프레임
   1  intervention : 사람이 교정한 프레임
 -10  pre-intv     : 각 개입 시작 직전 N개 프레임 (rollout에서 재라벨 — 실패로 이어진 행동)
"""

import torch
import numpy as np

LABEL_DEMO = -1
LABEL_ROLLOUT = 0
LABEL_INTV = 1
LABEL_PREINTV = -10

DESIRABLE_LABELS = (LABEL_DEMO, LABEL_ROLLOUT, LABEL_INTV)  # undesirable = 나머지(=PREINTV)


def desirable_mask(action_mode_window, preference_frames=8, desirable_labels=DESIRABLE_LABELS):
    """action_mode_window: (B, W) 라벨 텐서 -> (B,) bool.

    윈도우 앞 preference_frames개 프레임의 desirable 비율(평균) >= 0.5면 desirable(다수결).
    weighting/action_error.py·losses/apo_loss.py가 공유하는 preference 판정 로직
    (chosen/rejected를 나누는 기준은 하나여야 함 — 축마다 다시 정의하면 불일치 위험).
    """
    window = action_mode_window[:, :preference_frames]
    frame_desirable = torch.zeros_like(window, dtype=torch.float32)
    for label in desirable_labels:
        hit = torch.isclose(window.float(), torch.full_like(window.float(), float(label)))
        frame_desirable = torch.where(hit, torch.ones_like(frame_desirable), frame_desirable)
    return frame_desirable.mean(dim=1) >= 0.5


def relabel_preintv(action_modes, preintv_length):
    """각 개입 시작 직전의 rollout 프레임 preintv_length개를 PREINTV로 바꾼다.

    개입 시작 = 이전 프레임이 INTV가 아니고 현재가 INTV인 지점. 그 앞으로 거슬러
    올라가며 ROLLOUT인 프레임만 PREINTV로 바꾼다(다른 라벨은 건드리지 않는다).

    Args:
        action_modes (np.ndarray): (T,) int, 프레임별 라벨.
        preintv_length (int): 개입 시작 직전 재라벨할 프레임 수.

    Returns:
        np.ndarray: (T,) 재라벨된 복사본.
    """
    modes = np.asarray(action_modes, dtype=np.int64).copy()
    onsets = np.where((modes == LABEL_INTV) & (np.roll(modes, 1) != LABEL_INTV))[0]
    onsets = onsets[onsets > 0]  # index 0에서 시작하는 개입은 앞 프레임이 없음

    for onset in onsets:
        start = max(0, onset - preintv_length)
        window = modes[start:onset]
        window[window == LABEL_ROLLOUT] = LABEL_PREINTV
        modes[start:onset] = window

    return modes
