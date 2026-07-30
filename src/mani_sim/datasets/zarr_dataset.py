"""zarr(ReplayBuffer) -> Diffusion Policy 공통 batch dict 어댑터.

robomimic 벤치마크가 아닌 task(현재: Piper sort_return, collect.py로 수집(lerobot 저장) +
convert_lerobot_to_zarr.py로 변환)용. robomimic_dataset.py의 RobomimicSequenceDataset과
동일한 __getitem__ 계약(obs/action/action_mask/demo_id/index_in_demo)을 만족시켜
diffusion_trainer.py가 데이터 소스만 바꿔 그대로 재사용한다(weighting 모듈도 이 일반
계약만 보고 동작 - robomimic 내부 API에 의존 안 함, 2026-07-26 확인).

ReplayBuffer 자체는 mani_sim_external/piper_capstone/replay_buffer.py에 vendoring된
FLARE 원본(zarr+numcodecs만 있으면 됨, lerobot/flare 패키지 전체 불필요)을 그대로 쓴다.
"""

import sys
from pathlib import Path

import numpy as np
import torch

_PIPER_CAPSTONE_DIR = Path(__file__).resolve().parents[3] / "mani_sim_external" / "piper_capstone"
if str(_PIPER_CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPER_CAPSTONE_DIR))
from replay_buffer import ReplayBuffer  # noqa: E402


class ZarrSequenceDataset(torch.utils.data.Dataset):
    """robomimic의 pad_frame_stack=True/pad_seq_length=True와 동일하게, 데모 경계 밖은
    가장자리 프레임을 반복(pad)해서 (obs_horizon, pred_horizon) 윈도우를 만든다."""

    def __init__(
        self, zarr_path, obs_keys, obs_horizon, pred_horizon, normalizer=None,
        action_key="action", rgb_keys=(), extra_keys=(),
    ):
        self.rgb_keys = list(rgb_keys)
        self.obs_keys = list(obs_keys)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.normalizer = normalizer
        self.action_key = action_key
        self.extra_keys = list(extra_keys)

        self._buffer = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
        episode_ends = np.asarray(self._buffer.episode_ends[:])
        self._episode_ends = episode_ends
        self._episode_starts = np.concatenate([[0], episode_ends[:-1]])

        self._index = []  # (demo_id, index_in_demo) - 데모 안의 매 프레임 하나가 샘플 하나
        for demo_id, (start, end) in enumerate(zip(self._episode_starts, self._episode_ends)):
            self._index.extend((demo_id, t) for t in range(end - start))

    def get_action_mode_first_frame(self):
        """샘플(윈도우)별 대표 action_mode - robomimic_dataset.RobomimicSequenceDataset.
        get_action_mode_first_frame과 동일 계약(그 윈도우의 행동 청크가 시작하는 프레임의
        라벨). apo_sampler.build_balanced_sampler가 배치 구성 이전에 전체 인덱스에 대해
        한 번에 필요로 한다(2026-07-27 밤)."""
        action_mode = self._buffer.data["action_mode"][:]
        global_positions = np.array(
            [int(self._episode_starts[demo_id]) + t for demo_id, t in self._index]
        )
        return action_mode[global_positions]

    def get_action_mode_window(self, width):
        """샘플(윈도우)별 앞 width개 프레임의 action_mode (N, width).

        get_action_mode_first_frame이 첫 프레임 하나만 보는 것과 달리, loss 쪽
        `datasets/labels.desirable_mask`가 실제로 쓰는 것과 같은 윈도우를 돌려준다 —
        두 곳이 다른 기준으로 판정하면 샘플러가 목표한 배치 구성이 loss에서 그대로
        재현되지 않는다(EXP-10.md 2026-07-30~31: PREINTV로 뽑힌 윈도우의 26.7%가
        loss에선 desirable로 뒤집혀 목표 25%가 실제 18.3%만 반영됐음).
        데모 경계 밖은 __getitem__의 _get_window와 동일하게 가장자리 반복(clip)."""
        action_mode = self._buffer.data["action_mode"][:]
        out = np.empty((len(self._index), width), dtype=np.int64)
        for i, (demo_id, t) in enumerate(self._index):
            start = int(self._episode_starts[demo_id])
            demo_len = int(self._episode_ends[demo_id]) - start
            idxs = [start + int(np.clip(t + o, 0, demo_len - 1)) for o in range(width)]
            out[i] = action_mode[idxs]
        return out

    def __len__(self):
        return len(self._index)

    def _get_window(self, key, start, end, t, horizon, before):
        """before=True: [t-horizon+1, t] / False: [t, t+horizon-1] - 범위 밖은 가장자리
        프레임 반복(clip)으로 패딩."""
        arr = self._buffer.data[key]
        demo_len = end - start
        if before:
            offsets = range(-horizon + 1, 1)
        else:
            offsets = range(0, horizon)
        idxs = [int(np.clip(t + o, 0, demo_len - 1)) for o in offsets]
        return np.stack([arr[start + i] for i in idxs])

    def __getitem__(self, index):
        demo_id, t = self._index[index]
        start, end = int(self._episode_starts[demo_id]), int(self._episode_ends[demo_id])
        demo_len = end - start

        obs = {}
        for key in self.obs_keys:
            raw = self._get_window(key, start, end, t, self.obs_horizon, before=True)
            if key in self.rgb_keys:
                obs[key] = torch.as_tensor(raw, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            else:
                obs[key] = torch.as_tensor(raw, dtype=torch.float32)

        action = torch.as_tensor(
            self._get_window(self.action_key, start, end, t, self.pred_horizon, before=False),
            dtype=torch.float32,
        )
        valid_len = min(self.pred_horizon, demo_len - t)
        action_mask = torch.zeros(self.pred_horizon, dtype=torch.bool)
        action_mask[:valid_len] = True

        if self.normalizer is not None:
            low = {k: v for k, v in obs.items() if k not in self.rgb_keys}
            obs.update(self.normalizer.normalize_obs(low))
            action = self.normalizer.normalize_action(action)

        item = {
            "obs": obs, "action": action, "action_mask": action_mask,
            "demo_id": demo_id, "index_in_demo": t,
        }
        for key in self.extra_keys:
            item[key] = torch.as_tensor(
                self._get_window(key, start, end, t, self.pred_horizon, before=False), dtype=torch.float32,
            )
        return item
