"""robomimic HDF5 데이터셋의 min/max 통계 계산 및 MinMax 정규화.

정규화 공식은 Diffusion Policy(Chi et al.)에서 쓰는 표준 컨벤션을 따른다:
  normalize:   (x - min) / (max - min + eps) * 2 - 1   →  [-1, 1]
  unnormalize: (x + 1) / 2 * (max - min) + min
"""

import json

import h5py
import numpy as np
import torch

_EPS = 1e-8


def compute_minmax_stats(hdf5_path, obs_keys, action_key="actions"):
    """전체 demo를 스캔해 obs 키별·action의 min/max를 계산한다.

    Returns:
        dict: {"obs": {key: {"min": np.ndarray, "max": np.ndarray}}, "action": {"min": ..., "max": ...}}
    """
    obs_mins = {k: None for k in obs_keys}
    obs_maxs = {k: None for k in obs_keys}
    action_min = None
    action_max = None

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = list(f["data"].keys())
        for demo_key in demo_keys:
            demo = f["data"][demo_key]

            actions = demo["actions"][:]
            a_min = actions.min(axis=0)
            a_max = actions.max(axis=0)
            action_min = a_min if action_min is None else np.minimum(action_min, a_min)
            action_max = a_max if action_max is None else np.maximum(action_max, a_max)

            for key in obs_keys:
                values = demo["obs"][key][:]
                v_min = values.min(axis=0)
                v_max = values.max(axis=0)
                obs_mins[key] = v_min if obs_mins[key] is None else np.minimum(obs_mins[key], v_min)
                obs_maxs[key] = v_max if obs_maxs[key] is None else np.maximum(obs_maxs[key], v_max)

    stats = {
        "obs": {k: {"min": obs_mins[k].tolist(), "max": obs_maxs[k].tolist()} for k in obs_keys},
        "action": {"min": action_min.tolist(), "max": action_max.tolist()},
    }
    return stats


def compute_minmax_stats_zarr(zarr_path, obs_keys, action_key="action"):
    """zarr(ReplayBuffer) 버전의 compute_minmax_stats - robomimic처럼 demo 단위로 나눌
    필요 없이(min/max는 데모 경계와 무관) 전체 프레임 배열에서 바로 계산한다. 반환 형식은
    compute_minmax_stats와 동일(MinMaxNormalizer가 출처를 구분할 필요 없게)."""
    import sys
    from pathlib import Path

    piper_capstone_dir = Path(__file__).resolve().parents[3] / "mani_sim_external" / "piper_capstone"
    if str(piper_capstone_dir) not in sys.path:
        sys.path.insert(0, str(piper_capstone_dir))
    from replay_buffer import ReplayBuffer

    buffer = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
    stats = {
        "obs": {
            k: {"min": buffer.data[k][:].min(axis=0).tolist(), "max": buffer.data[k][:].max(axis=0).tolist()}
            for k in obs_keys
        },
        "action": {
            "min": buffer.data[action_key][:].min(axis=0).tolist(),
            "max": buffer.data[action_key][:].max(axis=0).tolist(),
        },
    }
    return stats


def save_stats(stats, path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path):
    with open(path) as f:
        return json.load(f)


class MinMaxNormalizer:
    """stats(dict, compute_minmax_stats 형식)를 받아 obs dict·action 텐서를 [-1, 1]로 정규화/역정규화."""

    def __init__(self, stats):
        self.obs_min = {k: torch.tensor(v["min"], dtype=torch.float32) for k, v in stats["obs"].items()}
        self.obs_max = {k: torch.tensor(v["max"], dtype=torch.float32) for k, v in stats["obs"].items()}
        self.action_min = torch.tensor(stats["action"]["min"], dtype=torch.float32)
        self.action_max = torch.tensor(stats["action"]["max"], dtype=torch.float32)

    @staticmethod
    def _normalize(x, vmin, vmax):
        return (x - vmin) / (vmax - vmin + _EPS) * 2 - 1

    @staticmethod
    def _unnormalize(x, vmin, vmax):
        return (x + 1) / 2 * (vmax - vmin) + vmin

    def normalize_obs(self, obs):
        return {k: self._normalize(v, self.obs_min[k], self.obs_max[k]) for k, v in obs.items()}

    def unnormalize_obs(self, obs):
        return {k: self._unnormalize(v, self.obs_min[k], self.obs_max[k]) for k, v in obs.items()}

    def normalize_action(self, action):
        return self._normalize(action, self.action_min, self.action_max)

    def unnormalize_action(self, action):
        return self._unnormalize(action, self.action_min, self.action_max)

    @classmethod
    def from_file(cls, path):
        return cls(load_stats(path))
