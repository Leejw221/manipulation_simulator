"""수집한 개입 에피소드를 robomimic 형식 HDF5로 저장 (+ 프레임별 action_mode).

robomimic SequenceDataset이 요구하는 최소 구조만 쓴다:
  data/                       (group)
    demo_0/  attrs: num_samples=T
      obs/<key>   : (T, D_k)
      actions     : (T, Da)
      action_mode : (T,)      ← 개입 라벨 (SIRIUS 스킴), robomimic 확장 필드
    demo_1/ ...
  data.attrs["total"] = 총 프레임 수
  data.attrs["env_args"] = 원본 task hdf5의 env_args JSON 문자열 그대로

env_args는 `utils.task_utils.derive_task_meta_from_hdf5`가 학습 시 이 파일에서 직접
읽는다(2026-07-27 발견 — round.py가 만드는 모든 hdf5에 이 필드가 빠져 있어서
round.py의 train 단계가 항상 KeyError로 죽던 gap). 라이브 env에서 새로 만들지 않고
그 task의 원본 robomimic hdf5(`cfg.task.hdf5_path`)에서 그대로 복사하는 이유: 값은
어차피 동일하고(같은 env_name/env_kwargs로 수집), env.serialize() 포맷을 다시 맞출
필요 없이 이미 검증된 문자열을 재사용할 수 있다.

이 파일은 robomimic import를 쓰지 않으므로 datasets 격리 원칙과 무관하지만,
저장 포맷을 robomimic 어댑터(robomimic_dataset.py)가 그대로 되읽도록 맞춘다.
"""

import os

import h5py
import numpy as np


def read_env_args(hdf5_path):
    """task의 원본 robomimic hdf5에서 env_args(JSON 문자열)를 그대로 읽어온다."""
    with h5py.File(hdf5_path, "r") as f:
        return f["data"].attrs["env_args"]


def write_intervention_hdf5(path, episodes, obs_keys, env_args):
    """수집 에피소드 리스트를 robomimic 형식 HDF5로 저장.

    Args:
        path (str): 출력 .hdf5 경로.
        episodes (list[dict]): 각 dict = collect_episode() 반환
            ({"obs": [프레임별 obs dict], "actions": (T,Da), "action_modes": (T,)}).
        obs_keys (list[str]): 저장할 low_dim obs 키.
        env_args (str): read_env_args()로 읽은 원본 task hdf5의 env_args JSON 문자열.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        total = 0

        for demo_idx, ep in enumerate(episodes):
            actions = np.asarray(ep["actions"], dtype=np.float32)
            modes = np.asarray(ep["action_modes"], dtype=np.int64)
            num_samples = actions.shape[0]

            demo_grp = data_grp.create_group(f"demo_{demo_idx}")
            demo_grp.attrs["num_samples"] = num_samples

            obs_grp = demo_grp.create_group("obs")
            for key in obs_keys:
                # rgb 키는 이미 runners.intervention_rollout._to_storage_obs가 raw
                # (HWC, uint8)로 변환해서 넘긴다 - 여기서 float32로 다시 캐스팅하면
                # 원본 데모 hdf5(uint8)와 dtype이 어긋나 병합 데이터셋 collate가 깨진다.
                obs_grp.create_dataset(key, data=np.stack([o[key] for o in ep["obs"]]))

            demo_grp.create_dataset("actions", data=actions)
            demo_grp.create_dataset("action_mode", data=modes)
            total += num_samples

        data_grp.attrs["total"] = total
        data_grp.attrs["env_args"] = env_args

    return path
