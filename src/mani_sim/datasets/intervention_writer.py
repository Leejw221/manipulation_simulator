"""수집한 개입 에피소드를 robomimic 형식 HDF5로 저장 (+ 프레임별 action_mode).

robomimic SequenceDataset이 요구하는 최소 구조만 쓴다:
  data/                       (group)
    demo_0/  attrs: num_samples=T
      obs/<key>   : (T, D_k)
      actions     : (T, Da)
      action_mode : (T,)      ← 개입 라벨 (SIRIUS 스킴), robomimic 확장 필드
    demo_1/ ...
  data.attrs["total"] = 총 프레임 수

이 파일은 robomimic import를 쓰지 않으므로 datasets 격리 원칙과 무관하지만,
저장 포맷을 robomimic 어댑터(robomimic_dataset.py)가 그대로 되읽도록 맞춘다.
"""

import os

import h5py
import numpy as np


def write_intervention_hdf5(path, episodes, obs_keys):
    """수집 에피소드 리스트를 robomimic 형식 HDF5로 저장.

    Args:
        path (str): 출력 .hdf5 경로.
        episodes (list[dict]): 각 dict = collect_episode() 반환
            ({"obs": [프레임별 obs dict], "actions": (T,Da), "action_modes": (T,)}).
        obs_keys (list[str]): 저장할 low_dim obs 키.
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
                obs_grp.create_dataset(
                    key, data=np.stack([o[key] for o in ep["obs"]]).astype(np.float32)
                )

            demo_grp.create_dataset("actions", data=actions)
            demo_grp.create_dataset("action_mode", data=modes)
            total += num_samples

        data_grp.attrs["total"] = total

    return path
