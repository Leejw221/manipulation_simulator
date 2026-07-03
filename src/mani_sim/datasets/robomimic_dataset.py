"""robomimic HDF5 → Diffusion Policy 공통 batch dict 어댑터.

robomimic 전용 코드(robomimic.utils.dataset.SequenceDataset, obs_utils)는 이 파일 안에만 둔다
(docs/plan.md 설계원칙 3: 벤치마크 코드 격리).

robomimic SequenceDataset은 frame_stack(=To)·seq_length(=Tp)를 합쳐 길이
(To - 1 + Tp)짜리 하나의 윈도우를 obs·action 양쪽에 동일하게 반환한다 — 즉 obs와
action의 윈도우가 같은 배열이고, 그 안에서 "관측 이력"과 "예측할 행동 구간"을
나누는 건 호출자 책임이다. 이 클래스가 그 슬라이싱(윈도우 앞쪽 To개=obs, 뒤쪽
Tp개=action)을 수행해 Diffusion Policy가 바로 쓸 수 있는 형태로 재구성한다.
"""

import robomimic.utils.obs_utils as ObsUtils
import torch
from robomimic.utils.dataset import SequenceDataset


def _ensure_obs_utils_initialized(obs_keys):
    if ObsUtils.OBS_KEYS_TO_MODALITIES is not None:
        return
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(obs_keys), "rgb": []}})


class RobomimicSequenceDataset(torch.utils.data.Dataset):
    """robomimic HDF5(low_dim)에서 (obs_horizon, pred_horizon) 시퀀스를 뽑아 DP용 batch로 변환.

    Args:
        hdf5_path (str): robomimic 형식 HDF5 경로.
        obs_keys (list[str]): 사용할 low_dim obs 키.
        obs_horizon (int): To, 관측 이력 길이.
        pred_horizon (int): Tp, 예측할 행동 청크 길이.
        normalizer (MinMaxNormalizer | None): 주어지면 obs/action을 [-1, 1]로 정규화해 반환.
        action_key (str): 기본 "actions".
        filter_key (str | None): robomimic demo 필터(예: "train"/"valid" 마스크). 없으면 전체 사용.
    """

    def __init__(
        self,
        hdf5_path,
        obs_keys,
        obs_horizon,
        pred_horizon,
        normalizer=None,
        action_key="actions",
        filter_key=None,
    ):
        _ensure_obs_utils_initialized(obs_keys)

        self.obs_keys = list(obs_keys)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.normalizer = normalizer
        self.action_key = action_key

        self._seq_dataset = SequenceDataset(
            hdf5_path=hdf5_path,
            obs_keys=self.obs_keys,
            dataset_keys=(action_key,),
            load_next_obs=False,
            frame_stack=obs_horizon,
            seq_length=pred_horizon,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=True,
            filter_by_attribute=filter_key,
            hdf5_cache_mode="low_dim",
        )

    def __len__(self):
        return len(self._seq_dataset)

    def __getitem__(self, index):
        raw = self._seq_dataset[index]

        # robomimic이 준 전체 윈도우(길이 To-1+Tp)에서 앞 To개=관측 이력, 뒤 Tp개=행동 청크로 슬라이싱.
        obs = {
            key: torch.as_tensor(raw["obs"][key][: self.obs_horizon], dtype=torch.float32)
            for key in self.obs_keys
        }
        action = torch.as_tensor(raw[self.action_key][-self.pred_horizon :], dtype=torch.float32)
        action_mask = torch.as_tensor(
            raw["pad_mask"][-self.pred_horizon :, 0], dtype=torch.bool
        )

        if self.normalizer is not None:
            obs = self.normalizer.normalize_obs(obs)
            action = self.normalizer.normalize_action(action)

        return {"obs": obs, "action": action, "action_mask": action_mask}
