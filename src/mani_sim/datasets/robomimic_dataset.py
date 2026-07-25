"""robomimic HDF5 → Diffusion Policy 공통 batch dict 어댑터.

robomimic 전용 코드(robomimic.utils.dataset.SequenceDataset, obs_utils)는 이 파일 안에만 둔다
(docs/plan.md 설계원칙 3: 벤치마크 코드 격리).

robomimic import는 함수/생성자 안에서 지연 로드한다(2026-07-26) — robomimic이 설치 안 된
env(예: Piper 텔레옵용 piper_collect env)에서도 이 모듈을 import만은 할 수 있어야
factory.py의 registry 등록 부수효과 import 체인이 안 깨진다(실제로 robosuite task를 쓸
때만 robomimic이 필요).

robomimic SequenceDataset은 frame_stack(=To)·seq_length(=Tp)를 합쳐 길이
(To - 1 + Tp)짜리 하나의 윈도우를 obs·action 양쪽에 동일하게 반환한다 — 즉 obs와
action의 윈도우가 같은 배열이고, 그 안에서 "관측 이력"과 "예측할 행동 구간"을
나누는 건 호출자 책임이다. 이 클래스가 그 슬라이싱(윈도우 앞쪽 To개=obs, 뒤쪽
Tp개=action)을 수행해 Diffusion Policy가 바로 쓸 수 있는 형태로 재구성한다.
"""

import torch


def _ensure_obs_utils_initialized(low_dim_keys, rgb_keys=()):
    import robomimic.utils.obs_utils as ObsUtils

    if ObsUtils.OBS_KEYS_TO_MODALITIES is not None:
        return
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": list(low_dim_keys), "rgb": list(rgb_keys)}}
    )


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
        rgb_keys=(),
        hdf5_cache_mode="low_dim",
        extra_keys=(),
    ):
        self.rgb_keys = list(rgb_keys)
        low_dim_keys = [k for k in obs_keys if k not in self.rgb_keys]
        _ensure_obs_utils_initialized(low_dim_keys, self.rgb_keys)

        self.obs_keys = list(obs_keys)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.normalizer = normalizer
        self.action_key = action_key
        self.extra_keys = list(extra_keys)  # 예: action_mode(SIRIUS/APO 4-class 라벨)

        from robomimic.utils.dataset import SequenceDataset

        self._seq_dataset = SequenceDataset(
            hdf5_path=hdf5_path,
            obs_keys=self.obs_keys,
            dataset_keys=(action_key,) + tuple(self.extra_keys),
            load_next_obs=False,
            frame_stack=obs_horizon,
            seq_length=pred_horizon,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=True,
            filter_by_attribute=filter_key,
            hdf5_cache_mode=hdf5_cache_mode,
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
        # rgb: (To,H,W,C) uint8값 → (To,C,H,W) float[0,1]
        for key in self.rgb_keys:
            obs[key] = obs[key].permute(0, 3, 1, 2) / 255.0
        action = torch.as_tensor(raw[self.action_key][-self.pred_horizon :], dtype=torch.float32)
        action_mask = torch.as_tensor(
            raw["pad_mask"][-self.pred_horizon :, 0], dtype=torch.bool
        )

        if self.normalizer is not None:
            # rgb 이미지는 이미 [0,1] → 정규화 제외, proprio·stage만 정규화
            low = {k: v for k, v in obs.items() if k not in self.rgb_keys}
            obs.update(self.normalizer.normalize_obs(low))
            action = self.normalizer.normalize_action(action)

        # RA-BC 등 외부 가중치가 demo 내 위치(progress lookup)를 찾을 수 있게 노출.
        # robomimic SequenceDataset이 내부적으로 계산하는 것과 동일한 정의(get_item 참고):
        # index_in_demo = 이 샘플의 obs 이력이 끝나는(=행동 청크가 시작하는) 프레임.
        seq = self._seq_dataset
        demo_id = seq._index_to_demo_id[index]
        demo_start = seq._demo_id_to_start_indices[demo_id]
        offset = 0 if seq.pad_frame_stack else (seq.n_frame_stack - 1)
        index_in_demo = index - demo_start + offset

        item = {
            "obs": obs,
            "action": action,
            "action_mask": action_mask,
            "demo_id": demo_id,
            "index_in_demo": index_in_demo,
        }
        # action_mode 등 extra_keys — action과 같은 뒤쪽 Tp 구간(행동 청크)으로 슬라이싱.
        for key in self.extra_keys:
            item[key] = torch.as_tensor(raw[key][-self.pred_horizon :], dtype=torch.float32)
        return item
