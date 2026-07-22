"""SARM(arXiv:2509.25358) reward model 학습용 데이터셋.

SARMWindowDataset — xdofai/opensarm(저자 공식 repo, lerobot/common/datasets/rm_lerobot_dataset.py
FrameGapLeRobotDataset) 원문 그대로 포팅[검증-원문, 2026-07-22]: 각 target 프레임 idx마다
[ep_start, ..., idx] causal 윈도우(길이 n_obs_steps+1)를 만든다. D=idx-ep_start가
frame_gap*n_obs_steps 이상이면 idx에 고정 간격(frame_gap)으로 앵커된 mid 프레임을, 부족하면
ep_start~idx 구간을 균등분할한 mid 프레임을 쓴다. rewind augmentation(원문 max_rewind_steps)은
augmentation일 뿐 핵심 메커니즘이 아니라 시간 제약상 생략[사용자 확인 대기 없이 진행,
2026-07-22 — 데이터 증강 스킵은 충실도(fidelity)와 무관].

정답 라벨은 VLM annotation 대신 heuristic stage 라벨(stage_labeler.py의 stage_onehot)을
sparse annotation으로 취급해서 그대로 쓴다(dense 브랜치는 사용자 지시로 아예 미구현).

targets(T,) = stage(정수부) + tau(소수부) — xdofai train_step의
`gt_stage, gt_sub_reward = floor(trg), remainder(trg, 1.0)`과 동일한 인코딩.
"""

import h5py
import numpy as np
from torch.utils.data import Dataset

from mani_sim.datasets.stage_labeler import stage_alpha_bar, stage_tau as _stage_tau


def _decode_stage_tau(onehot):
    """onehot(T,K) -> stage(T,) int, target(T,) float(=stage+tau, alpha_bar 미적용 — tau는
    순수 구간-내 상대위치). alpha_bar로 가중된 global progress([0,1))는 학습 타깃이 아니라
    추론 후 SARMFullModel.predict_progress()에서 Eq.3-4로 별도 합성한다[2026-07-22 정정 —
    이전 버전은 progress*num_stages를 floor/remainder로 쪼갰는데, alpha_bar가 균등이 아니면
    floor(progress*K)가 실제 stage 번호와 어긋남(예: stage2=35.5%라 25%가량 오분류)]."""
    stage = onehot.argmax(axis=1).astype(np.int64)
    num_stages = onehot.shape[1]
    tau = _stage_tau(stage, num=num_stages)
    target = stage.astype(np.float32) + tau
    return stage, target


def get_frame_indices(idx, n_obs_steps, frame_gap, ep_start, ep_end):
    """xdofai FrameGapLeRobotDataset.get_frame_indices 원문 그대로 포팅[검증-원문].

    [ep_start] + mid(n_obs_steps-1개) + [idx], 길이 n_obs_steps+1, 단조비감소."""
    idx = min(max(idx, ep_start), ep_end)
    if n_obs_steps == 0:
        return [idx]
    steps_between = n_obs_steps - 1
    if steps_between <= 0:
        return [ep_start, idx]

    D = idx - ep_start
    if D >= frame_gap * n_obs_steps:
        mid = [idx - frame_gap * j for j in range(steps_between, 0, -1)]
    else:
        mid = [ep_start + round(D * k / n_obs_steps) for k in range(1, n_obs_steps)]

    frames = [ep_start] + mid + [idx]
    for i in range(1, len(frames)):
        if frames[i] < frames[i - 1]:
            frames[i] = frames[i - 1]
    return frames


class SARMWindowDataset(Dataset):
    """hdf5(stage_onehot 라벨링됨) -> causal 윈도우 단위 (images, state, targets, lengths).

    image_key: 단일 카메라. state_keys: proprio 저차원 키 목록.
    """

    def __init__(self, hdf5_path, image_key, state_keys, state_dims, n_obs_steps, frame_gap):
        self.hdf5_path = hdf5_path
        self.image_key = image_key
        self.state_keys = list(state_keys)
        self.state_dims = state_dims
        self.n_obs_steps = n_obs_steps
        self.frame_gap = frame_gap
        self._h5 = None

        with h5py.File(hdf5_path, "r") as f:
            onehot_by_demo = {}
            stage_by_demo = {}
            self._T_by_demo = {}
            for demo_id in f["data"].keys():
                onehot = np.asarray(f["data"][demo_id]["obs"]["stage_onehot"][:])
                onehot_by_demo[demo_id] = onehot
                stage_by_demo[demo_id] = onehot.argmax(axis=1).astype(np.int64)
                self._T_by_demo[demo_id] = len(onehot)
            self.num_stages = onehot.shape[1]

        self.alpha_bar = stage_alpha_bar(stage_by_demo, self.num_stages)

        self.index = []  # (demo_id, t)
        self._target_by_demo = {}
        for demo_id, onehot in onehot_by_demo.items():
            _, target = _decode_stage_tau(onehot)
            self._target_by_demo[demo_id] = target
            self.index.extend([(demo_id, t) for t in range(self._T_by_demo[demo_id])])

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")  # 지연 오픈(DataLoader worker fork 후)
        return self._h5

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        demo_id, t = self.index[idx]
        T = self._T_by_demo[demo_id]
        ep_start, ep_end = 0, T - 1
        obs_idx = get_frame_indices(t, self.n_obs_steps, self.frame_gap, ep_start, ep_end)
        # obs_idx는 단조비감소일 뿐 중복 가능(에피소드 초반) — h5py fancy indexing은 순증가만
        # 허용해서 unique로 읽은 뒤 되매핑한다.
        uniq_idx = sorted(set(obs_idx))
        pos_of = {v: k for k, v in enumerate(uniq_idx)}
        remap = [pos_of[i] for i in obs_idx]

        grp = self._file()["data"][demo_id]["obs"]
        img_raw = np.asarray(grp[self.image_key][uniq_idx])[remap]  # (W,H,C) uint8, W=n_obs_steps+1
        images = np.transpose(img_raw.astype(np.float32) / 255.0, (0, 3, 1, 2))  # (W,C,H,W)
        state = np.concatenate(
            [np.asarray(grp[k][uniq_idx], dtype=np.float32)[remap] for k in self.state_keys], axis=1
        )  # (W, state_dim)
        target = self._target_by_demo[demo_id]
        targets = target[obs_idx]  # (W,)

        return {
            "images": images,
            "state": state,
            "targets": targets,
            "lengths": self.n_obs_steps + 1,
        }
