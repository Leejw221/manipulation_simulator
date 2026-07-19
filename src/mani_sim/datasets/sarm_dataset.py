"""SARM(arXiv:2509.25358) reward model 학습용 — 프레임 단위 (image, state, stage, tau) 데이터셋.

lerobot 공식 SARM은 CLIP으로 영상+언어를 인코딩하지만, 우리는 task가 Transport 하나뿐이라
언어로 task를 구분할 필요가 없어 언어/CLIP 없이(비전+proprio만) stage 분류 + tau(구간 내
상대위치) 회귀를 예측하는 경량 버전으로 만든다[사용자 판단, 2026-07-19]. 정답 라벨은 VLM
annotation 대신 이미 만든 heuristic stage 라벨(datasets/stage_labeler.py, label_stages.py가
hdf5에 기록해둔 stage_onehot)을 "사람이 단 annotation"처럼 취급해서 그대로 쓴다.

stage_onehot(T,K) -> stage(T,) int(argmax) + tau(T,) float[0,1)(stage_progress 역산:
progress=(stage+tau)/K 이므로 tau=progress*K-stage).
"""

import h5py
import numpy as np
from torch.utils.data import Dataset

from mani_sim.datasets.stage_labeler import stage_progress


def _decode_stage_tau(onehot):
    """onehot(T,K) -> (stage(T,) int, tau(T,) float[0,1))."""
    stage = onehot.argmax(axis=1).astype(np.int64)
    num_stages = onehot.shape[1]
    progress = stage_progress(stage, num=num_stages)
    tau = np.clip(progress * num_stages - stage, 0.0, 1.0).astype(np.float32)
    return stage, tau


class SARMFrameDataset(Dataset):
    """hdf5(이미 stage_onehot 라벨링됨) -> 프레임 단위 (image, state, stage, tau).

    image_key: 단일 카메라(SARM 원문도 단일 video-key 사용 관례, lerobot --video-key와 동일).
    state_keys: proprio 저차원 키 목록(policy와 별개로 reward model 전용 조합 가능).
    """

    def __init__(self, hdf5_path, image_key, state_keys, state_dims):
        self.hdf5_path = hdf5_path
        self.image_key = image_key
        self.state_keys = list(state_keys)
        self.state_dims = state_dims
        self._h5 = None

        with h5py.File(hdf5_path, "r") as f:
            self.index = []  # (demo_id, t)
            self._stage_tau_by_demo = {}
            for demo_id in f["data"].keys():
                onehot = np.asarray(f["data"][demo_id]["obs"]["stage_onehot"][:])
                stage, tau = _decode_stage_tau(onehot)
                self._stage_tau_by_demo[demo_id] = (stage, tau)
                self.index.extend([(demo_id, t) for t in range(len(stage))])
        self.num_stages = onehot.shape[1]

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")  # 지연 오픈(DataLoader worker fork 후, 기존 관례)
        return self._h5

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        demo_id, t = self.index[idx]
        grp = self._file()["data"][demo_id]["obs"]
        img_raw = np.asarray(grp[self.image_key][t])  # (H,W,C) uint8
        image = np.transpose(img_raw.astype(np.float32) / 255.0, (2, 0, 1))  # (C,H,W) float[0,1]
        state = np.concatenate([np.asarray(grp[k][t], dtype=np.float32) for k in self.state_keys])
        stage, tau = self._stage_tau_by_demo[demo_id]
        return {
            "image": image,
            "state": state,
            "stage": int(stage[t]),
            "tau": float(tau[t]),
        }
