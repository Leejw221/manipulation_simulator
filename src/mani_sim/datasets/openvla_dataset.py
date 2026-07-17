"""OpenVLA용 flat 데이터셋 — (image, instruction, action) 프레임 단위.

BC_RNN_GMM/DiffusionPolicy는 (obs_horizon, pred_horizon) 윈도우가 자연스러운 단위였지만,
OpenVLA는 원 논문·공식 구현 그대로 **단일 timestep**(이미지 1장 + 언어 지시 + action
7-dim) 단위로 학습한다 — 청크 개념 없음(π0-FAST 이식만 예외, 우리는 안 씀).

언어 지시는 Square PH 데이터에 없어서(robomimic HDF5엔 언어 라벨 필드가 없음) task
전체에 고정 문자열 하나를 쓴다 — round0(비가중 base)엔 문제없음(모든 프레임이 같은 task).
"""

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

SQUARE_INSTRUCTION = "pick up the square nut and place it on the peg"


class OpenVLAFrameDataset(Dataset):
    def __init__(self, hdf5_path, image_key="agentview_image", action_key="actions", instruction=SQUARE_INSTRUCTION):
        self.hdf5_path = hdf5_path
        self.image_key = image_key
        self.action_key = action_key
        self.instruction = instruction
        self._h5 = None  # 지연 오픈 — DataLoader worker 프로세스에서 각자 열게 함(fork 후
        # 공유 핸들 문제 회피, mani_sim_status.md에 기록된 BC job 크래시와 같은 함정 방지)

        with h5py.File(hdf5_path, "r") as f:
            self.index = []  # (demo_id, t)
            for demo_id in f["data"].keys():
                n = f["data"][demo_id][action_key].shape[0]
                self.index.extend([(demo_id, t) for t in range(n)])

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        demo_id, t = self.index[idx]
        grp = self._file()["data"][demo_id]
        image = Image.fromarray(np.asarray(grp["obs"][self.image_key][t]))
        action = np.asarray(grp[self.action_key][t], dtype=np.float32)
        return {"image": image, "instruction": self.instruction, "action": action}


def collate_openvla(batch):
    return {
        "images": [b["image"] for b in batch],
        "instructions": [b["instruction"] for b in batch],
        "actions": np.stack([b["action"] for b in batch]),
    }
