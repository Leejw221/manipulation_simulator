"""RA-BC(Reward-Aligned Behavior Cloning) 가중치 — lerobot `utils/rabc.py`(Eq.8-9)를 이식.

lerobot은 SARM 모델이 예측한 progress를 parquet에 저장해 전역 flat index로 lookup하지만,
여기서는 SARM(별도 reward 모델) 학습을 생략하고 heuristic stage 라벨에서 계산한 progress
(`stage_labeler.stage_progress`)를 demo_id별로 직접 lookup한다 — RobomimicSequenceDataset이
이미 `demo_id`·`index_in_demo`를 배치에 노출하므로(datasets/robomimic_dataset.py), lerobot처럼
전역 index 부기가 필요 없다.
"""

import h5py
import numpy as np
import torch

from mani_sim.datasets.stage_labeler import stage_progress


class RABCWeights:
    def __init__(
        self,
        hdf5_path,
        chunk_size,
        kappa=0.01,
        epsilon=1e-6,
        fallback_weight=1.0,
        device=None,
    ):
        self.chunk_size = chunk_size
        self.kappa = kappa
        self.epsilon = epsilon
        self.fallback_weight = fallback_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.progress_by_demo = {}
        with h5py.File(hdf5_path, "r") as f:
            for demo_id in f["data"].keys():
                stage_idx = np.array(f["data"][demo_id]["obs"]["stage_idx"])
                self.progress_by_demo[demo_id] = stage_progress(stage_idx)

        self._compute_global_stats()

    def _delta(self, demo_id, index_in_demo):
        """progress[t + chunk_size] - progress[t] (에피소드 끝 넘으면 마지막 프레임으로 클램프)."""
        prog = self.progress_by_demo.get(demo_id)
        if prog is None:
            return np.nan
        t = len(prog) - 1
        cur = min(max(index_in_demo, 0), t)
        future = min(index_in_demo + self.chunk_size, t)
        return float(prog[future] - prog[cur])

    def _compute_global_stats(self):
        deltas = []
        for prog in self.progress_by_demo.values():
            t = len(prog) - 1
            for i in range(len(prog)):
                future = min(i + self.chunk_size, t)
                deltas.append(prog[future] - prog[i])
        deltas = np.asarray(deltas, dtype=np.float32)
        self.delta_mean = max(float(np.mean(deltas)), 0.0) if len(deltas) else 0.0
        self.delta_std = max(float(np.std(deltas)), self.epsilon) if len(deltas) else self.epsilon

    def compute_batch_weights(self, batch):
        """batch: {'demo_id': list[str], 'index_in_demo': LongTensor(B,)} → (weights(B,), stats dict).

        lerobot rabc.py Eq.8-9와 동일한 공식:
          soft = clip((delta - (mu - 2*sigma)) / (4*sigma + eps), 0, 1)
          delta > kappa        -> weight = 1   (stage를 확실히 진전)
          0 <= delta <= kappa  -> weight = soft (완만히 진전)
          delta < 0            -> weight = 0   (정체/후퇴)
        배치 합이 batch_size가 되도록 정규화.
        """
        demo_ids = batch["demo_id"]
        idxs = batch["index_in_demo"]
        if torch.is_tensor(idxs):
            idxs = idxs.cpu().numpy().tolist()

        deltas = np.array([self._delta(d, i) for d, i in zip(demo_ids, idxs)], dtype=np.float32)

        lower = self.delta_mean - 2 * self.delta_std
        soft = np.clip((deltas - lower) / (4 * self.delta_std + self.epsilon), 0.0, 1.0)

        weights = np.zeros_like(deltas)
        valid = ~np.isnan(deltas)
        weights[deltas > self.kappa] = 1.0
        moderate = (deltas >= 0) & (deltas <= self.kappa)
        weights[moderate] = soft[moderate]
        weights[~valid] = self.fallback_weight

        stats = {
            "raw_mean_weight": float(np.nanmean(weights)),
            "num_zero_weight": int(np.sum(weights == 0)),
            "num_full_weight": int(np.sum(weights == 1.0)),
        }

        w = torch.tensor(weights, device=self.device, dtype=torch.float32)
        w = w * len(w) / (w.sum() + self.epsilon)
        return w, stats
