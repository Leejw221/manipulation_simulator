"""가중치 축 — action_variance (교수님 7/15 원문 정의: cross-demo 아님).

`scripts/compute_action_variance.py`로 미리 계산해둔 npz(demo_id별 (T,) 분산 배열)를 읽어
가중치로 변환한다. 분산 = 같은 관측을 diffusion policy가 반복 샘플링했을 때 action이 얼마나
안 일관되는가(생성 반복성) — 오차와 같은 방향으로 해석: 분산이 크면(=아직 확신 없음) 더
학습시켜야 한다는 뜻이라 가중치를 올린다.

rabc.py/phase_rule.py와 인터페이스를 맞춘다(demo_id + index_in_demo)."""

import numpy as np
import torch


class ActionVarianceWeight:
    def __init__(self, variance_path, w_min=0.2, w_max=5.0, eps=1e-6, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w_min = w_min
        self.w_max = w_max
        self.eps = eps

        npz = np.load(variance_path)
        self._variance_by_demo = {k: npz[k] for k in npz.files}

        all_var = np.concatenate(list(self._variance_by_demo.values()))
        self.mu = float(all_var.mean())
        print(
            f"action_variance weight: variance_path={variance_path} "
            f"mean_variance={self.mu:.6f} w_range=[{w_min},{w_max}]",
            flush=True,
        )

    def compute_weights(self, demo_ids, index_in_demos, **_unused):
        """demo_ids: list[str] 길이 B. index_in_demos: (B,) int. 반환: (B,) float weight —
        분산이 데이터셋 평균보다 크면 가중치>1(더 학습), 작으면 <1(덜 학습)."""
        weight = np.ones(len(demo_ids), dtype=np.float32)
        for i, (demo_id, idx) in enumerate(zip(demo_ids, index_in_demos)):
            var_arr = self._variance_by_demo[demo_id]
            idx = min(int(idx), len(var_arr) - 1)
            weight[i] = var_arr[idx] / (self.mu + self.eps)
        weight = np.clip(weight, self.w_min, self.w_max)
        return torch.as_tensor(weight, dtype=torch.float32, device=self.device)
