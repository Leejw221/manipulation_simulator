"""가중치 축 — stage_difficulty (단계별 이탈률에서 도출한 차등 가중치, EXP 3차 2026-08-17).

`phase_rule.py`(critical stage에 고정 배수)와 두 가지가 다르다:
  - **가중치를 고르지 않고 도출한다.** vanilla rollout에서 잰 **단계별 이탈률**(그 stage에
    진입했으나 다음 stage로 못 넘어간 비율) 순위로 정한다. `critical_weight=3.0` 같은
    임의 상수를 쓰지 않는 게 목적 — 이 트랙에서 `stage_embed_dim=32`를 근거 없이 고른 게
    내내 약점이었다(EXP-01 참고).
  - **zarr에서 읽는다.** phase_rule은 hdf5 `stage_onehot` 전용이라 RoboCasa zarr(`stage`,
    0-indexed int)를 못 읽는다.

범위는 [1.0, 2.0]을 stage 수만큼 균등 분할해 이탈률 순위대로 배정한다. **0을 안 쓰는 이유**:
종단 stage(스펀지 놓기)는 이탈률이 0이라 최하위가 되는데, 성공 조건에 반드시 필요한 동작이라
가중치 0을 주면 그 행동을 아예 안 배운다. 최소값 1.0이면 어느 단계도 baseline보다 덜 배우지
않는다.

⚠️ 배치 평균 정규화는 여기서 안 한다 — `diffusion_trainer.py`가 이미
`weight_b / weight_b.mean()`으로 처리한다(학습률 교란 방지). 즉 실제로 적용되는 건 절대값이
아니라 **상대 비율**이다(최대/최소 2.0배).
"""

import numpy as np
import torch


class StageDifficultyWeight:
    def __init__(self, zarr_path, dropout_rates, w_min=1.0, w_max=2.0, stage_key="stage",
                 device=None):
        """dropout_rates: stage별 이탈률 리스트(길이 = stage 수). 값이 클수록 높은 가중치를
        받는다. 종단 stage처럼 이탈률을 정의할 수 없으면 0(또는 최소값)을 넣으면 최하위가
        된다."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "mani_sim_external" / "piper_capstone"))
        from replay_buffer import ReplayBuffer

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n = len(dropout_rates)
        levels = np.linspace(w_max, w_min, n)          # 1등이 w_max, 꼴등이 w_min
        order = np.argsort(-np.asarray(dropout_rates, dtype=float))  # 이탈률 내림차순
        self.stage_weight = np.empty(n, dtype=np.float32)
        for rank, stage_idx in enumerate(order):
            self.stage_weight[stage_idx] = levels[rank]

        buf = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
        stages = np.asarray(buf.data[stage_key][:]).reshape(-1)
        ends = np.asarray(buf.episode_ends[:])
        starts = np.concatenate([[0], ends[:-1]])
        self._stage_by_demo = {
            int(d): stages[int(s):int(e)] for d, (s, e) in enumerate(zip(starts, ends))
        }

        frac = [float((stages == s).mean()) for s in range(n)]
        print(
            "stage_difficulty weight: "
            + " ".join(f"s{i+1}(이탈{dropout_rates[i]:.1%})→w{self.stage_weight[i]:.2f}[{frac[i]:.0%}]"
                       for i in range(n)),
            flush=True,
        )

    def compute_weights(self, demo_ids, index_in_demos, **_unused):
        """demo_ids: (B,) int(zarr는 demo_id가 정수). 반환: (B,) float weight."""
        w = np.empty(len(demo_ids), dtype=np.float32)
        for i, (demo_id, idx) in enumerate(zip(demo_ids, index_in_demos)):
            stages = self._stage_by_demo[int(demo_id)]
            w[i] = self.stage_weight[int(stages[min(int(idx), len(stages) - 1)])]
        return torch.as_tensor(w, dtype=torch.float32, device=self.device)
