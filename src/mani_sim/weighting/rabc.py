"""가중치 축 — RA-BC(SARM 논문의 Reward-Aligned Behavior Cloning) progress-delta 가중치.

원문 대조: SARM(arXiv:2509.25358) Eq 8-9, lerobot 공식 구현
(huggingface/lerobot `src/lerobot/rewards/sarm`, `docs/source/sarm.mdx`)[검증-원문, 2026-07-19
WebFetch로 확인]. progress 소스는 두 가지를 다 지원한다:
  1. `progress_path=None`(기본) — heuristic stage 라벨(`datasets/stage_labeler.py`의
     `stage_progress()`)을 그대로 씀. 빠른 baseline/스모크 테스트용.
  2. `progress_path=<npz>` — `scripts/train_sarm.py`로 학습하고 `scripts/
     compute_sarm_progress.py`로 미리 계산해둔 **학습된 reward model의 예측 progress**.
     lerobot의 실제 SARM+RA-BC 파이프라인과 동일 구조(reward model 예측을 가중치에 씀,
     2026-07-19 사용자 지시 — heuristic 라벨을 "annotation"으로 취급해 reward model을
     학습시킨 버전).

공식(원문 Eq 8-9 그대로):
  r_i = progress(o_{t+Δ}) - progress(o_t)          (Δ = 정책 chunk_size, 정책 종류와 무관)
  soft weight:  w̃ = clip((r - (μ-2σ)) / (4σ + ε), 0, 1)
  final weight: w  = 1{r>κ} + 1{0<=r<=κ} * w̃

μ, σ는 데이터셋 전체 r 분포의 평균/표준편차(배치 단위 아님 — ClassBasedWeight의 ratio 계산과
같은 이유, 배치 통계는 표본이 작아 불안정). κ(kappa)는 "이미 충분히 좋은 진행" 임계값 —
lerobot 문서의 튜닝 가이드(delta_mean 근처로 설정)를 그대로 따름.

class_based/action_error와 인터페이스가 다르다(action_mode(B,T) 대신 demo_id+index_in_demo가
필요) — `RobomimicSequenceDataset.__getitem__`이 이미 이 둘을 기본 반환값으로 노출해둔 상태라
extra_keys 지정 없이 바로 쓸 수 있다.
"""

import h5py
import numpy as np
import torch

from mani_sim.datasets.stage_labeler import stage_progress


class RABCWeight:
    def __init__(self, hdf5_path, chunk_size, stage_key="stage_onehot", kappa=0.01, eps=1e-6,
                 progress_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.kappa = kappa
        self.eps = eps

        if progress_path is not None:
            npz = np.load(progress_path)
            self._progress_by_demo = {k: npz[k] for k in npz.files}
            print(f"RABC weight: progress_path={progress_path}에서 로드(reward model 예측)", flush=True)
        else:
            self._progress_by_demo = {}
            with h5py.File(hdf5_path, "r") as f:
                for demo_id in f["data"].keys():
                    onehot = np.asarray(f["data"][demo_id]["obs"][stage_key][:])
                    stage = onehot.argmax(axis=1)
                    self._progress_by_demo[demo_id] = stage_progress(stage, num=onehot.shape[1])
            print("RABC weight: heuristic stage_progress()에서 직접 계산", flush=True)

        deltas = []
        for progress in self._progress_by_demo.values():
            T = len(progress)
            idx = np.arange(T)
            idx_delta = np.minimum(idx + chunk_size, T - 1)
            deltas.append(progress[idx_delta] - progress[idx])

        all_deltas = np.concatenate(deltas)
        self.mu = float(all_deltas.mean())
        self.sigma = float(all_deltas.std())
        print(
            f"RABC weight: chunk_size={chunk_size} kappa={kappa} "
            f"delta_mean={self.mu:.4f} delta_std={self.sigma:.4f}",
            flush=True,
        )

    def compute_weights(self, demo_ids, index_in_demos):
        """demo_ids: list[str] 길이 B(청크가 시작하는 demo). index_in_demos: (B,) int
        (그 demo 안에서 청크가 시작하는 프레임). 반환: (B,) float weight, device 위."""
        r = np.zeros(len(demo_ids), dtype=np.float32)
        for i, (demo_id, idx) in enumerate(zip(demo_ids, index_in_demos)):
            progress = self._progress_by_demo[demo_id]
            T = len(progress)
            idx = min(int(idx), T - 1)
            idx_delta = min(idx + self.chunk_size, T - 1)
            r[i] = progress[idx_delta] - progress[idx]

        soft = np.clip((r - (self.mu - 2 * self.sigma)) / (4 * self.sigma + self.eps), 0.0, 1.0)
        hard_high = (r > self.kappa).astype(np.float32)
        mid_mask = (r >= 0) & (r <= self.kappa)
        weight = hard_high + mid_mask.astype(np.float32) * soft
        return torch.as_tensor(weight, dtype=torch.float32, device=self.device)
