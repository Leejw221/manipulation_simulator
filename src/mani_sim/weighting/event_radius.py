"""가중치 축 — event_radius (STAIR Eq.2 critical-segment resampling 그대로).

원문 대조: STAIR(arXiv 2606.15587) Eq.2 — key moments K={k1,...,kM} 기준 radius r 안의
프레임 집합 W={t: min_{k in K}|t-k|<=r}, w_t=lambda(t in W) else 1,
L=(1/sum(w_t)) * sum(w_t * loss_t). 정규화는 diffusion_trainer.py에서 weight_b를 배치 평균
1로 스케일링해 동일 효과를 낸다(2026-07-22, sirius_loss는 안 건드림).

phase_rule.py(critical stage **전체 구간**에 가중치, DAISS류)와의 차이: 이건 critical stage로
**들어가는 경계 시점** 근방 ±radius 프레임만 가중치를 준다 — "전체 구간 vs 이벤트 근방"을
실제로 비교하기 위한 별도 arm.

key moments = stage_onehot에서 critical_stages로 진입하는 경계(offline 라벨, 이미 그리퍼
개폐+속도감속 둘 다로 계산된 경계 — Square align_A=속도, grasp/release=그리퍼,
Transport 전부 그리퍼. 새 신호 불필요, 기존 stage_labeler.py 경계 재사용)."""

import h5py
import numpy as np
import torch


class EventRadiusWeight:
    def __init__(self, hdf5_path, stage_key="stage_onehot", critical_weight=3.0,
                 critical_stages=(3,), radius=20, device=None):
        """radius: 이벤트(critical stage 진입 시점) 기준 ±radius 프레임 창."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critical_weight = critical_weight
        self.critical_stages = set(critical_stages)
        self.radius = radius

        self._events_by_demo = {}
        self._T_by_demo = {}
        with h5py.File(hdf5_path, "r") as f:
            for demo_id in f["data"].keys():
                onehot = np.asarray(f["data"][demo_id]["obs"][stage_key][:])
                stage = onehot.argmax(axis=1)
                T = len(stage)
                idx = np.arange(T)
                in_critical = np.isin(stage, list(self.critical_stages))
                entered = in_critical & np.concatenate([[in_critical[0]], np.diff(in_critical.astype(int)) == 1])
                events = idx[entered]
                self._events_by_demo[demo_id] = events
                self._T_by_demo[demo_id] = T

        n_events = sum(len(v) for v in self._events_by_demo.values())
        print(
            f"event_radius weight: critical_stages={sorted(self.critical_stages)} "
            f"radius={radius} critical_weight={critical_weight} "
            f"demos={len(self._events_by_demo)} total_events={n_events}",
            flush=True,
        )

    def compute_weights(self, demo_ids, index_in_demos, **_unused):
        """demo_ids: list[str] 길이 B. index_in_demos: (B,) int. 반환: (B,) float weight."""
        weight = np.ones(len(demo_ids), dtype=np.float32)
        for i, (demo_id, idx) in enumerate(zip(demo_ids, index_in_demos)):
            events = self._events_by_demo[demo_id]
            idx = min(int(idx), self._T_by_demo[demo_id] - 1)
            if len(events) and np.min(np.abs(events - idx)) <= self.radius:
                weight[i] = self.critical_weight
        return torch.as_tensor(weight, dtype=torch.float32, device=self.device)
