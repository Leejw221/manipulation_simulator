"""가중치 축 — phase_rule (DAISS 스타일 규칙기반 phase 가중치).

원문 동기: DAISS(2026-03 preprint, 양팔 초음파 시술 IL) — phase를 규칙/이벤트로 판정한 뒤
critical phase(접촉·삽입 등)에 고정 가중치를 더 준다, transit phase는 그대로 둔다. 우리는
학습된 reward model 없이 이미 있는 offline stage 라벨(`datasets/stage_labeler.py`,
`label_stages_transport`가 hdf5 `stage_onehot`에 미리 기록해둔 것)을 그대로 재사용한다.

rabc.py와 인터페이스를 맞춘다(demo_id + index_in_demo) — stage_onehot이 hdf5에 있어야 함
(task 설정과 무관하게, 정책 obs_keys에 stage_onehot을 안 넣어도 이 클래스는 직접 읽는다).

이 축은 학습 라벨(offline, 이미 stage 자체가 hdf5에 저장된 것)만 쓴다 — rollout 때 online
tracker를 다시 태울 필요가 없다(정책 입력이 아니라 loss 가중치 계산에만 쓰이므로, EXP-07에서
찾은 train/eval stage 라벨 mismatch 문제와 무관하다)."""

import h5py
import numpy as np
import torch


class PhaseRuleWeight:
    def __init__(self, hdf5_path, stage_key="stage_onehot", critical_weight=3.0,
                 critical_stages=(3,), device=None):
        """critical_stages: critical로 취급할 stage index 튜플. Transport 기본값=(3,)
        (payload_handoff — 실제 로봇0→로봇1 핸드오프 순간, STAGE_NAMES_TRANSPORT 참고.
        2026-07-21 trash_delivered 제거로 5단계로 축소되며 인덱스 4→3)."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critical_weight = critical_weight
        self.critical_stages = set(critical_stages)

        self._stage_by_demo = {}
        with h5py.File(hdf5_path, "r") as f:
            for demo_id in f["data"].keys():
                onehot = np.asarray(f["data"][demo_id]["obs"][stage_key][:])
                self._stage_by_demo[demo_id] = onehot.argmax(axis=1)

        n_critical = sum(int(s in self.critical_stages) for stages in self._stage_by_demo.values() for s in stages)
        n_total = sum(len(v) for v in self._stage_by_demo.values())
        print(
            f"phase_rule weight: critical_stages={sorted(self.critical_stages)} "
            f"critical_weight={critical_weight} critical_frac={n_critical / n_total:.3f}",
            flush=True,
        )

    def compute_weights(self, demo_ids, index_in_demos, **_unused):
        """demo_ids: list[str] 길이 B. index_in_demos: (B,) int. 반환: (B,) float weight."""
        weight = np.ones(len(demo_ids), dtype=np.float32)
        for i, (demo_id, idx) in enumerate(zip(demo_ids, index_in_demos)):
            stages = self._stage_by_demo[demo_id]
            idx = min(int(idx), len(stages) - 1)
            if int(stages[idx]) in self.critical_stages:
                weight[i] = self.critical_weight
        return torch.as_tensor(weight, dtype=torch.float32, device=self.device)
