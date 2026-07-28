"""ControlLoop(env.step 루프)와 PolicyServer(GPU 추론) 사이의 중간 계층.

매 반복:
    1. ObsProvider.get()에서 새 관측을 기다린다
    2. obs_horizon 프레임 버퍼링(콜드 스타트는 첫 프레임 복제 - collect_episode의
       deque([obs_raw] * obs_horizon, ...) 초기화와 동일 관례)
    3. (t_obs, obs_history)를 PolicyServer에 넘긴다
    4. chunk_queue에서 결과를 기다려 merger에 submit
    5. 즉시 반복 - 이게 "연속 추론": chunk가 도착하는 순간 최신 obs로 다음 추론을
       바로 시작한다(action_horizon 스텝마다 기다렸다 재계획하는 게 아님).

moai_policy(flare/inference/client_manager.py) 이식이지만 두 가지를 mani_sim에 맞게
바꿨다(2026-07-27):
  - obs를 텐서로 미리 쌓아 정규화하지 않는다 - obs_history를 raw obs dict 리스트
    그대로 predict_fn에 넘기고, 정규화·device 배치는 predict_fn 내부(_build_obs_batch)
    책임으로 남긴다. ClientManager는 버퍼링만 한다.
  - anchor 보정이 없다: moai_policy가 이식한 실물 로봇 정책은 ACT류(obs_horizon-1
    스텝 전까지 포함해 예측)라 `true_anchor = t_obs - (obs_horizon-1)`이 필요했지만,
    mani_sim의 diffusion policy는 표준 Diffusion Policy 방식이라 chunk[0]이
    obs_history 마지막 프레임 시점(t_obs) 자체의 행동이다(datasets/robomimic_dataset.py의
    frame_stack=obs_horizon/seq_length=pred_horizon 정렬로 실측 확인 - obs는 윈도우의
    앞 obs_horizon개, action은 윈도우의 뒤 pred_horizon개이므로 둘 다 인덱스 t에서
    맞물린다). 그래서 여기선 `merger.submit(t_obs_recv, chunk)`를 보정 없이 그대로 쓴다.
"""

from __future__ import annotations

import collections
import queue
import threading
from typing import Callable


class ClientManager(threading.Thread):
    def __init__(
        self,
        obs_provider,
        policy_obs_queue: "queue.Queue",
        policy_chunk_queue: "queue.Queue",
        merger,
        merger_lock: threading.Lock,
        obs_horizon: int,
        stop_event: threading.Event,
        on_chunk_submitted: Callable[[int, float], None] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="ClientManager")
        self.obs_provider = obs_provider
        self.policy_obs_queue = policy_obs_queue
        self.policy_chunk_queue = policy_chunk_queue
        self.merger = merger
        self.merger_lock = merger_lock
        self.obs_horizon = obs_horizon
        self.stop_event = stop_event
        self.on_chunk_submitted = on_chunk_submitted
        self._obs_buffer: collections.deque = collections.deque(maxlen=obs_horizon)

    def reset_buffer(self) -> None:
        self._obs_buffer.clear()

    def _stack(self, obs_raw) -> list:
        """콜드 스타트는 첫 프레임을 복제해 obs_horizon을 채운다."""
        self._obs_buffer.append(obs_raw)
        while len(self._obs_buffer) < self.obs_horizon:
            self._obs_buffer.appendleft(self._obs_buffer[0])
        return list(self._obs_buffer)

    def run(self) -> None:
        while not self.stop_event.is_set():
            item = self.obs_provider.get(timeout=0.5)
            if item is None:
                continue
            t_obs, obs_raw = item

            obs_history = self._stack(obs_raw)

            try:
                self.policy_obs_queue.put((t_obs, obs_history), timeout=0.1)
            except queue.Full:
                continue

            try:
                t_obs_recv, chunk, latency_ms = self.policy_chunk_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            with self.merger_lock:
                self.merger.submit(t_obs_recv, chunk)

            if self.on_chunk_submitted is not None:
                self.on_chunk_submitted(t_obs_recv, latency_ms)
