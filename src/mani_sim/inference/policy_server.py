"""Inference worker thread.

Pure inference loop: pull (t_obs, obs_history) from the input queue ->
predict_fn(obs_history) -> push (t_obs, chunk_np, latency_ms) to the output
queue.

moai_policy(flare/inference/policy_server.py)의 역할을 그대로 이식하되, LeRobot
정책 인터페이스(normalize_inputs/generate_actions/unnormalize_outputs) 대신
mani_sim이 이미 쓰던 `predict_fn(obs_history) -> (T, Da) ndarray` 경계를 그대로
받는다 - diffusion(policy.predict_action_chunk 기반)이든 robomimic이든 collect.py가
만들어 준 predict_fn을 그대로 재사용할 수 있어 정책별 분기가 필요 없다(2026-07-27).
"""

from __future__ import annotations

import queue
import threading
import time


class PolicyServer(threading.Thread):
    def __init__(
        self,
        predict_fn,
        obs_queue: "queue.Queue",
        chunk_queue: "queue.Queue",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="PolicyServer")
        self.predict_fn = predict_fn
        self.obs_queue = obs_queue
        self.chunk_queue = chunk_queue
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                t_obs, obs_history = self.obs_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t_start = time.time()
            chunk_np = self.predict_fn(obs_history)
            latency_ms = (time.time() - t_start) * 1000.0

            try:
                self.chunk_queue.put((t_obs, chunk_np, latency_ms), timeout=0.1)
            except queue.Full:
                # 수신 측(ClientManager)이 밀렸으면 버린다 - 곧 다음 chunk가 대체함.
                pass
