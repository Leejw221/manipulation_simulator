"""Single-slot observation buffer with new-data signaling.

The control loop publishes the latest (t_obs, obs) every tick. ClientManager
blocks on get() until new data is available since its last read. Stale
observations are silently overwritten - the inference path always sees the
freshest frame.

moai_policy(flare/inference/obs_provider.py)에서 그대로 이식 - 실물 로봇/시뮬레이션에
의존하는 부분이 전혀 없는 범용 구조라 수정 없음(2026-07-27).
"""

from __future__ import annotations

import threading
import time
from typing import Any


class ObsProvider:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._data: tuple[int, Any] | None = None
        self._seq = 0
        self._last_seen_seq = 0
        self._stopped = False

    def put(self, t_obs: int, obs: Any) -> None:
        with self._cond:
            self._data = (t_obs, obs)
            self._seq += 1
            self._cond.notify_all()

    def get(self, timeout: float | None = None) -> tuple[int, Any] | None:
        """Block until new data has been put since last get. Returns None on stop or timeout."""
        with self._cond:
            deadline = time.monotonic() + timeout if timeout is not None else None
            while self._seq == self._last_seen_seq and not self._stopped:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait()
            if self._stopped:
                return None
            self._last_seen_seq = self._seq
            return self._data

    def reset(self) -> None:
        """Drop any pending data so the next get() blocks until a fresh put()."""
        with self._cond:
            self._data = None
            self._seq = 0
            self._last_seen_seq = 0
            self._stopped = False

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
