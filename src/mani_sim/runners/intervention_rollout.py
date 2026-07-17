"""사람 개입이 가능한 닫힌 루프 rollout — 개입 데이터 수집용.

정책이 receding horizon으로 돌다가, 사람이 토글키를 누르면 그 순간부터 키보드가
로봇을 직접 몬다. 개입 중 프레임은 INTV, 정책 프레임은 ROLLOUT으로 라벨링하고,
에피소드가 끝나면 각 개입 직전 구간을 PREINTV로 재라벨한다(SIRIUS 스킴).

개입 판단은 `intervention_fn(step, obs_raw)`로 주입한다:
  - None 반환      → 그 스텝은 정책이 실행 (ROLLOUT)
  - action 반환    → 그 스텝은 사람이 실행 (INTV), 반환된 7-dim action을 그대로 step

이 주입 구조 덕에 키보드(사람) 없이도 스크립트 개입(테스트/오라클)으로 같은 경로를
돌릴 수 있다. 키보드용 구현은 아래 KeyboardIntervention.
"""

import time
from collections import deque

import numpy as np
import torch

from mani_sim.datasets.labels import LABEL_INTV, LABEL_ROLLOUT, relabel_preintv


def _predict_chunk(policy, normalizer, obs_history, obs_keys, device):
    # normalizer 통계는 CPU 텐서 → 정규화는 CPU에서 하고 policy forward만 device로 올린다.
    obs_batch = {
        key: torch.as_tensor(
            np.stack([o[key] for o in obs_history]), dtype=torch.float32
        ).unsqueeze(0)
        for key in obs_keys
    }
    obs_batch = normalizer.normalize_obs(obs_batch)
    obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
    chunk = policy.predict_action_chunk(obs_batch)  # (1, Tp, Da), normalized, on device
    chunk = normalizer.unnormalize_action(chunk[0].cpu())  # (Tp, Da), CPU
    return chunk.detach().numpy()


def collect_episode(
    env,
    policy,
    normalizer,
    obs_keys,
    obs_horizon,
    action_horizon,
    device,
    intervention_fn,
    max_steps=None,
    should_end_fn=None,
    preintv_length=15,
    render=False,
    render_fn=None,
    control_fps=0.0,
    predict_fn=None,
):
    """한 에피소드를 개입 가능 상태로 돌려 프레임별 (obs, action, action_mode)를 수집.

    에피소드는 성공·env done·사람 종료(should_end_fn)·max_steps 중 먼저 오는 것에서 끝난다.
    배포 데이터 수집에선 max_steps=None(무제한)으로 두고 사람이 종료키로 끝내는 게 기본.
    테스트·오프라인 수집에선 max_steps로 상한을 준다.

    개입에서 정책으로 돌아올 때는 남은 action chunk를 버리고 현재 관측에서 재계획한다
    (사람이 상태를 바꿔놨을 수 있으므로).

    control_fps > 0이면 각 스텝을 그 주기에 맞춰 실시간으로 늦춘다(사람이 보고 개입할
    시간 확보). 0이면 페이싱 없이 최대 속도(테스트·오프라인 수집용).

    render_fn이 주어지면 매 스텝 호출하고 False 반환 시 에피소드를 끝낸다(커스텀 뷰어용,
    render보다 우선). 없고 render=True면 기존 env.render(mode="human")를 쓴다.

    predict_fn(obs_history) -> (T, Da) ndarray로 청크 예측을 대체할 수 있다(예: robomimic
    체크포인트처럼 자체 정규화·RNN 은닉상태를 갖는 정책 — 이 경우 T=1로 매 스텝 재계획해도
    무방). None이면 기존 (policy, normalizer, obs_keys, obs_horizon, device) 기반
    _predict_chunk를 그대로 쓴다(우리 자체 DP/BC-RNN 정책, 기존 동작 그대로).
    """
    if predict_fn is None:
        predict_fn = lambda history: _predict_chunk(policy, normalizer, history, obs_keys, device)
    if hasattr(policy, "eval"):
        policy.eval()
    obs_raw = env.reset()
    obs_history = deque([obs_raw] * obs_horizon, maxlen=obs_horizon)

    obs_seq, action_seq, mode_seq = [], [], []
    chunk, chunk_ptr = None, 0
    success = False

    step = 0
    while max_steps is None or step < max_steps:
        if should_end_fn is not None and should_end_fn():
            break
        loop_start = time.time()

        intv_action = intervention_fn(step, obs_raw)

        if intv_action is not None:
            action = np.asarray(intv_action, dtype=np.float32)
            mode = LABEL_INTV
            chunk = None  # 개입 후 정책 복귀 시 강제 재계획
        else:
            if chunk is None or chunk_ptr >= action_horizon:
                chunk = predict_fn(obs_history)
                chunk_ptr = 0
            action = np.asarray(chunk[chunk_ptr], dtype=np.float32)
            chunk_ptr += 1
            mode = LABEL_ROLLOUT

        obs_seq.append({key: np.asarray(obs_raw[key], dtype=np.float32) for key in obs_keys})
        action_seq.append(action)
        mode_seq.append(mode)

        obs_raw, _reward, done, _info = env.step(action)
        obs_history.append(obs_raw)
        if render_fn is not None:
            if not render_fn():  # 창이 닫히면 에피소드 종료
                break
        elif render:
            env.render(mode="human")

        if env.is_success()["task"]:
            success = True
        if success or done:
            break

        if control_fps > 0:
            remaining = 1.0 / control_fps - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
        step += 1

    modes = relabel_preintv(np.asarray(mode_seq, dtype=np.int64), preintv_length)
    return {
        "obs": obs_seq,
        "actions": np.asarray(action_seq, dtype=np.float32),
        "action_modes": modes,
        "success": success,
    }


class KeyboardIntervention:
    """robosuite 키보드 device를 개입 신호원으로 감싼 intervention_fn.

    토글키(기본 좌·우 Ctrl)로 개입 on/off를 전환한다. 개입 on일 때만 키보드 입력을
    7-dim action으로 변환해 반환하고, off면 None(정책이 실행)을 반환한다. 종료키(기본
    Enter)로 현재 에피소드를 끝낸다(정책이 실패해 사람이 포기할 때) — should_end()로 노출.

    주의: pynput 리스너는 X 디스플레이가 있어야 동작하므로 실제 화면(로컬/포워딩)에서
    실행해야 한다. 헤드리스 환경에선 스크립트 intervention_fn을 대신 쓴다.
    """

    def __init__(
        self, raw_env, toggle_key="ctrl", end_key="enter", pos_sensitivity=1.0, rot_sensitivity=1.0
    ):
        from pynput import keyboard as pynput_keyboard

        from robosuite.devices import Keyboard

        self.raw_env = raw_env
        self.device = Keyboard(
            env=raw_env, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity
        )
        self.device.start_control()
        self.intervening = False
        self.end_requested = False

        self._toggle_keys = self._resolve_keys(toggle_key, pynput_keyboard)
        self._end_keys = self._resolve_keys(end_key, pynput_keyboard)
        self._listener = pynput_keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    @staticmethod
    def _resolve_keys(name, pynput_keyboard):
        """키 이름 → 허용 키 집합. "ctrl"/"alt"/"shift"는 좌·우 양쪽 모두 인식한다
        (pynput은 좌/우를 구분하므로 한쪽만 지정하면 반대쪽을 눌렀을 때 안 걸린다).
        단일 문자면 그 문자, 그 외엔 특수키 이름(enter, tab 등)."""
        Key = pynput_keyboard.Key
        aliases = {
            "ctrl": {Key.ctrl, Key.ctrl_l, Key.ctrl_r},
            "alt": {Key.alt, Key.alt_l, Key.alt_r},
            "shift": {Key.shift, Key.shift_l, Key.shift_r},
        }
        if name in aliases:
            return aliases[name]
        if len(name) == 1:
            return {pynput_keyboard.KeyCode.from_char(name)}
        return {getattr(Key, name)}

    def _on_press(self, key):
        if key in self._toggle_keys:
            self.intervening = not self.intervening
            if self.intervening:
                self.device.start_control()  # 델타 누적 초기화 (점프 방지)
        elif key in self._end_keys:
            self.end_requested = True

    def should_end(self):
        return self.end_requested

    def reset(self):
        """에피소드 시작 시 개입/종료 상태 초기화."""
        self.intervening = False
        self.end_requested = False

    def __call__(self, step, obs_raw):
        if not self.intervening:
            return None

        ac_dict = self.device.input2action()
        if ac_dict is None:  # device reset('q') → 개입 종료
            self.intervening = False
            return None

        active_robot = self.raw_env.robots[self.device.active_robot]
        action_dict = dict(ac_dict)
        for arm in active_robot.arms:
            action_dict[arm] = ac_dict[f"{arm}_delta"]  # OSC_POSE: delta 입력
        return active_robot.create_action_vector(action_dict)

    def close(self):
        self._listener.stop()
