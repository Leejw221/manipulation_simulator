"""Piper(raw MuJoCo, Placo IK) 텔레옵을 intervention_rollout.KeyboardIntervention과 동일한
인터페이스(`__call__(step,obs_raw)->action|None`, `.should_end()`, `.reset()`, `.close()`)로
감싼다. collect_episode()는 이 계약만 보고 동작해서(로봇 종류를 모름, 실측 확인 2026-07-26)
collect.py가 env_backend로 KeyboardIntervention/이 클래스만 바꿔 끼우면 된다.

FLARE의 KeyboardIKController(Placo IK, bi_piper_keyboard.py에서 팔 하나 몫만 재사용)로
EE를 조그한다 - flare/lerobot/placo가 설치된 piper_collect env(python 3.12)에서만
실제로 동작(지연 import, mani_sim의 기본 robomimic env엔 이 의존성 없음)."""

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 키를 계속 누르고 있어도 X11 autorepeat이 release+press를 아주 빠르게 반복 전송하는
# 환경이 있어(실측: 2026-07-26, "f"(토글, press 1회로 충분)는 되는데 이동키(지속적으로
# "눌려있나" 확인해야 함)는 전혀 반응 안 하는 증상 - _held를 release로 바로 비우면 이
# 깜빡임에 취약함) - "마지막 press 이후 이 시간 안이면 계속 눌려있는 걸로 간주"하는
# 타임아웃 방식으로 바꿔 release 이벤트 자체에 의존하지 않게 한다.
_KEY_HOLD_TIMEOUT_S = 0.15

_PIPER_CAPSTONE_DIR = Path(__file__).resolve().parents[3] / "mani_sim_external" / "piper_capstone"
# 심 home keyframe과 동일 자세(mani_sim_external/piper_capstone/piper.xml의 "home" 키) -
# IK 시드를 여기서 시작해야 조그 시작 시 팔이 안 튐.
_SIM_READY_RAD = [0.0, 1.57, -1.3485, 0.0, 0.0, 0.0]


class PiperKeyboardIntervention:
    """항상 사람이 조작(토글 없음) - round0(정책 없음) 수집이 주 용도라 매 프레임 LABEL_INTV로
    기록됨(KeyboardIntervention이 토글 켜진 동안과 동일 의미). 정책 배포 라운드가 생기면
    토글 추가를 검토한다(아직 불필요, 과설계 방지)."""

    def __init__(self, pos_step_m=0.003, rot_step_deg=0.8,
                 gripper_open_pos=101.4, gripper_close_pos=0.0, control_fps=30):
        from flare.teleoperators.bi_piper_keyboard.bi_piper_keyboard import KeyboardIKController
        from flare.teleoperators.bi_piper_xr.config_bi_piper_xr import PiperArmConfig
        from pynput import keyboard

        arm_cfg = PiperArmConfig(
            side="left", urdf_path=str(_PIPER_CAPSTONE_DIR / "urdf" / "piper_description.urdf"),
            link_name="link6", joints_init=list(_SIM_READY_RAD), motion_tracker=None,
        )
        self.ik = KeyboardIKController(arm_cfg, dt=1.0 / control_fps)
        self.pos_step_m = pos_step_m
        self.rot_step_deg = rot_step_deg
        self.gripper_open_pos = gripper_open_pos
        self.gripper_close_pos = gripper_close_pos

        self._last_press: dict[str, float] = {}  # key -> 마지막 press 시각(monotonic)
        self._gripper_closed = False
        self._end_requested = False
        self._press_count = 0  # 진단용(2026-07-26) - 실제로 press 이벤트가 오긴 오는지 확인

        self._Key = keyboard.Key
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        logger.info("[Piper keyboard] 연결됨. wasdqe=이동 uoikjl=회전 f=그리퍼 Enter=에피소드 종료")

    def _is_held(self, ch):
        t = self._last_press.get(ch)
        return t is not None and (time.monotonic() - t) < _KEY_HOLD_TIMEOUT_S

    def _on_press(self, key):
        ch = getattr(key, "char", None)
        if key == self._Key.enter:
            self._end_requested = True
        elif ch is not None:
            ch = ch.lower()
            self._press_count += 1
            if self._press_count <= 5 or self._press_count % 50 == 0:
                logger.info(f"[Piper keyboard][진단] press #{self._press_count}: {ch!r}")
            if ch == "f":
                self._gripper_closed = not self._gripper_closed
                logger.info(f"[Piper keyboard] gripper -> {'close' if self._gripper_closed else 'open'}")
            else:
                self._last_press[ch] = time.monotonic()

    def _on_release(self, key):
        pass  # release는 안 씀 - _is_held가 타임아웃으로 판단(autorepeat release+press 깜빡임 대비)

    def should_end(self):
        return self._end_requested

    def reset(self):
        self._end_requested = False
        self._gripper_closed = False
        self._last_press.clear()

    def __call__(self, step, obs_raw):
        """항상 action을 반환(None 없음) - 매 프레임 사람이 몬다."""
        p, r = self.pos_step_m, float(np.radians(self.rot_step_deg))
        dxyz = np.array([
            p * (self._is_held("w") - self._is_held("s")),
            p * (self._is_held("a") - self._is_held("d")),
            p * (self._is_held("q") - self._is_held("e")),
        ])
        drpy = np.array([
            r * (self._is_held("o") - self._is_held("u")),
            r * (self._is_held("i") - self._is_held("k")),
            r * (self._is_held("j") - self._is_held("l")),
        ])
        self.ik.nudge(dxyz, drpy)
        joints_deg = np.degrees(self.ik.solve())
        gripper = self.gripper_close_pos if self._gripper_closed else self.gripper_open_pos
        return np.concatenate([joints_deg, [gripper]]).astype(np.float32)

    def close(self):
        self._listener.stop()
