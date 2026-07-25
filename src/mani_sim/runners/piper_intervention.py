"""Piper(raw MuJoCo, Placo IK) 텔레옵을 intervention_rollout.KeyboardIntervention과 동일한
인터페이스(`__call__(step,obs_raw)->action|None`, `.should_end()`, `.reset()`, `.close()`)로
감싼다. collect_episode()는 이 계약만 보고 동작해서(로봇 종류를 모름, 실측 확인 2026-07-26)
collect.py가 env_backend로 KeyboardIntervention/이 클래스만 바꿔 끼우면 된다.

FLARE의 KeyboardIKController(Placo IK, bi_piper_keyboard.py에서 팔 하나 몫만 재사용)로
EE를 조그한다 - flare/lerobot/placo가 설치된 piper_collect env(python 3.12)에서만
실제로 동작(지연 import, mani_sim의 기본 robomimic env엔 이 의존성 없음)."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

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

        self._held: set[str] = set()
        self._gripper_closed = False
        self._end_requested = False

        self._Key = keyboard.Key
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        logger.info("[Piper keyboard] 연결됨. wasdqe=이동 uoikjl=회전 f=그리퍼 Enter=에피소드 종료")

    def _on_press(self, key):
        ch = getattr(key, "char", None)
        if key == self._Key.enter:
            self._end_requested = True
        elif ch is not None:
            if ch.lower() == "f":
                self._gripper_closed = not self._gripper_closed
                logger.info(f"[Piper keyboard] gripper -> {'close' if self._gripper_closed else 'open'}")
            else:
                self._held.add(ch.lower())

    def _on_release(self, key):
        ch = getattr(key, "char", None)
        if ch is not None:
            self._held.discard(ch.lower())

    def should_end(self):
        return self._end_requested

    def reset(self):
        self._end_requested = False
        self._gripper_closed = False
        self._held.clear()

    def __call__(self, step, obs_raw):
        """항상 action을 반환(None 없음) - 매 프레임 사람이 몬다."""
        p, r = self.pos_step_m, float(np.radians(self.rot_step_deg))
        held = self._held
        dxyz = np.array([
            p * (("w" in held) - ("s" in held)),
            p * (("a" in held) - ("d" in held)),
            p * (("q" in held) - ("e" in held)),
        ])
        drpy = np.array([
            r * (("o" in held) - ("u" in held)),
            r * (("i" in held) - ("k" in held)),
            r * (("j" in held) - ("l" in held)),
        ])
        self.ik.nudge(dxyz, drpy)
        joints_deg = np.degrees(self.ik.solve())
        gripper = self.gripper_close_pos if self._gripper_closed else self.gripper_open_pos
        return np.concatenate([joints_deg, [gripper]]).astype(np.float32)

    def close(self):
        self._listener.stop()
