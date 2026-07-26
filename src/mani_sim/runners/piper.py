"""Piper 조작 기기 구현체 - intervention_rollout.KeyboardIntervention과 동일한 인터페이스
(`__call__(step,obs_raw)->action|None`, `.should_end()`, `.reset()`, `.close()`)로 감싼다.
collect_episode()는 이 계약만 보고 동작해서(로봇 종류를 모름, 실측 확인 2026-07-26)
collect.py/teleoperate.py가 intervention_device로 이 파일의 기기만 바꿔 끼우면 된다.

FLARE의 KeyboardIKController/BiPiperXRTeleop(둘 다 Placo IK)를 재사용한다 - flare/lerobot/
placo(+PICO는 xrobotoolkit_sdk)가 설치된 piper_collect env(python 3.12)에서만 실제로
동작(지연 import, mani_sim의 기본 robomimic env엔 이 의존성 없음).

여기는 "조작 기기" 관심사만 담당한다 - 로봇(어떤 Piper 구성인지)은 기존 task=(hydra) 설정이
고르고, 뷰어(mujoco 온스크린)는 PiperSortReturnEnv.attach_viewer()가 담당한다(env 내부
model/data를 알아야 하므로 기기 파일이 아니라 env 쪽 소관, 2026-07-26 분리 - 안 그러면
로봇×기기 조합이 늘 때마다 같은 뷰어 연결 코드가 여러 곳에 중복될 위험). 새 기기를
추가하려면 클래스를 만들고 make_piper_intervention()의 _TELEOP_DEVICES에 등록하면 끝 -
collect.py/teleoperate.py는 안 건드림."""

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
# 심 home keyframe과 반드시 같아야 하는 값(piper_sort_return.xml의 "home" 키 qpos) - IK
# 시드가 여기서 안 어긋나야 조그 시작 시 팔이 안 튐. 2026-07-26: 사용자 요청으로 원점(전부
# 0도)으로 변경 - 물리 충돌 체크 완료(all-zero에서 팔 자체 충돌 없음, 블록-패드 접촉만
# 있음) - piper_sort_return.xml의 home keyframe도 같이 0으로 맞춰야 함(별도 수정).
_SIM_READY_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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


class PiperXRIntervention:
    """PICO 컨트롤러로 Piper 팔 하나를 조그 - PiperKeyboardIntervention과 동일 계약.

    flare의 BiPiperXRTeleop(양팔 하드코딩, connect() 시 왼팔·오른팔 IK를 둘 다 만듦 -
    실측 확인 2026-07-26, bi_piper_xr.py)를 포크하지 않고 그대로 재사용한다: side로 고른
    쪽의 PiperArmConfig에 이 sim의 URDF를 넣고, 반대쪽엔 같은 URDF를 더미로 채워
    connect()를 통과시킨 뒤 get_action()에서 고른 쪽의 관절만 뽑아 쓴다(반대쪽 IK는
    풀리지만 그냥 버림) - 클러치(grip)·그리퍼(trigger) 매핑은 flare 클래스가 이미
    구현하므로 여기서 새로 안 만든다. neck(Dynamixel)은 이 sim에 없는 하드웨어라
    enable_neck=False로 끈다."""

    # 연구실 실물 모션 트래커(팔뚝에 착용) 시리얼 - Cashier_policy/moai_policy 기본 설정과
    # 동일(2026-07-26 실측 확인: 이 PC의 PICO 서비스에 이 두 시리얼이 실제로 잡힘).
    _MOTION_TRACKER_SERIAL = {"left": "PC2310MLKB041941G", "right": "PC2310MLKB041978G"}

    def __init__(self, side="left", control_fps=30, gripper_open_pos=101.4,
                 gripper_close_pos=0.0, end_button="Y", axis_threshold=0.6):
        from flare.teleoperators.bi_piper_xr.bi_piper_xr import BiPiperXRTeleop
        from flare.teleoperators.bi_piper_xr.config_bi_piper_xr import (
            BiPiperXRTeleopConfig, MotionTrackerConfig, PiperArmConfig,
        )

        urdf_path = str(_PIPER_CAPSTONE_DIR / "urdf" / "piper_description.urdf")
        other_side = "right" if side == "left" else "left"
        arm_cfg = PiperArmConfig(
            side=side, pose_source=f"{side}_controller", control_trigger=f"{side}_grip",
            gripper_trigger=f"{side}_trigger", gripper_open_pos=gripper_open_pos,
            gripper_close_pos=gripper_close_pos, urdf_path=urdf_path,
            link_name="link6", joints_init=list(_SIM_READY_RAD),
            # 팔뚝 트래커로 팔꿈치(link3)도 실제 팔 자세를 따라가게 함(2026-07-26, 실측으로
            # 트래커를 착용하고도 안 켜져 있어 팔꿈치가 IK 자체 규칙대로만 움직이던 문제 발견).
            motion_tracker=MotionTrackerConfig(
                serial=self._MOTION_TRACKER_SERIAL[side], link_target="link3",
            ),
            # R_headset_world가 반사행렬(아래 주석)이라 flare의 quaternion conjugate 기반
            # 회전 매핑이 손대칭 뒤집힌 채로 나옴(실측: 위로 기울이면 반대로 회전) - 여긴
            # rotation_scale이 delta_rot에 그대로 곱해지는 걸 이용해 부호만 상쇄한다
            # (flare 코드 자체는 안 건드림, position_scale은 그대로 둬서 위치엔 영향 없음).
            rotation_scale=-1.0,
        )
        dummy_cfg = PiperArmConfig(
            side=other_side, pose_source=f"{other_side}_controller",
            control_trigger=f"{other_side}_grip", gripper_trigger=f"{other_side}_trigger",
            motion_tracker=None, urdf_path=urdf_path, link_name="link6",
            joints_init=list(_SIM_READY_RAD),
        )
        config = BiPiperXRTeleopConfig(
            dt=1.0 / control_fps,
            left_arm=arm_cfg if side == "left" else dummy_cfg,
            right_arm=dummy_cfg if side == "left" else arm_cfg,
            enable_neck=False,
            # 컨트롤러축->로봇 world축 매핑. flare 기본값(Cashier_policy 실물 로봇 기준)을
            # 그대로 뒀었는데(2026-07-26 최초 구현) 이 sim에서 좌우가 반대로 느껴진다는
            # 실측 피드백을 받아 좌우 담당 행(가운데 행)만 부호를 뒤집음 - 실측 확인:
            # 앞뒤·좌우 위치는 이걸로 맞음(2026-07-26). 이 행렬은 이제 반사(행렬식 -1)라
            # flare가 회전에 재사용하는 quaternion conjugate가 손대칭 뒤집힌 결과를 내는데,
            # 그건 위 rotation_scale=-1.0으로 따로 상쇄한다(행렬은 안 더 건드림).
            R_headset_world=[0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self._side = side
        self.teleop = BiPiperXRTeleop(config)
        self.teleop.connect(calibrate=False)

        import xrobotoolkit_sdk as xrt

        self.xrt = xrt  # collect.py/teleoperate.py가 A버튼(시점전환)을 직접 읽을 때 씀
        self._get_end = getattr(xrt, f"get_{end_button}_button")
        self._get_grip = getattr(xrt, f"get_{side}_grip")
        self._get_pose = getattr(xrt, f"get_{side}_controller_pose")
        self._get_axis = getattr(xrt, f"get_{side}_axis")
        self._axis_threshold = axis_threshold
        self._end_requested = False
        self._rerecord_requested = False
        self._stop_requested = False
        self._prev_end = False
        self._prev_axis_active = False
        self._step_count = 0  # 진단용(2026-07-26) - 시작하자마자 팔이 튀는 문제 원인 확인
        self._prev_joints_deg = None
        logger.info(
            f"[Piper PICO] 연결됨. {side} 컨트롤러=이동/그리퍼 · grip=클러치 · "
            f"스틱 오른쪽=에피소드 종료(저장) · 스틱 왼쪽=다시 녹화(버림) · "
            f"{end_button}=종료+세션 전체 중단 · A=시점전환"
        )

    def should_end(self):
        return self._end_requested

    def should_rerecord(self):
        """lerobot 관례의 '왼쪽 화살표'(다시 녹화) - True면 호출부가 이 에피소드를 버리고
        같은 인덱스로 재시도해야 한다."""
        return self._rerecord_requested

    def should_stop(self):
        """lerobot 관례의 'esc'(전체 세션 중단) - True면 호출부가 남은 에피소드를 건너뛰고
        루프를 끝내야 한다(2026-07-26: end_button을 이 역할로 승격 - 기존엔 이 에피소드만
        끝냈는데, 그 역할은 스틱 오른쪽이 대신 맡음)."""
        return self._stop_requested

    def reset(self):
        """에피소드 시작 시 호출됨(collect_episode 관례) - end 플래그뿐 아니라 Placo IK의
        내부 관절 상태도 home으로 되돌린다. 안 하면(2026-07-26 실측으로 발견한 버그) IK가
        직전 에피소드 종료 시점 자세를 계속 들고 있다가, env는 reset()으로 home에 가 있는데
        다음 스텝에 IK가 그 stale 자세를 액션으로 다시 내보내 "리셋됐다가 바로 이전 자세로
        튕겨 돌아가는" 것처럼 보인다.

        _prev_end는 False가 아니라 "지금 실제로 버튼이 눌려있는지"로 초기화한다 - 안 그러면
        (2026-07-26 실측으로 발견한 버그) 종료 버튼을 누른 채로 다음 에피소드가 시작될 때
        "새로 눌림"으로 오판해 바로 또 종료돼버린다(1프레임짜리 에피소드가 연달아 나옴).
        같은 이유로 _prev_axis_active도 현재 스틱 상태로 초기화한다. _stop_requested는 여기서
        안 지운다 - 세션 전체 중단 신호라 다음 에피소드로 넘어가면 안 되므로 호출부(루프)가
        직접 보고 끝내야 한다."""
        self._end_requested = False
        self._rerecord_requested = False
        self._prev_end = bool(self._get_end())
        self._prev_axis_active = abs(self._get_axis()[0]) > self._axis_threshold

    def __call__(self, step, obs_raw):
        end = bool(self._get_end())
        if end and not self._prev_end:
            self._end_requested = True
            self._stop_requested = True
        self._prev_end = end

        axis_x = self._get_axis()[0]
        axis_active = abs(axis_x) > self._axis_threshold
        if axis_active and not self._prev_axis_active:
            self._end_requested = True
            if axis_x < 0:
                self._rerecord_requested = True
        self._prev_axis_active = axis_active

        action = self.teleop.get_action()
        joints_deg = [action[f"{self._side}_joint{i}.pos"] for i in range(1, 7)]
        gripper = action[f"{self._side}_gripper.pos"]

        self._step_count += 1
        jump = (self._prev_joints_deg is not None
                and max(abs(a - b) for a, b in zip(joints_deg, self._prev_joints_deg)) > 15)
        if self._step_count <= 10 or jump or self._step_count % 15 == 0:
            tag = "점프!" if jump else "진단"
            logger.info(
                f"[Piper PICO][{tag}] step={self._step_count} grip={self._get_grip():.2f} "
                f"pose={[f'{v:.3f}' for v in self._get_pose()]} "
                f"joints_deg={[f'{j:.1f}' for j in joints_deg]}"
            )
        self._prev_joints_deg = joints_deg
        return np.array(joints_deg + [gripper], dtype=np.float32)

    def close(self):
        self.teleop.disconnect()


_TELEOP_DEVICES = {
    "keyboard": PiperKeyboardIntervention,
    "pico": PiperXRIntervention,
}


def _device_kwargs(device, cfg):
    """device별로 필요한 인자만 cfg에서 골라 뽑는다 - 새 기기 추가 시 여기 분기 하나만 늘면 됨."""
    if device == "pico":
        return dict(side=cfg.pico.side, end_button=cfg.pico.end_button)
    return {}


def make_piper_intervention(cfg):
    """cfg.intervention_device로 조작 기기 인스턴스를 만든다 - collect.py·teleoperate.py가
    공유(2026-07-26). env·뷰어와는 무관 - 뷰어는 PiperSortReturnEnv.attach_viewer()가 따로
    담당(호출부에서 필요하면 별도로 부를 것)."""
    device = cfg.intervention_device
    if device not in _TELEOP_DEVICES:
        raise ValueError(
            f"지원하지 않는 intervention_device={device!r} (piper task: {list(_TELEOP_DEVICES)})"
        )
    return _TELEOP_DEVICES[device](control_fps=cfg.control_fps, **_device_kwargs(device, cfg))


def make_camera_toggle_fn(env, intervention):
    """A버튼 누르면 env.cycle_camera() 호출(2026-07-26) - PICO 전용(intervention.xrt가 있을
    때만, 키보드는 해당 없어 None 반환). collect_episode()의 render_fn(obs_raw) 콜백에
    그대로 꽂는다 - 카메라 프리뷰(cv2/rerun)는 안 쓰지만 이 콜백 하나는 필요."""
    xrt = getattr(intervention, "xrt", None)
    if xrt is None:
        return None
    prev_a = [False]

    def render_fn(obs_raw):
        a_btn = bool(xrt.get_A_button())
        if a_btn and not prev_a[0]:
            env.cycle_camera()
        prev_a[0] = a_btn
        return True

    return render_fn
