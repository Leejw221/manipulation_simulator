"""PICO VR 컨트롤러를 개입 신호원으로 쓰는 intervention_fn.

`KeyboardIntervention`과 같은 계약(__call__ / should_end / reset / close)을 구현하므로
`collect_episode`에 그대로 끼워 쓴다. manipulation_pipeline(sirius)의 XR teleop 로직을
이 repo에 자립적으로 옮긴 것 — import 의존 없이 복사했다.

sirius와의 핵심 차이:
  sirius는 실물 PiPER를 관절공간(14-dim)으로 몰기 때문에 컨트롤러 pose → Placo IK로
  관절각을 푼다. 이 repo의 action space는 robosuite OSC_POSE(EE 6-DoF delta + gripper,
  7-dim)라 IK가 필요 없다 — 컨트롤러의 프레임간 이동/회전을 그대로 OSC delta로 낸다.
  또한 OSC delta 제어는 "engage 기준 절대 delta"가 아니라 "직전 프레임 대비 per-step
  delta"를 기대하므로(robosuite Keyboard/SpaceMouse device와 동일), 여기서도 프레임간
  delta로 만든다.

조작:
  toggle_button(기본 B, 우 컨트롤러) : 개입 on/off (정책 ↔ 사람)
  grip(클러치)                        : 잡고 있는 동안만 팔이 움직인다. 놓으면 정지 →
                                        손을 편한 위치로 옮긴 뒤 다시 잡으면 점프 없이 이어짐
  trigger                             : 그리퍼 (뗀 상태=열림, 당기면 닫힘)
  end_button(기본 Y, 좌 컨트롤러)     : 현재 에피소드 종료

보정 노브(실기에서 맞춰야 함):
  R_headset_world : PICO 헤드셋 → 로봇 base 축 대응(3x3). robosuite Panda 프레임은
                    sirius의 PiPER와 다를 수 있어 sirius 기본값을 출발점으로만 쓴다.
  pos_scale / rot_scale : 컨트롤러 이동/회전 → OSC 정규화 단위([-1,1]) 감도.
  gripper_sign : Panda 그리퍼 부호(+1=닫힘 가정). 반대로 움직이면 -1로 뒤집는다.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class OneEuroFilter:
    """VR pose 지터 완화용 1-Euro 필터 (sirius bi_piper_xr에서 그대로 옮김)."""

    def __init__(self, dim, rate, min_cutoff, beta, d_cutoff):
        self.rate = rate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = np.zeros(dim)

    def _alpha(self, cutoff):
        te = 1.0 / self.rate
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        x = np.asarray(x, dtype=float)
        if self.x_prev is None:
            self.x_prev = x.copy()
            return x.copy()
        a_d = self._alpha(self.d_cutoff)
        dx = (x - self.x_prev) * self.rate
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = np.array([self._alpha(c) for c in cutoff])
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        return x_hat


class PICOIntervention:
    """PICO 컨트롤러를 읽어 7-dim OSC_POSE action을 만드는 intervention_fn.

    개입 off면 None(정책 실행)을, 개입 on이면 매 스텝 action을 반환한다. 개입 on이라도
    클러치(grip)를 놓고 있으면 정지 action(그리퍼만 활성)을 낸다.

    주의: `xrobotoolkit_sdk`(PICO 앱과 통신하는 외부 SDK)와 실제 PICO 연결이 필요하다.
    헤드리스/SDK 미설치 환경에선 import 시점이 아니라 생성 시점에 실패한다(지연 import).
    """

    def __init__(
        self,
        raw_env,
        side="right",
        pos_scale=1.0,
        rot_scale=1.0,
        gripper_sign=1.0,
        toggle_button="B",
        end_button="Y",
        R_headset_world=(0, 0, -1, 1, 0, 0, 0, 1, 0),
        grip_threshold=0.9,
        control_hz=20.0,
        euro_min_cutoff=1.0,
        euro_beta=0.01,
        euro_d_cutoff=1.0,
    ):
        import xrobotoolkit_sdk as xrt

        self.xrt = xrt
        xrt.init()

        self.raw_env = raw_env
        self._get_pose = getattr(xrt, f"get_{side}_controller_pose")
        self._get_grip = getattr(xrt, f"get_{side}_grip")
        self._get_trigger = getattr(xrt, f"get_{side}_trigger")
        self._get_toggle = getattr(xrt, f"get_{toggle_button}_button")
        self._get_end = getattr(xrt, f"get_{end_button}_button")

        # 위치는 부호 반전(좌우 미러) 포함 signed permutation을 허용한다(det=-1 가능).
        # 회전 벡터는 유사벡터라, 반사일 때 det 부호를 곱해 축 매핑을 일관되게 만든다.
        self.M = np.asarray(R_headset_world, dtype=float).reshape(3, 3)
        self.det_M = float(round(np.linalg.det(self.M)))  # +1 회전 / -1 반사
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.gripper_sign = gripper_sign
        self.grip_threshold = grip_threshold
        self._euro = (control_hz, euro_min_cutoff, euro_beta, euro_d_cutoff)

        self.intervening = False
        self.end_requested = False
        self._prev_toggle = False
        self._prev_end = False
        self._reset_clutch()

    def _reset_clutch(self):
        """클러치(참조 pose·필터) 초기화 — 다음 grip 인게이지 때 점프 없이 재앵커."""
        self._ref_pos = None
        self._ref_rot = None
        self._pos_filter = None
        self._rot_filter = None

    def should_end(self):
        return self.end_requested

    def reset(self):
        """에피소드 시작 시 개입/종료 상태 초기화."""
        self.intervening = False
        self.end_requested = False
        self._prev_toggle = False
        self._prev_end = False
        self._reset_clutch()

    def __call__(self, step, obs_raw):
        # 버튼 엣지 감지 (collect_episode가 매 스텝 이 함수를 부르므로 여기서 폴링)
        toggle = bool(self._get_toggle())
        end = bool(self._get_end())
        if toggle and not self._prev_toggle:
            self.intervening = not self.intervening
            self._reset_clutch()
        if end and not self._prev_end:
            self.end_requested = True
        self._prev_toggle = toggle
        self._prev_end = end

        if not self.intervening:
            return None

        trigger = float(np.clip(self._get_trigger(), 0.0, 1.0))
        gripper = self.gripper_sign * (2.0 * trigger - 1.0)

        # 클러치 놓음 → 정지(그리퍼만). 참조를 비워 재인게이지 시 새로 앵커.
        if float(self._get_grip()) < self.grip_threshold:
            self._reset_clutch()
            return np.array([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32)

        pose = self._get_pose()  # [x,y,z,qx,qy,qz,qw], PICO 프레임
        pos_pico = np.asarray(pose[:3], dtype=float)
        rot_pico = Rotation.from_quat([pose[3], pose[4], pose[5], pose[6]])

        if self._pos_filter is None:
            self._pos_filter = OneEuroFilter(3, *self._euro)
            self._rot_filter = OneEuroFilter(3, *self._euro)
        f_pos = self._pos_filter.filter(pos_pico)
        f_rot = Rotation.from_rotvec(self._rot_filter.filter(rot_pico.as_rotvec()))

        # PICO 프레임에서 프레임간(per-step) delta — OSC_POSE delta 제어가 기대하는 형태
        if self._ref_pos is None:
            dpos_pico = np.zeros(3)
            drot_pico = np.zeros(3)
        else:
            dpos_pico = f_pos - self._ref_pos
            drot_pico = (f_rot * self._ref_rot.inv()).as_rotvec()
        self._ref_pos = f_pos
        self._ref_rot = f_rot

        # PICO → 로봇 world 축 매핑. 회전 벡터는 유사벡터라 반사(det=-1) 시 부호를 보정.
        delta_pos = (self.M @ dpos_pico) * self.pos_scale
        delta_rot = (self.det_M * (self.M @ drot_pico)) * self.rot_scale

        # 추적 글리치로 튀는 것 방지 (OSC 컨트롤러도 내부 클리핑하지만 여기서 한 번 더)
        action = np.concatenate([delta_pos, delta_rot, [gripper]]).astype(np.float32)
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        return action

    def close(self):
        try:
            self.xrt.close()
        except Exception:
            pass
