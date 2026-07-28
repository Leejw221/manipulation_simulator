"""PICO VR 컨트롤러를 개입 신호원으로 쓰는 intervention_fn.

`KeyboardIntervention`과 같은 계약(__call__ / should_end / reset / close)을 구현하므로
`collect_episode`에 그대로 끼워 쓴다. manipulation_pipeline(sirius)의 XR teleop 로직을
이 repo에 자립적으로 옮긴 것 — import 의존 없이 복사했다.

sirius와의 핵심 차이:
  sirius는 실물 PiPER를 관절공간(14-dim)으로 몰기 때문에 컨트롤러 pose → Placo IK로
  관절각을 푼다. 이 repo의 action space는 robosuite OSC_POSE(EE 6-DoF delta + gripper,
  7-dim, 팔 하나당)라 IK가 필요 없다 — 컨트롤러의 프레임간 이동/회전을 그대로 OSC delta로
  낸다. 또한 OSC delta 제어는 "engage 기준 절대 delta"가 아니라 "직전 프레임 대비 per-step
  delta"를 기대하므로(robosuite Keyboard/SpaceMouse device와 동일), 여기서도 프레임간
  delta로 만든다.

양팔 task(action_dim>7, 예: TwoArmTransport): 양손 다 쓴다(2026-07-26, 실측으로 왼손이
아예 배선 안 돼있던 걸 발견 - 원래 컨트롤러 하나만 처리하게 만들어져 있었음). 손마다
독립된 클러치(grip)·필터 상태를 갖고, 개입 on/off(B)는 양손 공용 - 켜져 있으면 각 손은
자기 grip을 잡고 있을 때만 그 손이 담당하는 팔이 움직이고, 안 잡으면 그 팔만 정지(그리퍼는
계속 반응). side_robot_index로 "어느 손이 어느 로봇 슬롯을 모는지" 정하고, 기본값은
실측 확인된 것(2026-07-26, TwoArmTransport 기준): left->로봇0(왼팔) · right->로봇1(오른팔).

조작:
  toggle_button(기본 B)     : 개입 on/off (정책 ↔ 사람, 양손 공용)
  grip(클러치, 손마다 독립)  : 잡고 있는 동안만 그 손이 담당하는 팔이 움직인다. 놓으면 정지
                              → 손을 편한 위치로 옮긴 뒤 다시 잡으면 점프 없이 이어짐
  trigger(손마다 독립)       : 그리퍼 (뗀 상태=열림, 당기면 닫힘)
  end_button(기본 Y)        : 현재 에피소드 종료(저장)
  오른손 스틱 왼쪽(x<-0.8)   : 진행 중인 에피소드 폐기하고 즉시 재시도(저장 안 함).
  오른손 스틱 오른쪽(x>0.8)  : 진행 중인 에피소드 폐기 + **직전에 이미 저장된 에피소드도
                              목록에서 삭제**하고 재시도 - 개입을 잘못했다는 걸 한 에피소드
                              늦게 깨달았을 때용(2026-07-27, 사용자 결정). moai_policy(실물)의
                              "오른쪽=저장+reset mode"는 시뮬엔 reset mode 자체가 없고
                              end_button과도 중복이라 다른 의미로 재배정함 - "왼쪽=폐기"만
                              원래 컨벤션 그대로. collect.py가 should_rerecord()/
                              should_delete_previous()를 보고 실제 목록 조작을 담당한다
                              (should_rerecord는 2026-07-26 Piper용으로 만들어둔 자리).

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


class _HandState:
    """손 하나(left|right)의 클러치 앵커·필터 상태 + xrt 게터. 양팔 개입에서 손마다
    독립적으로 관리해야 하는 것들을 여기 모아둔다."""

    def __init__(self, xrt, side):
        self.get_pose = getattr(xrt, f"get_{side}_controller_pose")
        self.get_grip = getattr(xrt, f"get_{side}_grip")
        self.get_trigger = getattr(xrt, f"get_{side}_trigger")
        self.reset()

    def reset(self):
        self.ref_pos = None
        self.ref_rot = None
        self.pos_filter = None
        self.rot_filter = None


class PICOIntervention:
    """PICO 컨트롤러를 읽어 OSC_POSE action(팔당 7-dim)을 만드는 intervention_fn.

    개입 off면 None(정책 실행)을, 개입 on이면 매 스텝 action을 반환한다. 개입 on이라도
    손마다 독립적으로 클러치(grip)를 놓고 있으면 그 손이 담당하는 팔은 정지(그리퍼만 활성).

    주의: `xrobotoolkit_sdk`(PICO 앱과 통신하는 외부 SDK)와 실제 PICO 연결이 필요하다.
    헤드리스/SDK 미설치 환경에선 import 시점이 아니라 생성 시점에 실패한다(지연 import).
    """

    def __init__(
        self,
        raw_env,
        side_robot_index=None,
        pos_scale=1.0,
        rot_scale=1.0,
        gripper_sign=1.0,
        toggle_button="B",
        end_button="Y",
        R_headset_world=(0, 0, 1, 1, 0, 0, 0, 1, 0),
        grip_threshold=0.9,
        control_hz=20.0,
        euro_min_cutoff=1.0,
        euro_beta=0.01,
        euro_d_cutoff=1.0,
        rerecord_stick_threshold=0.8,
    ):
        import xrobotoolkit_sdk as xrt

        self.xrt = xrt
        xrt.init()

        self.raw_env = raw_env
        self.arm_dim = 7
        self.full_action_dim = raw_env.action_dim
        self.bimanual = self.full_action_dim > self.arm_dim

        # side_robot_index: 어느 손이 어느 로봇 슬롯(0|1)을 모는지. 단일팔이면 오른손 하나만
        # (기존 동작 유지), 양팔이면 실측 확인된 매핑(2026-07-26, TwoArmTransport 기준
        # robot_index=0이 시각적으로 왼팔이었음) - 다른 task에서 반대면 override할 것.
        if side_robot_index is None:
            side_robot_index = {"left": 0, "right": 1} if self.bimanual else {"right": 0}
        self.side_robot_index = dict(side_robot_index)
        self._hands = {side: _HandState(xrt, side) for side in self.side_robot_index}

        self._get_toggle = getattr(xrt, f"get_{toggle_button}_button")
        self._get_end = getattr(xrt, f"get_{end_button}_button")
        self._get_stick = xrt.get_right_axis  # 재녹화 제스처(moai_policy와 동일하게 오른손 스틱)
        self.rerecord_stick_threshold = rerecord_stick_threshold

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
        self.rerecord_requested = False
        self.delete_previous_requested = False
        self._prev_toggle = False
        self._prev_end = False
        self._prev_stick_left = False   # 엣지 감지(홀드 중 재트리거 방지)
        self._prev_stick_right = False

    def _reset_clutch(self, side=None):
        """클러치(참조 pose·필터) 초기화 — 다음 grip 인게이지 때 점프 없이 재앵커.
        side=None이면 양손 다."""
        for s in ([side] if side else list(self._hands)):
            self._hands[s].reset()

    def should_end(self):
        return self.end_requested

    def should_rerecord(self):
        return self.rerecord_requested

    def should_delete_previous(self):
        return self.delete_previous_requested

    def reset(self):
        """에피소드 시작 시 개입/종료 상태 초기화."""
        self.intervening = False
        self.end_requested = False
        self.rerecord_requested = False
        self.delete_previous_requested = False
        self._prev_toggle = False
        self._prev_end = False
        self._prev_stick_left = False
        self._prev_stick_right = False
        self._reset_clutch()

    def _compute_arm_action(self, hand):
        trigger = float(np.clip(hand.get_trigger(), 0.0, 1.0))
        gripper = self.gripper_sign * (2.0 * trigger - 1.0)

        # 클러치 놓음 → 정지(그리퍼만). 참조를 비워 재인게이지 시 새로 앵커.
        if float(hand.get_grip()) < self.grip_threshold:
            hand.reset()
            return np.array([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32)

        pose = hand.get_pose()  # [x,y,z,qx,qy,qz,qw], PICO 프레임
        pos_pico = np.asarray(pose[:3], dtype=float)
        rot_pico = Rotation.from_quat([pose[3], pose[4], pose[5], pose[6]])

        if hand.pos_filter is None:
            hand.pos_filter = OneEuroFilter(3, *self._euro)
            hand.rot_filter = OneEuroFilter(3, *self._euro)
        f_pos = hand.pos_filter.filter(pos_pico)
        f_rot = Rotation.from_rotvec(hand.rot_filter.filter(rot_pico.as_rotvec()))

        # PICO 프레임에서 프레임간(per-step) delta — OSC_POSE delta 제어가 기대하는 형태
        if hand.ref_pos is None:
            dpos_pico = np.zeros(3)
            drot_pico = np.zeros(3)
        else:
            dpos_pico = f_pos - hand.ref_pos
            drot_pico = (f_rot * hand.ref_rot.inv()).as_rotvec()
        hand.ref_pos = f_pos
        hand.ref_rot = f_rot

        # PICO → 로봇 world 축 매핑. 회전 벡터는 유사벡터라 반사(det=-1) 시 부호를 보정.
        delta_pos = (self.M @ dpos_pico) * self.pos_scale
        delta_rot = (self.det_M * (self.M @ drot_pico)) * self.rot_scale

        # 추적 글리치로 튀는 것 방지 (OSC 컨트롤러도 내부 클리핑하지만 여기서 한 번 더)
        action = np.concatenate([delta_pos, delta_rot, [gripper]]).astype(np.float32)
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        return action

    def __call__(self, step, obs_raw):
        # 버튼 엣지 감지 (collect_episode가 매 스텝 이 함수를 부르므로 여기서 폴링, 양손 공용)
        toggle = bool(self._get_toggle())
        end = bool(self._get_end())
        if toggle and not self._prev_toggle:
            self.intervening = not self.intervening
            self._reset_clutch()
        if end and not self._prev_end:
            self.end_requested = True
        self._prev_toggle = toggle
        self._prev_end = end

        # 오른손 스틱(둘 다 진행 중인 에피소드는 저장 안 하고 즉시 종료+재시도):
        #   왼쪽  : 그냥 재시도
        #   오른쪽: 재시도 + 직전에 이미 저장된 에피소드도 목록에서 삭제(collect.py가 처리)
        # moai_policy(실물)의 "오른쪽=저장+reset mode"는 시뮬엔 reset mode 자체가 없고
        # end_button과도 중복이라 다른 의미로 재배정함(2026-07-27, 사용자 결정).
        # 홀드 중 반복 트리거 방지를 위해 "이미 임계값을 넘어있던 상태"를 엣지로만 감지한다.
        stick_x = float(self._get_stick()[0])
        stick_left = stick_x < -self.rerecord_stick_threshold
        stick_right = stick_x > self.rerecord_stick_threshold
        if stick_left and not self._prev_stick_left:
            self.end_requested = True
            self.rerecord_requested = True
        if stick_right and not self._prev_stick_right:
            self.end_requested = True
            self.rerecord_requested = True
            self.delete_previous_requested = True
        self._prev_stick_left = stick_left
        self._prev_stick_right = stick_right

        if not self.intervening:
            return None

        if not self.bimanual:
            side, robot_index = next(iter(self.side_robot_index.items()))
            return self._compute_arm_action(self._hands[side])

        full = np.zeros(self.full_action_dim, dtype=np.float32)
        for side, robot_index in self.side_robot_index.items():
            arm_action = self._compute_arm_action(self._hands[side])
            start = robot_index * self.arm_dim
            full[start:start + self.arm_dim] = arm_action
        return full

    def close(self):
        try:
            self.xrt.close()
        except Exception:
            pass
