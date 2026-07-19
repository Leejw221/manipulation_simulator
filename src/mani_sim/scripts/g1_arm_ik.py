"""G1 팔 IK — Pinocchio 기반, damped least squares + nullspace 보정.

2026-07-19 아침 검증 상태:
- 단독 기구학 테스트(물리 없이): 8/5/5cm 이동 목표에 대해 1.5cm 이내로 수렴, 관절값 전부
  물리적으로 타당한 범위(joint limit 안). [검증]
- **"6초 후 불안정" 버그 원인 규명·해결됨**: IK 자체 문제가 아니었음(q_target은 항상
  안정적이었고 q_actual만 발산 — 진단으로 확인). 진짜 원인은 `<motor>` 액추에이터 위에
  Python에서 직접 계산한 explicit PD가 MuJoCo 물리 루프에서 구조적으로 불안정했던 것.
  **`<position>` 액추에이터(게인 내장, kp/kv를 XML에 직접 지정)로 전환하고 Python에선
  `data.ctrl = target_qpos`만 넣도록 바꾸니 12초 합성 테스트에서 완전히 안정화됨**(오차
  0.02~0.07m 유지, 발산 없음). [검증] 이 IK의 damping/dq-clip 개선(아래 solve() 참고,
  moai_policy PiPER 텔레옵 코드 참고해 추가)은 안전장치로는 유효하나 이 버그의 직접
  원인은 아니었음.
- MuJoCo의 `right_ee_site`(수동으로 잡은 손목 오프셋)와 Pinocchio의
  `right_hand_palm_link`(URDF 명명 프레임)가 완전히 같은 지점이 아님(초기 오차 ~5cm) —
  둘을 같은 지점으로 맞추거나, 한쪽으로 통일해야 함. [미해결]

사용 패턴(실시간 텔레옵 루프에서):
    ik = ArmIK(urdf_path, RIGHT_ARM_JOINTS, "right_hand_palm_link")
    ...매 컨트롤 스텝...
    target_se3 = ...PICO delta를 반영해 갱신한 목표 pose...
    q_arm = ik.solve(target_se3, n_iter=5)  # 워밈스타트(이전 결과에서 이어서) 소수 반복
    target_qpos[arm_slice] = q_arm  # PD 컨트롤러가 이 목표를 물리적으로 추종
"""

import numpy as np
import pinocchio as pin

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]


class ArmIK:
    """Pinocchio 기반 단일 팔 IK. 매 스텝 이전 해에서 이어서(워밈스타트) 소수 반복만 돌리는
    실시간 사용을 전제로 한다(풀 수렴까지 안 기다림 — PICO 목표가 계속 조금씩 움직이므로
    "계속 따라가는 것"이 목적이지 "한 번에 정확히 도달"이 목적이 아님).

    nullspace 항(k_null)이 없으면 7-DOF 중복 자유도 때문에 관절이 여러 바퀴 돌아
    물리적으로 말이 안 되는 해로 발산한다(2026-07-19 밤 실측) — 반드시 포함해야 함.
    """

    def __init__(self, urdf_path, arm_joint_names, ee_frame_name, base_z=0.793):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(ee_frame_name)
        self.qi_idx = np.array([self.model.joints[self.model.getJointId(j)].idx_q for j in arm_joint_names])
        self.v_idx = np.array([self.model.joints[self.model.getJointId(j)].idx_v for j in arm_joint_names])
        self.q_lower = self.model.lowerPositionLimit[self.qi_idx]
        self.q_upper = self.model.upperPositionLimit[self.qi_idx]

        self.q_full = pin.neutral(self.model)
        self.q_full[2] = base_z  # MuJoCo pelvis 기준 world z와 맞춤(고정 베이스 가정)
        self.q_ref_arm = self.q_full[self.qi_idx].copy()  # nullspace가 당기는 기준 자세

    def set_arm_qpos(self, q_arm):
        self.q_full[self.qi_idx] = q_arm

    def get_ee_pose(self, q_arm=None):
        """현재(또는 지정한) 팔 자세에서 ee 프레임의 world pose(SE3)."""
        if q_arm is not None:
            self.set_arm_qpos(q_arm)
        pin.forwardKinematics(self.model, self.data, self.q_full)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id].copy()

    def solve(self, target_se3, n_iter=5, dt=0.15, damp_min=1e-3, manip_threshold=5e-3,
              damp_max=1.0, max_dq=0.3, k_null=0.1):
        """target_se3로 향해 n_iter번 갱신. 내부 상태(self.q_full)에서 이어서 진행(워밈스타트).

        moai_policy(PiPER 텔레옵, 실기 검증된 구현)의 패턴을 따름 — 2026-07-19 아침, 우리
        구현이 물리 루프에서 몇 초 후 튀는 문제의 원인으로 확인된 두 가지를 보완:
        (1) damping을 고정값 대신 **manipulability 기반 적응형**으로(특이점 근처서 키움),
        (2) 반복당 관절 이동량(dq)에 **명시적 상한**(그 전엔 최종 qpos만 joint limit으로
        클립했지, 스텝 자체의 크기는 안 막아서 오차가 커지면 한 스텝에 폭주할 수 있었음).
        """
        q = self.q_full
        for _ in range(n_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.frame_id]
            dMf = oMf.actInv(target_se3)  # 공식 예제와 동일한 부호(현재.actInv(목표))
            err = pin.log6(dMf).vector

            J_full = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id, pin.LOCAL)
            J_full = -np.dot(pin.Jlog6(dMf.inverse()), J_full)  # log6 오차에 맞춘 보정(필수)
            J = J_full[:, self.v_idx]

            manip = np.sqrt(max(np.linalg.det(J.dot(J.T)), 0.0))
            damp = damp_min if manip > manip_threshold else damp_max

            J_pinv = J.T.dot(np.linalg.inv(J.dot(J.T) + damp * np.eye(6)))
            v_primary = -J_pinv.dot(err)
            n_dof = len(self.v_idx)
            N = np.eye(n_dof) - J_pinv.dot(J)
            v_null = N.dot(k_null * (self.q_ref_arm - q[self.qi_idx]))

            dq = np.clip((v_primary + v_null) * dt, -max_dq, max_dq)
            q[self.qi_idx] = np.clip(q[self.qi_idx] + dq, self.q_lower, self.q_upper)
        return q[self.qi_idx].copy()
