"""G1 오른팔 단독 PICO 텔레옵 디버그 루프 — IK+PD 물리 루프를 실제 PICO로 움직이며
2026-07-19 새벽에 발견한 "6초 후 갑자기 불안정해짐" 버그를 재현·진단하기 위한 스크립트.

MuJoCo 뷰어 창이 뜨고(사람이 직접 봄), 콘솔/로그엔 매 컨트롤 스텝 진단 정보(팔 목표 대비
실제 오차, 손끝 위치, PD 토크 크기)를 찍는다 — Claude는 화면을 못 보므로 이 로그 + 사람의
설명으로 원인을 좁힌다.

조작: 오른쪽 컨트롤러 grip(클러치) 잡은 동안만 팔이 움직임(토글 없음). trigger=그리퍼(쥐기).
      Ctrl+C로 종료.

사용:
    python -m mani_sim.scripts.g1_teleop_debug
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from mani_sim.scripts.g1_arm_ik import ArmIK, RIGHT_ARM_JOINTS


class OneEuroFilter:
    """VR pose 지터 완화용 1-Euro 필터(pico_intervention.py에서 그대로 이식)."""

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

MJCF = "/home/moai/jungwook_ws/ljw_workspace/mani_sim_external/xr_teleoperate/assets/g1/scene_g1_hand_desk.xml"
URDF = "/home/moai/jungwook_ws/ljw_workspace/mani_sim_external/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"

HOLD_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]
# 2026-07-19 아침: Dex3 3지 -> Dex1-1 2지 그리퍼로 교체(구조적으로 서랍 손잡이를 못 쥘
# 것 같다는 실사용 판단). range min(-0.02)=열림, max(0.0245)=닫힘 확인됨.
RIGHT_HAND_JOINTS = ["right_gripper_joint1", "right_gripper_joint2"]
# 왼팔은 이번 디버그 대상 아님 -> 그냥 hold 취급(고정)해서 오른팔에만 집중
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
LEFT_HAND_JOINTS = [f"left_hand_{j}" for j in [
    "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
    "middle_0_joint", "middle_1_joint", "index_0_joint", "index_1_joint",
]]
FROZEN_JOINTS = HOLD_JOINTS + LEFT_ARM_JOINTS + LEFT_HAND_JOINTS

# sirius 기본값에 "사람이 로봇과 마주보는 방향"(앞뒤+좌우 반전, 위/z는 그대로) 보정을 얹음
# — 2026-07-19 아침 실기 확인: 컨트롤러를 앞으로 밀면 팔이 뒤로 뻗는 증상 -> 180도 뒤집힌 것으로 판단.
R_HEADSET_WORLD = (np.diag([-1, -1, 1]) @ np.array([0, 0, 1, 1, 0, 0, 0, 1, 0], dtype=float).reshape(3, 3))
DET_M = float(round(np.linalg.det(R_HEADSET_WORLD)))  # 반사(det=-1)면 회전 벡터(유사벡터) 부호 보정 필요
POS_SCALE = 1.0
GRIP_THRESHOLD = 0.9


def main():
    import xrobotoolkit_sdk as xrt
    xrt.init()
    print("[teleop] xrobotoolkit_sdk 초기화 완료")

    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ik = ArmIK(URDF, RIGHT_ARM_JOINTS, "right_hand_palm_link")

    all_names = FROZEN_JOINTS + RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS
    qpos_adrs = np.array([model.joint(n).qposadr[0] for n in all_names])
    dof_adrs = np.array([model.joint(n).dofadr[0] for n in all_names])

    # 2026-07-19 아침: motor+수동PD -> position 액추에이터(게인 내장)로 전환, 12초 합성
    # 테스트로 불안정 해소 확인됨. ctrl에 목표 qpos를 직접 넣으면 됨(토크 계산 불필요).
    drive_names = RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS
    drive_act_ids = np.array([model.actuator(n).id for n in drive_names])
    drive_slice = slice(len(FROZEN_JOINTS), len(all_names))

    frozen_qpos_adrs = np.array([model.joint(n).qposadr[0] for n in FROZEN_JOINTS])
    frozen_dof_adrs = np.array([model.joint(n).dofadr[0] for n in FROZEN_JOINTS])

    target_qpos = data.qpos[qpos_adrs].copy()
    arm_slice = slice(len(FROZEN_JOINTS), len(FROZEN_JOINTS) + len(RIGHT_ARM_JOINTS))
    hand_slice = slice(len(FROZEN_JOINTS) + len(RIGHT_ARM_JOINTS), len(all_names))
    hand_qi = np.array([model.joint(n).qposadr[0] for n in RIGHT_HAND_JOINTS])
    hand_lo = model.jnt_range[[model.joint(n).id for n in RIGHT_HAND_JOINTS], 0]
    hand_hi = model.jnt_range[[model.joint(n).id for n in RIGHT_HAND_JOINTS], 1]

    site_id = model.site("right_ee_site").id

    ref_pos = None
    ref_rot = None
    pos_filter = None
    rot_filter = None
    EURO_ARGS = (60.0, 1.0, 0.01, 1.0)  # (control_hz 근사, min_cutoff, beta, d_cutoff) — pico_intervention.py와 동일값
    target_se3 = ik.get_ee_pose(target_qpos[arm_slice])

    step_i = 0
    t_engage_start = None
    print("[teleop] B버튼 토글 없이, grip(클러치)만 잡으면 바로 팔이 따라감. trigger=그리퍼. Ctrl+C 종료.")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                data.qpos[frozen_qpos_adrs] = 0.0
                data.qvel[frozen_dof_adrs] = 0.0

                trigger = float(np.clip(xrt.get_right_trigger(), 0.0, 1.0))
                hand_target = hand_lo + trigger * (hand_hi - hand_lo)  # 균일 min/max 가정(미검증)
                target_qpos[hand_slice] = hand_target

                grip_val = float(xrt.get_right_grip())
                engaged = grip_val >= GRIP_THRESHOLD
                if engaged:
                    if t_engage_start is None:
                        t_engage_start = time.time()
                    pose = xrt.get_right_controller_pose()  # [x,y,z,qx,qy,qz,qw]
                    pos_pico_raw = np.asarray(pose[:3], dtype=float)
                    rot_pico_raw = Rotation.from_quat([pose[3], pose[4], pose[5], pose[6]])

                    if pos_filter is None:
                        pos_filter = OneEuroFilter(3, *EURO_ARGS)
                        rot_filter = OneEuroFilter(3, *EURO_ARGS)
                    pos_pico = pos_filter.filter(pos_pico_raw)
                    rot_pico = Rotation.from_rotvec(rot_filter.filter(rot_pico_raw.as_rotvec()))

                    if ref_pos is None:
                        ref_pos, ref_rot = pos_pico, rot_pico
                    dpos_pico = pos_pico - ref_pos
                    drot_pico = (rot_pico * ref_rot.inv()).as_rotvec()
                    ref_pos, ref_rot = pos_pico, rot_pico

                    delta_pos = (R_HEADSET_WORLD @ dpos_pico) * POS_SCALE
                    delta_rot = DET_M * (R_HEADSET_WORLD @ drot_pico)

                    target_se3.translation = target_se3.translation + delta_pos
                    dq_rot = Rotation.from_rotvec(delta_rot)
                    cur_rotmat = target_se3.rotation
                    new_rotmat = (dq_rot * Rotation.from_matrix(cur_rotmat)).as_matrix()
                    target_se3.rotation = new_rotmat
                else:
                    t_engage_start = None
                    ref_pos, ref_rot = None, None
                    pos_filter, rot_filter = None, None

                q_arm_new = ik.solve(target_se3, n_iter=5, dt=0.15, k_null=0.1)
                target_qpos[arm_slice] = q_arm_new
                # 방향 목표가 위치 수렴을 방해하는 문제(2026-07-19 아침 발견) 회피 —
                # 방향은 별도로 안 몰고 매 스텝 "지금 도달한 방향"을 목표로 다시 세팅
                target_se3.rotation = ik.get_ee_pose().rotation

                data.ctrl[drive_act_ids] = target_qpos[drive_slice]
                mujoco.mj_step(model, data)
                viewer.sync()

                step_i += 1
                if step_i % 50 == 0:  # ~0.2s마다
                    ee_now = data.site_xpos[site_id].copy()
                    err = np.linalg.norm(ee_now - target_se3.translation)
                    elapsed = f"{time.time() - t_engage_start:.1f}s" if t_engage_start else "-"
                    print(f"[diag] engaged={engaged} grip_raw={grip_val:.2f} trigger_raw={trigger:.2f} "
                          f"engage_elapsed={elapsed} "
                          f"ee={np.round(ee_now, 3)} target={np.round(target_se3.translation, 3)} "
                          f"err={err:.4f}", flush=True)
    finally:
        try:
            xrt.close()
        except Exception:
            pass
        print("[teleop] xrt 정리 완료, 종료")


if __name__ == "__main__":
    main()
