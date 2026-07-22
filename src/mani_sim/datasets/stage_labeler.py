"""Square(NutAssemblySquare) 데모의 프레임별 stage heuristic 라벨링.

1차 stage feasibility 실험용 수동 heuristic 라벨러. obs만으로 계산되므로 학습 데이터
라벨링과 롤아웃 실시간 stage 주입에 같은 함수를 쓴다(별도 예측기 불필요).

핵심 신호 = eef 속도(좌표변화). align 구간은 "속도가 near-zero로 떨어지는 감속 지점"으로
잡는다(근접·최고점이 아니라). Square PH 200개 실측:
  approach-A(0.005~) → align-A(speed<0.0015, 너트 앞) → grasp(sep↓) → lift(nut_z↑) →
  approach-B(peg로 수평) → align-B(speed<0.0015, peg 위) → insert(nut_z↓) → release(sep↑)

경계 앵커:
  LIFT   nut_z > 0.91 (테이블 z≈0.89)
  GRASP  finger_sep(=qpos0-qpos1) < 0.055 & 너트 근접
  ALIGN  speed < 0.0015 (감속) — A=너트 근접, B=peg 근접
  PLACE  peg([0.230,0.101]) 근접 후 nut_z 최고점 지나 하강
  RELEASE finger_sep > 0.06
순서는 advance-only로 강제.

stage(7): approach_A → align_A → grasp → lift → approach_B → insert → release
  데이터 판정(200개 실측): align_A(너트 앞 감속)는 뚜렷해서 살렸으나, align_B(peg 위
  감속)는 0.6%로 죽음 — peg 쪽은 감속과 하강이 동시라 별도 구간이 안 생겨 insert에 흡수.

object obs(NutAssemblySquare, 14): [0:3]nut_to_eef_pos·[3:7]quat·[7:10]nut_pos·[10:14]quat
"""

import numpy as np

STAGE_NAMES = [
    "approach_A", "align_A", "grasp", "lift", "approach_B", "insert", "release",
]
NUM_STAGES = len(STAGE_NAMES)

PEG_XY = np.array([0.2298, 0.1006])
LIFT_Z = 0.91
GRASP_SEP = 0.055     # 이 값 아래 = 그리퍼 닫힘(잡는 중)
OPEN_SEP = 0.06       # 이 값 위 = 열림
NEAR_NUT = 0.09       # nut_to_eef 근접(잡기/정렬 판정용)
PEG_R = 0.05          # 너트 xy가 peg 근접(align_B/insert 판정용)
V_SLOW = 0.0015       # eef 프레임당 이동 — 이 값 아래 = 감속(align)


def _boundaries(nut_pos, nut_to_eef, finger_sep, speed):
    """경계 [t_alignA, t_grasp, t_lift, t_apB, t_insert, t_release] (advance-only)."""
    T = len(nut_pos)
    nz = nut_pos[:, 2]
    idx = np.arange(T)

    def first(mask, default):
        w = np.where(mask)[0]
        return int(w[0]) if len(w) else default

    # crisp 이벤트
    t_grasp = first((finger_sep < GRASP_SEP) & (nut_to_eef < NEAR_NUT), T)
    t_lift = first((idx >= t_grasp) & (nz > LIFT_Z), T)
    t_release = first((idx > t_lift) & (finger_sep > OPEN_SEP), T)

    # align-A: grasp 직전 너트 근처에서 감속 시작 (속도 기반)
    t_alignA = first((idx < t_grasp) & (nut_to_eef < NEAR_NUT) & (speed < V_SLOW), t_grasp)

    # peg 쪽: 최고점 지나 하강 시작 = insert (align_B는 데이터상 안 잡혀 insert에 흡수)
    t_peak = t_lift + int(np.argmax(nz[t_lift:t_release])) if t_lift < t_release else t_lift
    t_peak = min(t_peak, T - 1)  # lift 미검출 데모에서 t_peak=T 방지
    t_insert = first((idx > t_peak) & (nz < nz[t_peak] - 0.015), t_release)
    # lift → approach-B: 너트가 목표 높이(구간 최고점 근처)에 도달한 시점
    t_apB = first((idx > t_lift) & (nz > nz[t_peak] - 0.02), t_insert)

    ts = [t_alignA, t_grasp, t_lift, t_apB, t_insert, t_release]
    for i in range(1, len(ts)):
        ts[i] = max(ts[i], ts[i - 1])
    return ts


def label_stages_square(obs):
    """obs(dict of numpy) → (T,) int stage. 필요 키: object, robot0_eef_pos, robot0_gripper_qpos."""
    obj = np.asarray(obs["object"], dtype=np.float64)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    T = len(obj)

    nut_to_eef = np.linalg.norm(obj[:, 0:3], axis=1)
    nut_pos = obj[:, 7:10]
    finger_sep = grip[:, 0] - grip[:, 1]
    speed = np.concatenate([[0.0], np.linalg.norm(np.diff(eef, axis=0), axis=1)])

    b = _boundaries(nut_pos, nut_to_eef, finger_sep, speed)
    t_alignA, t_grasp, t_lift, t_apB, t_insert, t_release = b

    stage = np.zeros(T, dtype=np.int64)
    stage[:t_alignA] = 0        # approach_A
    stage[t_alignA:t_grasp] = 1  # align_A
    stage[t_grasp:t_lift] = 2    # grasp
    stage[t_lift:t_apB] = 3      # lift
    stage[t_apB:t_insert] = 4    # approach_B
    stage[t_insert:t_release] = 5  # insert
    stage[t_release:] = 6        # release
    return stage


def onehot(stage, num=NUM_STAGES):
    oh = np.zeros((len(stage), num), dtype=np.float32)
    oh[np.arange(len(stage)), stage] = 1.0
    return oh


def stage_alpha_bar(stage_by_demo, num):
    """SARM(arXiv:2509.25358) Eq.1 — ᾱ_k = (1/M)Σᵢ(L_{i,k}/Tᵢ), 데이터셋 전체 M개 데모에서
    stage k가 차지하는 시간비율의 평균[검증-원문, xdofai/opensarm 및 lerobot 포크
    compute_temporal_proportions와 대조 확인, 2026-07-22]. stage_by_demo: {demo_id: stage(T,) int}.
    반환: (num,) float, sum=1."""
    ratio_sum = np.zeros(num, dtype=np.float64)
    M = len(stage_by_demo)
    for stage in stage_by_demo.values():
        T = len(stage)
        counts = np.bincount(np.asarray(stage), minlength=num)
        ratio_sum += counts / T
    alpha_bar = ratio_sum / M
    return (alpha_bar / alpha_bar.sum()).astype(np.float32)  # 수치오차 방지용 재정규화


def stage_tau(stage, num=NUM_STAGES):
    """stage(T,) int(0..num-1, 단조증가) → tau(T,) float[0,1) — stage 내 상대위치
    (SARM Eq.2의 τ_t, alpha_bar와 무관한 순수 지역 위치). stage_progress()와 stage_alpha_bar()
    양쪽에서 재사용(구간 경계 계산은 동일 로직)."""
    stage = np.asarray(stage, dtype=np.int64)
    T = len(stage)
    starts = np.searchsorted(stage, np.arange(num), side="left")  # stage 단조증가라 유효
    ends = np.append(starts[1:], T)

    tau = np.zeros(T, dtype=np.float32)
    for s in range(num):
        s0, s1 = int(starts[s]), int(ends[s])
        if s1 <= s0:
            continue
        tau[s0:s1] = (np.arange(s0, s1) - s0) / max(s1 - s0, 1)
    return tau


def stage_progress(stage, num=NUM_STAGES, alpha_bar=None):
    """stage(T,) int(0..num-1, 단조증가) → stage-aware progress(T,) float[0,1), 단조증가.

    SARM Eq.2 그대로: y_t = P_{k-1} + ᾱ_k·τ_t (P_k=Σᵢ≤k ᾱ_i, τ_t=stage 내 상대위치).
    alpha_bar: (num,) — stage_alpha_bar()로 데이터셋 전체에서 미리 계산해 넘길 것. None이면
    균등(1/num)으로 근사(구버전 호환용 fallback — 원문과 다름, RQ1 진단 등 급할 때만 사용,
    실제 학습 타깃 생성에는 항상 alpha_bar를 계산해서 넘길 것[2026-07-22 정정]).
    """
    stage = np.asarray(stage, dtype=np.int64)
    if alpha_bar is None:
        alpha_bar = np.full(num, 1.0 / num, dtype=np.float32)
    cum = np.concatenate([[0.0], np.cumsum(alpha_bar)]).astype(np.float32)  # P_0..P_num(=1)
    tau = stage_tau(stage, num=num)
    return cum[stage] + alpha_bar[stage] * tau


STAGE_NAMES_DOOR = ["approach", "align", "pull_open", "push_close", "withdraw", "return"]
NUM_STAGES_DOOR = len(STAGE_NAMES_DOOR)

# door 배치 랜덤화 폭이 작아서(x:[0.07,0.09],y:[-0.01,0.01] 오프셋) 고정 근사값으로 충분
# (2026-07-18 실측 handle_pos 여러 번: 대략 x=-0.13,y=-0.25,z=1.075).
NOMINAL_HANDLE_POS = np.array([-0.13, -0.25, 1.075])
NEAR_HANDLE = 0.08
RETURN_DIST = 0.08        # DoorCabinet._check_success()의 RETURN_DIST_THRESHOLD와 동일
MIN_RUN = 8                # 노이즈 방지: 이 프레임 수 이상 연속 증가/감소해야 극점으로 인정


def _smooth(x, w=5):
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    pad = w // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def _next_local_min(x, start, end):
    """[start,end) 안에서 x가 감소하다 증가로 바뀌는 첫 지점(지역 최소)."""
    for t in range(start + MIN_RUN, end - MIN_RUN):
        if x[t] <= x[t - MIN_RUN] and x[t] <= x[t + MIN_RUN] and x[t] < x[start]:
            # 이 지점 이후로도 계속 오르는지(진짜 극소인지) 확인
            if x[min(t + MIN_RUN, end - 1)] > x[t]:
                return t
    return int(start + np.argmin(x[start:end])) if end > start else start


def _next_local_max(x, start, end):
    for t in range(start + MIN_RUN, end - MIN_RUN):
        if x[t] >= x[t - MIN_RUN] and x[t] >= x[t + MIN_RUN] and x[t] > x[start]:
            if x[min(t + MIN_RUN, end - 1)] < x[t]:
                return t
    return int(start + np.argmax(x[start:end])) if end > start else start


def _boundaries_door(eef, t_intv_start):
    """[t_align, t_min1, t_peak, t_min2, t_return] (advance-only).

    실측(2026-07-18, 30 에피소드): 그리퍼를 닫지 않고(열린 채로 3지 그리퍼로 손잡이를
    걸어서) 밀고 당김 - action[6](그리퍼 채널)이 개입 내내 -1(open) 고정이라 grasp
    상태로는 구간을 못 나눔(첫 시도, 버그로 확인됨). 대신 "고정 손잡이 위치까지 거리"가
    감소(접근/열기 위해 당김)->증가(문이 열리며 손잡이가 원위치에서 멀어짐)->감소(밀어서
    닫으며 손잡이가 원위치로 복귀)->증가(놓고 복귀)의 뚜렷한 4구간 파동을 그린다 - 이
    파동의 극점(지역 최소/최대)으로 경계를 나눈다.

    t_intv_start 이전(개입 전 zero-action stub 구간, 로봇 정지)은 탐색에서 제외한다.
    """
    T = len(eef)
    dist_handle = _smooth(np.linalg.norm(eef - NOMINAL_HANDLE_POS, axis=1))
    initial_eef_pos = eef[0]

    t_min1 = _next_local_min(dist_handle, t_intv_start, T)   # 첫 접근(당기기 시작점)
    t_peak = _next_local_max(dist_handle, t_min1, T)          # 문 최대로 열린 시점
    t_min2 = _next_local_min(dist_handle, t_peak, T)          # 밀어서 다시 손잡이 근접

    # align: t_min1 직전, 손잡이 근접권(NEAR_HANDLE)에 처음 들어온 시점
    idx = np.arange(T)
    w = np.where((idx >= t_intv_start) & (idx < t_min1) & (dist_handle < NEAR_HANDLE))[0]
    t_align = int(w[0]) if len(w) else t_min1

    # withdraw 끝(복귀 시작): t_min2 이후 초기 위치까지 거리가 넉넉히 가까워지는 시점
    dist_initial = np.linalg.norm(eef - initial_eef_pos, axis=1)
    w2 = np.where((idx > t_min2) & (dist_initial < RETURN_DIST * 2))[0]
    t_return = int(w2[0]) if len(w2) else T

    ts = [t_align, t_min1, t_peak, t_min2, t_return]
    for i in range(1, len(ts)):
        ts[i] = max(ts[i], ts[i - 1])
    return ts


def label_stages_door_cabinet(obs, actions, action_mode):
    """obs(dict) + actions(T,7, 미사용) + action_mode(T,) -> (T,) int stage. 필요 키: robot0_eef_pos.

    actions는 인터페이스 호환을 위해 받지만 안 쓴다(그리퍼 채널이 이 데이터에서 신호가
    없어서 - 위 _boundaries_door docstring 참고). action_mode: datasets/labels.py의
    LABEL_INTV(1)/LABEL_ROLLOUT(0)/LABEL_PREINTV(-10) 스킴 - 개입 시작 전 구간을 탐색에서
    제외한다.
    """
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    mode = np.asarray(action_mode)
    T = len(eef)

    intv_idx = np.where(mode == 1)[0]  # LABEL_INTV
    t_intv_start = int(intv_idx[0]) if len(intv_idx) else 0

    t_align, t_min1, t_peak, t_min2, t_return = _boundaries_door(eef, t_intv_start)

    stage = np.zeros(T, dtype=np.int64)
    stage[:t_align] = 0          # approach
    stage[t_align:t_min1] = 1    # align
    stage[t_min1:t_peak] = 2     # pull_open
    stage[t_peak:t_min2] = 3     # push_close
    stage[t_min2:t_return] = 4   # withdraw
    stage[t_return:] = 5         # return
    return stage


STAGE_NAMES_TRANSPORT = [
    "lid_present", "lid_removed", "payload_approach", "payload_handoff", "payload_delivered",
]
NUM_STAGES_TRANSPORT = len(STAGE_NAMES_TRANSPORT)
# 2026-07-21 밤 trash_delivered 제거([검증-실측], 200demo 온라인/오프라인 경계 비교):
# 순수 그리퍼 이벤트 경계(lid_removed·payload_handoff)는 online-offline lag 거의 0(평균
# -0.2f/0f)인데, trash_delivered(ground-truth flag)는 최대 -140프레임까지 어긋남 — 신호
# 자체 문제가 아니라 "두 팔이 겹칠 때(28.5% demo) offline이 trash_done을 lid_removed
# 이후로 강제 clamp하지만 online은 즉시 반응"하는 게 원인(EXP-07의 stage mismatch와
# 동일 유형). robot1의 trash 처리는 로봇0 관점의 payload_approach 구간에 자연히 흡수됨.

# object obs(TwoArmTransport, 41) 레이아웃 — 2026-07-19 실측 확정([검증-실측], robosuite
# TwoArmTransport의 raw named sensor 값과 object-state 벡터를 슬라이딩윈도우로 대조해 offset
# 확인, 추측 아님):
#   0:3 payload_pos · 3:7 payload_quat · 7:10 trash_pos · 10:14 trash_quat ·
#   14:17 lid_handle_pos · 17:21 lid_handle_quat · 21:24 target_bin_pos · 24:27 trash_bin_pos ·
#   27 payload_in_target_bin · 28 trash_in_trash_bin ·
#   29:32 gripper0_to_payload · 32:35 gripper0_to_lid_handle ·
#   35:38 gripper1_to_payload · 38:41 gripper1_to_trash
#
# 이 데이터셋(PH 200 demo) 실측(30개 확인): trash_in_trash_bin은 항상 payload_in_target_bin보다
# 먼저 참이 됨(30/30) — target_bin이 trash로 막혀 있어 물리적으로 trash를 먼저 치워야 payload를
# 놓을 수 있는 구조로 보임[추정, 실측 순서 기반]. robot0는 lid를 다룬 뒤 payload_pos(41 벡터의
# [0:3])를 start_bin 쪽(y≈-0.45)에서 잡아 옮기고, robot1이 도중에(로봇0이 놓기 10~30스텝 전부터
# 겹쳐서) grasp해 target_bin 쪽(y≈+0.4)까지 마저 배송한다(payload_pos 시작·끝 좌표 직접 확인
# — **실제 핸드오프**, 로봇0이 trash는 안 건드리고 payload만 로봇1에 전달). trash는 로봇1이
# 단독으로 처리(gripper1의 첫 close 구간, d_payload1이 이때는 0.8~0.9로 커서 payload 관련
# close(0.02~0.15)와 거리로 구분됨).
#
# **핸드오프 세부구조 실측(40/40 demo 확인, 2026-07-19 사용자 요청으로 추가 검증)**: robot0가
# payload를 잡은 뒤(lid_removed 이후 gripper0의 두 번째 close) 두 팔 eef 거리(norm(eef0-eef1))가
# 뚜렷하게 수렴(예: demo_1은 0.99→0.13)했다가 핸드오프 순간 최소(~0.12~0.14)를 찍고 로봇0이
# 놓으면서 다시 발산 — "접근(중앙이동)→정렬→넘겨주기→후퇴"가 전 데모에서 일관되게 나타남
# (Square의 align_B(0.6%만 존재)와 달리 거의 보편적이라 별도 stage로 승격: payload_approach).
# 후퇴(로봇0 복귀)는 로봇1의 target_bin 이동과 시간이 겹쳐 별도 전역 마일스톤으로 못 잡음(팔
# 병렬성) — 그래서 여기서 멈추고 6단계로 확정. 두 팔이 시간상 겹쳐 동작(순차적이지 않음, 예:
# demo_10은 로봇1의 trash 배송이 로봇0의 lid 완전 해제보다 먼저 끝남)하므로 나머지도 전부
# **전역 마일스톤**(ground-truth 플래그 또는 그리퍼 개폐 이벤트로 크리스프하게 잡히고
# advance-only로 클램프)만으로 구성해 팔 간 병렬성 문제를 피한다.
TRANSPORT_GRASP_SEP = 0.055  # Square와 동일 임계(같은 Panda 그리퍼 하드웨어)
TRANSPORT_OPEN_SEP = 0.06
TRANSPORT_NEAR_PAYLOAD = 0.15  # gripper1 close 구간이 trash(0.8~0.9)가 아니라 payload(0.02~0.15)인지 구분


def _first(mask, default):
    w = np.where(mask)[0]
    return int(w[0]) if len(w) else default


def _close_open_events(sep, grasp_sep=TRANSPORT_GRASP_SEP, open_sep=TRANSPORT_OPEN_SEP):
    """sep(T,) -> (close_starts, open_starts) 인덱스 배열(그리퍼 닫힘 시작/열림 시작 시점)."""
    closed = sep < grasp_sep
    trans = np.diff(closed.astype(int))
    return np.where(trans == 1)[0] + 1, np.where(trans == -1)[0] + 1


def label_stages_transport(obs):
    """obs(dict of numpy) → (T,) int stage(0..4). 필요 키: object, robot0_gripper_qpos,
    robot1_gripper_qpos.

    stage 0 lid_present: robot0이 아직 lid를 안 치움.
    stage 1 lid_removed: robot0이 lid handle을 한 번 쥐었다 놓음(제거 완료). robot1의 trash
      처리(trash_in_trash_bin)는 별도 stage로 안 잡고 이 구간에 흡수(위 모듈독스트링 참고).
    stage 2 payload_approach: robot0이 payload를 grasp해 robot1 쪽으로 옮기는 중(핸드오프 전).
    stage 3 payload_handoff: robot1이 payload를 grasp(핸드오프 발생, 이후 target_bin으로 운반).
    stage 4 payload_delivered: payload_in_target_bin=1(보통 에피소드 거의 끝).
    """
    obj = np.asarray(obs["object"], dtype=np.float64)
    g0 = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    g1 = np.asarray(obs["robot1_gripper_qpos"], dtype=np.float64)
    T = len(obj)
    idx = np.arange(T)

    sep0 = g0[:, 0] - g0[:, 1]
    sep1 = g1[:, 0] - g1[:, 1]
    payload_flag = obj[:, 27]
    d_payload1 = np.linalg.norm(obj[:, 35:38], axis=1)

    # reset 직후 neutral pose가 우연히 GRASP_SEP 아래일 수 있어(오탐), 첫 "진짜 열림" 이후만 탐색.
    t_first_open0 = _first(sep0 > TRANSPORT_OPEN_SEP, 0)
    t_lid_grasp = _first((idx > t_first_open0) & (sep0 < TRANSPORT_GRASP_SEP), T)
    t_lid_removed = _first((idx > t_lid_grasp) & (sep0 > TRANSPORT_OPEN_SEP), T)

    # robot0의 lid_removed 이후 첫 close = payload grasp(lid grasp는 이미 지나감).
    c0, _o0 = _close_open_events(sep0)
    c0_after = c0[c0 > t_lid_removed]
    t_payload_grasp0 = int(c0_after[0]) if len(c0_after) else T

    # robot1의 close 구간들 중 payload에 가까운(=trash 아닌) 첫 구간 시작 = 핸드오프(robot1 grasp).
    c1, o1 = _close_open_events(sep1)
    t_handoff = T
    for cs in c1:
        later_opens = o1[o1 > cs]
        oe = int(later_opens[0]) if len(later_opens) else T
        if oe > cs and d_payload1[cs:oe].min() < TRANSPORT_NEAR_PAYLOAD:
            t_handoff = cs
            break

    t_payload_done = _first(payload_flag > 0.5, T)

    ts = [t_lid_removed, t_payload_grasp0, t_handoff, t_payload_done]
    for i in range(1, len(ts)):
        ts[i] = max(ts[i], ts[i - 1])
    t_lid_removed, t_payload_grasp0, t_handoff, t_payload_done = ts

    stage = np.zeros(T, dtype=np.int64)
    stage[:t_lid_removed] = 0
    stage[t_lid_removed:t_payload_grasp0] = 1
    stage[t_payload_grasp0:t_handoff] = 2
    stage[t_handoff:t_payload_done] = 3
    stage[t_payload_done:] = 4
    return stage


class OnlineStageTrackerTransport:
    """롤아웃 때 프레임별로 현재 stage를 인과적으로 추정(advance-only). label_stages_transport와
    동일 임계값·정의 공유 — 오프라인은 close~open 구간의 최소거리로 판정하지만, 온라인은 매
    스텝 "지금 닫혀있고 payload에 가까운가"를 바로 확인해 같은 이벤트를 인과적으로 잡는다."""

    def reset(self):
        self.stage = 0
        self._opened0 = False   # reset neutral pose 오탐 방지(로봇0 그리퍼가 한 번 열린 뒤에만 grasp 인정)
        self._grasped_lid = False
        self._grasped_payload0 = False

    def step(self, obs):
        obj = np.asarray(obs["object"], dtype=np.float64)
        g0 = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
        g1 = np.asarray(obs["robot1_gripper_qpos"], dtype=np.float64)
        sep0 = float(g0[0] - g0[1])
        sep1 = float(g1[0] - g1[1])
        d_payload1 = float(np.linalg.norm(obj[35:38]))
        payload_done = obj[27] > 0.5

        if sep0 > TRANSPORT_OPEN_SEP:
            self._opened0 = True
        if self._opened0 and sep0 < TRANSPORT_GRASP_SEP:
            self._grasped_lid = True
        if self.stage >= 1 and sep0 < TRANSPORT_GRASP_SEP:
            self._grasped_payload0 = True  # lid_removed(stage>=1) 이후 gripper0 재닫힘 = payload grasp

        s = self.stage
        if s == 0 and self._grasped_lid and sep0 > TRANSPORT_OPEN_SEP:
            s = 1
        if s <= 1 and self._grasped_payload0:
            s = 2
        if s <= 2 and sep1 < TRANSPORT_GRASP_SEP and d_payload1 < TRANSPORT_NEAR_PAYLOAD:
            s = 3
        if s <= 3 and payload_done:
            s = 4

        self.stage = max(self.stage, s)
        return self.stage


class OnlineStageTracker:
    """롤아웃 때 프레임별로 현재 stage를 인과적(causal)으로 추정. advance-only.

    오프라인 라벨러는 t_peak(전체 궤적 최고점, 비인과)로 lift/approach_B/insert를 나누지만,
    롤아웃은 미래를 모르므로 running-max로 대체한다. 그래서 lift/approach_B/insert 경계는
    오프라인 라벨과 약간 다를 수 있다(gross 구조·순서는 동일). 오프라인과 임계값 공유.
    """

    def reset(self):
        self.stage = 0
        self._prev_eef = None
        self._max_nz = -1e9
        self._opened = False  # 그리퍼가 한 번 열렸는지 (reset 중립 sep 오발 방지)

    def step(self, obs):
        """obs: 단일 프레임 dict (object, robot0_eef_pos, robot0_gripper_qpos) → 현재 stage int."""
        obj = np.asarray(obs["object"], dtype=np.float64)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)

        n2e = float(np.linalg.norm(obj[0:3]))
        nz = float(obj[9])
        dpeg = float(np.linalg.norm(obj[7:9] - PEG_XY))
        sep = float(grip[0] - grip[1])
        speed = 0.0 if self._prev_eef is None else float(np.linalg.norm(eef - self._prev_eef))
        self._prev_eef = eef.copy()

        # reset 직후 relative obs는 nut_to_eef≈0 garbage(실제 최소 ~0.053) → 유효 근접만 인정
        near_nut = 0.02 < n2e < NEAR_NUT
        if sep > OPEN_SEP:
            self._opened = True

        s = self.stage
        if s == 0 and near_nut and speed < V_SLOW:
            s = 1
        # grasp = 그리퍼가 한 번 열린 뒤(approach 중) 너트 근처에서 닫힘
        if s <= 1 and self._opened and sep < GRASP_SEP and near_nut:
            s = 2
        if s <= 2 and nz > LIFT_Z:
            s = 3
            self._max_nz = nz
        if s >= 3:
            self._max_nz = max(self._max_nz, nz)
        if s == 3 and nz < self._max_nz - 0.003:   # 상승 멈춤 = 최고점 도달
            s = 4
        if s == 4 and nz < self._max_nz - 0.02 and dpeg < PEG_R:  # 하강 = 삽입
            s = 5
        if s == 5 and sep > OPEN_SEP:               # 안착 후 그리퍼 열림
            s = 6

        self.stage = max(self.stage, s)
        return self.stage


class ShiftedStageTracker:
    """base tracker를 감싸서, target_stage 진입 후 처음 window 프레임만 stage+1(clamp)을
    주입한다 — 그 뒤(또는 target_stage가 아닐 때)는 base tracker 그대로. stage-as-input
    정책이 critical stage 진입 직후 짧게 틀린 신호를 받았을 때 얼마나 민감한지 보는 강건성
    체크용(2026-07-22, 학습 불필요 — 순수 추론 후처리). 전체 궤적을 다 틀리게 하면 붕괴 원인이
    "stage 의존"인지 "그냥 다 망가짐"인지 구분이 안 돼서, 구간을 짧게 국소화했다."""

    def __init__(self, base_tracker, target_stage, window, num_stages):
        self.base = base_tracker
        self.target_stage = target_stage
        self.window = window
        self.num_stages = num_stages
        self._steps_in_target = 0

    def reset(self):
        self.base.reset()
        self._steps_in_target = 0

    def step(self, obs):
        s = self.base.step(obs)
        if s == self.target_stage:
            self._steps_in_target += 1
            if self._steps_in_target <= self.window:
                return min(s + 1, self.num_stages - 1)
        return s


def make_online_tracker(task_name):
    """task 이름 -> 해당 task용 causal(rollout) stage tracker 인스턴스.
    task별 object obs 레이아웃이 달라 OnlineStageTracker(Square 전용)를 그대로 못 씀 —
    diffusion_trainer.py/eval.py가 이 함수를 통해서만 트래커를 만들어야 함(하드코딩 금지)."""
    if task_name.startswith("transport"):
        return OnlineStageTrackerTransport()
    return OnlineStageTracker()


def num_stages_for_task(task_name):
    if task_name.startswith("transport"):
        return NUM_STAGES_TRANSPORT
    return NUM_STAGES
