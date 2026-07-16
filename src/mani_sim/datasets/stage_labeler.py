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


def stage_progress(stage, num=NUM_STAGES):
    """stage(T,) int(0..num-1, 단조증가) → stage-aware progress(T,) float[0,1), 단조증가.

    SARM(arXiv:2509.25358)의 "전체 에피소드 단일 선형 진행도 대신, stage+stage 내 상대위치로
    진행도를 표현" 정의를 그대로 따른다: progress = (stage_idx + stage 내 상대위치) / num_stages.
    RA-BC 가중치의 입력이며, 학습 라벨(오프라인, 전체 궤적 기준)이라 롤아웃 인과성 문제는 없다
    — OnlineStageTracker(온라인, stage 입력용)와는 별도 용도.
    """
    stage = np.asarray(stage, dtype=np.int64)
    T = len(stage)
    starts = np.searchsorted(stage, np.arange(num), side="left")  # stage 단조증가라 유효
    ends = np.append(starts[1:], T)

    progress = np.zeros(T, dtype=np.float32)
    for s in range(num):
        s0, s1 = int(starts[s]), int(ends[s])
        if s1 <= s0:
            continue
        local = (np.arange(s0, s1) - s0) / max(s1 - s0, 1)  # 0..<1, stage 내 상대위치
        progress[s0:s1] = (s + local) / num
    return progress


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
