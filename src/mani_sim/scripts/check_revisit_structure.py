"""HDF5 데모 데이터셋에서 "같은 위치를 시간차 두고 재방문하면서 이후 이동 방향이 크게
다른" state-aliasing 구조를 정량적으로 탐지 — 2026-07-19, Square vs Transport 비교에 사용.

방법: 궤적 상의 두 시점 t1<t2가 (a) eef 위치가 dist_thresh 이내로 가깝고, (b) min_gap
스텝 이상 떨어져 있고(단순 왕복이 아님), (c) 각 시점 이후 horizon 스텝 동안의 평균 이동
방향이 angle_thresh_deg 이상 다르면 "재방문"으로 센다.

결과(2026-07-19, low_dim PH 20 demo 기준, min_gap=100):
- Square(robot0): 0/20 demo에서 재방문 없음 -> EXP-01 stage-as-input null의 구조적 근거.
- Transport(robot0): 10/20(50%) demo에서 재방문 있음 -> stage 재검증 후보로 결정.

사용:
    python -m mani_sim.scripts.check_revisit_structure \
        --hdf5 data/robomimic/square/ph/v1.5/square/ph/square_image_v15.hdf5 \
        --obs-key robot0_eef_pos --num-demos 20 --min-gap 100
"""

import argparse

import h5py
import numpy as np


def future_direction(pos, t, horizon=15):
    """t 시점 이후 horizon 스텝 동안의 평균 이동 방향(unit vector)."""
    end = min(len(pos) - 1, t + horizon)
    if end <= t:
        return None
    d = pos[end] - pos[t]
    n = np.linalg.norm(d)
    if n < 1e-4:
        return None
    return d / n


def find_revisits(pos, dist_thresh=0.02, min_gap=30, angle_thresh_deg=90, horizon=15):
    """같은 위치(dist_thresh 이내)를 min_gap 이상 떨어진 시점에 재방문하면서,
    그 이후 이동 방향이 angle_thresh_deg 이상 다른 경우를 찾는다."""
    revisits = []
    T = len(pos)
    for t2 in range(min_gap, T):
        for t1 in range(0, t2 - min_gap):
            if np.linalg.norm(pos[t2] - pos[t1]) < dist_thresh:
                d1 = future_direction(pos, t1, horizon)
                d2 = future_direction(pos, t2, horizon)
                if d1 is None or d2 is None:
                    continue
                cos_angle = np.clip(np.dot(d1, d2), -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                if angle > angle_thresh_deg:
                    revisits.append((t1, t2, angle, pos[t1].copy()))
    return revisits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--obs-key", default="robot0_eef_pos", help="위치로 쓸 obs 키(여러 개면 콤마로)")
    p.add_argument("--num-demos", type=int, default=20)
    p.add_argument("--dist-thresh", type=float, default=0.02)
    p.add_argument("--min-gap", type=int, default=100)
    p.add_argument("--angle-thresh", type=float, default=90)
    args = p.parse_args()

    obs_keys = args.obs_key.split(",")
    f = h5py.File(args.hdf5, "r")

    for obs_key in obs_keys:
        demos_with_revisit = 0
        total = 0
        for i in range(args.num_demos):
            demo = f["data"][f"demo_{i}"]
            pos = demo["obs"][obs_key][:]
            revisits = find_revisits(pos, args.dist_thresh, args.min_gap, args.angle_thresh)
            if revisits:
                demos_with_revisit += 1
                total += len(revisits)
                gaps = [t2 - t1 for t1, t2, a, p_ in revisits]
                print(f"demo_{i} {obs_key}: 재방문 {len(revisits)}개, 최대 gap={max(gaps)}")
        print(f"\n[{obs_key}] 재방문 있는 demo: {demos_with_revisit}/{args.num_demos} "
              f"(min_gap={args.min_gap}, dist_thresh={args.dist_thresh}, "
              f"angle_thresh={args.angle_thresh}), 총 쌍수={total}\n")


if __name__ == "__main__":
    main()
