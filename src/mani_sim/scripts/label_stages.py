"""HDF5에 stage_labeler의 heuristic 라벨링 결과를 obs/stage_onehot(T,num_stages)로 기록.
outputs/label_square_stages.py를 대체(경로를 인자로 받게 일반화). idempotent(있으면 skip).

사용:
    python -m mani_sim.scripts.label_stages --task square --hdf5 data/robomimic/square/ph/v1.5/square/ph/square_image_v15.hdf5
    python -m mani_sim.scripts.label_stages --task door_cabinet --hdf5 outputs/intervention/door_cabinet_round.hdf5
"""

import argparse

import h5py
import numpy as np

from mani_sim.datasets.stage_labeler import (
    NUM_STAGES_DOOR,
    STAGE_NAMES,
    STAGE_NAMES_DOOR,
    label_stages_door_cabinet,
    label_stages_square,
    onehot,
)

TASK_CONFIGS = {
    "square": {"names": STAGE_NAMES, "num": len(STAGE_NAMES)},
    "door_cabinet": {"names": STAGE_NAMES_DOOR, "num": NUM_STAGES_DOOR},
}


def _compute_stage(task, obs_grp, actions, action_mode):
    if task == "square":
        obs = {
            "object": obs_grp["object"][:],
            "robot0_eef_pos": obs_grp["robot0_eef_pos"][:],
            "robot0_gripper_qpos": obs_grp["robot0_gripper_qpos"][:],
        }
        return label_stages_square(obs)
    obs = {"robot0_eef_pos": obs_grp["robot0_eef_pos"][:]}
    return label_stages_door_cabinet(obs, actions, action_mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASK_CONFIGS), default="square")
    p.add_argument("--hdf5", default="data/robomimic/square/ph/v1.5/square/ph/square_image_v15.hdf5")
    args = p.parse_args()
    names = TASK_CONFIGS[args.task]["names"]
    num = TASK_CONFIGS[args.task]["num"]

    with h5py.File(args.hdf5, "a") as f:
        demo_keys = list(f["data"].keys())
        n_written, n_skipped = 0, 0
        for dk in demo_keys:
            obs_grp = f["data"][dk]["obs"]
            if "stage_onehot" in obs_grp:
                n_skipped += 1
                continue
            actions = f["data"][dk]["actions"][:]
            action_mode = f["data"][dk]["action_mode"][:] if "action_mode" in f["data"][dk] else None
            stage = _compute_stage(args.task, obs_grp, actions, action_mode)
            obs_grp.create_dataset("stage_onehot", data=onehot(stage, num=num), dtype="float32")
            n_written += 1
        print(f"[label] demos total={len(demo_keys)} written={n_written} skipped(existing)={n_skipped}")

        first = demo_keys[0]
        stage0 = f["data"][first]["obs"]["stage_onehot"][:].argmax(axis=1)
        counts = np.bincount(stage0, minlength=num)
        print(f"[label] demo0 T={len(stage0)} stage 프레임수:", dict(zip(names, counts.tolist())))


if __name__ == "__main__":
    main()
