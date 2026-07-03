"""M1 육안 검증: 데이터셋에서 뽑은 샘플의 obs/action 시퀀스를 그려서 확인한다.

사용:
    python -m mani_sim.scripts.visualize_dataset_sample \\
        --hdf5-path data/robomimic/lift/ph/v1.5/lift/ph/low_dim_v15.hdf5 \\
        --index 0
"""

import argparse
import os

import matplotlib.pyplot as plt

from mani_sim.datasets.normalization import MinMaxNormalizer, compute_minmax_stats
from mani_sim.datasets.robomimic_dataset import RobomimicSequenceDataset

DEFAULT_OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="outputs/m1_dataset_check/sample_traj.png")
    args = parser.parse_args()

    stats = compute_minmax_stats(args.hdf5_path, DEFAULT_OBS_KEYS)
    normalizer = MinMaxNormalizer(stats)

    raw_dataset = RobomimicSequenceDataset(
        args.hdf5_path, DEFAULT_OBS_KEYS, args.obs_horizon, args.pred_horizon, normalizer=None
    )
    norm_dataset = RobomimicSequenceDataset(
        args.hdf5_path, DEFAULT_OBS_KEYS, args.obs_horizon, args.pred_horizon, normalizer=normalizer
    )

    raw_batch = raw_dataset[args.index]
    norm_batch = norm_dataset[args.index]

    print("obs shapes:", {k: tuple(v.shape) for k, v in raw_batch["obs"].items()})
    print("action shape:", tuple(raw_batch["action"].shape))
    print("action_mask:", raw_batch["action_mask"].tolist())

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    eef_pos = raw_batch["obs"]["robot0_eef_pos"].numpy()
    axes[0].set_title(f"obs_horizon window (index={args.index}) - robot0_eef_pos (raw scale)")
    for dim, label in enumerate(["x", "y", "z"]):
        axes[0].plot(eef_pos[:, dim], marker="o", label=label)
    axes[0].legend()
    axes[0].set_xlabel("obs timestep (0..To-1)")

    action_norm = norm_batch["action"].numpy()
    mask = raw_batch["action_mask"].numpy()
    axes[1].set_title("pred_horizon window - action (normalized [-1,1], right of dashed line = padding)")
    for dim in range(action_norm.shape[1]):
        axes[1].plot(action_norm[:, dim], marker="o", markevery=list(range(len(mask))), alpha=0.7)
    valid_len = int(mask.sum())
    axes[1].axvline(valid_len - 0.5, color="red", linestyle="--", label="valid/padding boundary")
    axes[1].legend()
    axes[1].set_xlabel("action timestep (0..Tp-1)")
    axes[1].set_ylim(-1.1, 1.1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
