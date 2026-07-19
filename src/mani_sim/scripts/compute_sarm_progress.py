"""학습된 SARM reward model로 hdf5 전체 프레임의 progress를 미리 계산해 저장.
lerobot의 compute_rabc_weights.py(--visualize-only 아닌 기본 경로, sarm_progress.parquet 생성)와
같은 역할 — 우리는 npz로 저장(demo_id별 배열, 포맷 단순화).

사용:
    python -m mani_sim.scripts.compute_sarm_progress \
        --checkpoint outputs/sarm/transport/policy_epoch20.pt \
        --hdf5 data/transport_check/transport/ph/transport_image_v15.hdf5 \
        --output outputs/sarm/transport/sarm_progress.npz
"""

import argparse

import h5py
import numpy as np
import torch

from mani_sim.policies.sarm.sarm_reward_model import SARMRewardModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--image-key", default="agentview_image")
    p.add_argument("--state-keys", default="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,"
                                            "robot1_eef_pos,robot1_eef_quat,robot1_gripper_qpos")
    p.add_argument("--num-stages", type=int, default=6)
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_keys = args.state_keys.split(",")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    with h5py.File(args.hdf5, "r") as f:
        d0 = f["data"][list(f["data"].keys())[0]]["obs"]
        state_dim = sum(np.asarray(d0[k][0]).size for k in state_keys)

    model = SARMRewardModel(num_stages=args.num_stages, state_dim=state_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint: {args.checkpoint}")

    progress_by_demo = {}
    with h5py.File(args.hdf5, "r") as f:
        demo_keys = list(f["data"].keys())
        for i, demo_id in enumerate(demo_keys):
            grp = f["data"][demo_id]["obs"]
            T = grp[args.image_key].shape[0]
            progress = np.zeros(T, dtype=np.float32)
            for start in range(0, T, args.batch_size):
                end = min(start + args.batch_size, T)
                img_raw = np.asarray(grp[args.image_key][start:end])  # (b,H,W,C) uint8
                image = torch.as_tensor(
                    np.transpose(img_raw.astype(np.float32) / 255.0, (0, 3, 1, 2)), device=device,
                )
                state = torch.as_tensor(
                    np.concatenate([np.asarray(grp[k][start:end], dtype=np.float32) for k in state_keys], axis=1),
                    device=device,
                )
                with torch.no_grad():
                    p_batch = model.predict_progress(image, state)
                progress[start:end] = p_batch.cpu().numpy()
            progress_by_demo[demo_id] = progress
            if (i + 1) % 20 == 0:
                print(f"[{i + 1}/{len(demo_keys)}] {demo_id} done", flush=True)

    np.savez(args.output, **progress_by_demo)
    print(f"saved: {args.output} ({len(progress_by_demo)} demos)")


if __name__ == "__main__":
    main()
