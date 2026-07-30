"""학습된 SARM(StageTransformer+SubtaskTransformer) reward model로 hdf5 전체 프레임의
progress를 미리 계산해 저장. lerobot의 compute_rabc_weights.py와 같은 역할 — 우리는 npz로
저장(demo_id별 배열).

각 target 프레임 t마다 causal window([ep_start,...,t], sarm_dataset.get_frame_indices와
동일)를 만들어 모델에 넣고, 윈도우의 마지막 위치(=t) 예측만 취한다.

사용:
    python -m mani_sim.scripts.compute_sarm_progress \
        --checkpoint outputs/sarm/transport_v2/policy_epoch3.pt \
        --hdf5 data/robomimic/transport/transport/ph/transport_image_v15.hdf5 \
        --output outputs/sarm/transport_v2/sarm_progress.npz
"""

import argparse

import h5py
import numpy as np
import torch

from mani_sim.datasets.sarm_dataset import get_frame_indices
from mani_sim.policies.sarm.clip_text_encoder import ClipTextEncoder
from mani_sim.policies.sarm.sarm_reward_model import ClipVisionEncoder
from mani_sim.policies.sarm.sarm_transformer import SARMFullModel, StageTransformer, SubtaskTransformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--image-key", default="agentview_image")
    p.add_argument("--state-keys", default="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,"
                                            "robot1_eef_pos,robot1_eef_quat,robot1_gripper_qpos")
    p.add_argument("--num-stages", type=int, default=5)
    p.add_argument("--task-description",
                    default="remove the lid, hand off the payload between the two arms, and deliver it to the target bin")
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--n-obs-steps", type=int, default=8)
    p.add_argument("--frame-gap", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_keys = args.state_keys.split(",")

    with h5py.File(args.hdf5, "r") as f:
        d0 = f["data"][list(f["data"].keys())[0]]["obs"]
        state_dim = sum(np.asarray(d0[k][0]).size for k in state_keys)
        vmin = np.concatenate([np.asarray(d0[k][:]).min(axis=0) for k in state_keys])
        vmax = np.concatenate([np.asarray(d0[k][:]).max(axis=0) for k in state_keys])
        for demo_id in f["data"].keys():
            grp = f["data"][demo_id]["obs"]
            vmin = np.minimum(vmin, np.concatenate([np.asarray(grp[k][:]).min(axis=0) for k in state_keys]))
            vmax = np.maximum(vmax, np.concatenate([np.asarray(grp[k][:]).max(axis=0) for k in state_keys]))
        vmin = vmin.astype(np.float32)
        vmax = vmax.astype(np.float32)
    state_vmin = torch.as_tensor(vmin, device=device)
    state_vmax = torch.as_tensor(vmax, device=device)

    vision_encoder = ClipVisionEncoder(freeze=True).to(device).eval()
    text_encoder = ClipTextEncoder(freeze=True).to(device).eval()
    with torch.no_grad():
        lang_emb_single = text_encoder([args.task_description], device=device)

    stage_model = StageTransformer(
        d_model=args.d_model, vis_emb_dim=vision_encoder.out_dim, text_emb_dim=text_encoder.out_dim,
        state_dim=state_dim, n_layers=args.n_layers, n_heads=args.n_heads, num_cameras=1,
        num_classes_sparse=args.num_stages,
    ).to(device)
    subtask_model = SubtaskTransformer(
        d_model=args.d_model, vis_emb_dim=vision_encoder.out_dim, text_emb_dim=text_encoder.out_dim,
        state_dim=state_dim, n_layers=args.n_layers, n_heads=args.n_heads, num_cameras=1,
    ).to(device)
    full_model = SARMFullModel(stage_model, subtask_model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    full_model.load_state_dict(ckpt["model"])
    full_model.eval()
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
                batch_targets = list(range(start, end))
                windows = [get_frame_indices(t, args.n_obs_steps, args.frame_gap, 0, T - 1) for t in batch_targets]
                flat_idx = sorted(set(i for w in windows for i in w))
                pos_of = {v: k for k, v in enumerate(flat_idx)}

                img_raw = np.asarray(grp[args.image_key][flat_idx])
                images = torch.as_tensor(
                    np.transpose(img_raw.astype(np.float32) / 255.0, (0, 3, 1, 2)), device=device,
                )
                state = torch.as_tensor(
                    np.concatenate([np.asarray(grp[k][flat_idx], dtype=np.float32) for k in state_keys], axis=1),
                    device=device,
                )
                with torch.no_grad():
                    img_emb_flat = vision_encoder(images)  # (len(flat_idx), vis_dim)
                    state_norm_flat = (state - state_vmin) / (state_vmax - state_vmin + 1e-8) * 2 - 1

                B = len(windows)
                W = args.n_obs_steps + 1
                img_emb = torch.stack(
                    [img_emb_flat[[pos_of[idx] for idx in w]] for w in windows]
                ).unsqueeze(1)  # (B,1,W,vis_dim)
                state_seq = torch.stack(
                    [state_norm_flat[[pos_of[idx] for idx in w]] for w in windows]
                )  # (B,W,state_dim)
                lengths = torch.full((B,), W, dtype=torch.long, device=device)
                lang_emb = lang_emb_single.expand(B, -1)

                with torch.no_grad():
                    p_batch = full_model.predict_progress(img_emb, lang_emb, state_seq, lengths)  # (B,W)
                progress[start:end] = p_batch[:, -1].cpu().numpy()
            progress_by_demo[demo_id] = progress
            if (i + 1) % 20 == 0:
                print(f"[{i + 1}/{len(demo_keys)}] {demo_id} done", flush=True)

    np.savez(args.output, **progress_by_demo)
    print(f"saved: {args.output} ({len(progress_by_demo)} demos)")


if __name__ == "__main__":
    main()
