"""이미 학습된 체크포인트로 "행동 분산"(교수님 7/15 원문: 같은 관측을 diffusion policy가
여러 번 반복 샘플링했을 때 action이 얼마나 일관되는가)을 미리 계산해 저장.

**cross-demo variance 아님** — 같은 (demo, t)에서 seed만 바꿔가며 N번 샘플링한 action 청크들의
분산. demo 정렬이 필요 없다(compute_sarm_progress.py와 같은 npz 저장 패턴, stage 불필요).

계산 비용 때문에 stride로 듬성듬성 계산한 뒤 각 구간에 값을 그대로 채운다(step function) —
전체 프레임 dense 계산은 200 demo x 700 step x N sample이라 너무 비쌈.

사용:
    python -m mani_sim.scripts.compute_action_variance \
        --checkpoint outputs/train/transport_diffusion_seed0/policy_epoch500.pt \
        --hdf5 data/robomimic/transport/transport/ph/transport_image_v15.hdf5 \
        --output outputs/weighting/transport_action_variance.npz --stride 20 --num-samples 4
"""

import argparse

import h5py
import numpy as np
import torch
from hydra import compose, initialize

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.factory import registry

OBS_HORIZON = 2


def _to_chw01(img):
    return np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))


def build_obs(f, demo_key, t, task_cfg, normalizer, device):
    obs_grp = f["data"][demo_key]["obs"]
    t0 = max(0, t - OBS_HORIZON + 1)
    idx = list(range(t0, t + 1))
    if len(idx) < OBS_HORIZON:
        idx = [idx[0]] * (OBS_HORIZON - len(idx)) + idx

    batch = {}
    for k in task_cfg.rgb_keys:
        frames = np.stack([_to_chw01(obs_grp[k][i]) for i in idx])
        batch[k] = torch.as_tensor(frames, dtype=torch.float32).unsqueeze(0)

    proprio = {}
    for k in task_cfg.lowdim_keys:
        frames = np.stack([np.asarray(obs_grp[k][i], dtype=np.float32) for i in idx])
        proprio[k] = torch.as_tensor(frames, dtype=torch.float32).unsqueeze(0)
    proprio = normalizer.normalize_obs(proprio)
    batch.update(proprio)

    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict_unseeded(policy, obs_batch):
    """seed 고정 안 함 — diffusion denoising의 자체 stochasticity로 반복마다 다른 샘플."""
    return policy.predict_action_chunk(obs_batch)[0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="transport")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--num-samples", type=int, default=4)
    args = p.parse_args()

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="eval", overrides=[f"task={args.task}", "policy_name=diffusion"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "/".join(args.checkpoint.split("/")[:-1])
    normalizer = MinMaxNormalizer(load_stats(f"{out_dir}/normalization_stats.json"))

    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"[var] loaded {args.checkpoint}")

    f = h5py.File(args.hdf5, "r")
    demo_keys = list(f["data"].keys())
    variance_by_demo = {}

    for i, demo_id in enumerate(demo_keys):
        T = f["data"][demo_id]["actions"].shape[0]
        var_arr = np.zeros(T, dtype=np.float32)
        sample_points = list(range(0, T, args.stride))
        for t in sample_points:
            obs_batch = build_obs(f, demo_id, t, cfg.task, normalizer, device)
            samples = np.stack([predict_unseeded(policy, obs_batch) for _ in range(args.num_samples)])  # (N, Tp, A)
            var_scalar = float(samples.var(axis=0).mean())  # 청크·행동차원 평균 분산
            t_end = min(t + args.stride, T)
            var_arr[t:t_end] = var_scalar
        variance_by_demo[demo_id] = var_arr
        if (i + 1) % 20 == 0:
            print(f"[var] [{i + 1}/{len(demo_keys)}] {demo_id} done", flush=True)

    f.close()
    np.savez(args.output, **variance_by_demo)
    print(f"[var] saved: {args.output} ({len(variance_by_demo)} demos, "
          f"stride={args.stride}, num_samples={args.num_samples})")


if __name__ == "__main__":
    main()
