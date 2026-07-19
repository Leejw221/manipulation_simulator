"""Stage counterfactual 진단: 같은 관측에서 stage 라벨만 바꿔 끼웠을 때 예측 행동이 실제로
달라지는지(mechanism check) — "stage 입력이 성공률을 올리는가"(EXP-01, null)와 다른 질문.
outputs/stage_counterfactual.py를 registry 기반으로 이식.

통제: DDIM 초기 노이즈 시드를 real/counterfactual 두 호출에서 동일 고정 → 차이의 원인을
stage 변경 하나로 격리. noise floor(같은 stage·다른 시드)와 비교해 유의미한 차이인지 판정.

사용:
    python -m mani_sim.scripts.stage_counterfactual --checkpoint outputs/train/square_stage_diffusion_unet/policy_epoch300.pt
"""

import argparse
import os

import h5py
import numpy as np
import torch

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.datasets.stage_labeler import NUM_STAGES, STAGE_NAMES, onehot
from mani_sim.factory import registry

TASK_NAME = "square_stage"
OBS_HORIZON = 2


def _to_chw01(img):
    return np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))


def build_obs(f, demo_key, t, task_cfg, normalizer, device, stage_override=None):
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
        if k == "stage_onehot":
            continue
        frames = np.stack([np.asarray(obs_grp[k][i], dtype=np.float32) for i in idx])
        proprio[k] = torch.as_tensor(frames, dtype=torch.float32).unsqueeze(0)
    proprio = normalizer.normalize_obs(proprio)
    batch.update(proprio)

    if stage_override is None:
        stage_frames = np.stack([obs_grp["stage_onehot"][i] for i in idx])
    else:
        stage_frames = np.stack([onehot(np.array([stage_override]))[0] for _ in idx])
    stage_t = torch.as_tensor(stage_frames, dtype=torch.float32).unsqueeze(0)
    batch["stage_onehot"] = normalizer.normalize_obs({"stage_onehot": stage_t})["stage_onehot"]

    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(policy, obs_batch, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return policy.predict_action_chunk(obs_batch)[0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hdf5", default="data/robomimic/square/ph/v1.5/square/ph/square_image_v15.hdf5")
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="eval", overrides=[f"task={TASK_NAME}", "policy_name=diffusion"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    out_dir = os.path.dirname(args.checkpoint)
    normalizer = MinMaxNormalizer(load_stats(os.path.join(out_dir, "normalization_stats.json")))

    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"[cf] loaded {args.checkpoint}")

    f = h5py.File(args.hdf5, "r")
    demo_keys = list(f["data"].keys())

    swap_dists, noise_floor_dists = [], []
    per_stage_swap = {s: [] for s in range(NUM_STAGES)}

    for _ in range(args.num_samples):
        dk = demo_keys[rng.integers(len(demo_keys))]
        T = f["data"][dk]["actions"].shape[0]
        t = int(rng.integers(OBS_HORIZON - 1, T))
        real_stage = int(f["data"][dk]["obs"]["stage_onehot"][t].argmax())

        obs_real = build_obs(f, dk, t, cfg.task, normalizer, device, stage_override=None)
        seed_a, seed_b = int(rng.integers(1e6)), int(rng.integers(1e6))
        action_real = predict(policy, obs_real, device, seed_a)
        action_real_2 = predict(policy, obs_real, device, seed_b)
        noise_floor_dists.append(float(np.linalg.norm(action_real - action_real_2)))

        other_stage = int(rng.integers(NUM_STAGES))
        while other_stage == real_stage:
            other_stage = int(rng.integers(NUM_STAGES))
        obs_swap = build_obs(f, dk, t, cfg.task, normalizer, device, stage_override=other_stage)
        action_swap = predict(policy, obs_swap, device, seed_a)

        d = float(np.linalg.norm(action_real - action_swap))
        swap_dists.append(d)
        per_stage_swap[real_stage].append(d)

    f.close()

    swap_dists, noise_floor_dists = np.array(swap_dists), np.array(noise_floor_dists)
    print(f"\n[cf] N={args.num_samples}")
    print(f"[cf] stage-swap distance   : mean={swap_dists.mean():.4f}  std={swap_dists.std():.4f}")
    print(f"[cf] noise-floor distance  : mean={noise_floor_dists.mean():.4f}  std={noise_floor_dists.std():.4f}")
    ratio = swap_dists.mean() / max(noise_floor_dists.mean(), 1e-8)
    print(f"[cf] ratio (swap/floor)    : {ratio:.2f}x")
    print("[cf] 해석: ratio >> 1 이면 stage가 예측에 실제로 영향(mechanism 살아있음). "
          "ratio ~ 1 이면 stage를 사실상 무시(주입이 안 먹힘).\n")

    print("[cf] stage별 swap distance (그 stage에서 다른 stage로 바꿨을 때):")
    for s in range(NUM_STAGES):
        vals = per_stage_swap[s]
        if vals:
            print(f"    {STAGE_NAMES[s]:12s} n={len(vals):3d}  mean={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
