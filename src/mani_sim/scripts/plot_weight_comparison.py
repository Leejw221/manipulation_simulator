"""학습 없이(정책 재학습 대기 없이) 4개 가중치 방식이 실제로 각 샘플에 얼마의 값을 주는지
episode progress(%) 기준으로 비교 플롯. ablation 학습 결과 기다리기 전에 방식별 정성적 차이를
빠르게 확인하는 진단용(사용자가 보여준 예시 그림과 같은 형식).

사용:
    python -m mani_sim.scripts.plot_weight_comparison \
        --checkpoint outputs/train/transport_diffusion_seed0/policy_epoch500.pt \
        --hdf5 data/transport_check/transport/ph/transport_image_v15.hdf5 \
        --variance-npz outputs/weighting/transport_action_variance.npz \
        --output outputs/weighting/weight_comparison.png --stride 20
"""

import argparse

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.factory import registry
from mani_sim.weighting.action_error import ActionErrorWeight
from mani_sim.weighting.action_variance import ActionVarianceWeight
from mani_sim.weighting.phase_rule import PhaseRuleWeight
from mani_sim.weighting.rabc import RABCWeight

OBS_HORIZON = 2
PRED_HORIZON = 16


def _to_chw01(img):
    return np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))


def build_batch(f, demo_key, t, task_cfg, normalizer, device):
    obs_grp = f["data"][demo_key]["obs"]
    T = f["data"][demo_key]["actions"].shape[0]

    t0 = max(0, t - OBS_HORIZON + 1)
    idx = list(range(t0, t + 1))
    if len(idx) < OBS_HORIZON:
        idx = [idx[0]] * (OBS_HORIZON - len(idx)) + idx

    obs = {}
    for k in task_cfg.rgb_keys:
        frames = np.stack([_to_chw01(obs_grp[k][i]) for i in idx])
        obs[k] = torch.as_tensor(frames, dtype=torch.float32).unsqueeze(0)
    proprio = {}
    for k in task_cfg.lowdim_keys:
        frames = np.stack([np.asarray(obs_grp[k][i], dtype=np.float32) for i in idx])
        proprio[k] = torch.as_tensor(frames, dtype=torch.float32).unsqueeze(0)
    proprio = normalizer.normalize_obs(proprio)
    obs.update(proprio)
    obs = {k: v.to(device) for k, v in obs.items()}

    a_idx = list(range(t, min(t + PRED_HORIZON, T)))
    n_pad = PRED_HORIZON - len(a_idx)
    action_raw = np.stack([np.asarray(f["data"][demo_key]["actions"][i], dtype=np.float32) for i in a_idx])
    if n_pad > 0:
        action_raw = np.concatenate([action_raw, np.repeat(action_raw[-1:], n_pad, axis=0)], axis=0)
    action = normalizer.normalize_action(torch.as_tensor(action_raw, dtype=torch.float32)).unsqueeze(0).to(device)
    mask = torch.as_tensor([True] * len(a_idx) + [False] * n_pad).unsqueeze(0).to(device)

    return {"obs": obs, "action": action, "action_mask": mask}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="transport")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--variance-npz", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stride", type=int, default=20)
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
    print(f"[plot] loaded {args.checkpoint}")

    action_error_w = ActionErrorWeight(beta_d=1.0, beta_u=1.0, gamma=1.0)
    action_variance_w = ActionVarianceWeight(args.variance_npz, w_min=0.2, w_max=5.0, device=device)
    phase_rule_w = PhaseRuleWeight(args.hdf5, critical_weight=3.0, critical_stages=(3,), device=device)
    rabc_w = RABCWeight(args.hdf5, chunk_size=PRED_HORIZON, kappa=0.01, device=device)

    f = h5py.File(args.hdf5, "r")
    demo_keys = list(f["data"].keys())

    demo_ids_all, t_all, errors_all, progress_pct = [], [], [], []

    for i, demo_id in enumerate(demo_keys):
        T = f["data"][demo_id]["actions"].shape[0]
        for t in range(0, T, args.stride):
            batch = build_batch(f, demo_id, t, cfg.task, normalizer, device)
            with torch.no_grad():
                error = policy.compute_loss(batch, reduction="none")  # (1,)
            demo_ids_all.append(demo_id)
            t_all.append(t)
            errors_all.append(error.item())
            progress_pct.append(100.0 * t / max(T - 1, 1))
        if (i + 1) % 40 == 0:
            print(f"[plot] [{i + 1}/{len(demo_keys)}] error 계산 완료", flush=True)
    f.close()

    # action_error는 "그 배치 안에서의 상대적" 정규화(오차/배치 오차합)라 실제 학습(batch_size)과
    # 다른 크기로 한꺼번에 정규화하면 스케일이 왜곡된다(7000개를 통째로 넣으면 분모가 커져서
    # 전부 0에 수렴 — 진단 스크립트 버그였음). 실제 학습과 같은 batch_size로 무작위로 묶어서 계산.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(errors_all))
    errors_t = torch.as_tensor(errors_all, dtype=torch.float32, device=device)
    w_err_all = np.zeros(len(errors_all), dtype=np.float32)
    train_batch_size = 64
    for start in range(0, len(perm), train_batch_size):
        chunk = perm[start:start + train_batch_size]
        action_mode_chunk = torch.full((len(chunk), PRED_HORIZON), -1.0, device=device)
        w_chunk = action_error_w.compute_weights(action_mode_chunk, error=errors_t[chunk]).mean(dim=1)
        w_err_all[chunk] = w_chunk.cpu().numpy()
    w_var_all = action_variance_w.compute_weights(demo_ids_all, t_all).cpu().numpy()
    w_phase_all = phase_rule_w.compute_weights(demo_ids_all, t_all).cpu().numpy()
    w_rabc_all = rabc_w.compute_weights(demo_ids_all, t_all).cpu().numpy()

    progress_pct = np.array(progress_pct)
    records = {
        "none": np.ones(len(errors_all)),
        "action_error": w_err_all,
        "action_variance": w_var_all,
        "phase_rule": w_phase_all,
        "rabc": w_rabc_all,
    }

    progress_pct = np.array(progress_pct)
    bins = np.linspace(0, 100, 11)
    bin_idx = np.digitize(progress_pct, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 9)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    colors = {
        "none": "#888888", "action_error": "#B4131C", "action_variance": "#2ec4b6",
        "phase_rule": "#163A78", "rabc": "#E67E22",
    }
    labels = {
        "none": "가중치 없음", "action_error": "행동오차(APO)", "action_variance": "행동분산",
        "phase_rule": "phase규칙(criticality)", "rabc": "SARM-RABC(progress, stage축)",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, values in records.items():
        values = np.array(values)
        means = np.array([values[bin_idx == b].mean() if np.any(bin_idx == b) else np.nan for b in range(10)])
        stds = np.array([values[bin_idx == b].std() if np.any(bin_idx == b) else 0 for b in range(10)])
        ax.plot(bin_centers, means, marker="o", label=labels[method], color=colors[method], linewidth=2)
        ax.fill_between(bin_centers, means - stds, means + stds, color=colors[method], alpha=0.15)

    ax.set_xlabel("Episode progress (%)")
    ax.set_ylabel("Weight")
    ax.set_title("Transport — 가중치 산정 방식별 비교 (학습 전 진단, stride=%d)" % args.stride)
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"[plot] saved: {args.output}")


if __name__ == "__main__":
    main()
