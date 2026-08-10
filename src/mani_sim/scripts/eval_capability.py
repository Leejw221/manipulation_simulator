"""체크포인트별 '능력 지표'를 고정 세트에서 오프라인 측정한다(2026-08-11 신설).

**왜 필요한가**
학습 중 로그(`raw_model_mse_*`)는 (1) **raw 가중치**로 (2) **매 스텝 다른 랜덤 배치**에서 잰다.
그런데 우리가 실제로 배포·평가하는 건 `policy_epoch*.pt`에 담긴 **EMA 가중치**다. 즉
"정책이 이렇다"는 능력 주장을 학습 로그로 하면 **배포하지 않는 모델**을 근거로 삼게 된다.
in-training 로그는 "최적화가 설계대로 도는가"(z0·margin·util)에는 맞는 기준이지만, 능력
주장에는 이 스크립트의 값을 써야 한다.

**무엇을 재나**
- `e_theta(chosen)` / `e_theta(rejected)`: 각 그룹에서 정책의 노이즈 예측 오차
- **`e_theta(mismatch)`**: 관측 obs_i에 **다른 샘플의 액션** action_j를 붙였을 때의 오차.
  이 값이 **커져야** 정책이 관측을 실제로 쓰고 있다는 뜻이다(조건 판별력). EXP-10 (115)절의
  0% 붕괴를 진단한 지표 — 정상 런은 1.71배 증가, 붕괴 런은 0.58배로 감소했다.
  `chosen`/`rejected`만 보면 이걸 못 잡는다(둘 다 "맞는 쌍"만 보므로).
- 각각의 `e_ref`(reference 오차)와 비율. reward = e_ref - e_theta이므로 비율 < 1이면
  정책이 reference보다 정확하다는 뜻.

**측정을 고정하는 것들** — 체크포인트 간 차이가 오직 모델 차이가 되게 한다:
샘플 집합(seed 고정, 그룹별 층화) · timestep 격자 · 노이즈. mismatch 짝짓기는 학습과 동일한
결정적 circular shift(`torch.roll(action, -1)`).

사용:
    python -m mani_sim.scripts.eval_capability \
        --run-dir outputs/square_r1_fix_a \
        --reference outputs/square_base_sirius_matched/policy_epoch1060.pt \
        --zarr data/intervention/square_round1_new_merged_k10.zarr \
        --out outputs/square_r1_fix_a/capability.csv
"""

import argparse
import copy
import csv
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from mani_sim.datasets.labels import DESIRABLE_LABELS
from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.datasets.zarr_dataset import ZarrSequenceDataset
from mani_sim.factory import registry
from mani_sim.utils.checkpoints import load_run_config
from mani_sim.utils.task_utils import task_obs_keys

# 정책·reference를 만들 때 쓰는 policy 등록(import 부작용으로 registry에 올라간다)
import mani_sim.policies.diffusion.diffusion_policy_registrations  # noqa: F401


def _ckpt_epoch(path):
    m = re.search(r"policy_epoch(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def build_fixed_batches(cfg, stats_dir, pf, zarr_path, n_chosen, n_rejected, batch_size, seed, device):
    """그룹별로 층화 추출한 고정 샘플 집합을 (obs, action, mask, is_chosen) 배치 리스트로."""
    normalizer = MinMaxNormalizer(load_stats(os.path.join(stats_dir, "normalization_stats.json")))
    ds = ZarrSequenceDataset(
        zarr_path=zarr_path,
        obs_keys=task_obs_keys(cfg.task),
        obs_horizon=cfg.policy.obs_horizon,
        pred_horizon=cfg.policy.pred_horizon,
        normalizer=normalizer,
        rgb_keys=cfg.task.rgb_keys,
        extra_keys=("action_mode",),
    )
    # 전체 윈도우의 라벨을 훑어 chosen/rejected 인덱스를 만든다(학습 loss와 동일 기준:
    # 앞 preference_frames 프레임의 desirable 비율 다수결).
    chosen_idx, rejected_idx = [], []
    for i in range(len(ds)):
        am = ds[i]["action_mode"][:pf].numpy()
        if np.isin(am, list(DESIRABLE_LABELS)).mean() >= 0.5:
            chosen_idx.append(i)
        else:
            rejected_idx.append(i)
    rng = np.random.default_rng(seed)
    chosen_idx = rng.permutation(chosen_idx)[:n_chosen]
    rejected_idx = rng.permutation(rejected_idx)[:n_rejected]
    print(f"고정 세트: chosen {len(chosen_idx)} / rejected {len(rejected_idx)} "
          f"(풀 {len(ds)} 윈도우 중)")

    def collate(indices, is_chosen):
        out = []
        for s in range(0, len(indices), batch_size):
            chunk = [ds[int(i)] for i in indices[s:s + batch_size]]
            if len(chunk) < 2:
                continue  # mismatch(roll)에는 최소 2개 필요
            obs = {k: torch.stack([c["obs"][k] for c in chunk]).to(device) for k in chunk[0]["obs"]}
            action = torch.stack([c["action"] for c in chunk]).to(device)
            mask = torch.stack([c["action_mask"] for c in chunk]).to(device).float()
            out.append((obs, action, mask, is_chosen))
        return out

    return collate(chosen_idx, True) + collate(rejected_idx, False)


@torch.no_grad()
def measure(policy, reference, batches, timesteps_grid, seed, device):
    """고정 배치·고정 timestep 격자·고정 노이즈에서 세 종류 MSE를 잰다."""
    sched = policy.train_scheduler
    acc = {k: [0.0, 0] for k in
           ("e_th_chosen", "e_ref_chosen", "e_th_rejected", "e_ref_rejected",
            "e_th_mismatch", "e_ref_mismatch")}

    def add(key, value, n):
        acc[key][0] += float(value) * n
        acc[key][1] += n

    for bi, (obs, action, mask, is_chosen) in enumerate(batches):
        B = action.shape[0]
        cond = policy.get_global_cond(obs)
        ref_cond = reference.get_global_cond(obs)   # 둘 다 eval 모드 -> 같은 CenterCrop
        for t_val in timesteps_grid:
            g = torch.Generator(device="cpu").manual_seed(seed + 1000 * bi + t_val)
            noise = torch.randn(action.shape, generator=g).to(device)
            t = torch.full((B,), int(t_val), device=device, dtype=torch.long)
            noisy = sched.add_noise(action, noise, t)

            def mse(pred):
                per = F.mse_loss(pred, noise, reduction="none").mean(dim=-1)  # (B, Tp)
                return ((per * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).mean()

            tag = "chosen" if is_chosen else "rejected"
            add(f"e_th_{tag}", mse(policy.unet(noisy, t, cond)), B)
            add(f"e_ref_{tag}", mse(reference.unet(noisy, t, ref_cond)), B)

            # mismatch: 관측은 그대로, 액션만 한 칸 밀어서(학습과 동일한 결정적 shift)
            mm_action = torch.roll(action, shifts=-1, dims=0)
            mm_mask = mask * torch.roll(mask, shifts=-1, dims=0)
            mm_noisy = sched.add_noise(mm_action, noise, t)

            def mse_mm(pred):
                per = F.mse_loss(pred, noise, reduction="none").mean(dim=-1)
                return ((per * mm_mask).sum(dim=1) / mm_mask.sum(dim=1).clamp(min=1.0)).mean()

            add("e_th_mismatch", mse_mm(policy.unet(mm_noisy, t, cond)), B)
            add("e_ref_mismatch", mse_mm(reference.unet(mm_noisy, t, ref_cond)), B)

    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="policy_epoch*.pt와 run_config.yaml이 있는 학습 출력 디렉토리")
    ap.add_argument("--reference", required=True, help="학습 시작점(reference) 체크포인트")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-chosen", type=int, default=512)
    ap.add_argument("--n-rejected", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--timesteps", default="4,14,24,34,44,54,64,74,84,94",
                    help="측정할 timestep 격자(쉼표 구분). 학습은 랜덤 t지만 측정은 고정해야 비교된다.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preference-frames", type=int, default=None,
                    help="chosen/rejected 판정 구간. 생략하면 run_config의 system.preference_frames(없으면 8).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(
        [os.path.join(args.run_dir, f) for f in os.listdir(args.run_dir) if _ckpt_epoch(f) >= 0],
        key=lambda p: _ckpt_epoch(p),
    )
    if not ckpts:
        raise SystemExit(f"체크포인트 없음: {args.run_dir}")
    print(f"체크포인트 {len(ckpts)}개: epoch {[_ckpt_epoch(p) for p in ckpts]}")

    cfg = load_run_config(ckpts[0])
    # BC/SIRIUS 학습 출력에는 system 블록이 없다(apo 전용) — 그땐 CLI 기본값을 쓴다.
    pf = args.preference_frames
    if pf is None:
        try:
            pf = int(cfg.system.preference_frames)
        except Exception:
            pf = 8
    grid = [int(x) for x in args.timesteps.split(",")]

    batches = build_fixed_batches(cfg, args.run_dir, pf, args.zarr, args.n_chosen,
                                  args.n_rejected, args.batch_size, args.seed, device)

    reference = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy).to(device)
    reference.load_state_dict(torch.load(args.reference, map_location=device, weights_only=False)["model"])
    reference.eval()
    policy = copy.deepcopy(reference)

    rows = []
    for path in ckpts:
        policy.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
        policy.eval()
        m = measure(policy, reference, batches, grid, args.seed, device)
        m["epoch"] = _ckpt_epoch(path)
        for tag in ("chosen", "rejected", "mismatch"):
            m[f"ratio_{tag}"] = m[f"e_th_{tag}"] / m[f"e_ref_{tag}"]
        rows.append(m)
        print(f"epoch {m['epoch']:>4}  "
              f"chosen {m['e_th_chosen']:.5f}(x{m['ratio_chosen']:.2f})  "
              f"rejected {m['e_th_rejected']:.5f}(x{m['ratio_rejected']:.2f})  "
              f"mismatch {m['e_th_mismatch']:.5f}(x{m['ratio_mismatch']:.2f})")

    out = args.out or os.path.join(args.run_dir, "capability.csv")
    cols = ["epoch", "e_th_chosen", "e_ref_chosen", "ratio_chosen",
            "e_th_rejected", "e_ref_rejected", "ratio_rejected",
            "e_th_mismatch", "e_ref_mismatch", "ratio_mismatch"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
