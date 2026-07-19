"""합성(synthetic) stage 필요성 진단 — Square 실측(counterfactual ratio~1.15x, stage 사실상
무시)이 "task가 stage를 안 필요로 해서"인지 "우리 주입 방식(FiLM 경유 concat) 자체가 stage를
못 배우는 구조적 문제"인지 가른다.

방법: 진짜 이미지·proprio는 그대로 두고, stage_onehot과 정답 action만 **이미지와 완전히
무관하게 무작위**로 새로 붙인다(무작위 stage 번호 → 그 번호만으로 정해지는 단순 상수 action).
이러면 네트워크가 이미지를 아무리 봐도 정답을 못 맞히고, **주입된 stage 입력을 실제로 읽어야만**
loss가 떨어진다 — 즉 이 task에선 stage가 "반드시 필요"하도록 강제된 상태.

같은 정책 클래스(DiffusionPolicyImage)·같은 주입 방식(FiLM)으로 이 인공 task를 학습시켜:
  - 학습이 실제로 stage를 따라가면(예측 action이 주입한 stage에 맞게 바뀌면)
    → 주입 방식 자체는 건강함, Square null은 진짜 "이 task엔 안 필요해서".
  - 그래도 stage를 못 배우면 → 주입 방식(차원 희석 등) 자체가 문제 → 서랍 실험 전에 먼저 고쳐야 함.

실행:
    MUJOCO_GL=egl python -m mani_sim.scripts.stage_sanity_check --num-epochs 30
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mani_sim.datasets.normalization import MinMaxNormalizer, compute_minmax_stats
from mani_sim.datasets.robomimic_dataset import RobomimicSequenceDataset
from mani_sim.datasets.stage_labeler import NUM_STAGES, STAGE_NAMES, onehot
from mani_sim.policies.diffusion.diffusion_policy_image import DiffusionPolicyImage

RGB_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
BASE_LOWDIM = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
LOWDIM_KEYS = BASE_LOWDIM + ["stage_onehot"]
OBS_DIMS = {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2, "stage_onehot": 7}
ACTION_DIM = 7
OBS_HORIZON = 2
PRED_HORIZON = 16


def stage_to_action_value(stage_idx):
    """무작위 stage 번호 -> [-1,1] 스칼라, action 전 차원에 동일 방송. 이미지 내용과 무관한 규칙."""
    return (stage_idx / (NUM_STAGES - 1)) * 2 - 1


class SyntheticStageDataset(Dataset):
    """진짜 obs(이미지·proprio)는 base_dataset 그대로, stage_onehot·action만 매번 새로
    무작위 배정 — 이미지에서 라벨을 추론할 수 없게 만드는 게 핵심."""

    def __init__(self, base_dataset, seed=0):
        self.base = base_dataset
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        stage_idx = int(self.rng.integers(0, NUM_STAGES))

        oh = onehot(np.array([stage_idx] * OBS_HORIZON))  # (obs_horizon,7) raw 0/1
        item["obs"]["stage_onehot"] = torch.as_tensor(oh * 2 - 1, dtype=torch.float32)  # MinMax[-1,1] 규약과 맞춤

        target = stage_to_action_value(stage_idx)
        item["action"] = torch.full((PRED_HORIZON, ACTION_DIM), target, dtype=torch.float32)
        return item


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", default="data/robomimic/square/ph/v1.5/square/ph/square_image_v15.hdf5")
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # RobomimicSequenceDataset.normalize_obs가 LOWDIM_KEYS 전체(stage_onehot 포함)를 순회하므로
    # stats에 stage_onehot 항목이 없으면 KeyError. 여기서 계산되는 stage_onehot 정규화값은 base_dataset
    # 단계에서만 쓰이고, SyntheticStageDataset이 곧바로 덮어써서 실제로는 안 쓰인다.
    stats = compute_minmax_stats(args.hdf5, LOWDIM_KEYS)
    normalizer = MinMaxNormalizer(stats)

    cache_mode = "all" if args.num_workers >= 1 else "low_dim"
    base_dataset = RobomimicSequenceDataset(
        hdf5_path=args.hdf5,
        obs_keys=LOWDIM_KEYS + RGB_KEYS,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON,
        normalizer=normalizer,  # BASE_LOWDIM만 정규화됨(stage_onehot은 normalizer.obs_min에 없어 skip)
        rgb_keys=RGB_KEYS,
        hdf5_cache_mode=cache_mode,
    )
    dataset = SyntheticStageDataset(base_dataset, seed=args.seed)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers >= 1,
    )
    print(f"[sanity] dataset len={len(dataset)} (매 샘플마다 stage 무작위 재배정)")

    policy = DiffusionPolicyImage(
        rgb_keys=RGB_KEYS, lowdim_keys=LOWDIM_KEYS, obs_dims=OBS_DIMS,
        obs_horizon=OBS_HORIZON, action_dim=ACTION_DIM, pred_horizon=PRED_HORIZON,
    ).to(device)
    print(f"[sanity] policy params: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M "
          f"(실제 square_stage 학습과 동일 아키텍처)")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.num_epochs * len(dataloader), 1)
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            batch = {
                "obs": {k: v.to(device) for k, v in batch["obs"].items()},
                "action": batch["action"].to(device),
                "action_mask": batch["action_mask"].to(device),
            }
            loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if global_step % args.log_every == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}", flush=True)
            global_step += 1

    print("\n[sanity] === 학습 끝, 검증 시작 ===")
    print("[sanity] 같은 이미지에 stage만 바꿔가며 예측 action이 인공 규칙(stage_to_action_value)을 따라가는지 확인")

    # 검증용 실제 이미지 배치 하나(라벨 내용은 안 씀 — 이미지·proprio만 재사용)
    policy.eval()
    raw_item = base_dataset[0]
    img_batch = {k: raw_item["obs"][k].unsqueeze(0).to(device) for k in RGB_KEYS}
    proprio_batch = {k: raw_item["obs"][k].unsqueeze(0).to(device) for k in BASE_LOWDIM}

    results = []
    with torch.no_grad():
        for stage_idx in range(NUM_STAGES):
            oh = onehot(np.array([stage_idx] * OBS_HORIZON))
            stage_t = torch.as_tensor(oh * 2 - 1, dtype=torch.float32).unsqueeze(0).to(device)
            obs_batch = dict(img_batch)
            obs_batch.update(proprio_batch)
            obs_batch["stage_onehot"] = stage_t

            pred = policy.predict_action_chunk(obs_batch)  # (1,Tp,7) normalized, already in [-1,1] scale
            pred_mean = pred[0].mean().item()
            expected = stage_to_action_value(stage_idx)
            results.append((stage_idx, expected, pred_mean))
            print(f"  stage={stage_idx}({STAGE_NAMES[stage_idx]:12s}) expected={expected:+.2f}  "
                  f"predicted_mean={pred_mean:+.2f}  |diff|={abs(expected - pred_mean):.3f}")

    diffs = [abs(e - p) for _, e, p in results]
    mean_abs_diff = float(np.mean(diffs))
    # expected가 stage마다 다른 값을 갖게 설계했으므로, "따라간다"의 기준 =
    # 예측값들 자체가 stage마다 유의미하게 갈라지는지(표준편차)로도 같이 본다.
    pred_std = float(np.std([p for _, _, p in results]))
    expected_std = float(np.std([e for _, e, p in results]))

    print(f"\n[sanity] mean |expected - predicted| = {mean_abs_diff:.3f} (0에 가까울수록 잘 따라감)")
    print(f"[sanity] predicted 값들의 stage간 표준편차 = {pred_std:.3f} "
          f"(참고: expected 표준편차 = {expected_std:.3f} — 이만큼 갈라져야 '따라간다')")
    if mean_abs_diff < 0.3 and pred_std > 0.3 * expected_std:
        print("[sanity] RESULT: 학습이 stage를 실제로 사용함 — 주입 방식 자체는 건강함 "
              "(Square null은 task 특성 때문이라는 해석에 힘 실림)")
    else:
        print("[sanity] RESULT: 이 강제 상황에서도 stage를 거의 못 씀 — 주입 방식/차원 희석 등 "
              "구조적 문제 의심, 서랍 실험 전에 먼저 짚어야 함")


if __name__ == "__main__":
    main()
