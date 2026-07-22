"""RQ1: 가중치 지표가 성공률과 실제로 관련 있는가?

policy_evaluate가 아니라 별도 롤아웃(성공/실패 섞임)에서 궤적을 저장한 뒤, 에피소드별로
phase규칙/RABC-progress 지표를 계산해 성공 vs 실패 그룹 간 분포를 비교한다.

행동오차(APO)는 여기 없다 — 롤아웃 중엔 비교할 데모 정답 행동이 없어 정의 자체가 안 됨
(demo 궤적에서만 계산 가능, action_error.py 참고). 행동분산은 diffusion 재샘플링이 필요해
스텝마다 추가 forward pass가 드니 이번 1차 분석에선 제외(다음 단계로 미룸).

--use-sarm: heuristic stage_progress() 대신 학습된 SARM(v3) reward model로 progress 계산.
SARM은 200개 순수 데모(항상 성공)로만 학습됐는데, 롤아웃엔 실패 궤적(분포 밖)이 섞여 있음 —
"OOD인 실패 궤적에서도 progress가 정체/하락하는가"를 미리 보는 가벼운 사전 점검
(2026-07-22, round-1 실제 개입 데이터 검증 전 단계).

사용:
    python -m mani_sim.scripts.rq1_weight_vs_success \
        --checkpoint outputs/train/transport_diffusion_bs128_seed0/policy_epoch100.pt \
        --task transport --num-episodes 100 --max-steps 700 \
        --use-sarm --sarm-checkpoint outputs/sarm/transport_v2/policy_epoch3.pt
"""

import argparse

import numpy as np
import torch
from hydra import compose, initialize

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.datasets.sarm_dataset import get_frame_indices
from mani_sim.datasets.stage_labeler import label_stages_transport, stage_progress
from mani_sim.factory import registry
from mani_sim.policies.sarm.clip_text_encoder import ClipTextEncoder
from mani_sim.policies.sarm.sarm_reward_model import ClipVisionEncoder
from mani_sim.policies.sarm.sarm_transformer import SARMFullModel, StageTransformer, SubtaskTransformer
from mani_sim.runners.rollout import rollout_policy
from mani_sim.utils.task_utils import make_eval_env, task_obs_keys


TRAJ_KEYS = ["object", "robot0_gripper_qpos", "robot1_gripper_qpos"]
CRITICAL_STAGES = (3,)  # payload_handoff, 5단계 스킴

SARM_IMAGE_KEY = "agentview_image"
SARM_STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                    "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]


def episode_stats(traj, sarm=None):
    obs = {k: traj[k] for k in TRAJ_KEYS}
    stage = label_stages_transport(obs)
    num_stages = int(stage.max()) + 1

    if sarm is None:
        progress = stage_progress(stage, num=num_stages)
    else:
        progress = sarm.predict_episode(traj)

    frac_critical = float(np.mean(np.isin(stage, CRITICAL_STAGES)))
    final_progress = float(progress[-1])
    delta = np.diff(progress, prepend=progress[0])
    stall_frac = float(np.mean(np.abs(delta) < 1e-4))
    mean_delta = float(delta.mean())

    return {
        "success": bool(traj["success"]),
        "T": len(stage),
        "frac_critical_stage": frac_critical,
        "final_progress": final_progress,
        "stall_frac": stall_frac,
        "mean_progress_delta": mean_delta,
    }


class SarmEpisodeProgress:
    """롤아웃 저장 trajectory(dict of (T,...) arrays)에서 SARM v3로 progress(T,) 계산."""

    def __init__(self, checkpoint, device, d_model=384, n_layers=6, n_heads=6,
                 n_obs_steps=8, frame_gap=15, num_stages=5,
                 task_description="remove the lid, hand off the payload between the two arms, "
                                   "and deliver it to the target bin"):
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.frame_gap = frame_gap

        self.vision_encoder = ClipVisionEncoder(freeze=True).to(device).eval()
        self.text_encoder = ClipTextEncoder(freeze=True).to(device).eval()
        with torch.no_grad():
            self.lang_emb_single = self.text_encoder([task_description], device=device)

        state_dim = sum({"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2,
                          "robot1_eef_pos": 3, "robot1_eef_quat": 4, "robot1_gripper_qpos": 2}[k]
                         for k in SARM_STATE_KEYS)
        stage_model = StageTransformer(
            d_model=d_model, vis_emb_dim=self.vision_encoder.out_dim, text_emb_dim=self.text_encoder.out_dim,
            state_dim=state_dim, n_layers=n_layers, n_heads=n_heads, num_cameras=1,
            num_classes_sparse=num_stages,
        ).to(device)
        subtask_model = SubtaskTransformer(
            d_model=d_model, vis_emb_dim=self.vision_encoder.out_dim, text_emb_dim=self.text_encoder.out_dim,
            state_dim=state_dim, n_layers=n_layers, n_heads=n_heads, num_cameras=1,
        ).to(device)
        self.model = SARMFullModel(stage_model, subtask_model).to(device)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        print(f"[rq1] SARM checkpoint loaded: {checkpoint}", flush=True)

    @torch.no_grad()
    def predict_episode(self, traj, batch_size=64):
        T = len(traj[SARM_IMAGE_KEY])
        state_full = np.concatenate([np.asarray(traj[k], dtype=np.float32) for k in SARM_STATE_KEYS], axis=1)
        progress = np.zeros(T, dtype=np.float32)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            targets = list(range(start, end))
            windows = [get_frame_indices(t, self.n_obs_steps, self.frame_gap, 0, T - 1) for t in targets]
            flat_idx = sorted(set(i for w in windows for i in w))
            pos_of = {v: k for k, v in enumerate(flat_idx)}

            # rollout_policy가 저장한 raw obs는 robomimic ObsUtils.process_obs가 이미 적용된
            # CHW float[0,1](hdf5의 HWC uint8 저장 포맷과 다름 — eval.py `_to_pil` 참고).
            img_raw = np.asarray(traj[SARM_IMAGE_KEY][flat_idx], dtype=np.float32)
            images = torch.as_tensor(img_raw, device=self.device)
            state = torch.as_tensor(state_full[flat_idx], device=self.device)
            img_emb_flat = self.vision_encoder(images)

            B = len(windows)
            W = self.n_obs_steps + 1
            img_emb = torch.stack([img_emb_flat[[pos_of[i] for i in w]] for w in windows]).unsqueeze(1)
            state_seq = torch.stack([state[[pos_of[i] for i in w]] for w in windows])
            lengths = torch.full((B,), W, dtype=torch.long, device=self.device)
            lang_emb = self.lang_emb_single.expand(B, -1)

            p_batch = self.model.predict_progress(img_emb, lang_emb, state_seq, lengths)
            progress[start:end] = p_batch[:, -1].cpu().numpy()
        return progress


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="transport")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=700)
    p.add_argument("--use-sarm", action="store_true")
    p.add_argument("--sarm-checkpoint", default=None)
    p.add_argument("--eval-seed", type=int, default=None)
    args = p.parse_args()

    if args.eval_seed is not None:
        np.random.seed(args.eval_seed)

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="eval", overrides=[f"task={args.task}", "policy_name=diffusion"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "/".join(args.checkpoint.split("/")[:-1])
    normalizer = MinMaxNormalizer(load_stats(f"{out_dir}/normalization_stats.json"))

    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"[rq1] loaded {args.checkpoint}")

    sarm = None
    if args.use_sarm:
        assert args.sarm_checkpoint, "--use-sarm이면 --sarm-checkpoint 필요"
        sarm = SarmEpisodeProgress(args.sarm_checkpoint, device)

    traj_keys = list(TRAJ_KEYS)
    if args.use_sarm:
        traj_keys = list(dict.fromkeys(traj_keys + [SARM_IMAGE_KEY] + SARM_STATE_KEYS))

    env = make_eval_env(cfg.task)
    obs_keys = task_obs_keys(cfg.task)

    metrics = rollout_policy(
        env=env, policy=policy, normalizer=normalizer, obs_keys=obs_keys,
        obs_horizon=cfg.policy.obs_horizon, action_horizon=cfg.policy.action_horizon,
        max_steps=args.max_steps, num_episodes=args.num_episodes, device=device,
        rgb_keys=cfg.task.rgb_keys, save_trajectory_keys=traj_keys,
    )
    env.env.close()
    print(f"[rq1] rollout success_rate={metrics['success_rate']:.3f} "
          f"({metrics['num_successes']}/{metrics['num_episodes']})")

    rows = [episode_stats(t, sarm=sarm) for t in metrics["trajectories"]]
    succ = [r for r in rows if r["success"]]
    fail = [r for r in rows if not r["success"]]
    print(f"[rq1] success={len(succ)} fail={len(fail)}")

    for key in ["frac_critical_stage", "final_progress", "stall_frac", "mean_progress_delta"]:
        s_vals = np.array([r[key] for r in succ])
        f_vals = np.array([r[key] for r in fail])
        s_str = f"{s_vals.mean():.4f}±{s_vals.std():.4f}" if len(s_vals) else "n/a"
        f_str = f"{f_vals.mean():.4f}±{f_vals.std():.4f}" if len(f_vals) else "n/a"
        print(f"[rq1] {key}: success={s_str}  fail={f_str}")


if __name__ == "__main__":
    main()
