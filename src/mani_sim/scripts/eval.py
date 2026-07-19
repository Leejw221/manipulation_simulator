"""통합 rollout 평가(DP: low_dim·image 공용) — outputs/final_eval.py를 대체.
OpenVLA는 predict_action_chunk가 아니라 매 스텝 단일 이미지 재계획이라 인터페이스가 달라
별도 스크립트(eval_openvla.py)로 남긴다(flare의 eval.py/eval_real.py 분리와 같은 이유).

사용:
    python -m mani_sim.scripts.eval checkpoint_path=outputs/train/square_diffusion_unet/policy_epoch300.pt
    python -m mani_sim.scripts.eval task=square_stage checkpoint_path=... render=true
"""

import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.factory import registry
from mani_sim.runners.rollout import rollout_policy
from mani_sim.utils.task_utils import is_image_task, make_eval_env, task_obs_keys

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy).to(device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    stats_path = os.path.join(os.path.dirname(cfg.checkpoint_path), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))

    env = make_eval_env(cfg.task)

    extra_obs_fn = extra_reset_fn = None
    if cfg.task.get("use_online_stage_tracker", False):
        from mani_sim.datasets.stage_labeler import OnlineStageTracker, onehot as stage_onehot
        tracker = OnlineStageTracker()

        def extra_obs_fn(obs_raw):
            import numpy as np
            return stage_onehot(np.array([tracker.step(obs_raw)]))[0]

        extra_reset_fn = tracker.reset

    on_episode_step = None
    if cfg.render:
        import cv2
        window_name = f"eval {os.path.basename(cfg.checkpoint_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        def on_episode_step(env_, ep, step):
            frame = env_.render(mode="rgb_array", height=640, width=640, camera_name="agentview")
            bgr = frame[:, :, ::-1].copy()
            cv2.putText(bgr, f"ep {ep + 1}/{cfg.num_episodes} step {step}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window_name, bgr)
            cv2.waitKey(1)

    if cfg.eval_seed is not None:
        np.random.seed(cfg.eval_seed)  # env reset(너트 초기 위치) 시퀀스 고정 — 공정 비교용

    metrics = rollout_policy(
        env=env, policy=policy, normalizer=normalizer,
        obs_keys=task_obs_keys(cfg.task), obs_horizon=cfg.policy.obs_horizon,
        action_horizon=cfg.policy.action_horizon, max_steps=cfg.max_steps, num_episodes=cfg.num_episodes,
        device=device, rgb_keys=cfg.task.rgb_keys if is_image_task(cfg.task) else (),
        extra_obs_fn=extra_obs_fn, extra_obs_reset_fn=extra_reset_fn, on_episode_step=on_episode_step,
    )
    if cfg.render:
        import cv2
        cv2.destroyAllWindows()
    env.env.close()

    logger.info(f"checkpoint={cfg.checkpoint_path} {metrics}")
    print(metrics)
    return metrics


if __name__ == "__main__":
    main()
