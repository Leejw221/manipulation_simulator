"""Diffusion Policy(low-dim) rollout 평가 스크립트.

사용:
    python -m mani_sim.scripts.eval checkpoint_path=outputs/train/.../policy_epoch10.pt
"""

import os

import hydra
import torch
from omegaconf import DictConfig

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.envs.robomimic.factory import make_lowdim_env
from mani_sim.policies.diffusion.diffusion_policy import DiffusionPolicyLowDim
from mani_sim.runners.rollout import rollout_policy


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)

    stats_path = os.path.join(os.path.dirname(cfg.checkpoint_path), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))

    policy = DiffusionPolicyLowDim(
        obs_keys=cfg.task.obs_keys,
        obs_dims=cfg.task.obs_dims,
        obs_horizon=cfg.policy.obs_horizon,
        action_dim=cfg.task.action_dim,
        pred_horizon=cfg.policy.pred_horizon,
        down_dims=cfg.policy.down_dims,
        kernel_size=cfg.policy.kernel_size,
        n_groups=cfg.policy.n_groups,
        diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
        num_train_timesteps=cfg.policy.num_train_timesteps,
        beta_schedule=cfg.policy.beta_schedule,
        num_inference_steps=cfg.policy.num_inference_steps,
    ).to(device)
    policy.load_state_dict(ckpt["model"])

    env = make_lowdim_env(cfg.task.env_name, cfg.task.robots, cfg.task.obs_keys)

    metrics = rollout_policy(
        env=env,
        policy=policy,
        normalizer=normalizer,
        obs_keys=cfg.task.obs_keys,
        obs_horizon=cfg.policy.obs_horizon,
        action_horizon=cfg.policy.action_horizon,
        max_steps=cfg.max_steps,
        num_episodes=cfg.num_episodes,
        device=device,
    )
    print(metrics)
    return metrics


if __name__ == "__main__":
    main()
