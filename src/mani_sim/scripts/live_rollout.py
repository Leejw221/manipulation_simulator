"""학습된 정책을 화면(DISPLAY)에 실시간 MuJoCo 뷰어로 띄워서 눈으로 확인.

주의: MUJOCO_GL=egl로 설정된 셸에서는 창이 안 뜬다(오프스크린 강제). 이 스크립트를 실행하는
터미널에서는 MUJOCO_GL을 unset하거나 설정하지 않은 채로 실행해야 한다. 원격 접속 중이고
X forwarding이 없으면 DISPLAY가 가리키는 화면에 창이 떠도 사용자 눈에는 안 보일 수 있다.

rollout.rollout_policy와 로직은 같지만(receding horizon), 매 env.step 후 env.render()를
호출해야 해서 별도 루프로 둔다.

사용:
    python -m mani_sim.scripts.live_rollout checkpoint_path=outputs/train/.../policy_epoch50.pt
"""

import os
import time
from collections import deque

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.envs.robomimic.factory import make_lowdim_env
from mani_sim.policies.diffusion.diffusion_policy import DiffusionPolicyLowDim


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cpu")

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
    policy.eval()

    env = make_lowdim_env(cfg.task.env_name, cfg.task.robots, cfg.task.obs_keys, render=True)
    obs_keys = cfg.task.obs_keys
    obs_horizon = cfg.policy.obs_horizon
    action_horizon = cfg.policy.action_horizon

    for episode in range(cfg.num_episodes):
        obs_raw = env.reset()
        env.render()
        obs_history = deque([obs_raw] * obs_horizon, maxlen=obs_horizon)

        success = False
        step_count = 0
        while step_count < cfg.max_steps:
            obs_batch = {
                key: torch.as_tensor(
                    np.stack([o[key] for o in obs_history]), dtype=torch.float32, device=device
                ).unsqueeze(0)
                for key in obs_keys
            }
            obs_batch = normalizer.normalize_obs(obs_batch)
            action_chunk = policy.predict_action_chunk(obs_batch)
            action_chunk = normalizer.unnormalize_action(action_chunk[0])

            for t in range(action_horizon):
                action = action_chunk[t].detach().cpu().numpy()
                obs_raw, _reward, done, _info = env.step(action)
                env.render()
                time.sleep(0.03)  # 사람 눈으로 따라갈 수 있게 살짝 속도 조절
                obs_history.append(obs_raw)
                step_count += 1

                if env.is_success()["task"]:
                    success = True
                if success or done or step_count >= cfg.max_steps:
                    break

            if success or step_count >= cfg.max_steps:
                break

        print(f"episode {episode}: success={success} steps={step_count}")

    env.env.close()  # 뷰어·sim을 명시적으로 정리 (안 하면 종료 시 GLXBadWindow 발생)


if __name__ == "__main__":
    main()
