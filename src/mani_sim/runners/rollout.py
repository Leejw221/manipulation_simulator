"""Diffusion Policy 닫힌 루프(receding horizon) rollout 평가.

DP 표준 실행 방식: obs_horizon(To)개 과거 관측으로 pred_horizon(Tp) 길이 action chunk를
예측하고, 그중 action_horizon(Ta <= Tp)개만 실행한 뒤 다시 관측해 재예측한다.
"""

from collections import deque

import numpy as np
import torch


def rollout_policy(env, policy, normalizer, obs_keys, obs_horizon, action_horizon, max_steps, num_episodes, device):
    policy.eval()
    successes = []

    for _ in range(num_episodes):
        obs_raw = env.reset()
        obs_history = deque([obs_raw] * obs_horizon, maxlen=obs_horizon)

        success = False
        step_count = 0
        while step_count < max_steps:
            # normalizer 통계는 CPU 텐서 → CPU에서 정규화 후 device로 이동(intervention_rollout.py와 동일 관례)
            obs_batch = {
                key: torch.as_tensor(
                    np.stack([o[key] for o in obs_history]), dtype=torch.float32
                ).unsqueeze(0)
                for key in obs_keys
            }
            obs_batch = normalizer.normalize_obs(obs_batch)
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}

            action_chunk = policy.predict_action_chunk(obs_batch)  # (1, Tp, Da), normalized, on device
            action_chunk = normalizer.unnormalize_action(action_chunk[0].cpu())  # (Tp, Da)

            for t in range(action_horizon):
                action = action_chunk[t].detach().cpu().numpy()
                obs_raw, _reward, done, _info = env.step(action)
                obs_history.append(obs_raw)
                step_count += 1

                if env.is_success()["task"]:
                    success = True

                if success or done or step_count >= max_steps:
                    break

            if success or step_count >= max_steps:
                break

        successes.append(success)

    return {
        "success_rate": float(np.mean(successes)),
        "num_episodes": num_episodes,
        "num_successes": int(np.sum(successes)),
    }
