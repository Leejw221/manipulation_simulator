"""Diffusion Policy 닫힌 루프(receding horizon) rollout 평가.

DP 표준 실행 방식: obs_horizon(To)개 과거 관측으로 pred_horizon(Tp) 길이 action chunk를
예측하고, 그중 action_horizon(Ta <= Tp)개만 실행한 뒤 다시 관측해 재예측한다.

low_dim/image 공용(rgb_keys가 비어있으면 기존 low_dim 동작 그대로). image 키는 정규화하지
않고(rgb→[0,1]은 env가 이미 반환), CHW로만 변환해서 쌓는다 — 학습 데이터셋 어댑터
(datasets/robomimic_dataset.py)와 동일 규약.

stage_onehot처럼 env가 직접 안 주는 합성 obs 키는 `extra_obs_fn(obs_raw) -> value` 콜백으로
매 스텝 계산해 채운다(예: OnlineStageTracker) — 이 함수는 stage를 모른 채로 남긴다.
"""

from collections import deque

import numpy as np
import torch


def _to_chw01(img):
    """(H,W,C) uint8 또는 float → (C,H,W) float[0,1]. rgb 키 전용."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
    if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
        img = np.transpose(img, (2, 0, 1))
    return img


def _build_obs_batch(obs_history, obs_keys, rgb_keys, normalizer, device, extra_obs_fn):
    rgb_keys = set(rgb_keys)
    batch = {}
    for key in obs_keys:
        if key in rgb_keys:
            frames = np.stack([_to_chw01(o[key]) for o in obs_history])
        elif extra_obs_fn is not None and key not in obs_history[0]:
            frames = np.stack([extra_obs_fn(o) for o in obs_history])
        else:
            frames = np.stack([np.asarray(o[key], dtype=np.float32) for o in obs_history])
        batch[key] = torch.as_tensor(frames, dtype=torch.float32).unsqueeze(0)

    lowdim = {k: v for k, v in batch.items() if k not in rgb_keys}
    batch.update(normalizer.normalize_obs(lowdim))
    return {k: v.to(device) for k, v in batch.items()}


def rollout_policy(
    env, policy, normalizer, obs_keys, obs_horizon, action_horizon, max_steps, num_episodes, device,
    rgb_keys=(), extra_obs_fn=None, extra_obs_reset_fn=None, on_episode_step=None,
):
    """extra_obs_fn: 합성 obs 키 계산 콜백(예: stage_onehot). extra_obs_reset_fn: 에피소드
    시작마다 그 콜백의 내부 상태를 초기화(예: OnlineStageTracker.reset()). on_episode_step:
    디버깅/시각화용 훅(env, episode_idx, step) — 화면 렌더 등에 사용, 로직에 영향 없음."""
    policy.eval()
    successes = []

    for ep in range(num_episodes):
        obs_raw = env.reset()
        if extra_obs_reset_fn is not None:
            extra_obs_reset_fn()
        obs_history = deque([obs_raw] * obs_horizon, maxlen=obs_horizon)
        if on_episode_step is not None:
            on_episode_step(env, ep, 0)

        success = False
        step_count = 0
        while step_count < max_steps:
            obs_batch = _build_obs_batch(obs_history, obs_keys, rgb_keys, normalizer, device, extra_obs_fn)

            with torch.no_grad():
                action_chunk = policy.predict_action_chunk(obs_batch)  # (1, Tp, Da), normalized, on device
            action_chunk = normalizer.unnormalize_action(action_chunk[0].cpu())  # (Tp, Da)

            for t in range(action_horizon):
                action = action_chunk[t].detach().cpu().numpy()
                obs_raw, _reward, done, _info = env.step(action)
                obs_history.append(obs_raw)
                step_count += 1
                if on_episode_step is not None:
                    on_episode_step(env, ep, step_count)

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
