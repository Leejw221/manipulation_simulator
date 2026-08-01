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


def maybe_speed_up_inference(policy, deploy_num_inference_steps):
    """실시간/라이브 뷰어용 추론 가속(원래 collect.py 전용이었다가 2026-07-27 공용으로 이동
    - eval.py의 render=true 뷰어도 옛날 rollout_policy()의 동기 청크 계산(DDPM 100-step,
    청크당 ~500ms) 그대로라 collect.py가 겪었던 것과 똑같이 끊겨서 여기로 옮겨 재사용).
    학습/성공률 측정용 호출(diffusion_trainer.py의 주기 eval, rq1_weight_vs_success.py 등,
    전부 deploy_num_inference_steps 인자 자체를 안 넘김)은 전혀 안 건드림 - 호출부가 명시적으로
    값을 넘길 때만(collect.py의 배포, eval.py의 render=true) 적용되는 opt-in.

    이미 학습된 noise-prediction 네트워크(가중치)는 그대로 두고 **추론 시점 스케줄러만**
    DDIM으로 바꿔치기한다(DDIM은 DDPM으로 학습된 모델에 재학습 없이 그대로 적용 가능한 결정론적
    샘플러 - low_dim 정책이 이미 이 조합을 기본으로 씀). deploy_num_inference_steps=None이면
    아무 것도 안 바꿈(기존 동작 그대로).

    실측 배경: Transport 이미지 정책(DDPM 100-step)의 청크 계산이 평균 513ms 걸려 pred_horizon
    (16스텝) x control_fps(20Hz) 예산인 800ms의 64%를 먹어치웠고, 순차 요청(비파이프라인)
    구조라 거의 매 청크마다 merger가 바닥나 stall이 발생함(2026-07-27 실측, HANDOFF 참고)."""
    if deploy_num_inference_steps is None or not hasattr(policy, "inference_scheduler"):
        return
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    train_cfg = policy.train_scheduler.config
    policy.inference_scheduler = DDIMScheduler(
        num_train_timesteps=train_cfg.num_train_timesteps,
        beta_schedule=train_cfg.beta_schedule,
        clip_sample=True,
        prediction_type="epsilon",
    )
    policy.num_inference_steps = deploy_num_inference_steps
    print(
        f"[추론 가속] 추론 스케줄러를 DDIM({deploy_num_inference_steps} step)으로 교체 "
        f"(학습 가중치·성공률 평가용 DDPM {train_cfg.num_train_timesteps}-step은 그대로)"
    )


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
    save_trajectory_keys=None, verbose=False,
):
    """extra_obs_fn: 합성 obs 키 계산 콜백(예: stage_onehot). extra_obs_reset_fn: 에피소드
    시작마다 그 콜백의 내부 상태를 초기화(예: OnlineStageTracker.reset()). on_episode_step:
    디버깅/시각화용 훅(env, episode_idx, step) — 화면 렌더 등에 사용. False를 반환하면 rollout
    전체를 그 자리에서 깨끗이 중단한다(예: render 창에서 ESC/닫기 감지 — Ctrl+C만으로는
    MuJoCo GL 컨텍스트가 지저분하게 남을 때가 있어서 추가함, 2026-07-29). 문자열 "skip"을
    반환하면 rollout 전체는 그대로 두고 **현재 에피소드만** 실패(success=False)로 즉시
    종료하고 다음 에피소드로 넘어간다(예: render 창에서 Enter — 사람이 보다가 이미 실패라고
    판단했을 때 max_steps 타임아웃까지 기다릴 필요 없게, 2026-07-30).

    save_trajectory_keys: None이면 기존과 동일(집계만 반환). 키 목록(예: ["object",
    "robot0_gripper_qpos","robot1_gripper_qpos"])을 주면 에피소드별 (T, ...) 배열로 저장해
    반환 dict의 "trajectories"에 담는다 — RQ1(지표-성공률 상관) 분석용, 학습에는 안 씀.

    verbose=True면 에피소드가 끝날 때마다 `ep {i}: steps=... success=...`를 즉시 출력한다
    (2026-07-27 추가 - eval.py에서 render=true로 여러 에피소드를 눈으로 지켜볼 때, 화면만
    보고 일일이 세지 않아도 되게). 기본 False라 diffusion_trainer.py의 학습 중 주기 평가·
    rq1_weight_vs_success.py 등 기존 호출부는 로그가 안 늘어남. 에피소드별 success/steps는
    verbose와 무관하게 항상 가볍게(트래젝토리 전체 저장 아님) `metrics["episodes"]`로 반환.

    extra_obs_fn/extra_obs_reset_fn/save_trajectory_keys/on_episode_step이 전부 없으면
    (=eval.py의 기본 헤드리스 호출) intervention_rollout.collect_episode()에 그대로
    위임한다(2026-07-28) — collect.py 배포와 "청크→액션 실행" 로직을 완전히 같은 코드로
    공유하기 위함(둘이 따로 구현돼 있으면 한쪽만 고치고 잊는 사고가 또 날 수 있음, 실제로
    DDIM 가속이 collect.py에만 들어갔다가 나중에 발견된 전례가 있음). 저 네 기능은
    collect_episode가 지원하지 않아서(예: save_trajectory_keys는 obs_keys 밖의 키까지
    저장해야 하는데 PICO 배포엔 그 기능이 전혀 필요 없음 - 억지로 넣으면 배포 스크립트가
    연구 분석 전용 기능으로 지저분해짐) 이 경우만 기존 구현을 그대로 유지."""
    if (extra_obs_fn is None and extra_obs_reset_fn is None
            and save_trajectory_keys is None and on_episode_step is None):
        from mani_sim.runners.intervention_rollout import _predict_chunk, collect_episode

        policy.eval()

        def predict_fn(obs_history):
            return _predict_chunk(policy, normalizer, obs_history, obs_keys, device, rgb_keys=rgb_keys)

        successes, episodes = [], []
        for ep in range(num_episodes):
            result = collect_episode(
                env, policy, normalizer, obs_keys, obs_horizon, action_horizon, device,
                lambda step, obs_raw: None, max_steps=max_steps, render=False, control_fps=0.0,
                predict_fn=predict_fn, async_infer=False, print_diagnostics=False,
            )
            success = result["success"]
            step_count = len(result["actions"])
            successes.append(success)
            episodes.append({"episode": ep, "success": success, "steps": step_count})
            if verbose:
                print(f"ep {ep}: steps={step_count} success={success}")
        return {
            "success_rate": float(np.mean(successes)),
            "num_episodes": num_episodes,
            "num_successes": int(np.sum(successes)),
            "episodes": episodes,
        }

    policy.eval()
    successes = []
    episodes = []
    trajectories = [] if save_trajectory_keys is not None else None
    interrupted = False

    for ep in range(num_episodes):
        obs_raw = env.reset()
        if extra_obs_reset_fn is not None:
            extra_obs_reset_fn()
        obs_history = deque([obs_raw] * obs_horizon, maxlen=obs_horizon)
        skip = False
        step0_signal = on_episode_step(env, ep, 0) if on_episode_step is not None else None
        if step0_signal is False:
            interrupted = True
            break
        elif step0_signal == "skip":
            skip = True
        ep_log = {k: [np.asarray(obs_raw[k], dtype=np.float32)] for k in save_trajectory_keys} \
            if save_trajectory_keys is not None else None

        success = False
        step_count = 0
        while not skip and step_count < max_steps:
            obs_batch = _build_obs_batch(obs_history, obs_keys, rgb_keys, normalizer, device, extra_obs_fn)

            with torch.no_grad():
                action_chunk = policy.predict_action_chunk(obs_batch)  # (1, Tp, Da), normalized, on device
            action_chunk = normalizer.unnormalize_action(action_chunk[0].cpu())  # (Tp, Da)

            for t in range(action_horizon):
                action = action_chunk[t].detach().cpu().numpy()
                obs_raw, _reward, done, _info = env.step(action)
                obs_history.append(obs_raw)
                step_count += 1
                if ep_log is not None:
                    for k in save_trajectory_keys:
                        ep_log[k].append(np.asarray(obs_raw[k], dtype=np.float32))
                # on_episode_step may return False to request an early, clean stop for the whole
                # rollout (e.g. ESC/window close -- Ctrl+C alone can leave the MuJoCo GL context
                # stuck), or "skip" to abandon just this episode as a failure and move to the next
                # (e.g. Enter key -- for when a human watching render=true can already tell it's
                # failed and doesn't want to wait out max_steps).
                step_signal = on_episode_step(env, ep, step_count) if on_episode_step is not None else None
                if step_signal is False:
                    interrupted = True
                elif step_signal == "skip":
                    skip = True

                if env.is_success()["task"]:
                    success = True

                if success or done or interrupted or skip or step_count >= max_steps:
                    break

            if success or interrupted or skip or step_count >= max_steps:
                break

        successes.append(success)
        episodes.append({"episode": ep, "success": success, "steps": step_count})
        if verbose:
            print(f"ep {ep}: steps={step_count} success={success}")
        if ep_log is not None:
            trajectories.append({
                "success": success,
                **{k: np.stack(v) for k, v in ep_log.items()},
            })
        if interrupted:
            break

    metrics = {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "num_episodes": len(successes),
        "num_successes": int(np.sum(successes)),
        "episodes": episodes,
        "interrupted": interrupted,
    }
    if trajectories is not None:
        metrics["trajectories"] = trajectories
    return metrics


def rollout_policy_async(
    env, policy, normalizer, obs_keys, obs_horizon, action_horizon, max_steps, num_episodes, device,
    rgb_keys=(), verbose=False, merger_name="overwrite", te_coeff=0.01, render=False, control_fps=0.0,
):
    """rollout_policy()와 같은 metrics를 반환하지만, action_horizon 스텝마다 멈춰서
    재계획하는 대신 moai_policy 스타일 연속 재계획(매 스텝 최신 관측으로 재추론)을 쓴다.
    collect.py의 PICO 배포에서만 쓰던 intervention_rollout.collect_episode()의 async_infer
    경로를 그대로 재사용(2026-07-28) — round0 배포 때 체감 성공률이 헤드리스 eval보다
    좋아 보인다는 관찰이 나와, 8스텝 열린 루프 재생 대신 이 방식으로도 성공률을 잴 수
    있게 함. stage tracker 등 extra_obs_fn 계열은 아직 미지원(필요해지면 추가)."""
    from mani_sim.runners.intervention_rollout import _predict_chunk, collect_episode

    policy.eval()

    def predict_fn(obs_history):
        return _predict_chunk(policy, normalizer, obs_history, obs_keys, device, rgb_keys=rgb_keys)

    def no_intervention(step, obs_raw):
        return None

    successes, episodes = [], []
    for ep in range(num_episodes):
        result = collect_episode(
            env, policy, normalizer, obs_keys, obs_horizon, action_horizon, device,
            no_intervention, max_steps=max_steps, render=render, control_fps=control_fps,
            predict_fn=predict_fn, async_infer=True, merger_name=merger_name, te_coeff=te_coeff,
        )
        success = result["success"]
        step_count = len(result["actions"])
        successes.append(success)
        episodes.append({"episode": ep, "success": success, "steps": step_count})
        if verbose:
            print(f"ep {ep}: steps={step_count} success={success}")

    return {
        "success_rate": float(np.mean(successes)),
        "num_episodes": num_episodes,
        "num_successes": int(np.sum(successes)),
        "episodes": episodes,
    }
