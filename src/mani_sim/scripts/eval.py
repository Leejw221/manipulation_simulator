"""통합 rollout 평가(DP: low_dim·image 공용) — outputs/final_eval.py를 대체.
render=true면 live_rollout.py가 하던 것과 동일하게 화면에 띄운다(low_dim=mjviewer 온스크린,
image=cv2 오프스크린 렌더 — 동시 사용 시 GL 컨텍스트 충돌로 세그폴트하므로 분기, live_rollout.py
에 있던 지뢰 그대로 유지). OpenVLA는 predict_action_chunk가 아니라 매 스텝 단일 이미지
재계획이라 인터페이스가 달라 policy_name=="openvla"일 때 별도 루프로 분기한다.

사용:
    python -m mani_sim.scripts.eval checkpoint_path=outputs/train/square_diffusion_unet/policy_epoch300.pt
    python -m mani_sim.scripts.eval task=square_stage checkpoint_path=... render=true
    unset MUJOCO_GL  # low_dim 화면(mjviewer)일 때만 필요
    python -m mani_sim.scripts.eval task=square policy_name=openvla policy=openvla \
        checkpoint_path=outputs/train/square_openvla/lora_final/policy \
        stats_path=outputs/train/square_openvla/normalization_stats.json save_gif=outputs/openvla_rollout.gif
"""

import json
import logging
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.factory import registry
from mani_sim.runners.rollout import rollout_policy
from mani_sim.utils.checkpoints import load_run_config
from mani_sim.utils.task_utils import is_image_task, is_piper_task, make_eval_env, task_obs_keys

logger = logging.getLogger(__name__)


def _apply_run_config(cfg):
    """checkpoint_path 옆 run_config.yaml(학습 시 자동 저장)이 있으면 task/policy/policy_name을
    거기서 통째로 교체해 학습·추론 설정이 어긋나는 걸 막는다(2026-07-25). 없으면(구 체크포인트,
    openvla는 아직 저장 안 함) 아무것도 안 하고 기존 CLI 값 그대로 진행.

    ⚠ OmegaConf.merge가 아니라 직접 대입으로 교체한다 — merge는 하위 dict까지 깊은 병합이라,
    eval.yaml 기본 task(square)에만 있고 실제 학습 task(예: transport)엔 없는 필드가
    살아남는 걸 실측으로 확인함(예: transport 기본값일 때 square 체크포인트를 평가하면
    transport의 env_kwargs가 안 지워지고 섞여 들어감 — 정확히 7/21에 사고 낸 그 종류의
    필드). 직접 대입은 완전 교체라 이 문제가 없음(마찬가지로 실측 확인)."""
    if not cfg.get("use_run_config", True) or not cfg.checkpoint_path:
        return cfg
    saved = load_run_config(cfg.checkpoint_path)
    if saved is None:
        return cfg
    cfg.task = saved.task
    cfg.policy = saved.policy
    cfg.policy_name = saved.policy_name
    logger.info(f"run_config.yaml에서 자동 적용: task={saved.task.name} policy_name={saved.policy_name}")
    return cfg


def _to_pil(img_chw01):
    """make_image_env가 주는 CHW float[0,1](robomimic ObsUtils 규약) -> PIL Image(HWC uint8).
    hdf5 raw(HWC uint8)와 형식이 달라 변환 필요(EXP-01에 기록된 지뢰, OpenVLA rollout에서
    처음 실제로 밟음)."""
    from PIL import Image
    img_hwc = np.transpose(np.asarray(img_chw01), (1, 2, 0))
    img_uint8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


def _run_dp_eval(cfg, device):
    """DiffusionPolicy/BC 등 predict_action_chunk 인터페이스 공용 경로(receding-horizon)."""
    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy).to(device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    stats_path = cfg.stats_path or os.path.join(os.path.dirname(cfg.checkpoint_path), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))

    extra_obs_fn = extra_reset_fn = None
    if cfg.task.get("use_online_stage_tracker", False):
        from mani_sim.datasets.stage_labeler import (
            ShiftedStageTracker, make_online_tracker, num_stages_for_task, onehot as stage_onehot,
        )
        tracker = make_online_tracker(cfg.task.name)
        n_stages = num_stages_for_task(cfg.task.name)
        if cfg.get("stage_shift_target", None) is not None:
            tracker = ShiftedStageTracker(
                tracker, target_stage=cfg.stage_shift_target, window=cfg.stage_shift_window,
                num_stages=n_stages,
            )
            logger.info(
                f"stage shift 강건성 체크: target_stage={cfg.stage_shift_target} "
                f"window={cfg.stage_shift_window}"
            )

        def extra_obs_fn(obs_raw):
            return stage_onehot(np.array([tracker.step(obs_raw)]), num=n_stages)[0]

        extra_reset_fn = tracker.reset

    if cfg.render and is_image_task(cfg.task):
        # cv2.namedWindow가 이 PC에서 MuJoCo 오프스크린 GL 컨텍스트와 충돌해 멈춤(2026-07-20
        # 실측 — 렌더 자체는 14ms/frame으로 빠름, cv2 창 생성 시점부터 CPU 100%로 무한 대기,
        # SIGINT도 안 먹힘 — native 루프 추정, 원인 미해결). image task는 save_gif로 대체.
        raise RuntimeError(
            "render=true(cv2 라이브 창)는 이 PC에서 행 걸림(2026-07-20 확인) — 대신 "
            "save_gif=<path>를 쓰세요(오프스크린 렌더만 사용, cv2 창 없음)."
        )

    on_episode_step = None
    gif_frames = None
    if cfg.save_gif and is_image_task(cfg.task):
        env = make_eval_env(cfg.task)
        gif_frames = []
        # Piper task는 3인칭 카메라 이름이 다르다(camera_names.front="front_cam" vs
        # robosuite의 "agentview") - env_backend로 분기해서 고른다(2026-07-26).
        gif_camera = cfg.task.camera_names.front if is_piper_task(cfg.task) else "agentview"

        def on_episode_step(env_, ep, step):
            if ep == 0 and step % 4 == 0:
                frame = env_.render(mode="rgb_array", height=256, width=256, camera_name=gif_camera)
                from PIL import Image
                gif_frames.append(Image.fromarray(frame))
    elif cfg.render:
        # low_dim: MuJoCo 네이티브 mjviewer(온스크린). (image+render는 위에서 이미 막힘)
        env = make_eval_env(cfg.task, render=True)

        def on_episode_step(env_, ep, step):
            env_.render()
            time.sleep(0.03)  # 사람 눈으로 따라갈 수 있게 살짝 속도 조절
    else:
        env = make_eval_env(cfg.task)

    if cfg.eval_seed is not None:
        np.random.seed(cfg.eval_seed)  # env reset(너트 초기 위치) 시퀀스 고정 — 공정 비교용

    metrics = rollout_policy(
        env=env, policy=policy, normalizer=normalizer,
        obs_keys=task_obs_keys(cfg.task), obs_horizon=cfg.policy.obs_horizon,
        action_horizon=cfg.policy.action_horizon, max_steps=cfg.max_steps, num_episodes=cfg.num_episodes,
        device=device, rgb_keys=cfg.task.rgb_keys if is_image_task(cfg.task) else (),
        extra_obs_fn=extra_obs_fn, extra_obs_reset_fn=extra_reset_fn, on_episode_step=on_episode_step,
    )
    if gif_frames:
        gif_frames[0].save(cfg.save_gif, save_all=True, append_images=gif_frames[1:],
                            duration=60, loop=0, optimize=True)
        logger.info(f"GIF 저장: {cfg.save_gif} ({len(gif_frames)} 프레임)")
    env.env.close()
    return metrics


def _run_openvla_eval(cfg):
    """OpenVLA는 predict_action_chunk가 아니라 매 스텝 단일 이미지로 재계획(원 논문 방식) —
    rollout_policy()의 청크 인터페이스와 안 맞아 별도 루프."""
    from mani_sim.datasets.openvla_dataset import SQUARE_INSTRUCTION

    if cfg.stats_path is None:
        raise ValueError(
            "policy_name=openvla는 stats_path를 명시해야 함 — checkpoint_path가 "
            "{output_dir}/lora_final/policy라 dirname 추정(DP 관례)이 안 맞음"
            "(stats는 {output_dir}/normalization_stats.json)."
        )

    cfg.policy.policy_adapter_path = cfg.checkpoint_path
    normalizer = MinMaxNormalizer(load_stats(cfg.stats_path))
    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy)
    policy.model.eval()
    logger.info(f"loaded adapter: {cfg.checkpoint_path}")

    # OpenVLA는 224 사전학습 해상도를 기대(task.image_size가 84여도 여기선 224로 별도 렌더).
    # (2026-07-25: env_kwargs를 여기서 아예 안 넘기던 버그도 make_eval_env 경유로 같이 해결됨
    # — Transport 등 env_kwargs가 실제로 의미 있는 task에서 OpenVLA를 평가하면 로봇이
    # 잘못된 위치에 스폰됐을 것, collect.py와 같은 종류.)
    env = make_eval_env(cfg.task, image_size_override=224)

    if cfg.eval_seed is not None:
        np.random.seed(cfg.eval_seed)

    image_key = cfg.task.rgb_keys[0]
    instruction = cfg.openvla_instruction or SQUARE_INSTRUCTION

    successes = []
    t0 = time.time()
    for ep in range(cfg.num_episodes):
        obs = env.reset()
        success = False
        steps = 0
        gif_frames = [] if (cfg.save_gif and ep == 0) else None
        while steps < cfg.max_steps:
            img = _to_pil(obs[image_key])
            action_norm = policy.predict_action(img, instruction)
            action = normalizer.unnormalize_action(torch.as_tensor(action_norm, dtype=torch.float32)).numpy()
            obs, _r, done, _i = env.step(action)
            steps += 1
            if gif_frames is not None and steps % 2 == 0:  # 매 2프레임마다(용량 절반)
                frame = env.render(mode="rgb_array", height=224, width=224, camera_name="agentview")
                from PIL import Image
                gif_frames.append(Image.fromarray(frame))
            if env.is_success()["task"]:
                success = True
            if success or done or steps >= cfg.max_steps:
                break
        if gif_frames:
            gif_frames[0].save(cfg.save_gif, save_all=True, append_images=gif_frames[1:],
                                duration=60, loop=0, optimize=True)
            logger.info(f"GIF 저장: {cfg.save_gif} ({len(gif_frames)} 프레임)")
        successes.append(success)
        logger.info(f"episode {ep}: steps={steps} success={success}")

    env.env.close()
    logger.info(f"elapsed={time.time() - t0:.0f}s")
    return {
        "success_rate": float(np.mean(successes)),
        "num_episodes": cfg.num_episodes,
        "num_successes": int(np.sum(successes)),
    }


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    cfg = _apply_run_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if cfg.policy_name == "openvla":
        metrics = _run_openvla_eval(cfg)
    else:
        metrics = _run_dp_eval(cfg, device)

    logger.info(f"checkpoint={cfg.checkpoint_path} {metrics}")
    print(metrics)

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump({"checkpoint_path": cfg.checkpoint_path, **metrics}, f, indent=2)
    return metrics


if __name__ == "__main__":
    main()
