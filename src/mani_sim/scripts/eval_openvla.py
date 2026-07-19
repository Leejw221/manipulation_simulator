"""OpenVLA(round0 base) rollout 평가 — outputs/rollout_openvla.py를 이식.

DP의 receding-horizon(청크)과 인터페이스가 달라(매 스텝 단일 이미지로 재계획, 원 논문 방식)
scripts/eval.py(rollout_policy 공용 경로)와 통합하지 않고 별도 스크립트로 둔다.

사용:
    python -m mani_sim.scripts.eval_openvla --adapter-path outputs/train/square_openvla/lora_final/policy \
        --stats-path outputs/train/square_openvla/normalization_stats.json --num-episodes 10
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.datasets.openvla_dataset import SQUARE_INSTRUCTION
from mani_sim.factory import registry
from mani_sim.utils.task_utils import make_eval_env


def _to_pil(img_chw01):
    """make_image_env가 주는 CHW float[0,1](robomimic ObsUtils 규약) -> PIL Image(HWC uint8).
    hdf5 raw(HWC uint8)와 형식이 달라 변환 필요(EXP-01에 기록된 지뢰, 여기서도 처음 실제로 밟음)."""
    img_hwc = np.transpose(np.asarray(img_chw01), (1, 2, 0))
    img_uint8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", required=True, help="예: outputs/train/square_openvla/lora_final/policy")
    p.add_argument("--stats-path", required=True)
    p.add_argument("--image-key", default="agentview_image")
    p.add_argument("--instruction", default=SQUARE_INSTRUCTION)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--eval-seed", type=int, default=None,
                    help="scripts/eval.py와 동일 관례 — 지정하면 np.random.seed()로 env reset(너트 위치) 시퀀스 고정")
    p.add_argument("--save-gif", default=None,
                    help="지정하면 첫 에피소드의 agentview 프레임을 GIF로 저장(라이브 창 없이 "
                         "헤드리스로 눈으로 확인하는 용도 — 2026-07-19, cv2 라이브 창이 이 PC "
                         "터미널 환경에서 원인 불명으로 계속 멈춰서(GPU도 안 도는 채로 정지) "
                         "대안으로 씀). 예: --save-gif outputs/openvla_rollout.gif")
    args = p.parse_args()

    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="eval", overrides=["task=square", "policy=openvla", "policy_name=openvla"])
    cfg.policy.policy_adapter_path = args.adapter_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = MinMaxNormalizer(load_stats(args.stats_path))
    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy)
    policy.model.eval()
    print(f"[rollout] loaded adapter: {args.adapter_path}")

    env = make_eval_env(cfg.task)  # square task는 image_size=84로 정의돼 있지만
    # OpenVLA는 224 사전학습 해상도를 기대(processor가 리사이즈는 하지만 원본 해상도를
    # 맞춰 렌더하는 게 더 정확) — env를 224로 별도 생성.
    env.env.close()
    from mani_sim.envs.robomimic.factory import make_image_env
    env = make_image_env(cfg.task.env_name, cfg.task.robots, list(cfg.task.lowdim_keys),
                          list(cfg.task.rgb_keys), list(cfg.task.camera_names), image_size=224)

    if args.eval_seed is not None:
        np.random.seed(args.eval_seed)  # DP 쪽 matched-eval과 동일 관례 — 너트 위치 시퀀스 고정

    successes = []
    t0 = time.time()
    for ep in range(args.num_episodes):
        obs = env.reset()
        success = False
        steps = 0
        gif_frames = [] if (args.save_gif and ep == 0) else None
        while steps < args.max_steps:
            img = _to_pil(obs[args.image_key])
            action_norm = policy.predict_action(img, args.instruction)
            action = normalizer.unnormalize_action(torch.as_tensor(action_norm, dtype=torch.float32)).numpy()
            obs, _r, done, _i = env.step(action)
            steps += 1
            if gif_frames is not None and steps % 2 == 0:  # 매 2프레임마다(용량 절반)
                frame = env.render(mode="rgb_array", height=224, width=224, camera_name="agentview")
                gif_frames.append(Image.fromarray(frame))
            if env.is_success()["task"]:
                success = True
            if success or done or steps >= args.max_steps:
                break
        if gif_frames:
            gif_frames[0].save(args.save_gif, save_all=True, append_images=gif_frames[1:],
                                duration=60, loop=0, optimize=True)
            print(f"[rollout] GIF 저장: {args.save_gif} ({len(gif_frames)} 프레임)", flush=True)
        successes.append(success)
        print(f"[rollout] episode {ep}: steps={steps} success={success}", flush=True)

    env.env.close()
    sr = float(np.mean(successes))
    print(f"\n[rollout] success_rate={sr:.3f} (n={args.num_episodes}) elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
