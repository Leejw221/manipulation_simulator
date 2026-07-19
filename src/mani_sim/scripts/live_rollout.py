"""학습된 정책을 화면에 실시간으로 띄워서 눈으로 확인 — low_dim·image 공용.

low_dim: MuJoCo 네이티브 mjviewer 창(env.render(), DISPLAY 필요, MUJOCO_GL unset).
image: 오프스크린(egl) 렌더 + OpenCV 창(cv2.imshow). **onscreen(mjviewer)+오프스크린(이미지 obs)을
동시에 켜면 GL 컨텍스트 충돌로 세그폴트한다(이번 세션에 실측한 지뢰)** — 그래서 image는
mjviewer를 아예 안 쓰고 cv2로만 그린다. outputs/live_image.py를 이 갈래로 흡수.

두 경로 다 receding-horizon 루프 자체는 runners/rollout.py의 rollout_policy()를 재사용하고
(on_episode_step 훅으로 렌더만 얹음) — 예전처럼 루프를 통째로 복붙하지 않는다.

사용:
    unset MUJOCO_GL   # low_dim(mjviewer)일 때만 필요
    python -m mani_sim.scripts.live_rollout checkpoint_path=...

    MUJOCO_GL=egl DISPLAY=:1 python -m mani_sim.scripts.live_rollout \
        task=square checkpoint_path=... num_episodes=5
"""

import os
import time

import hydra
import torch
from omegaconf import DictConfig

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.factory import registry
from mani_sim.runners.rollout import rollout_policy
from mani_sim.utils.task_utils import is_image_task, task_obs_keys


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cpu")

    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy).to(device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    stats_path = os.path.join(os.path.dirname(cfg.checkpoint_path), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))

    image = is_image_task(cfg.task)
    if image:
        import cv2
        from mani_sim.envs.robomimic.factory import make_image_env
        env = make_image_env(cfg.task.env_name, cfg.task.robots, list(cfg.task.lowdim_keys),
                              list(cfg.task.rgb_keys), list(cfg.task.camera_names), image_size=cfg.task.image_size)
        window = f"live {os.path.basename(cfg.checkpoint_path)}"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

        def on_episode_step(env_, ep, step):
            frame = env_.render(mode="rgb_array", height=640, width=640, camera_name="agentview")
            bgr = frame[:, :, ::-1].copy()
            cv2.putText(bgr, f"ep {ep + 1}/{cfg.num_episodes} step {step}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window, bgr)
            cv2.waitKey(30)
    else:
        from mani_sim.envs.robomimic.factory import make_lowdim_env
        env = make_lowdim_env(cfg.task.env_name, cfg.task.robots, list(cfg.task.obs_keys), render=True)

        def on_episode_step(env_, ep, step):
            env_.render()
            time.sleep(0.03)  # 사람 눈으로 따라갈 수 있게 살짝 속도 조절

    metrics = rollout_policy(
        env=env, policy=policy, normalizer=normalizer,
        obs_keys=task_obs_keys(cfg.task), obs_horizon=cfg.policy.obs_horizon,
        action_horizon=cfg.policy.action_horizon, max_steps=cfg.max_steps, num_episodes=cfg.num_episodes,
        device=device, rgb_keys=cfg.task.rgb_keys if image else (), on_episode_step=on_episode_step,
    )
    print(metrics)

    if image:
        import cv2
        cv2.destroyAllWindows()
    env.env.close()  # 뷰어·sim을 명시적으로 정리 (안 하면 종료 시 GLXBadWindow 발생)


if __name__ == "__main__":
    main()
