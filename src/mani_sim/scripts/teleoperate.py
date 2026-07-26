"""순수 텔레오퍼 - 데이터 저장 없이 Piper 팔을 조작 기기로 움직여보기만 한다(PICO 페어링·
캘리브레이션 확인, 새 기기 동작 확인용). collect.py와 같은 hydra 설정(task=가 로봇/env,
intervention_device=가 조작 기기)을 그대로 재사용하고, collect_episode()도 그대로 써서
collect.py와 동일하게 동작하되 마지막 저장(_write_piper_lerobot) 단계만 없다.

사용:
    <piper_collect>/bin/python -m mani_sim.scripts.teleoperate task=piper_sort_return intervention_device=pico
    <piper_collect>/bin/python -m mani_sim.scripts.teleoperate task=piper_sort_return intervention_device=keyboard
"""

import hydra
import numpy as np
from omegaconf import DictConfig

from mani_sim.runners.intervention_rollout import collect_episode
from mani_sim.runners.piper import make_camera_toggle_fn, make_piper_intervention
from mani_sim.utils.task_utils import is_piper_task, make_eval_env


@hydra.main(config_path="../configs", config_name="collect", version_base=None)
def main(cfg: DictConfig):
    if not is_piper_task(cfg.task):
        raise NotImplementedError(
            "지금은 Piper task(env_backend=piper_mujoco)만 지원 - robosuite task의 순수 "
            "텔레오퍼는 collect.py num_episodes=1로 대신하세요."
        )

    env = make_eval_env(cfg.task)
    intervention = make_piper_intervention(cfg)
    # 카메라 라이브 프리뷰는 안 띄운다 - cv2 창이 MuJoCo GL 컨텍스트와 같이 뜨면 이 PC에서
    # 멈추는 기존 지뢰(eval.py에 2026-07-20 기록됨)를 그대로 재현함(2026-07-26 실측 재확인).
    # mujoco 온스크린 뷰어(아래)가 실시간 시각 피드백을 이미 준다.
    render_fn = None
    if cfg.render:
        env.attach_viewer()
        render_fn = make_camera_toggle_fn(env, intervention)
    zero_chunk = np.zeros((cfg.policy.action_horizon, cfg.task.action_dim), dtype=np.float32)

    print("순수 텔레오퍼 - 데이터 저장 안 함. 종료 버튼(Y)으로 끝내거나 Ctrl+C.")
    try:
        while True:
            intervention.reset()
            ep = collect_episode(
                env, None, None, [], cfg.policy.obs_horizon, cfg.policy.action_horizon, "cpu",
                intervention, should_end_fn=intervention.should_end, render_fn=render_fn,
                control_fps=cfg.control_fps, predict_fn=lambda history: zero_chunk,
            )
            print(f"에피소드 종료: {len(ep['actions'])} 프레임 (저장 안 함)")
            if getattr(intervention, "should_stop", lambda: False)():
                print("세션 종료 요청 - 끝냅니다.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        intervention.close()
        env.close_viewer()


if __name__ == "__main__":
    main()
