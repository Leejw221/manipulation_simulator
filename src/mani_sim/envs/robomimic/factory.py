"""robomimic/robosuite env 생성 — robomimic import는 이 파일(과 datasets 어댑터)에만 존재.

docs/plan.md M0 확정 결과 참고: ObsUtils 초기화가 env 생성 전에 필요, MUJOCO_GL=egl 권장(오프스크린 렌더).
"""

import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_robosuite import EnvRobosuite


def _ensure_obs_utils_initialized(obs_keys, rgb_keys=()):
    if ObsUtils.OBS_KEYS_TO_MODALITIES is not None:
        return
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(obs_keys), "rgb": list(rgb_keys)}})


def make_lowdim_env(env_name, robots, obs_keys, render=False, renderer="mjviewer"):
    """low_dim obs만 쓰는 robosuite env 생성.

    Args:
        env_name (str): 예: "Lift".
        robots (str | list[str]): 예: "Panda".
        obs_keys (list[str]): SequenceDataset과 동일한 low_dim obs 키 목록.
        render (bool): True면 화면(DISPLAY)에 실시간 뷰어 창을 띄운다.
            이 프로세스의 DISPLAY 환경변수가 가리키는 화면에 창이 뜨므로, 원격 접속 중이고
            X forwarding이 없으면 창이 안 보일 수 있다. MUJOCO_GL은 설정하지 않아야 한다
            (egl로 두면 오프스크린 강제라 사람이 보는 창이 안 뜬다).
        renderer (str): "mjviewer"(MuJoCo 네이티브 뷰어, 마우스 카메라 조작 가능) 또는
            "mujoco"(OpenCV 창으로 프레임만 표시). robomimic의 `EnvRobosuite`는 render=True일 때
            내부적으로 renderer를 "mujoco"(OpenCV)로 강제 덮어쓰므로, mjviewer를 쓰려면 일단
            render=False로 생성한 뒤 내부 raw robosuite env(`env.env`)의 렌더러 속성을 직접
            설정해 우회한다.
    """
    _ensure_obs_utils_initialized(obs_keys)
    env = EnvRobosuite(
        env_name=env_name,
        robots=robots,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        reward_shaping=False,
    )
    if render:
        env.env.has_renderer = True
        env.env.renderer = renderer
    return env
