"""robomimic/robosuite env 생성 — robomimic import는 이 파일(과 datasets 어댑터)에만 존재.

docs/plan.md M0 확정 결과 참고: ObsUtils 초기화가 env 생성 전에 필요, MUJOCO_GL=egl 권장(오프스크린 렌더).
"""

import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_robosuite import EnvRobosuite

# 커스텀 robosuite env 클래스 등록 부수효과 (robosuite는 MujocoEnv 서브클래스가 import되는
# 순간 메타클래스가 자동으로 전역 레지스트리에 등록 - suite.make(env_name=...)가 찾으려면
# 이 import가 먼저 일어나야 한다).
from mani_sim.envs.custom import door_cabinet_env  # noqa: E402,F401


def _ensure_obs_utils_initialized(obs_keys, rgb_keys=()):
    if ObsUtils.OBS_KEYS_TO_MODALITIES is not None:
        return
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(obs_keys), "rgb": list(rgb_keys)}})


def make_lowdim_env(env_name, robots, obs_keys, render=False, renderer="mjviewer", gripper_types=None, env_kwargs=None):
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
        gripper_types (str | None): 예: "RobotiqThreeFingerGripper". None이면 로봇 기본 그리퍼.
        env_kwargs (dict | None): 커스텀 env 클래스별 추가 kwargs 통과용(예: DoorCabinet의
            outside_color). 범용 팩토리에 task별 파라미터를 하나씩 하드코딩하지 않기 위함.
    """
    _ensure_obs_utils_initialized(obs_keys)
    kwargs = dict(env_kwargs) if env_kwargs else {}
    if gripper_types is not None:
        kwargs["gripper_types"] = gripper_types
    env = EnvRobosuite(
        env_name=env_name,
        robots=robots,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        reward_shaping=False,
        **kwargs,
    )
    if render:
        env.env.has_renderer = True
        env.env.renderer = renderer
    return env


def make_image_env(env_name, robots, lowdim_keys, rgb_keys, camera_names, image_size=84, gripper_types=None, env_kwargs=None):
    """이미지(픽셀) obs를 쓰는 robosuite env 생성 (오프스크린 렌더).

    Args:
        env_name (str): 예: "NutAssemblySquare".
        robots (str | list[str]): 예: "Panda".
        lowdim_keys (list[str]): proprio 등 low_dim obs 키(예: robot0_eef_pos).
        rgb_keys (list[str]): 이미지 obs 키(예: agentview_image). camera_names와 순서 대응.
        camera_names (list[str]): 렌더할 robosuite 카메라(예: agentview, robot0_eye_in_hand).
            각 카메라 <cam>의 이미지는 obs 키 "<cam>_image"로 나온다.
        image_size (int): 정사각 렌더 해상도(H=W). 학습 데이터셋과 맞춰야 함(기본 84).
        gripper_types (str | None): 예: "RobotiqThreeFingerGripper". None이면 로봇 기본 그리퍼.
        env_kwargs (dict | None): 커스텀 env 클래스별 추가 kwargs 통과용(예: DoorCabinet의
            outside_color).

    오프스크린 렌더이므로 MUJOCO_GL=egl 환경에서 실행한다.
    """
    _ensure_obs_utils_initialized(lowdim_keys, rgb_keys)
    kwargs = dict(env_kwargs) if env_kwargs else {}
    if gripper_types is not None:
        kwargs["gripper_types"] = gripper_types
    env = EnvRobosuite(
        env_name=env_name,
        robots=robots,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        camera_names=list(camera_names),
        camera_heights=image_size,
        camera_widths=image_size,
        reward_shaping=False,
        **kwargs,
    )
    return env
