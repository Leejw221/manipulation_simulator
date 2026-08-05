"""task config(low_dim vs image) 판별 + eval env 생성 — runners/diffusion_trainer.py와
scripts/eval.py가 공유(중복 방지)."""

import json

import h5py
import numpy as np
from omegaconf import OmegaConf


def is_image_task(task_cfg):
    return "rgb_keys" in task_cfg


def is_piper_task(task_cfg):
    return task_cfg.get("env_backend", "robosuite") == "piper_mujoco"


def uses_zarr_dataset(task_cfg):
    """학습 데이터를 Zarr(ReplayBuffer)에서 읽을지 여부 - is_piper_task와 독립된 축이다.

    env_backend는 "시뮬레이터로 뭘 쓸지"(로봇수트 vs Piper 전용 raw MuJoCo)를 결정하고,
    collect.py/eval.py가 이 값만 본다. 반면 이 함수는 "학습 데이터를 어느 포맷으로 읽을지"만
    결정한다 - Piper는 시뮬레이터가 raw MuJoCo라 항상 Zarr(기존 그대로)지만, 로봇수트 task도
    task.yaml에 dataset_backend=zarr를 명시하면 시뮬레이터(로봇수트)는 안 건드리고 학습 데이터
    저장 포맷만 hdf5->Zarr로 바꿀 수 있다(2026-07-27 밤 - env_backend 하나가 이 둘을 같이
    결정하던 결합을 풀기 위해 추가. scripts/convert_hdf5_to_zarr.py로 변환한 파일을 씀)."""
    return is_piper_task(task_cfg) or task_cfg.get("dataset_backend", None) == "zarr"


def read_all_action_modes(task_cfg):
    """전체 데이터셋의 프레임별 action_mode를 하나로 이어붙여 반환.

    weighting/class_based.py(SIRIUS 4-class 고정 가중치)처럼 개별 샘플이 아니라 데이터셋
    전체의 클래스 비율(P(demo)/P(rollout)/P(intv)/P(preintv))이 필요한 축이 쓴다.
    uses_zarr_dataset에 따라 hdf5/zarr 중 맞는 소스에서 읽는다(2026-07-27 밤 - Zarr task
    에서도 SIRIUS 조건이 돌아가게 하려고 추가, weighting 클래스 자체는 포맷을 몰라도 됨)."""
    if uses_zarr_dataset(task_cfg):
        from mani_sim.datasets.zarr_dataset import ReplayBuffer

        buffer = ReplayBuffer.create_from_path(str(task_cfg.zarr_path), mode="r")
        return np.asarray(buffer.data["action_mode"][:])
    with h5py.File(task_cfg.hdf5_path, "r") as f:
        modes = [np.asarray(f["data"][demo_id]["action_mode"]) for demo_id in f["data"].keys()]
    return np.concatenate(modes)


def task_obs_keys(task_cfg):
    if is_image_task(task_cfg):
        keys = list(task_cfg.rgb_keys) + list(task_cfg.lowdim_keys)
    else:
        keys = list(task_cfg.obs_keys)
    # stage(카테고리 인덱스, nn.Embedding으로 별도 처리)는 lowdim_keys가 아니라 여기서만
    # 추가한다 - lowdim_keys에 넣으면 DiffusionPolicyImage._encode()가 raw concat과 임베딩
    # 둘 다에 중복으로 넣게 된다(2026-08-04, robocasa stage-conditioning 실험).
    if task_cfg.get("num_stages", None):
        keys = keys + ["stage"]
    return keys


def task_lowdim_keys(task_cfg):
    return list(task_cfg.lowdim_keys) if is_image_task(task_cfg) else list(task_cfg.obs_keys)


_RELEVANT_ENV_KWARGS = ("env_configuration", "controller_configs", "lite_physics")


def derive_task_meta_from_hdf5(task_cfg):
    """env_name/robots/env_kwargs/obs_dims/action_dim/camera_names는 데이터 수집 시점의
    '사실'이라 robomimic hdf5(env_args + 실측 배열 shape)에서 그대로 읽을 수 있다 —
    task.yaml에 손으로 다시 적으면 둘이 어긋날 수 있다(예: stage 스킴이 6→5단계로
    바뀌었는데 obs_dims를 안 고침, env_configuration을 깜빡함 — 2026-07-21 실제 사고).
    검증 대신 hdf5를 유일한 출처로 삼아 task_cfg를 여기서 덮어쓴다(2026-07-25,
    학습 시작 시 1회 호출 — 이후 run_config.yaml에 저장돼 eval까지 그대로 전파됨).

    rgb_keys/lowdim_keys(어떤 키를 쓸지)·image_size·name·hdf5_path는 데이터의 사실이
    아니라 실험 설계 선택이라 안 건드린다(예: lowdim task는 `object`를 일부러 쓰고
    image task는 정보 중복 방지로 일부러 뺌, EXP-01)."""
    with h5py.File(task_cfg.hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        demo0 = f["data/demo_0"]
        action_dim = int(demo0["actions"].shape[-1])
        obs_dims = {k: int(demo0["obs"][k].shape[-1]) for k in task_lowdim_keys(task_cfg) if k in demo0["obs"]}

    env_kwargs_src = env_args.get("env_kwargs", {})

    OmegaConf.set_struct(task_cfg, False)
    task_cfg.env_name = env_args["env_name"]
    if "robots" in env_kwargs_src:
        task_cfg.robots = env_kwargs_src["robots"]
    task_cfg.env_kwargs = {k: env_kwargs_src[k] for k in _RELEVANT_ENV_KWARGS if k in env_kwargs_src}
    task_cfg.obs_dims = obs_dims
    task_cfg.action_dim = action_dim
    if is_image_task(task_cfg):
        task_cfg.camera_names = [k[: -len("_image")] for k in task_cfg.rgb_keys]
    OmegaConf.set_struct(task_cfg, True)
    return task_cfg


def derive_task_meta_from_zarr(task_cfg):
    """derive_task_meta_from_hdf5의 zarr 버전 - env_name/robots/env_kwargs는 손대지 않고
    obs_dims/action_dim만 zarr 배열 shape에서 그대로 derive한다(2026-07-26).

    Piper(raw MuJoCo, env_backend=piper_mujoco)는 애초에 env_name/robots/env_kwargs라는
    개념이 없어서(make_eval_env가 xml_path/camera_names를 직접 읽음) 이 함수가 그 셋을
    안 건드리는 게 자연스럽다. 로봇수트 task가 dataset_backend=zarr로 학습 데이터만
    Zarr를 쓰는 경우(2026-07-27 밤, uses_zarr_dataset 참고)는 env_name/robots/env_kwargs가
    task.yaml에 이미 손으로 정확히 적혀 있고(eval.py/collect.py도 원래 이 값을 그대로
    믿고 씀 - derive_task_meta 자체를 안 부름), 굳이 hdf5처럼 데이터에서 재검증할
    필요가 없어 이 함수 그대로 재사용해도 안전하다."""
    import sys
    from pathlib import Path

    piper_capstone_dir = Path(__file__).resolve().parents[3] / "mani_sim_external" / "piper_capstone"
    if str(piper_capstone_dir) not in sys.path:
        sys.path.insert(0, str(piper_capstone_dir))
    from replay_buffer import ReplayBuffer

    buffer = ReplayBuffer.create_from_path(str(task_cfg.zarr_path), mode="r")
    obs_dims = {k: int(buffer.data[k].shape[-1]) for k in task_lowdim_keys(task_cfg) if k in buffer.data}
    action_dim = int(buffer.data["action"].shape[-1])

    OmegaConf.set_struct(task_cfg, False)
    task_cfg.obs_dims = obs_dims
    task_cfg.action_dim = action_dim
    OmegaConf.set_struct(task_cfg, True)
    return task_cfg


def derive_task_meta(task_cfg):
    """학습 데이터 저장 포맷(uses_zarr_dataset)에 따라 derive_task_meta_from_hdf5/_from_zarr
    중 맞는 쪽으로 분기 - 시뮬레이터 선택(is_piper_task)과는 별개다(uses_zarr_dataset 참고).
    train.py는 이 함수 하나만 호출하면 됨(2026-07-26)."""
    if uses_zarr_dataset(task_cfg):
        return derive_task_meta_from_zarr(task_cfg)
    return derive_task_meta_from_hdf5(task_cfg)


def make_eval_env(task_cfg, render=False, renderer="mjviewer", image_size_override=None, env_kwargs_override=None):
    """train/eval/collect 3곳에서 각자 env를 만들던 걸 통합(2026-07-25) — task_cfg 필드를
    풀어쓰는 로직이 세 군데 복사돼 있었고, 그중 하나(collect.py)는 env_kwargs를 통째로
    빠뜨리는 버그로 이어졌었다(직전 커밋). 실제로 다른 건 render 시점·image_size·env_kwargs
    출처(collect.py는 outside_color를 더 얹음) 셋뿐이라 인자로 흡수한다.

    image task + render=True는 여기서 처리하지 않는다(호출부 책임) — cv2 오프스크린 렌더와
    mjviewer 온스크린이 GL 컨텍스트 충돌로 세그폴트하는 게 문서화된 지뢰라, image 쪽은
    make_image_env 생성 *후에* `env.env.has_renderer` 등을 직접 패치하는 방식을 그대로 둔다
    (collect.py 참고, eval.py는 image+render 자체를 막음).

    env_backend="piper_mujoco"(2026-07-26)면 robosuite 경로를 아예 안 타고 raw MuJoCo
    어댑터(PiperSortReturnEnv)로 분기한다 — robosuite가 지원 안 하는 로봇(Piper)용."""
    if is_piper_task(task_cfg):
        from mani_sim.envs.piper.piper_sort_return_env import PiperSortReturnEnv

        image_size = image_size_override or tuple(task_cfg.image_size)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return PiperSortReturnEnv(
            xml_path=task_cfg.xml_path,
            camera_names=dict(task_cfg.camera_names) if task_cfg.get("camera_names") else None,
            image_size=image_size,
        )

    gripper_types = task_cfg.get("gripper_types", None)
    if env_kwargs_override is not None:
        env_kwargs = env_kwargs_override
    else:
        env_kwargs = OmegaConf.to_container(task_cfg.env_kwargs, resolve=True) if task_cfg.get("env_kwargs", None) else None
    if is_image_task(task_cfg):
        from mani_sim.envs.robomimic.factory import make_image_env
        return make_image_env(
            task_cfg.env_name, task_cfg.robots,
            list(task_cfg.lowdim_keys), list(task_cfg.rgb_keys),
            list(task_cfg.camera_names), image_size=image_size_override or task_cfg.image_size,
            gripper_types=gripper_types, env_kwargs=env_kwargs,
        )
    from mani_sim.envs.robomimic.factory import make_lowdim_env
    return make_lowdim_env(task_cfg.env_name, task_cfg.robots, list(task_cfg.obs_keys),
                            render=render, renderer=renderer, gripper_types=gripper_types, env_kwargs=env_kwargs)
