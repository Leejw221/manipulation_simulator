"""task config(low_dim vs image) 판별 + eval env 생성 — runners/diffusion_trainer.py와
scripts/eval.py가 공유(중복 방지)."""

import json

import h5py
from omegaconf import OmegaConf


def is_image_task(task_cfg):
    return "rgb_keys" in task_cfg


def task_obs_keys(task_cfg):
    if is_image_task(task_cfg):
        return list(task_cfg.rgb_keys) + list(task_cfg.lowdim_keys)
    return list(task_cfg.obs_keys)


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


def make_eval_env(task_cfg, render=False, renderer="mjviewer", image_size_override=None, env_kwargs_override=None):
    """train/eval/collect 3곳에서 각자 env를 만들던 걸 통합(2026-07-25) — task_cfg 필드를
    풀어쓰는 로직이 세 군데 복사돼 있었고, 그중 하나(collect.py)는 env_kwargs를 통째로
    빠뜨리는 버그로 이어졌었다(직전 커밋). 실제로 다른 건 render 시점·image_size·env_kwargs
    출처(collect.py는 outside_color를 더 얹음) 셋뿐이라 인자로 흡수한다.

    image task + render=True는 여기서 처리하지 않는다(호출부 책임) — cv2 오프스크린 렌더와
    mjviewer 온스크린이 GL 컨텍스트 충돌로 세그폴트하는 게 문서화된 지뢰라, image 쪽은
    make_image_env 생성 *후에* `env.env.has_renderer` 등을 직접 패치하는 방식을 그대로 둔다
    (collect.py 참고, eval.py는 image+render 자체를 막음)."""
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
