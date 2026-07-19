"""task config(low_dim vs image) 판별 + eval env 생성 — runners/diffusion_trainer.py와
scripts/eval.py·scripts/live_rollout.py가 공유(중복 방지)."""


def is_image_task(task_cfg):
    return "rgb_keys" in task_cfg


def task_obs_keys(task_cfg):
    if is_image_task(task_cfg):
        return list(task_cfg.rgb_keys) + list(task_cfg.lowdim_keys)
    return list(task_cfg.obs_keys)


def task_lowdim_keys(task_cfg):
    return list(task_cfg.lowdim_keys) if is_image_task(task_cfg) else list(task_cfg.obs_keys)


def make_eval_env(task_cfg):
    gripper_types = task_cfg.get("gripper_types", None)
    if is_image_task(task_cfg):
        from mani_sim.envs.robomimic.factory import make_image_env
        return make_image_env(
            task_cfg.env_name, task_cfg.robots,
            list(task_cfg.lowdim_keys), list(task_cfg.rgb_keys),
            list(task_cfg.camera_names), image_size=task_cfg.image_size,
            gripper_types=gripper_types,
        )
    from mani_sim.envs.robomimic.factory import make_lowdim_env
    return make_lowdim_env(task_cfg.env_name, task_cfg.robots, list(task_cfg.obs_keys), gripper_types=gripper_types)
