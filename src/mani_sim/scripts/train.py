"""통합 학습 진입점 — policy_name/runner_name(factory registry 키)로 bc/diffusion/openvla ×
low_dim/image를 전부 이 스크립트 하나로 다룬다. outputs/train_image.py·train_image_stage.py·
train_openvla_base.py(gitignore라 매번 유실됐던 스크립트들)를 대체.

사용:
    # DP(image), 기본 300 epoch
    python -m mani_sim.scripts.train use_wandb=true

    # DP(image) + stage conditioning
    python -m mani_sim.scripts.train task=square_stage

    # DP(low_dim), 예전 M2 경로
    python -m mani_sim.scripts.train task=lift_low_dim policy=diffusion_unet_lowdim policy_name=diffusion_lowdim

    # OpenVLA round0
    python -m mani_sim.scripts.train policy=openvla policy_name=openvla runner_name=openvla_trainer \
        batch_size=2 max_steps=3000 use_wandb=true

    # 중단된 학습 이어받기(같은 output_dir의 최신 체크포인트/adapter에서)
    python -m mani_sim.scripts.train resume=true
"""

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from mani_sim.factory import registry
from mani_sim.utils.task_utils import derive_task_meta

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    # env_name/robots/env_kwargs/obs_dims/action_dim/camera_names는 데이터(hdf5 또는 zarr)가
    # 유일한 출처 — task.yaml 손으로 적은 값이 데이터와 어긋나는 걸 검증하는 대신 여기서
    # 덮어써 원천 차단(2026-07-25, env_backend=piper_mujoco 분기 2026-07-26 추가).
    # policy 생성(obs_dims/action_dim 필요) 전에 반드시 먼저 호출해야 함.
    derive_task_meta(cfg.task)
    logger.info(f"policy={cfg.policy_name} runner={cfg.runner_name} task={cfg.task.name} device={device}")

    runner_cls = registry.get_runner_class(cfg.runner_name)

    if cfg.resume and hasattr(runner_cls, "prepare_for_resume"):
        runner_cls.prepare_for_resume(cfg)

    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy)
    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"policy params: {n_params / 1e6:.1f}M")

    if cfg.use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or f"{cfg.task.name}_{cfg.policy_name}",
            group=cfg.get("wandb_group", None),
            tags=list(cfg.get("wandb_tags", [])),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    runner = runner_cls(cfg, policy, device)

    if cfg.runner_name == "openvla_trainer":
        start_step = runner.resume_start_step() if cfg.resume else 0
        runner.train(cfg.max_steps, start_step=start_step, save_every=cfg.save_every,
                     log_every=cfg.log_every, use_wandb=cfg.use_wandb)
    else:
        start_epoch = runner.resume_start_epoch() if cfg.resume else 0
        runner.train(cfg.num_epochs, start_epoch=start_epoch, use_wandb=cfg.use_wandb)

    if cfg.use_wandb:
        import wandb
        wandb.finish()

    logger.info("training complete")


if __name__ == "__main__":
    main()
