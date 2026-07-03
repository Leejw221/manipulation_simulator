"""Diffusion Policy(low-dim) 학습 스크립트.

사용:
    python -m mani_sim.scripts.train                          # configs/train.yaml 기본값
    python -m mani_sim.scripts.train num_epochs=1              # 스모크 테스트용 오버라이드
"""

import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from mani_sim.datasets.normalization import MinMaxNormalizer, compute_minmax_stats, save_stats
from mani_sim.datasets.robomimic_dataset import RobomimicSequenceDataset
from mani_sim.policies.diffusion.diffusion_policy import DiffusionPolicyLowDim


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=f"{cfg.task.name}_{cfg.policy.name}_ep{cfg.num_epochs}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    stats = compute_minmax_stats(cfg.task.hdf5_path, cfg.task.obs_keys)
    save_stats(stats, os.path.join(cfg.output_dir, "normalization_stats.json"))
    normalizer = MinMaxNormalizer(stats)

    dataset = RobomimicSequenceDataset(
        hdf5_path=cfg.task.hdf5_path,
        obs_keys=cfg.task.obs_keys,
        obs_horizon=cfg.policy.obs_horizon,
        pred_horizon=cfg.policy.pred_horizon,
        normalizer=normalizer,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    policy = DiffusionPolicyLowDim(
        obs_keys=cfg.task.obs_keys,
        obs_dims=cfg.task.obs_dims,
        obs_horizon=cfg.policy.obs_horizon,
        action_dim=cfg.task.action_dim,
        pred_horizon=cfg.policy.pred_horizon,
        down_dims=cfg.policy.down_dims,
        kernel_size=cfg.policy.kernel_size,
        n_groups=cfg.policy.n_groups,
        diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
        num_train_timesteps=cfg.policy.num_train_timesteps,
        beta_schedule=cfg.policy.beta_schedule,
        num_inference_steps=cfg.policy.num_inference_steps,
    ).to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.num_epochs * len(dataloader), 1)
    )

    global_step = 0
    for epoch in range(cfg.num_epochs):
        for batch in dataloader:
            batch = {
                "obs": {k: v.to(device) for k, v in batch["obs"].items()},
                "action": batch["action"].to(device),
                "action_mask": batch["action_mask"].to(device),
            }
            loss = policy.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if global_step % cfg.log_every == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")
                if cfg.use_wandb:
                    wandb.log(
                        {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch},
                        step=global_step,
                    )
            global_step += 1

        is_last_epoch = epoch == cfg.num_epochs - 1
        if (epoch + 1) % cfg.ckpt_every_epochs == 0 or is_last_epoch:
            ckpt_path = os.path.join(cfg.output_dir, f"policy_epoch{epoch + 1}.pt")
            torch.save(
                {"model": policy.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)},
                ckpt_path,
            )
            print("saved checkpoint:", ckpt_path)

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
