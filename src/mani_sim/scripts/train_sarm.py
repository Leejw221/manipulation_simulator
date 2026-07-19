"""SARM reward model 학습(policies/sarm/sarm_reward_model.py) — heuristic stage 라벨을
"사람이 단 annotation"처럼 취급해서 stage 분류 + tau 회귀를 학습한다(사용자 지시,
2026-07-19). train.py의 policy/runner registry와 별개(reward model은 action을 안 만들어
기존 정책 인터페이스와 안 맞음, 그래서 독립 스크립트).

사용:
    python -m mani_sim.scripts.train_sarm num_epochs=20
    python -m mani_sim.scripts.train_sarm resume=true
"""

import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from mani_sim.datasets.sarm_dataset import SARMFrameDataset
from mani_sim.policies.sarm.sarm_reward_model import SARMRewardModel
from mani_sim.utils.checkpoints import get_latest_epoch_checkpoint, save_epoch_checkpoint

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="sarm", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    dataset = SARMFrameDataset(cfg.hdf5_path, cfg.image_key, cfg.state_keys, cfg.state_dims)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, persistent_workers=cfg.num_workers >= 1,
    )
    logger.info(f"dataset len={len(dataset)} num_stages={dataset.num_stages}")

    state_dim = sum(cfg.state_dims[k] for k in cfg.state_keys)
    model = SARMRewardModel(num_stages=cfg.num_stages, state_dim=state_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"SARM reward model params: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    start_epoch = 0
    if cfg.get("resume", False):
        path, epoch = get_latest_epoch_checkpoint(cfg.output_dir)
        if path is not None:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            start_epoch = epoch
            logger.info(f"resumed from {path} (epoch {epoch})")

    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name="sarm_transport", config=OmegaConf.to_container(cfg, resolve=True))

    global_step = start_epoch * len(dataloader)
    for epoch in range(start_epoch, cfg.num_epochs):
        for batch in dataloader:
            batch = {
                "image": batch["image"].float().to(device),
                "state": batch["state"].float().to(device),
                "stage": batch["stage"].long().to(device),
                "tau": batch["tau"].float().to(device),  # default_collate가 python float를 float64로 만듦
            }
            loss, metrics = model.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
            optimizer.step()

            if global_step % cfg.log_every == 0:
                logger.info(f"epoch {epoch} step {global_step} loss {loss.item():.4f} "
                            f"stage_loss {metrics['stage_loss']:.4f} tau_loss {metrics['tau_loss']:.4f} "
                            f"grad_norm {grad_norm:.3f}")
                if cfg.use_wandb:
                    wandb.log({"loss": loss.item(), "stage_loss": metrics["stage_loss"],
                               "tau_loss": metrics["tau_loss"], "grad_norm": float(grad_norm),
                               "epoch": epoch}, step=global_step)
            global_step += 1

        is_last = epoch == cfg.num_epochs - 1
        if (epoch + 1) % cfg.ckpt_every_epochs == 0 or is_last:
            path = save_epoch_checkpoint(cfg.output_dir, epoch + 1, model)
            logger.info(f"saved checkpoint: {path}")

    if cfg.use_wandb:
        import wandb
        wandb.finish()
    logger.info("SARM training complete")


if __name__ == "__main__":
    main()
