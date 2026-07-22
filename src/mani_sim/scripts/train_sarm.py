"""SARM(arXiv:2509.25358) reward model 학습 — xdofai/opensarm 원문 아키텍처
(StageTransformer+SubtaskTransformer, 50/50 teacher forcing, 독립 optimizer 2개) 그대로
포팅[검증-원문, 2026-07-22]. heuristic stage 라벨(stage_labeler.py)을 sparse annotation으로
취급해서 학습한다(dense 브랜치·SARM2·rewind augmentation은 사용자 지시로 미구현/생략).

train.py의 policy/runner registry와 별개(reward model은 action을 안 만들어 기존 정책
인터페이스와 안 맞음, 그래서 독립 스크립트).

사용:
    python -m mani_sim.scripts.train_sarm
    python -m mani_sim.scripts.train_sarm resume=true
"""

import logging
import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from mani_sim.datasets.normalization import compute_minmax_stats
from mani_sim.datasets.sarm_dataset import SARMWindowDataset
from mani_sim.policies.sarm.clip_text_encoder import ClipTextEncoder
from mani_sim.policies.sarm.sarm_reward_model import ClipVisionEncoder
from mani_sim.policies.sarm.sarm_transformer import (
    SARMFullModel,
    StageTransformer,
    SubtaskTransformer,
    gen_stage_emb,
)
from mani_sim.utils.checkpoints import get_latest_epoch_checkpoint, save_epoch_checkpoint

logger = logging.getLogger(__name__)


def _state_minmax(hdf5_path, state_keys, device):
    """datasets/normalization.py 기존 MinMax([-1,1]) 컨벤션 재사용 — state_keys 순서대로
    concat한 min/max 벡터를 반환."""
    stats = compute_minmax_stats(hdf5_path, state_keys)
    vmin = np.concatenate([stats["obs"][k]["min"] for k in state_keys]).astype(np.float32)
    vmax = np.concatenate([stats["obs"][k]["max"] for k in state_keys]).astype(np.float32)
    return torch.as_tensor(vmin, device=device), torch.as_tensor(vmax, device=device)


def _normalize_state(state, vmin, vmax, eps=1e-8):
    return (state - vmin) / (vmax - vmin + eps) * 2 - 1


@hydra.main(config_path="../configs", config_name="sarm", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    dataset = SARMWindowDataset(
        cfg.hdf5_path, cfg.image_key, cfg.state_keys, cfg.state_dims,
        n_obs_steps=cfg.model.n_obs_steps, frame_gap=cfg.model.frame_gap,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, persistent_workers=cfg.num_workers >= 1,
    )
    logger.info(f"dataset len={len(dataset)} num_stages={dataset.num_stages} "
                f"alpha_bar={dataset.alpha_bar.tolist()}")

    state_dim = sum(cfg.state_dims[k] for k in cfg.state_keys)
    state_vmin, state_vmax = _state_minmax(cfg.hdf5_path, cfg.state_keys, device)

    vision_encoder = ClipVisionEncoder(freeze=True).to(device).eval()
    text_encoder = ClipTextEncoder(freeze=True).to(device).eval()
    with torch.no_grad():
        lang_emb_single = text_encoder([cfg.task_description], device=device)  # (1, txt_dim)

    stage_model = StageTransformer(
        d_model=cfg.model.d_model, vis_emb_dim=vision_encoder.out_dim, text_emb_dim=text_encoder.out_dim,
        state_dim=state_dim, n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout, num_cameras=1, num_classes_sparse=cfg.num_stages,
    ).to(device)
    subtask_model = SubtaskTransformer(
        d_model=cfg.model.d_model, vis_emb_dim=vision_encoder.out_dim, text_emb_dim=text_encoder.out_dim,
        state_dim=state_dim, n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout, num_cameras=1,
    ).to(device)
    n_params = sum(p.numel() for p in stage_model.parameters()) + sum(p.numel() for p in subtask_model.parameters())
    logger.info(f"SARM stage+subtask params: {n_params / 1e6:.1f}M")

    start_epoch = 0
    if cfg.get("resume", False):
        path, epoch = get_latest_epoch_checkpoint(cfg.output_dir)
        if path is not None:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            stage_model.load_state_dict({k[len("stage_model."):]: v for k, v in ckpt["model"].items() if k.startswith("stage_model.")})
            subtask_model.load_state_dict({k[len("subtask_model."):]: v for k, v in ckpt["model"].items() if k.startswith("subtask_model.")})
            start_epoch = epoch
            logger.info(f"resumed from {path} (epoch {epoch})")

    total_steps = cfg.num_epochs * len(dataloader)
    warmup_steps = min(cfg.optim.warmup_steps, max(1, total_steps // 10))

    def make_optimizer_and_scheduler(model):
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.optim.lr, betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps, weight_decay=cfg.optim.weight_decay,
        )
        warmup = LinearLR(opt, start_factor=1e-6 / cfg.optim.lr, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=0.0)
        sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
        return opt, sched

    stage_optimizer, stage_scheduler = make_optimizer_and_scheduler(stage_model)
    subtask_optimizer, subtask_scheduler = make_optimizer_and_scheduler(subtask_model)

    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name="sarm_transport_v2", config=OmegaConf.to_container(cfg, resolve=True))

    global_step = start_epoch * len(dataloader)
    for epoch in range(start_epoch, cfg.num_epochs):
        stage_model.train()
        subtask_model.train()
        for batch in dataloader:
            B, W = batch["images"].shape[:2]  # W = n_obs_steps+1
            images = batch["images"].float().to(device)  # (B,W,C,H,W)
            state = batch["state"].float().to(device)  # (B,W,state_dim)
            targets = batch["targets"].float().to(device)  # (B,W) = stage+tau
            lengths = batch["lengths"].long().to(device)  # (B,) 항상 W(padding 없음)

            with torch.no_grad():
                img_flat = images.flatten(0, 1)  # (B*W,C,H,W)
                img_emb = vision_encoder(img_flat).view(B, W, -1).unsqueeze(1)  # (B,1,W,vis_dim) N=1 카메라
                lang_emb = lang_emb_single.expand(B, -1)  # (B,txt_dim)
            state_norm = _normalize_state(state, state_vmin, state_vmax)

            gt_stage = torch.floor(targets).long()
            gt_tau = torch.remainder(targets, 1.0)

            stage_logits = stage_model(img_emb, lang_emb, state_norm, lengths)  # (B,W,num_stages)

            if torch.rand(1).item() < 0.5:
                stage_emb = gen_stage_emb(cfg.num_stages, targets)  # GT teacher forcing
            else:
                stage_idx = stage_logits.argmax(dim=-1)
                stage_emb = F.one_hot(stage_idx, num_classes=cfg.num_stages).float().unsqueeze(1)
            tau_pred = subtask_model(img_emb, lang_emb, state_norm, lengths, stage_emb)

            stage_loss = F.cross_entropy(stage_logits.reshape(-1, cfg.num_stages), gt_stage.reshape(-1))
            subtask_loss = F.mse_loss(tau_pred, gt_tau)

            subtask_optimizer.zero_grad()
            subtask_loss.backward()
            subtask_grad_norm = torch.nn.utils.clip_grad_norm_(subtask_model.parameters(), cfg.grad_clip)
            subtask_optimizer.step()
            subtask_scheduler.step()

            stage_optimizer.zero_grad()
            stage_loss.backward()
            stage_grad_norm = torch.nn.utils.clip_grad_norm_(stage_model.parameters(), cfg.grad_clip)
            stage_optimizer.step()
            stage_scheduler.step()

            if global_step % cfg.log_every == 0:
                logger.info(
                    f"epoch {epoch} step {global_step} stage_loss {stage_loss.item():.4f} "
                    f"subtask_loss {subtask_loss.item():.4f} stage_grad {stage_grad_norm:.3f} "
                    f"subtask_grad {subtask_grad_norm:.3f} lr {stage_scheduler.get_last_lr()[0]:.2e}"
                )
                if cfg.use_wandb:
                    wandb.log({
                        "stage_loss": stage_loss.item(), "subtask_loss": subtask_loss.item(),
                        "stage_grad_norm": float(stage_grad_norm), "subtask_grad_norm": float(subtask_grad_norm),
                        "lr": stage_scheduler.get_last_lr()[0], "epoch": epoch,
                    }, step=global_step)
            global_step += 1

        is_last = epoch == cfg.num_epochs - 1
        if (epoch + 1) % cfg.ckpt_every_epochs == 0 or is_last:
            full_model = SARMFullModel(stage_model, subtask_model, alpha_bar=dataset.alpha_bar)
            path = save_epoch_checkpoint(cfg.output_dir, epoch + 1, full_model)
            logger.info(f"saved checkpoint: {path}")

    if cfg.use_wandb:
        import wandb
        wandb.finish()
    logger.info("SARM training complete")


if __name__ == "__main__":
    main()
