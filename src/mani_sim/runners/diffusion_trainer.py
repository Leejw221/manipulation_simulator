"""Diffusion/BC 계열 학습 루프 — low_dim·image 공용(정책이 `compute_loss(batch)`만
지원하면 이 러너는 policy 종류를 모른다). train_image.py/train_image_stage.py(구, outputs/,
gitignore로 유실 반복됐던 스크립트들)를 여기 하나로 합친다.

resume은 `utils/checkpoints.get_latest_epoch_checkpoint`로 `policy_epoch<N>.pt` 중 최신을
찾아 이어간다(오늘 PC 리부트로 학습이 중간에 죽었을 때 겪은 문제의 재발 방지).
"""

import logging
import os

import torch
from torch.utils.data import DataLoader

from mani_sim.datasets.normalization import MinMaxNormalizer, compute_minmax_stats, save_stats
from mani_sim.datasets.robomimic_dataset import RobomimicSequenceDataset
from mani_sim.factory import registry
from mani_sim.runners.rollout import rollout_policy
from mani_sim.utils.checkpoints import get_latest_epoch_checkpoint, save_epoch_checkpoint
from mani_sim.utils.task_utils import is_image_task, make_eval_env, task_lowdim_keys, task_obs_keys

logger = logging.getLogger(__name__)


@registry.register_runner("diffusion_trainer")
class DiffusionTrainer:
    def __init__(self, cfg, policy, device):
        self.cfg = cfg
        self.policy = policy.to(device)
        self.device = device
        self.task_cfg = cfg.task
        self.is_image = is_image_task(cfg.task)
        os.makedirs(cfg.output_dir, exist_ok=True)

        lowdim_keys = task_lowdim_keys(cfg.task)
        stats = compute_minmax_stats(cfg.task.hdf5_path, lowdim_keys)
        save_stats(stats, os.path.join(cfg.output_dir, "normalization_stats.json"))
        self.normalizer = MinMaxNormalizer(stats)

        cache_mode = "all" if cfg.num_workers >= 1 else "low_dim"  # h5py fork 크래시 회피(지뢰, mani_sim_status.md)
        self.dataset = RobomimicSequenceDataset(
            hdf5_path=cfg.task.hdf5_path,
            obs_keys=task_obs_keys(cfg.task),
            obs_horizon=cfg.policy.obs_horizon,
            pred_horizon=cfg.policy.pred_horizon,
            normalizer=self.normalizer,
            rgb_keys=cfg.task.rgb_keys if self.is_image else (),
            hdf5_cache_mode=cache_mode,
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, drop_last=True, persistent_workers=cfg.num_workers >= 1,
        )
        logger.info(f"dataset len={len(self.dataset)} cache={cache_mode} image={self.is_image}")

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(cfg.num_epochs * len(self.dataloader), 1)
        )
        self._eval_env = None  # lazy(첫 eval 때 생성 — dataloader worker fork 이후가 안전, EXP-01 지뢰)
        self._stage_tracker = None
        if cfg.task.get("use_online_stage_tracker", False):
            from mani_sim.datasets.stage_labeler import OnlineStageTracker
            self._stage_tracker = OnlineStageTracker()

    def _stage_extra_obs_fn(self, obs_raw):
        from mani_sim.datasets.stage_labeler import onehot as stage_onehot
        import numpy as np
        s = self._stage_tracker.step(obs_raw)
        return stage_onehot(np.array([s]))[0]

    def resume_start_epoch(self):
        path, epoch = get_latest_epoch_checkpoint(self.cfg.output_dir)
        if path is None:
            return 0
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["model"])
        logger.info(f"resumed from {path} (epoch {epoch})")
        return epoch

    def evaluate(self, num_episodes, max_steps):
        if self._eval_env is None:
            self._eval_env = make_eval_env(self.task_cfg)
        extra_obs_fn = self._stage_extra_obs_fn if self._stage_tracker is not None else None
        extra_reset_fn = self._stage_tracker.reset if self._stage_tracker is not None else None
        metrics = rollout_policy(
            env=self._eval_env, policy=self.policy, normalizer=self.normalizer,
            obs_keys=task_obs_keys(self.task_cfg), obs_horizon=self.cfg.policy.obs_horizon,
            action_horizon=self.cfg.policy.action_horizon, max_steps=max_steps, num_episodes=num_episodes,
            device=self.device, rgb_keys=self.task_cfg.rgb_keys if self.is_image else (),
            extra_obs_fn=extra_obs_fn, extra_obs_reset_fn=extra_reset_fn,
        )
        self.policy.train()
        return metrics

    def train(self, num_epochs, start_epoch=0, use_wandb=False):
        if use_wandb:
            import wandb
        global_step = start_epoch * len(self.dataloader)
        for epoch in range(start_epoch, num_epochs):
            for batch in self.dataloader:
                batch = {
                    "obs": {k: v.to(self.device) for k, v in batch["obs"].items()},
                    "action": batch["action"].to(self.device),
                    "action_mask": batch["action_mask"].to(self.device),
                }
                loss = self.policy.compute_loss(batch)

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1e9)
                self.optimizer.step()
                self.lr_scheduler.step()

                if global_step % self.cfg.log_every == 0:
                    logger.info(f"epoch {epoch} step {global_step} loss {loss.item():.4f} grad_norm {grad_norm:.3f}")
                    if use_wandb:
                        wandb.log({"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0],
                                   "grad_norm": float(grad_norm), "epoch": epoch}, step=global_step)
                global_step += 1

            is_last = epoch == num_epochs - 1
            if (epoch + 1) % self.cfg.ckpt_every_epochs == 0 or is_last:
                path = save_epoch_checkpoint(self.cfg.output_dir, epoch + 1, self.policy)
                logger.info(f"saved checkpoint: {path}")

            if self.cfg.eval_every_epochs > 0 and ((epoch + 1) % self.cfg.eval_every_epochs == 0 or is_last):
                metrics = self.evaluate(self.cfg.eval_episodes, self.cfg.eval_max_steps)
                logger.info(f"[eval] epoch {epoch + 1} success_rate {metrics['success_rate']:.3f} "
                            f"(n={self.cfg.eval_episodes})")
                if use_wandb:
                    wandb.log({"eval/success_rate": metrics["success_rate"], "epoch": epoch + 1}, step=global_step)

        if self._eval_env is not None:
            self._eval_env.env.close()
