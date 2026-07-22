"""Diffusion/BC 계열 학습 루프 — low_dim·image 공용(정책이 `compute_loss(batch)`만
지원하면 이 러너는 policy 종류를 모른다). train_image.py/train_image_stage.py(구, outputs/,
gitignore로 유실 반복됐던 스크립트들)를 여기 하나로 합친다.

resume은 `utils/checkpoints.get_latest_epoch_checkpoint`로 `policy_epoch<N>.pt` 중 최신을
찾아 이어간다(오늘 PC 리부트로 학습이 중간에 죽었을 때 겪은 문제의 재발 방지).

**가중치 학습(옵션, cfg.weighting.kind)**: round.py가 배포+개입으로 모은 데이터(action_mode
라벨 포함)로 학습할 때 SIRIUS 스타일 고정 가중치(class_based) 또는 action_error 가중치를
켤 수 있다 — losses/sirius_loss.py(reference 모델 없는 단순 가중 손실) + weighting/*.py를
그대로 재사용. **APO(적응형, reference 모델+KTO)는 아직 미구현** — BC/DiffusionPolicy 둘 다
reference-context(OpenVLA만 있음)가 없어 이번 리팩터 범위에서 제외, 별도 과제로 남김.

**rabc**(2026-07-19 추가) — SARM 논문(arXiv:2509.25358)의 RA-BC를 progress-delta 가중치로
이식(weighting/rabc.py). class_based/action_error와 인터페이스가 달라(action_mode(B,T) 대신
demo_id+index_in_demo 필요) 아래서 별도 분기로 처리.

**phase_rule/action_variance**(2026-07-21 추가) — 개입 라운드 없이(순수 PH demo) 가중치
산정방식 비교 ablation용. rabc와 같은 demo_id/index_in_demo 인터페이스. phase_rule=규칙기반
(DAISS 스타일, offline stage 라벨), action_variance=`scripts/compute_action_variance.py`로
미리 계산한 반복샘플링 분산(교수님 7/15 원문 정의 — cross-demo 아님)."""

import logging
import os

import torch
from torch.utils.data import DataLoader

from mani_sim.datasets.normalization import MinMaxNormalizer, compute_minmax_stats, save_stats
from mani_sim.datasets.robomimic_dataset import RobomimicSequenceDataset
from mani_sim.factory import registry
from mani_sim.losses.sirius_loss import sirius_loss
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

        self.weighting = None
        self.weighting_kind = None
        weighting_cfg = cfg.get("weighting", None)
        weighting_kind = weighting_cfg.kind if weighting_cfg else None
        extra_keys = ()
        if weighting_kind == "class_based":
            from mani_sim.weighting.class_based import ClassBasedWeight
            self.weighting = ClassBasedWeight(
                cfg.task.hdf5_path, target_intv=weighting_cfg.target_intv,
                target_preintv=weighting_cfg.target_preintv, device=device,
            )
            extra_keys = ("action_mode",)
        elif weighting_kind == "action_error":
            from mani_sim.weighting.action_error import ActionErrorWeight
            self.weighting = ActionErrorWeight(
                beta_d=weighting_cfg.beta_d, beta_u=weighting_cfg.beta_u, gamma=weighting_cfg.gamma,
            )
            extra_keys = ("action_mode",)
        elif weighting_kind == "rabc":
            from mani_sim.weighting.rabc import RABCWeight
            self.weighting = RABCWeight(
                cfg.task.hdf5_path, chunk_size=cfg.policy.action_horizon,
                kappa=weighting_cfg.kappa, eps=weighting_cfg.epsilon,
                progress_path=weighting_cfg.get("progress_path", None), device=device,
            )
            # demo_id/index_in_demo는 RobomimicSequenceDataset.__getitem__ 기본 반환값이라
            # extra_keys 불필요(action_mode도 안 씀 — SARM은 라벨 종류와 무관).
        elif weighting_kind == "phase_rule":
            from mani_sim.weighting.phase_rule import PhaseRuleWeight
            self.weighting = PhaseRuleWeight(
                cfg.task.hdf5_path, critical_weight=weighting_cfg.critical_weight,
                critical_stages=tuple(weighting_cfg.critical_stages), device=device,
            )
        elif weighting_kind == "action_variance":
            from mani_sim.weighting.action_variance import ActionVarianceWeight
            self.weighting = ActionVarianceWeight(
                weighting_cfg.variance_path, w_min=weighting_cfg.w_min, w_max=weighting_cfg.w_max,
                device=device,
            )
        elif weighting_kind == "event_radius":
            from mani_sim.weighting.event_radius import EventRadiusWeight
            self.weighting = EventRadiusWeight(
                cfg.task.hdf5_path, critical_weight=weighting_cfg.critical_weight,
                critical_stages=tuple(weighting_cfg.critical_stages), radius=weighting_cfg.radius,
                device=device,
            )
        elif weighting_kind is not None:
            raise ValueError(
                f"weighting.kind={weighting_kind!r} 미지원 "
                "(class_based|action_error|rabc|phase_rule|action_variance|event_radius) — "
                "APO(적응형, reference 모델 필요)는 BC/DiffusionPolicy에 아직 미구현"
            )
        self.weighting_kind = weighting_kind

        cache_mode = "all" if cfg.num_workers >= 1 else "low_dim"  # h5py fork 크래시 회피(지뢰, mani_sim_status.md)
        self.dataset = RobomimicSequenceDataset(
            hdf5_path=cfg.task.hdf5_path,
            obs_keys=task_obs_keys(cfg.task),
            obs_horizon=cfg.policy.obs_horizon,
            pred_horizon=cfg.policy.pred_horizon,
            normalizer=self.normalizer,
            rgb_keys=cfg.task.rgb_keys if self.is_image else (),
            hdf5_cache_mode=cache_mode,
            extra_keys=extra_keys,
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
            from mani_sim.datasets.stage_labeler import make_online_tracker
            self._stage_tracker = make_online_tracker(cfg.task.name)

    def _stage_extra_obs_fn(self, obs_raw):
        from mani_sim.datasets.stage_labeler import num_stages_for_task, onehot as stage_onehot
        import numpy as np
        s = self._stage_tracker.step(obs_raw)
        return stage_onehot(np.array([s]), num=num_stages_for_task(self.task_cfg.name))[0]

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
            for raw_batch in self.dataloader:
                batch = {
                    "obs": {k: v.to(self.device) for k, v in raw_batch["obs"].items()},
                    "action": raw_batch["action"].to(self.device),
                    "action_mask": raw_batch["action_mask"].to(self.device),
                }
                if self.weighting is not None:
                    per_sample_loss = self.policy.compute_loss(batch, reduction="none")  # (B,)
                    if self.weighting_kind in ("rabc", "phase_rule", "action_variance", "event_radius"):
                        weight_b = self.weighting.compute_weights(raw_batch["demo_id"], raw_batch["index_in_demo"])
                    else:
                        action_mode = raw_batch["action_mode"].to(self.device)
                        weight_bt = self.weighting.compute_weights(action_mode, error=per_sample_loss.detach())
                        weight_b = weight_bt.mean(dim=1)
                    if self.weighting_kind != "class_based":
                        # STAIR(2606.15587) Eq.2 정규화(1/Σw)·Σw·ℓ와 동치 — sirius_loss 자체는
                        # SIRIUS 원문 재현을 위해 안 건드리고(docstring 참고), 배치 평균이 1이
                        # 되도록 weight만 미리 스케일링(수학적으로 동일 효과). class_based만 SIRIUS
                        # 원문 그대로 두기 위해 제외(2026-07-22, 방식 간 실효 loss 스케일이 달라
                        # 공정 비교가 안 된다는 지적으로 추가 — phase_rule 배치평균 1.522,
                        # action_variance 0.922 등 방식마다 제각각이었음).
                        weight_b = weight_b / weight_b.mean().clamp(min=1e-8)
                    loss = sirius_loss(per_sample_loss, weight_b)  # (B,) x (B,) -> scalar
                else:
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
