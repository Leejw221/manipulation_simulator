"""Diffusion/BC 계열 학습 루프 — low_dim·image 공용(정책이 `compute_loss(batch)`만
지원하면 이 러너는 policy 종류를 모른다). train_image.py/train_image_stage.py(구, outputs/,
gitignore로 유실 반복됐던 스크립트들)를 여기 하나로 합친다.

resume은 `utils/checkpoints.get_latest_epoch_checkpoint`로 `policy_epoch<N>.pt` 중 최신을
찾아 이어간다(오늘 PC 리부트로 학습이 중간에 죽었을 때 겪은 문제의 재발 방지).

**가중치 학습(옵션, cfg.weighting.kind)**: round.py가 배포+개입으로 모은 데이터(action_mode
라벨 포함)로 학습할 때 SIRIUS 스타일 고정 가중치(class_based) 또는 action_error 가중치를
켤 수 있다 — losses/sirius_loss.py(reference 모델 없는 단순 가중 손실) + weighting/*.py를
그대로 재사용.

**system 축(옵션, cfg.system.kind, 2026-07-27)**: null(기본)이면 위 weighting 경로 그대로
(=SIRIUS 스타일). "apo"면 reference 모델 + KTO 앵커(policies/diffusion/apo_system.py) —
self.policy는 그대로 두고 self.system이 policy를 인자로 받아 손실만 계산해 돌려준다
(체크포인트·optimizer가 reference를 절대 못 보게 하기 위한 설계, apo_system.py 참고).
weighting 축과 독립이라 그대로 같이 켤 수 있다.

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

from mani_sim.datasets.apo_sampler import build_balanced_sampler, build_interaction_exhaustion_sampler
from mani_sim.utils.ema import EMAModel
from mani_sim.datasets.normalization import (
    MinMaxNormalizer, compute_minmax_stats, compute_minmax_stats_zarr, load_stats, save_stats,
)
from mani_sim.datasets.robomimic_dataset import RobomimicSequenceDataset
from mani_sim.datasets.zarr_dataset import ZarrSequenceDataset
from mani_sim.factory import registry
from mani_sim.losses.sirius_loss import sirius_loss
from mani_sim.runners.rollout import rollout_policy
from mani_sim.utils.checkpoints import (
    get_latest_epoch_checkpoint, load_resume_state, save_epoch_checkpoint, save_resume_state, save_run_config,
)
from mani_sim.utils.task_utils import (
    is_image_task, is_piper_task, is_robocasa_task, make_eval_env, read_all_action_modes, task_lowdim_keys,
    task_obs_keys, uses_zarr_dataset,
)

logger = logging.getLogger(__name__)


@registry.register_runner("diffusion_trainer")
class DiffusionTrainer:
    def __init__(self, cfg, policy, device):
        self.cfg = cfg
        self.policy = policy.to(device)
        # init_from 체크포인트 가중치 warm-start. 예전엔 이 로딩이 system.kind=apo일 때만
        # (apo_system.py의 ApoSystem.__init__ 안에서) 일어났다 — system.kind=null(SIRIUS류
        # weighting 경로)에서 init_from을 줘도 정규화 통계만 승계되고 정책 가중치는 계속
        # 무작위 초기화 상태로 학습이 시작되는 버그였다(2026-08-06 발견 — round2-new SIRIUS
        # 로그의 step0 loss가 1.12로 시작한 게 무작위 초기화 시그니처였음, 실측으로 확인).
        # 여기서 무조건 먼저 불러오면 system.kind=apo 경로(ApoSystem이 같은 init_state_dict를
        # 다시 한번 로드)와 겹치지만 동일한 가중치를 다시 씌우는 것뿐이라 안전하다.
        init_from = cfg.get("init_from", None)
        if init_from:
            ckpt = torch.load(init_from, map_location=device, weights_only=False)
            self.policy.load_state_dict(ckpt["model"])
            logger.info(f"init_from 가중치 승계: {init_from}")
        self.device = device
        self.task_cfg = cfg.task
        self.is_image = is_image_task(cfg.task)
        self.is_piper = is_piper_task(cfg.task)  # 시뮬레이터 선택(make_eval_env가 내부에서 씀)
        self.uses_zarr = uses_zarr_dataset(cfg.task)  # 학습 데이터 저장 포맷 - is_piper와 독립
        os.makedirs(cfg.output_dir, exist_ok=True)

        lowdim_keys = task_lowdim_keys(cfg.task)
        # init_from으로 파인튜닝할 땐 그 체크포인트의 정규화 통계를 그대로 승계한다
        # (2026-07-31). 여태 무조건 학습 데이터로 새로 계산했는데, 불러온 가중치는 원래
        # 통계에 맞춰져 있으므로 통계를 갈아끼우면 같은 물리값이 다른 정규화값으로 들어가
        # 학습 전부터 정책이 어긋난 입력을 보게 된다(실측: base 가중치 그대로 두고 통계만
        # demo50->merged로 바꾸니 demo 재현 L1 +28.4%). EXP-10.md 2026-07-31 절 참고.
        init_from = cfg.get("init_from", None)
        init_stats_path = (
            os.path.join(os.path.dirname(init_from), "normalization_stats.json") if init_from else None
        )
        if init_stats_path and cfg.get("inherit_init_stats", True) and os.path.exists(init_stats_path):
            stats = load_stats(init_stats_path)
            logger.info(f"정규화 통계 승계: {init_stats_path}")
        else:
            if init_stats_path and cfg.get("inherit_init_stats", True):
                logger.warning(f"init_from 옆에 통계 파일이 없어 새로 계산함: {init_stats_path}")
            if self.uses_zarr:
                stats = compute_minmax_stats_zarr(cfg.task.zarr_path, lowdim_keys)
            else:
                stats = compute_minmax_stats(cfg.task.hdf5_path, lowdim_keys)
        # stage(카테고리 인덱스, task.num_stages 있을 때만)는 lowdim_keys가 아니라 별도
        # obs_key로 들어와서(task_obs_keys 참고) 위 stats엔 없음 — ZarrSequenceDataset이
        # rgb 아닌 모든 obs_key를 일괄 normalize_obs에 넘기므로, 최소/최대를 trivial하게
        # [0, num_stages-1]로 넣어 정규화를 통과시키고 학습 루프에서 정수 인덱스로 되돌린다
        # (policies/diffusion/diffusion_policy_image.py의 nn.Embedding이 정수를 기대함).
        num_stages = cfg.task.get("num_stages", None)
        if num_stages and "stage" not in stats["obs"]:
            stats["obs"]["stage"] = {"min": [0.0], "max": [float(num_stages - 1)]}
        save_stats(stats, os.path.join(cfg.output_dir, "normalization_stats.json"))
        self.normalizer = MinMaxNormalizer(stats)
        self.num_stages = num_stages

        self.weighting = None
        self.weighting_kind = None
        weighting_cfg = cfg.get("weighting", None)
        weighting_kind = weighting_cfg.kind if weighting_cfg else None
        extra_keys = ()
        if weighting_kind == "class_based":
            from mani_sim.weighting.class_based import ClassBasedWeight
            self.weighting = ClassBasedWeight(
                read_all_action_modes(cfg.task), target_intv=weighting_cfg.target_intv,
                target_preintv=weighting_cfg.target_preintv, device=device,
            )
            extra_keys = ("action_mode",)
            if cfg.get("use_wandb", False):
                import wandb
                w = self.weighting
                wandb.log({
                    "weighting/w_demo": w.w_demos, "weighting/w_rollout": w.w_rollouts,
                    "weighting/w_intv": w.w_intvs, "weighting/w_preintv": w.w_pre_intvs,
                }, step=0)
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
                "(class_based|action_error|rabc|phase_rule|action_variance|event_radius)"
            )
        self.weighting_kind = weighting_kind

        self.system = None
        system_cfg = cfg.get("system", None)
        system_kind = system_cfg.kind if system_cfg else None
        if system_kind is not None:
            init_state_dict = None
            init_from = cfg.get("init_from", None)
            if init_from:
                ckpt = torch.load(init_from, map_location=device, weights_only=False)
                init_state_dict = ckpt["model"]
            self.system = registry.create_system(system_kind, cfg, self.policy, self.weighting, device, init_state_dict)
            if "action_mode" not in extra_keys:
                extra_keys = extra_keys + ("action_mode",)

        if self.uses_zarr:
            self.dataset = ZarrSequenceDataset(
                zarr_path=cfg.task.zarr_path,
                obs_keys=task_obs_keys(cfg.task),
                obs_horizon=cfg.policy.obs_horizon,
                pred_horizon=cfg.policy.pred_horizon,
                normalizer=self.normalizer,
                rgb_keys=cfg.task.rgb_keys if self.is_image else (),
                extra_keys=extra_keys,
            )
            cache_mode = "n/a(zarr)"
        else:
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
                filter_key=cfg.task.get("filter_key", None),
            )
        # system=apo면 균등 셔플 대신 APO 원문의 50/25/25(정상/개입/사전개입) balanced
        # sampling을 쓴다(2026-07-27 발견 - 이 배선이 없으면 개입 비율이 낮은 라운드
        # 데이터에서 배치 대부분이 "전부 desirable"이 되어 z0=배치평균 reward가 그
        # 배치 자체의 chosen 평균과 같아져 KTO loss가 beta와 무관하게 상수 0.5로
        # 죽는다 - apo_sampler.py가 2026-07-17부터 있었지만 한 번도 연결된 적 없었음).
        sampler = None
        if self.system is not None:
            # undesirable 판정은 loss(datasets/labels.desirable_mask)와 같은 기준(앞
            # preference_frames 프레임 다수결)을 써야 목표 배치 구성이 실제로 재현된다 —
            # 첫 프레임만 보던 기존 방식은 목표 25%가 loss에선 18.3%로 새어나갔다
            # (EXP-10.md 2026-07-30~31 전수조사). 데이터셋이 윈도우 라벨을 제공하지 못하면
            # 기존 동작으로 자동 폴백.
            pref_frames = cfg.system.get("preference_frames", 8)
            window_labels = (
                self.dataset.get_action_mode_window(pref_frames)
                if hasattr(self.dataset, "get_action_mode_window")
                else None
            )
            # sampler_kind=interaction_exhaustion(2026-08-05 추가)이면 원문처럼 "1 epoch =
            # interaction 소진 시점"으로 정의하는 sampler를 쓴다 — 기본(weighted)은 "1 epoch
            # = len(dataset)" 고정이라 interaction이 적은 라운드는 그 소수 샘플이 한 epoch
            # 안에서도 반복 노출되는 문제가 있음(EXP-10.md 2026-08-04/05 절).
            sampler_kind = cfg.system.get("sampler_kind", "weighted")
            if sampler_kind == "interaction_exhaustion":
                sampler = build_interaction_exhaustion_sampler(
                    self.dataset.get_action_mode_first_frame(),
                    batch_size=cfg.batch_size,
                    target_correct=cfg.system.sampler_target_correct,
                    target_interaction=cfg.system.sampler_target_interaction,
                    target_incorrect=cfg.system.sampler_target_incorrect,
                    action_mode_window=window_labels,
                    preference_frames=pref_frames,
                )
            else:
                sampler = build_balanced_sampler(
                    self.dataset.get_action_mode_first_frame(),
                    target_correct=cfg.system.sampler_target_correct,
                    target_interaction=cfg.system.sampler_target_interaction,
                    target_incorrect=cfg.system.sampler_target_incorrect,
                    action_mode_window=window_labels,
                    preference_frames=pref_frames,
                )
        self.dataloader = DataLoader(
            self.dataset, batch_size=cfg.batch_size, shuffle=(sampler is None), sampler=sampler,
            num_workers=cfg.num_workers, drop_last=True, persistent_workers=cfg.num_workers >= 1,
        )
        logger.info(f"dataset len={len(self.dataset)} cache={cache_mode} image={self.is_image}")

        # LoRA(system.use_lora=true)면 policy.parameters()의 대부분이 requires_grad=False —
        # optimizer엔 학습 가능한 것만 넘긴다(LoRA 없을 때는 전부 True라 동작 그대로).
        self.trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        # encoder_lr_scale>0(system.kind=apo 전용, 2026-07-31 추가)이면 인코더 파라미터를
        # base lr의 이 배율로 별도 그룹에 넣는다 — ApoSystem이 requires_grad만 켜고 lr
        # 분리는 여기서 한다(다른 시스템은 encoder_lr_scale 자체가 없어 항상 단일 그룹).
        encoder_lr_scale = float(getattr(self.system, "encoder_lr_scale", 0.0) or 0.0)
        if encoder_lr_scale > 0 and hasattr(self.policy, "encoders"):
            enc_ids = {id(p) for p in self.policy.encoders.parameters()}
            enc_params = [p for p in self.trainable_params if id(p) in enc_ids]
            other_params = [p for p in self.trainable_params if id(p) not in enc_ids]
            logger.info(
                f"encoder_lr_scale={encoder_lr_scale}: encoder {sum(p.numel() for p in enc_params)/1e6:.1f}M "
                f"params at lr={cfg.lr * encoder_lr_scale:.2e}, 나머지 {sum(p.numel() for p in other_params)/1e6:.1f}M "
                f"at lr={cfg.lr:.2e}"
            )
            param_groups = [
                {"params": other_params, "lr": cfg.lr},
                {"params": enc_params, "lr": cfg.lr * encoder_lr_scale},
            ]
        else:
            param_groups = self.trainable_params
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.95, 0.999), eps=1e-8,
        )
        # 정책 가중치 EMA — DP 공식 레시피(`use_ema: True`)이고 평가도 EMA 가중치로 한다.
        # 2026-08-10까지 누락돼 있었다(lerobot엔 이 필드가 없어서 대조 목록에서 빠졌음).
        self.ema = None
        if cfg.get("use_ema", True):
            self.ema = EMAModel(
                self.policy,
                inv_gamma=cfg.get("ema_inv_gamma", 1.0), power=cfg.get("ema_power", 0.75),
                min_value=cfg.get("ema_min_value", 0.0), max_value=cfg.get("ema_max_value", 0.9999),
            )

        total_steps = max(cfg.num_epochs * len(self.dataloader), 1)
        warmup_steps = 500
        if total_steps > warmup_steps:
            # DP 논문 학습 디테일 + 공식코드(lr_warmup_steps=500)/lerobot(scheduler_warmup_steps=500) 일치.
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
            self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps)
        save_run_config(cfg.output_dir, cfg.task, cfg.policy, cfg.policy_name)  # eval.py 자동 설정용
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
        resume_state = load_resume_state(self.cfg.output_dir, self.device)
        if resume_state is not None:
            self.policy.load_state_dict(resume_state["model"])
            self.optimizer.load_state_dict(resume_state["optimizer"])
            if self.ema is not None and resume_state.get("ema") is not None:
                self.ema.load_state_dict(resume_state["ema"])
            epoch = resume_state["epoch"]
            # lr_scheduler는 저장된 state_dict를 그대로 복원하지 않고 step()으로 진행시킨다 -
            # state_dict에는 T_max(스케줄이 몇 step에 걸쳐 감쇠하는지)도 들어있어서, cfg.num_epochs를
            # 체크포인트 때보다 늘려서 재개(예: 80->120epoch 이어서 더 학습)하면 그대로 로드된 옛
            # T_max(80epoch 기준, 이미 0 근처까지 감쇠 완료) 때문에 이후 LR이 0 근처에 고정되거나
            # 코사인 주기성으로 오히려 다시 올라가는 문제가 있었다(2026-08-07, 사용자와 확인).
            # step()으로 진행시키면 위 __init__에서 이미 새 cfg.num_epochs 기준 T_max로 만들어둔
            # 스케줄러를 그대로 쓰면서 지금까지 진행된 step 수만큼만 앞으로 감아, 새 목표에 맞춰
            # 자연스럽게 이어지는 감쇠 곡선이 된다. num_epochs가 그대로인 일반 resume(중단된 학습
            # 재개)에서도 결과는 동일하다 — 두 스케줄러 모두 last_epoch(진행된 step 수)만으로 LR이
            # 결정되는 순수 함수라, step() N번 호출은 "last_epoch=N인 state_dict를 로드"와 같다.
            for _ in range(epoch * len(self.dataloader)):
                self.lr_scheduler.step()
            logger.info(
                f"resumed from resume_state.pt (epoch {epoch}, optimizer 복원 + "
                f"lr_scheduler는 num_epochs={self.cfg.num_epochs} 기준으로 fast-forward)"
            )
            return epoch

        path, epoch = get_latest_epoch_checkpoint(self.cfg.output_dir)
        if path is None:
            return 0
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["model"])
        logger.warning(
            f"resumed from {path} (epoch {epoch}) — 구 학습이라 resume_state.pt 없음, "
            "optimizer/lr_scheduler는 처음부터 다시 시작됨(warmup 포함)"
        )
        return epoch

    def evaluate(self, num_episodes, max_steps):
        if is_robocasa_task(self.task_cfg):
            return self._evaluate_robocasa(num_episodes, max_steps)
        if self._eval_env is None:
            self._eval_env = make_eval_env(self.task_cfg)
        extra_obs_fn = self._stage_extra_obs_fn if self._stage_tracker is not None else None
        extra_reset_fn = self._stage_tracker.reset if self._stage_tracker is not None else None
        # policy_epoch*.pt에 EMA 가중치를 담으므로 학습 중 평가도 EMA로 해야 나중에
        # 그 체크포인트로 다시 잰 성공률과 같은 모델의 값이 된다(2026-08-11).
        eval_policy = self.ema.averaged_model if self.ema is not None else self.policy
        metrics = rollout_policy(
            env=self._eval_env, policy=eval_policy, normalizer=self.normalizer,
            obs_keys=task_obs_keys(self.task_cfg), obs_horizon=self.cfg.policy.obs_horizon,
            action_horizon=self.cfg.policy.action_horizon, max_steps=max_steps, num_episodes=num_episodes,
            device=self.device, rgb_keys=self.task_cfg.rgb_keys if self.is_image else (),
            extra_obs_fn=extra_obs_fn, extra_obs_reset_fn=extra_reset_fn,
        )
        self.policy.train()
        return metrics

    def _evaluate_robocasa(self, num_episodes, max_steps):
        """RoboCasa(RestockBowls 등) 학습 중 주기 평가 — is_robocasa_task() 참고: robosuite/
        robocasa 패키지가 이 프로세스(robomimic conda env)엔 없어서 env를 직접 못 만든다.
        `robocasa/scripts/eval_rollout.py`를 robocasa conda env 서브프로세스로 돌리고
        표준출력의 "성공률: n/N = X%"·"stage>=k: n/N = X%" 줄을 파싱한다(2026-08-06,
        EXP-01 "학습 중 rollout eval 연동" — 스크립트를 새로 만들지 않고 이미 검증된
        eval_rollout.py를 그대로 재사용, 출력 포맷 안 바뀌면 파싱도 안 깨짐).

        평가마다 현재 policy 가중치를 고정 경로(policy_eval_latest.pt)에 따로 저장한다 —
        ckpt_every_epochs 주기와 eval_every_epochs 주기가 다를 수 있어서, 정기 체크포인트에
        기대면 "지금 epoch"이 아니라 오래된 가중치를 평가할 위험이 있음.

        경로는 전부 절대경로로 넘긴다 — 서브프로세스의 cwd가 robocasa/scripts라 상대경로를
        주면 거기 기준으로 잘못 풀림(스모크 테스트에서 실제로 FileNotFoundError로 확인됨,
        2026-08-06)."""
        import re
        import subprocess

        eval_ckpt_path = os.path.abspath(os.path.join(self.cfg.output_dir, "policy_eval_latest.pt"))
        torch.save({"model": self.policy.state_dict()}, eval_ckpt_path)
        stats_path = os.path.abspath(os.path.join(self.cfg.output_dir, "normalization_stats.json"))
        episode_ids = ",".join(str(i) for i in range(num_episodes))

        cmd = [
            "conda", "run", "-n", "robocasa", "python", "eval_rollout.py",
            "--ckpt", eval_ckpt_path, "--stats", stats_path,
            "--episode_ids", episode_ids, "--max_steps", str(max_steps),
        ]
        if self.num_stages:
            cmd += ["--num_stages", str(self.num_stages)]

        result = subprocess.run(
            cmd, cwd="/home/moai/jungwook_ws/ljw_workspace/robocasa/scripts",
            capture_output=True, text=True, timeout=3600,
        )
        out = result.stdout
        sr_match = re.search(r"성공률: \d+/\d+ = ([\d.]+)%", out)
        if sr_match is None:
            logger.warning(
                f"[eval] robocasa 서브프로세스 출력 파싱 실패(success_rate=0.0으로 처리) — "
                f"stderr 마지막 1000자: {result.stderr[-1000:]}"
            )
            return {"success_rate": 0.0}

        metrics = {"success_rate": float(sr_match.group(1)) / 100}
        for m in re.finditer(r"stage>=(\d+): \d+/\d+ = ([\d.]+)%", out):
            metrics[f"stage_ge_{m.group(1)}"] = float(m.group(2)) / 100
        return metrics

    def train(self, num_epochs, start_epoch=0, use_wandb=False):
        if use_wandb:
            import wandb
        z0_calib_buffer = None
        z0_calib_steps = 0
        z0_calib_percentile = 99.0
        if self.system is not None:
            system_cfg = self.cfg.get("system", None)
            if system_cfg and system_cfg.get("z0_clamp_auto", False):
                # 학습 시작 전(정책=reference라 reward가 전부 ~0) forward-only로 재는 건
                # 무의미하다는 걸 실측으로 발견(2026-08-07 밤, z0_clamp 자동 캘리브레이션
                # 첫 시도가 ±0.000으로 붕괴) — z0는 정책이 reference에서 실제로 벌어져야
                # 의미 있는 스케일이 생기므로, **진짜 학습을 1epoch 진행시키면서**(그동안
                # clamp는 사실상 풀어둠) z0를 수집하고, 그 구간 끝에 percentile로 clamp를
                # 확정해 나머지 학습에 적용한다. 근거: EXP-10.md 2026-08-07 밤 절.
                z0_calib_steps = system_cfg.get("z0_clamp_calib_steps", None) or len(self.dataloader)
                z0_calib_percentile = system_cfg.get("z0_clamp_calib_percentile", 99.0)
                z0_calib_buffer = []
                self.system.kto_loss.z0_clamp_min = -1e6
                self.system.kto_loss.z0_clamp_max = 1e6
                logger.info(f"z0_clamp 자동 캘리브레이션 시작 — 앞 {z0_calib_steps}step 동안 z0 수집(clamp 임시 해제)")
        global_step = start_epoch * len(self.dataloader)
        for epoch in range(start_epoch, num_epochs):
            for raw_batch in self.dataloader:
                obs = {k: v.to(self.device) for k, v in raw_batch["obs"].items()}
                if self.num_stages and "stage" in obs:
                    obs["stage"] = ((obs["stage"] + 1) / 2 * (self.num_stages - 1)).round().long()
                batch = {
                    "obs": obs,
                    "action": raw_batch["action"].to(self.device),
                    "action_mask": raw_batch["action_mask"].to(self.device),
                }
                metrics = {}
                if self.system is not None:
                    action_mode = raw_batch["action_mode"].to(self.device)
                    loss, metrics = self.system.compute_loss(self.policy, batch, action_mode)
                    if z0_calib_buffer is not None:
                        z0_calib_buffer.append(metrics["z0"])
                        if len(z0_calib_buffer) >= z0_calib_steps:
                            bound = float(torch.tensor(z0_calib_buffer).abs().quantile(z0_calib_percentile / 100.0))
                            self.system.kto_loss.z0_clamp_min = -bound
                            self.system.kto_loss.z0_clamp_max = bound
                            logger.info(
                                f"z0_clamp 자동 캘리브레이션 완료(n={len(z0_calib_buffer)}, "
                                f"p{z0_calib_percentile}): ±{bound:.3f}"
                            )
                            z0_calib_buffer = None
                elif self.weighting is not None:
                    per_sample_loss = self.policy.compute_loss(batch, reduction="none")  # (B,)
                    if self.weighting_kind in ("rabc", "phase_rule", "action_variance", "event_radius"):
                        weight_b = self.weighting.compute_weights(raw_batch["demo_id"], raw_batch["index_in_demo"])
                    else:
                        action_mode = raw_batch["action_mode"].to(self.device)
                        # weighting.ref_timestep(기본 null): adaptive weight는 배치 안 샘플들의
                        # 오차를 서로 비교해 순위를 매기는데, per_sample_loss는 샘플별 랜덤 t라
                        # 오차 크기가 t에 따라 100배 넘게 달라진다(실측: t=0-9 MSE 0.1485 vs
                        # t=90-99 0.0012, EXP-10.md 2026-07-31 절). 그러면 순위가 "이 샘플이
                        # 어려운가"가 아니라 "어떤 t를 뽑았는가"를 반영한다. 값을 주면 가중치
                        # 계산만 고정 t에서 별도 forward로 구한다(학습 손실은 랜덤 t 그대로).
                        # system.weight_ref_timestep(apo 경로)과 같은 처방의 sirius 경로판.
                        ref_t = self.cfg.weighting.get("ref_timestep", None)
                        if ref_t is None:
                            error = per_sample_loss.detach()
                        else:
                            with torch.no_grad():
                                t_fixed = torch.full(
                                    (batch["action"].shape[0],), int(ref_t),
                                    device=self.device, dtype=torch.long,
                                )
                                error = self.policy.compute_loss(
                                    batch, reduction="none", timesteps=t_fixed
                                )
                        weight_bt = self.weighting.compute_weights(action_mode, error=error)
                        weight_b = weight_bt.mean(dim=1)
                    # 배치 평균이 1이 되도록 정규화(=moai_policy WeightedDiffusionPolicy의
                    # `(loss*w).sum()/w.sum()`과 수학적으로 동치). class_based도 포함(2026-07-28
                    # — 근거: EXP-10.md "SIRIUS loss 정규화" 절).
                    weight_b = weight_b / weight_b.mean().clamp(min=1e-8)
                    loss = sirius_loss(per_sample_loss, weight_b)  # (B,) x (B,) -> scalar
                    metrics["unweighted_loss"] = per_sample_loss.mean().item()
                else:
                    loss = self.policy.compute_loss(batch)

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.trainable_params, max_norm=self.cfg.get("max_grad_norm", 1e9)
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                if self.ema is not None:
                    self.ema.step(self.policy)
                if self.system is not None:
                    self.system.update_reference_ema(self.policy, global_step)

                if global_step % self.cfg.log_every == 0:
                    logger.info(f"epoch {epoch} step {global_step} loss {loss.item():.4f} grad_norm {grad_norm:.3f}"
                                + (f" {dict(metrics)}" if metrics else ""))
                    if use_wandb:
                        log_dict = {"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0],
                                    "grad_norm": float(grad_norm), "epoch": epoch}
                        # "system/" 접두사는 wandb가 예약해둔 System 섹션 이름과 겹쳐서
                        # 그 섹션의 기본 X축(_runtime)을 물려받아 그래프가 안 뜨는 문제가
                        # 있었음(2026-08-06 발견) — "diag/"로 변경.
                        log_dict.update({f"diag/{k}": v for k, v in metrics.items()})
                        wandb.log(log_dict, step=global_step)
                global_step += 1

            is_last = epoch == num_epochs - 1
            if (epoch + 1) % self.cfg.ckpt_every_epochs == 0 or is_last:
                # LoRA면 policy_epoch*.pt는 merge된 일반 아키텍처로 저장한다(eval.py 등
                # LoRA를 모르는 코드가 그대로 불러올 수 있게, 표준 LoRA 배포 방식). resume용
                # state는 원본(LoRA 구조 유지, 그대로 학습 이어갈 수 있어야 함)을 그대로 쓴다.
                # DP 원문은 평가를 EMA 가중치로 한다 -> policy_epoch*.pt에 EMA를 담아
                # eval.py가 별도 수정 없이 그대로 쓰게 한다. raw 가중치는 resume_state.pt에 남는다.
                # EMA를 먼저 고르고 그 위에 merge_lora를 적용해야 한다(2026-08-11 수정) —
                # 순서가 반대면 merge된 결과를 EMA가 덮어써서 LoRA 구조 그대로 저장되고,
                # LoRA를 모르는 eval.py가 load_state_dict에서 키 불일치로 죽는다.
                save_policy = self.ema.averaged_model if self.ema is not None else self.policy
                if self.system is not None and getattr(self.system, "use_lora", False):
                    import copy
                    from mani_sim.networks.lora import merge_lora
                    save_policy = copy.deepcopy(save_policy)
                    merge_lora(save_policy.unet)
                    if getattr(self.system, "encoder_lora", False) and hasattr(save_policy, "encoders"):
                        merge_lora(save_policy.encoders)
                path = save_epoch_checkpoint(self.cfg.output_dir, epoch + 1, save_policy)
                save_resume_state(self.cfg.output_dir, epoch + 1, self.policy, self.optimizer,
                                  self.lr_scheduler, ema=self.ema)
                logger.info(f"saved checkpoint: {path} (+ resume_state.pt)"
                            + (f" [EMA decay={self.ema.decay:.5f}]" if self.ema is not None else " [EMA off]"))

            if self.cfg.eval_every_epochs > 0 and ((epoch + 1) % self.cfg.eval_every_epochs == 0 or is_last):
                metrics = self.evaluate(self.cfg.eval_episodes, self.cfg.eval_max_steps)
                logger.info(f"[eval] epoch {epoch + 1} success_rate {metrics['success_rate']:.3f} "
                            f"(n={self.cfg.eval_episodes})")
                if use_wandb:
                    # stage_ge_N(robocasa 전용, _evaluate_robocasa 참고)이 있으면 eval/stage_ge_N으로
                    # 같이 로깅 — 다른 task는 이 키가 없어서 success_rate만 그대로 찍힘(하위호환).
                    log_dict = {"eval/success_rate": metrics["success_rate"], "epoch": epoch + 1}
                    log_dict.update({f"eval/{k}": v for k, v in metrics.items() if k != "success_rate"})
                    wandb.log(log_dict, step=global_step)

        if self._eval_env is not None:
            self._eval_env.env.close()
