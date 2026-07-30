"""system=apo — reference 모델 + KTO 앵커(Diffusion-KTO 다리로 diffusion에 이식).

2026-07-17 커밋(`combined_diffusion.py`, nn.Module 래퍼)의 후신. 그 버전은 `outputs/train.py`
(gitignore 대상)가 사라지며 고아 코드가 됐고, `reference`를 서브모듈로 등록해 optimizer/
checkpoint에 딸려가는 결함이 있었다 — 이번엔 **정책을 감싸지 않는다**: `diffusion_trainer.py`의
`self.policy`는 그대로 두고(체크포인트·optimizer·eval 전부 회귀 없음), 이 클래스는 policy를
매 스텝 인자로 받아 손실만 계산해 돌려주는 보조 객체다.

reference는 일반 파이썬 속성(`self.reference`)이지 nn.Module 서브모듈이 아니다 — 이 클래스
자체가 nn.Module이 아니므로 optimizer.parameters()/model.state_dict()에 애초에 안 잡힌다.
APO 원문(reference_model=완전히 별개 PeftModel 인스턴스)·Diffusion-KTO 원문(EMAModel로 파라미터
스왑, LoRA면 disable_adapters())도 둘 다 reference를 학습 모듈 바깥에 둔다[검증-원문,
2026-07-27] — 방식은 다르지만 "reference는 옵티마이저가 못 보는 곳에" 원칙은 같다.

**reference EMA 업데이트(2026-07-28 추가)**: Diffusion-KTO 공식 코드(`train_kto_sd_v1.5.py`,
GitHub `jacklishufan/diffusion-kto`, 직접 조회 확인)는 LoRA 안 쓸 때(=우리 상황, 전체 파라미터
학습) reference를 학습 시작 시점에 고정하지 않고 **`--reference_ema`로 20 optimizer step마다
`EMAModel.step()`으로 계속 업데이트**한다(`reference_ema_momentum` 기본 0.999). 처음(2026-07-27
새벽) 구현할 땐 이걸 놓치고 `copy.deepcopy` 정적 스냅샷으로만 뒀는데, round0 재학습에서
정책이 reference 대비 상대적 우위는 얻으면서도 실제 task 성공률이 무너지는 현상(loss는
건강한데 rollout은 0%에 가까움)의 유력 원인으로 지목되어 EMA로 교체함 — reference가 정책을
계속 느슨하게 따라가야 "그 시점의 나 자신"이라는 기준이 유지되고, 무한정 표류해도 이기는
상태(정적 reference의 함정)를 막는다.

diffusion↔KTO 다리: log(π/π_ref)가 diffusion에선 intractable이라 Diffusion-KTO(arXiv:2404.04465)
방식대로 노이즈예측 MSE 차이로 근사한다(papers/Diffusion-KTO_analysis.md). `compute_loss`가
policy.compute_loss()를 안 쓰고 policy.unet/get_global_cond/train_scheduler를 직접 호출하는
이유가 이것 — 정책이 최종 반환하는 스칼라 손실이 아니라 (B,Tp) 단위 per-step 오차가 필요하다.
"""

import copy

import torch
import torch.nn.functional as F

from mani_sim.datasets.labels import desirable_mask as _desirable_mask
from mani_sim.factory import registry
from mani_sim.losses.apo_loss import APOKTOLoss
from mani_sim.networks.lora import apply_lora

# OSC_POSE 컨벤션(각 팔 7dim=[dx,dy,dz,drx,dry,drz,gripper])에서 각 블록의 마지막 차원이
# gripper다. Square(7)/Transport(14) 전제 — 다른 액션 레이아웃(예: piper joint-space)엔 안 맞음.
_ARM_ACTION_DIM = 7


def _gripper_free_mask(action_dim, device):
    """(action_dim,) bool — gripper 차원만 False. APO Appendix C: undesirable(reject) reward
    계산에서 gripper를 빼기 위해 씀 — gripper는 desirable/undesirable 무관하게 열림/닫힘을
    오래 유지하는 게 정상이라, 그대로 두면 reject 항이 정상적인 유지 동작까지 밀어낸다."""
    mask = torch.ones(action_dim, dtype=torch.bool, device=device)
    mask[_ARM_ACTION_DIM - 1::_ARM_ACTION_DIM] = False
    return mask


class ApoSystem:
    def __init__(self, policy, weighting, beta, desirable_weight, undesirable_weight,
                 preference_frames, init_state_dict=None, reference_ema_enabled=False,
                 reference_ema_momentum=0.999, reference_ema_every=20, z0_clamp_min=-50.0,
                 z0_clamp_max=50.0, use_lora=False, lora_rank=32, lora_alpha=32,
                 bc_aux_weight=0.0, z0_method="match", ars_enabled=False, ars_k1=1.0, ars_eps=1e-4):
        """policy: DiffusionPolicyLowDim | DiffusionPolicyImage(이미 device에 올라간 것).
        weighting: weighting/*.py 인스턴스 또는 None(가중치 축 미사용 — 균등 가중).
        init_state_dict: 직전 라운드 체크포인트의 model state_dict. 주어지면 policy를 여기서
        warm-start하고 reference를 그 시점에서 복제한다 — APO 원문이 policy/reference를
        같은 vla_path(직전 라운드 정책)에서 로드해 둘 다 같은 지점에서 출발시키는 것과 동일
        [검증-원문]. 학습 시작 시 reward가 정확히 0이 되는 게 이 설계의 의도된 결과다.

        reference_ema_enabled(기본 False): APO 원문(갱신 코드 없음)·Diffusion-KTO 원문
        (`--reference_ema` 기본값 False, 논문도 언급 없음) 둘 다 reference를 고정해서 쓴다
        — 6차 시도부터 기본값을 이걸로 되돌림(근거: EXP-10.md "6차 시도" 절).

        use_lora(기본 False, APO는 항상 True): policy.unet의 Linear/Conv1d를
        LoRA(r=lora_rank)로 래핑, 나머지는 전부 freeze. reference는 LoRA 적용 *전* 상태를
        복제하므로 자동으로 "순수 base 가중치"가 된다(APO 공식 코드의
        `reference_model.disable_adapters()`와 수학적으로 동일 — LoRA는 가산적이라
        adapter 없는 forward와 base-only 복제본은 항상 같은 출력을 낸다).

        z0_method(기본 "match"): z0(KL anchor) 추정 방식 — "match"(같은 인덱스, 셔플 없이
        배치 reward 평균) | "mismatch"(state는 그대로 두고 action만 다른 샘플 것으로 섞은
        pair로 추정, KTO 원문 방식). diffusion-kto 공식 코드(`jacklishufan/diffusion-kto`,
        `train_kto_sd_v1.5.py`)의 `--bce_offset` argparse 기본값은 "none"(=match)이고, 논문
        결과를 낸 `exps/example.sh`도 이 플래그를 안 넘겨 기본값 그대로 씀[검증-코드,
        2026-07-30] — mismatch는 U-Net을 2회 더 돌려야 해서 SD 규모에선 비쌌을 것으로 추정.
        우리 UNet은 작아 두 방식 다 실험 대상(EXP-10.md 2026-07-30 절).

        ars_enabled/ars_k1/ars_eps: Adaptive Rejection Scaling(PG-DPO Eq.7 기반, K1/eps는
        원문과 동일한 이름) — losses/apo_loss.py 모듈 docstring 참고(arXiv:2511.19049,
        chosen과 구별하기 어려운 rejected 샘플의 밀어내는 힘을 줄여 spillover로 인한
        chosen 품질 붕괴를 완화)."""
        if init_state_dict is not None:
            policy.load_state_dict(init_state_dict)

        reference = copy.deepcopy(policy)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        self.reference = reference

        self.use_lora = use_lora
        if use_lora:
            for p in policy.parameters():
                p.requires_grad_(False)
            apply_lora(policy.unet, rank=lora_rank, alpha=lora_alpha)

        assert z0_method in ("match", "mismatch"), f"z0_method={z0_method!r} 미지원"
        self.z0_method = z0_method

        self.reference_ema_enabled = reference_ema_enabled
        self.reference_ema_momentum = reference_ema_momentum
        self.reference_ema_every = reference_ema_every

        self.weighting = weighting
        self.kto_loss = APOKTOLoss(
            beta=beta, desirable_weight=desirable_weight, undesirable_weight=undesirable_weight,
            preference_frames=preference_frames, z0_clamp_min=z0_clamp_min, z0_clamp_max=z0_clamp_max,
            ars_enabled=ars_enabled, ars_k1=ars_k1, ars_eps=ars_eps,
        )
        self.last_ref_distance = 0.0
        # bc_aux_weight(기본 0=원문 그대로): KTO sigmoid는 이미 reward>z0인 desirable
        # 샘플에서 gradient가 saturate돼(Prop 4.1) 그 샘플을 "지켜주는" 힘이 없다 — round0에서
        # 실측 확인(EXP-10.md "Degradation 직접 검증" 절): base가 이미 잘하던 reach가 8번의
        # 시도(가중치 축 유무 무관) 전부에서 무너짐. 이 항은 desirable 샘플에 sigmoid와 무관한
        # 표준 noise-prediction MSE를 더해, saturate돼도 "이 행동을 계속 재현하라"는 gradient가
        # 남게 한다 — 원문에는 없는 항이며, 우리가 실측으로 특정한 실패 메커니즘을 겨냥한 수정.
        self.bc_aux_weight = bc_aux_weight

    @torch.no_grad()
    def update_reference_ema(self, policy, global_step):
        """trainer가 매 optimizer.step() 직후 호출 - reference_ema_enabled=False(기본,
        6차부터)면 아무것도 안 함. True면 reference_ema_every 배수 스텝에서만 실제로
        갱신한다(Diffusion-KTO 공식 코드의 `(global_step+1) % 20 == 0` 그대로)."""
        if not self.reference_ema_enabled:
            return
        if (global_step + 1) % self.reference_ema_every != 0:
            return
        decay = self.reference_ema_momentum
        sq_dist = 0.0
        for ref_p, policy_p in zip(self.reference.parameters(), policy.parameters()):
            sq_dist += (ref_p - policy_p).pow(2).sum().item()
            ref_p.mul_(decay).add_(policy_p, alpha=1.0 - decay)
        self.last_ref_distance = sq_dist ** 0.5

    def compute_loss(self, policy, batch, action_mode):
        """batch: {'obs','action','action_mask'} — trainer가 다른 경로와 똑같이 만드는 그대로.
        action_mode: (B, Tp) — trainer가 raw_batch에서 직접 꺼내 넘긴다(다른 가중치 축과 동일
        관례, batch dict엔 안 섞음).

        반환: (scalar loss, dict 진단 지표) — APOKTOLoss.compute와 동일 반환 형태."""
        obs, action, action_mask = batch["obs"], batch["action"], batch["action_mask"]
        device = action.device
        B = action.shape[0]

        cond = policy.get_global_cond(obs)
        scheduler = policy.train_scheduler
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
        noise = torch.randn_like(action)
        noisy = scheduler.add_noise(action, noise, timesteps)

        pred = policy.unet(noisy, timesteps, cond)
        sq_err = F.mse_loss(pred, noise, reduction="none")  # (B, Tp, Da)
        model_mse = sq_err.mean(dim=-1)  # (B, Tp)
        with torch.no_grad():
            ref_cond = self.reference.get_global_cond(obs)
            ref_pred = self.reference.unet(noisy, timesteps, ref_cond)
        ref_sq_err = F.mse_loss(ref_pred, noise, reduction="none")
        ref_mse = ref_sq_err.mean(dim=-1)

        # gripper 제외 MSE(APO Appendix C) — undesirable(reject) reward 전용, chosen/z0/
        # weighting은 전체 액션(gripper 포함) 그대로 씀. gripper_free_mask=_gripper_free_mask
        # 참고(OSC_POSE 7dim/팔 컨벤션, Square/Transport 전제).
        gripper_free = _gripper_free_mask(sq_err.shape[-1], device)
        model_mse_ng = sq_err[..., gripper_free].mean(dim=-1)  # (B, Tp)
        ref_mse_ng = ref_sq_err[..., gripper_free].mean(dim=-1)

        mask = action_mask.float()

        # z0(KL anchor) — z0_method="mismatch"면 KTO 원문 방식(state는 그대로 두고 action만
        # 다른 샘플 것으로 섞은 mismatched pair로 별도 추정, KTO paper Implementation 절
        # "matching inputs x′ with unrelated outputs yU′", APO 원문 코드의 `mismatch_label`
        # 확인, 2026-07-29). "match"(기본, 2026-07-30)면 diffusion-kto 공식 코드
        # (`train_kto_sd_v1.5.py`) 기본값(`--bce_offset=none`)과 동일하게 셔플 없이 배치
        # reward 평균을 씀(APOKTOLoss.compute의 mismatch_reward=None 분기) — 자세한 트레이드오프는
        # apo_system.py 모듈 docstring/EXP-10.md 2026-07-30 절 참고.
        # mismatch일 땐 KL 항 역전파 안 함(원문 "we do not back-propagate through the KL term").
        mismatch_reward = None
        if self.z0_method == "mismatch":
            with torch.no_grad():
                if B > 1:
                    perm = torch.randperm(B, device=device)
                    if (perm == torch.arange(B, device=device)).all():
                        perm = torch.roll(perm, 1)
                    mismatch_action = action[perm]
                    mismatch_mask = mask * mask[perm]
                    mismatch_noisy = scheduler.add_noise(mismatch_action, noise, timesteps)
                    mismatch_pred = policy.unet(mismatch_noisy, timesteps, cond)
                    mismatch_model_mse = F.mse_loss(mismatch_pred, noise, reduction="none").mean(dim=-1)
                    mismatch_ref_pred = self.reference.unet(mismatch_noisy, timesteps, ref_cond)
                    mismatch_ref_mse = F.mse_loss(mismatch_ref_pred, noise, reduction="none").mean(dim=-1)
                    mismatch_reward = (-mismatch_model_mse * mismatch_mask).sum(dim=1) - (
                        -mismatch_ref_mse * mismatch_mask
                    ).sum(dim=1)

        if self.weighting is not None:
            error = (model_mse.detach() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B,)
            weight = self.weighting.compute_weights(action_mode, error=error)  # (B, Tp)
        else:
            weight = torch.ones_like(model_mse)

        # 패딩 프레임을 0으로 눌러 sum(dim=1)에서 자동 제외(같은 처리를 policy/ref 양쪽에
        # 동일하게 적용하니 reward=차이 계산에서 마스킹 효과가 상쇄되지 않고 그대로 남는다).
        log_probs = -model_mse * mask
        ref_log_probs = -ref_mse * mask
        log_probs_reject = -model_mse_ng * mask
        ref_log_probs_reject = -ref_mse_ng * mask
        total_loss, metrics = self.kto_loss.compute(
            log_probs, ref_log_probs, weight, action_mode, mismatch_reward=mismatch_reward,
            log_probs_reject=log_probs_reject, ref_log_probs_reject=ref_log_probs_reject,
        )

        des = _desirable_mask(action_mode, preference_frames=self.kto_loss.preference_frames)

        if self.bc_aux_weight > 0:
            per_sample_error = (model_mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B,)
            bc_loss = (per_sample_error * des.float()).sum()  # sum, not mean — matches kto_loss's sum convention
            total_loss = total_loss + self.bc_aux_weight * bc_loss
            metrics["bc_aux_loss"] = bc_loss.item()

        # raw(=reference와 무관한 절대) MSE — reward(=ref-model 상대값)만으로는 "정말 model이
        # desirable에서 정밀해지고 undesirable에서 뭉툭해지는지" 직접 볼 수 없다(EXP-10.md
        # 2026-07-30 밤 "z0 공유로 인한 비대칭 결합" 절 — LoRA 저랭크 제약 때문에 두 그룹을
        # 따로 조각하는 대신 전체적으로 뭉툭해지는 지름길을 택했을 가능성 가설, 아직 미검증).
        # chosen/rejected 각각 raw model_mse가 이후 라운드에서 실제로 같이 움직이는지(=뭉툭해짐의
        # 증거) 확인하기 위한 진단 전용 로그 — loss 계산엔 관여 안 함.
        with torch.no_grad():
            per_sample_mse = (model_mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            per_sample_mse_ng = (model_mse_ng * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            per_sample_ref_mse = (ref_mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            per_sample_ref_mse_ng = (ref_mse_ng * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            n_ch, n_rej = int(des.sum().item()), int((~des).sum().item())
            metrics["raw_model_mse_chosen"] = float(per_sample_mse[des].mean().item()) if n_ch else float("nan")
            metrics["raw_model_mse_rejected"] = (
                float(per_sample_mse_ng[~des].mean().item()) if n_rej else float("nan")
            )
            metrics["raw_ref_mse_chosen"] = float(per_sample_ref_mse[des].mean().item()) if n_ch else float("nan")
            metrics["raw_ref_mse_rejected"] = (
                float(per_sample_ref_mse_ng[~des].mean().item()) if n_rej else float("nan")
            )

        metrics["ref_distance"] = self.last_ref_distance
        return total_loss, metrics


@registry.register_system("apo")
def _build_apo(cfg, policy, weighting, device, init_state_dict):
    sys_cfg = cfg.system
    return ApoSystem(
        policy, weighting, beta=sys_cfg.beta, desirable_weight=sys_cfg.desirable_weight,
        undesirable_weight=sys_cfg.undesirable_weight, preference_frames=sys_cfg.preference_frames,
        init_state_dict=init_state_dict,
        reference_ema_enabled=sys_cfg.get("reference_ema_enabled", False),
        reference_ema_momentum=sys_cfg.get("reference_ema_momentum", 0.999),
        reference_ema_every=sys_cfg.get("reference_ema_every", 20),
        z0_clamp_min=sys_cfg.get("z0_clamp_min", -50.0),
        z0_clamp_max=sys_cfg.get("z0_clamp_max", 50.0),
        use_lora=sys_cfg.get("use_lora", False),
        lora_rank=sys_cfg.get("lora_rank", 32),
        lora_alpha=sys_cfg.get("lora_alpha", 16),
        bc_aux_weight=sys_cfg.get("bc_aux_weight", 0.0),
        z0_method=sys_cfg.get("z0_method", "match"),
        ars_enabled=sys_cfg.get("ars_enabled", False),
        ars_k1=sys_cfg.get("ars_k1", 1.0),
        ars_eps=sys_cfg.get("ars_eps", 1e-4),
    )
