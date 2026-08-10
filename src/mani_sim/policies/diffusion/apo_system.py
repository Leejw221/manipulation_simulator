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

from mani_sim.datasets.labels import LABEL_INTV
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
                 bc_aux_weight=0.0, z0_method="match", ars_enabled=False, ars_k1=1.0, ars_eps=1e-4,
                 ars_ratio_clip=10.0, hinge_weight=0.0, kl_retain_weight=0.0,
                 kl_retain_scope="all", kl_retain_ck=True, reward_reduction="sum", z0_gripper_free=True,
                 loss_frames="all", diag_mismatch_every=0,
                 action_error_source="noise_mse", weight_ref_timestep=None, encoder_lr_scale=0.0,
                 encoder_lora=False):
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
        pair로 추정, KTO 원문 방식) | "zero"(앵커 없음, z0를 항상 0으로 고정). diffusion-kto
        공식 코드(`jacklishufan/diffusion-kto`, `train_kto_sd_v1.5.py`)의 `--bce_offset`
        argparse 기본값은 "none"(=match)이고, 논문 결과를 낸 `exps/example.sh`도 이 플래그를
        안 넘겨 기본값 그대로 씀[검증-코드, 2026-07-30] — mismatch는 U-Net을 2회 더 돌려야
        해서 SD 규모에선 비쌌을 것으로 추정. 우리 UNet은 작아 두 방식 다 실험 대상(EXP-10.md
        2026-07-30 절).

        "zero"는 TRL(HuggingFace) `KTOTrainer`의 `loss_type="apo_zero_unpaired"`와 동일 —
        Anchored Preference Optimization(D'Oosterlinck et al., arXiv:2408.06266)의 unpaired
        변종[검증-코드, `trl/trainer/kto_trainer.py` 직접 확인, 2026-08-08]. z0를 배치에서
        추정하지 않고 상수 0으로 고정하므로, EXP-10.md (65)(84)절이 계속 추적한 "z0가 학습 중
        하락해서 margin_chosen이 진짜 reward 개선 없이 오른다"는 경로를 구조적으로 차단한다
        — mismatch pair 계산(U-Net 2회 추가 forward)도 필요 없어 match/mismatch보다 가볍다.

        ars_enabled/ars_k1/ars_eps: Adaptive Rejection Scaling(PG-DPO Eq.7 기반, K1/eps는
        원문과 동일한 이름) — losses/apo_loss.py 모듈 docstring 참고(arXiv:2511.19049,
        chosen과 구별하기 어려운 rejected 샘플의 밀어내는 힘을 줄여 spillover로 인한
        chosen 품질 붕괴를 완화).

        hinge_weight(기본 0.0=비활성, 2026-08-08 추가): DPOP(Pal et al., Smaug,
        arXiv:2402.13228) 스타일 — chosen이 reference보다 **절대적으로 나빠질 때만**
        벌점을 준다: `max(0, model_mse_chosen - ref_mse_chosen)`(desirable 샘플 평균).
        KTO의 chosen 항(`1-sigmoid(beta*(r-z0))`)은 r이 z0보다 계속 커지면(=이미 잘함)
        sigmoid 도함수가 0으로 죽어 그 샘플을 사실상 포기하는데, DPOP hinge는 MSE에
        선형이라 gradient가 안 죽는다 — 이미 만족된 샘플은 자동으로 hinge도 0(=max(0,·)
        조건)이라 서로 방해 없이 KTO 위에 얹을 수 있다. `bc_aux_weight`(항상 켜진 절대
        MSE)와 다른 점은 "reference를 이기면 꺼지는가"뿐 — 같이 켜면 desirable 보호항이
        (KTO chosen + bc_aux + hinge) 3중이 되므로, hinge를 쓸 땐 `bc_aux_weight=0`을
        권장(EXP-10.md 2026-08-08 절 논의). (75)절이 발견한 "desirable 샘플 약
        25~40%가 reward_chosen≤0"이 곧 이 hinge가 활성화되는 비율과 같은 양 —
        `hinge_active_frac` 메트릭으로 직접 검증 가능해짐.

        encoder_lr_scale(기본 0.0=완전 동결, 2026-07-31 추가): use_lora=True일 때도
        policy.encoders(비전 인코더)를 학습 가능하게 풀지 여부 — 0보다 크면 unfreeze하고,
        트레이너가 이 값을 base lr에 곱한 lr로 별도 파라미터 그룹을 만든다(실제 스케일링은
        diffusion_trainer.py에서 수행, 여기선 requires_grad만 켠다).
        APO 원문(OpenVLA)은 비전 백본이 인터넷 규모 foundation model이라 동결해도 downstream
        재조합만으로 충분하지만, 우리 인코더는 demo 50개로 처음부터 학습한 ResNet18이라
        원문의 전제(백본에 이미 충분한 지각이 있음)가 성립하지 않는다 — 실측(round1):
        인코더를 동결한 채로는 손실 형태를 KTO<->순수 BC로 바꿔도(26%<->18%), undesirable_weight를
        0~3으로 바꿔도(16%~2%) base(30~32%)를 넘지 못했고, 전부 epoch이 늘수록 단조 감소했다
        (파인튜닝이 base 위에 뭔가를 더 배우는 게 아니라 계속 깎아먹기만 하는 모양). Diffusion
        Policy/OpenVLA 문헌 둘 다 인코더 동결이 성능을 떨어뜨리고(약한 lr로 함께 학습이 최선),
        DP는 정책 네트워크의 1/10 lr을 권장한다[문헌, 2026-07-31 WebSearch로 확인]. 근거 상세는
        EXP-10.md 2026-07-31 절.

        encoder_lora(기본 False, 2026-08-01 추가): encoder_lr_scale>0일 때 인코더를 여는 방식.
        False면 기존 동작(전체 파라미터 직접 학습, full fine-tune) — 그런데 이동량을
        키울수록(scale 0.1->1.0) 오히려 성공률이 떨어지는 패턴이 관측됐다(base 36% > 34% >
        29%/28%, EXP-10.md 2026-08-01 절). UNet은 LoRA(저랭크 제약)라 망각에 덜 취약한데
        인코더는 무제약이었다는 게 차이일 수 있다는 가설[산출, 미검증] — True면 인코더도
        UNet과 같은 방식(apply_lora)으로 저랭크 제약을 걸어 이 가설을 검증한다."""
        if init_state_dict is not None:
            policy.load_state_dict(init_state_dict)

        reference = copy.deepcopy(policy)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        self.reference = reference

        self.use_lora = use_lora
        self.encoder_lr_scale = encoder_lr_scale
        self.encoder_lora = encoder_lora
        if use_lora:
            for p in policy.parameters():
                p.requires_grad_(False)
            apply_lora(policy.unet, rank=lora_rank, alpha=lora_alpha)
            if encoder_lr_scale > 0 and hasattr(policy, "encoders"):
                if encoder_lora:
                    apply_lora(policy.encoders, rank=lora_rank, alpha=lora_alpha)
                else:
                    for p in policy.encoders.parameters():
                        p.requires_grad_(True)

        assert z0_method in ("match", "mismatch", "zero"), f"z0_method={z0_method!r} 미지원"
        assert loss_frames in ("all", "preference"), f"loss_frames={loss_frames!r} 미지원"
        self.loss_frames = loss_frames
        self.diag_mismatch_every = int(diag_mismatch_every)
        self._diag_step = 0
        self.z0_method = z0_method

        # action_error_source: weighting(action_error 축)에 넘길 "오차"의 정의.
        #   "noise_mse"(기존 기본값) — 노이즈 예측 MSE. reward와 같은 양을 재사용.
        #   "x0_l1"(APO 원문 충실) — 노이즈 예측에서 복원한 액션(x0)의 GT 대비 L1.
        # 원문은 reward(토큰 log확률)와 weight(디코딩된 연속 액션 L1)를 의도적으로 다른 양으로
        # 쓴다(논문 4.2절) — compute_loss의 해당 분기 주석 참고. 2026-07-30~31 실측으로
        # "노이즈 예측 MSE는 실제 액션 품질을 대변하지 못한다"가 확인됐으므로(EXP-10.md),
        # x0_l1이 원문 의도에 맞는 선택이다. 기존 런 재현성을 위해 기본값은 noise_mse 유지.
        assert action_error_source in ("noise_mse", "x0_l1"), \
            f"action_error_source={action_error_source!r} 미지원"
        self.action_error_source = action_error_source
        # weight_ref_timestep(기본 None=기존 동작): adaptive weight를 계산할 때 쓸
        # 고정 기준 timestep. None이면 학습 손실과 같은 샘플별 랜덤 t를 그대로 쓴다.
        # 값을 주면 가중치만 그 t에서 별도 forward로 계산 — compute_loss 해당 분기 주석 참고.
        self.weight_ref_timestep = weight_ref_timestep
        self.last_action_error = 0.0

        self.reference_ema_enabled = reference_ema_enabled
        self.reference_ema_momentum = reference_ema_momentum
        self.reference_ema_every = reference_ema_every

        self.weighting = weighting
        self.kto_loss = APOKTOLoss(
            beta=beta, desirable_weight=desirable_weight, undesirable_weight=undesirable_weight,
            preference_frames=preference_frames, z0_clamp_min=z0_clamp_min, z0_clamp_max=z0_clamp_max,
            ars_enabled=ars_enabled, ars_k1=ars_k1, ars_eps=ars_eps, ars_ratio_clip=ars_ratio_clip,
            reward_reduction=reward_reduction, z0_gripper_free=z0_gripper_free,
        )
        self.last_ref_distance = 0.0
        # bc_aux_weight(기본 0=원문 그대로): KTO sigmoid는 이미 reward>z0인 desirable
        # 샘플에서 gradient가 saturate돼(Prop 4.1) 그 샘플을 "지켜주는" 힘이 없다 — round0에서
        # 실측 확인(EXP-10.md "Degradation 직접 검증" 절): base가 이미 잘하던 reach가 8번의
        # 시도(가중치 축 유무 무관) 전부에서 무너짐. 이 항은 desirable 샘플에 sigmoid와 무관한
        # 표준 noise-prediction MSE를 더해, saturate돼도 "이 행동을 계속 재현하라"는 gradient가
        # 남게 한다 — 원문에는 없는 항이며, 우리가 실측으로 특정한 실패 메커니즘을 겨냥한 수정.
        self.bc_aux_weight = bc_aux_weight
        self.hinge_weight = hinge_weight
        self.kl_retain_weight = kl_retain_weight
        self.kl_retain_scope = kl_retain_scope
        self.kl_retain_ck = kl_retain_ck

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

    def _reference_cond(self, obs, crop_rng, crop_rng_cuda, device):
        """reference의 conditioning을 policy와 **같은 크롭**에서 뽑는다(2026-08-11).

        난수 상태를 policy가 크롭을 뽑기 직전으로 되감아 reference의 RandomCrop이 같은 오프셋을
        고르게 하고, 끝나면 원래 스트림으로 복구해 이후 난수(timestep·noise 등)에 영향을 주지
        않는다. reference는 train 모드여야 RandomCrop 경로를 탄다."""
        saved, saved_cuda = torch.get_rng_state(), (
            torch.cuda.get_rng_state(device) if crop_rng_cuda is not None else None
        )
        torch.set_rng_state(crop_rng)
        if crop_rng_cuda is not None:
            torch.cuda.set_rng_state(crop_rng_cuda, device)
        was_training = self.reference.training
        self.reference.train()
        try:
            ref_cond = self.reference.get_global_cond(obs)
        finally:
            self.reference.train(was_training)
            torch.set_rng_state(saved)
            if saved_cuda is not None:
                torch.cuda.set_rng_state(saved_cuda, device)
        return ref_cond

    def compute_loss(self, policy, batch, action_mode):
        """batch: {'obs','action','action_mask'} — trainer가 다른 경로와 똑같이 만드는 그대로.
        action_mode: (B, Tp) — trainer가 raw_batch에서 직접 꺼내 넘긴다(다른 가중치 축과 동일
        관례, batch dict엔 안 섞음).

        반환: (scalar loss, dict 진단 지표) — APOKTOLoss.compute와 동일 반환 형태."""
        obs, action, action_mask = batch["obs"], batch["action"], batch["action_mask"]
        device = action.device
        B = action.shape[0]

        # reward = ref_mse - model_mse는 "같은 관측에서 policy가 reference보다 얼마나 나은가"여야
        # 하는데, 비전 인코더가 학습 중엔 RandomCrop·평가 중엔 CenterCrop을 쓰므로
        # (diffusion_policy_image.py `x = self.random_crop(x) if self.training else self.center_crop(x)`)
        # reference를 eval()로 두면 두 모델이 **서로 다른 크롭을 본다**. 실측(2026-08-11): 가중치가
        # 100% 동일해도 conditioning이 feature 최댓값의 72%만큼 달라지고, 그 결과 reward가 0이
        # 아니라 std 0.379(신호 0.489의 78%)의 잡음을 갖는다. 그래서 (1) reference도 train 모드로
        # 두고 (2) 크롭 난수를 policy 호출 시점으로 되감아 **같은 크롭**을 보게 한다. 이 네트워크엔
        # dropout이 없고 BatchNorm은 전부 GroupNorm이라, train 모드 전환이 바꾸는 건 크롭뿐이다.
        crop_rng = torch.get_rng_state()
        crop_rng_cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

        cond = policy.get_global_cond(obs)
        scheduler = policy.train_scheduler
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
        noise = torch.randn_like(action)
        noisy = scheduler.add_noise(action, noise, timesteps)

        pred = policy.unet(noisy, timesteps, cond)
        sq_err = F.mse_loss(pred, noise, reduction="none")  # (B, Tp, Da)
        model_mse = sq_err.mean(dim=-1)  # (B, Tp)
        with torch.no_grad():
            ref_cond = self._reference_cond(obs, crop_rng, crop_rng_cuda, device)
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
        if self.loss_frames == "preference":
            # 라벨 판정 구간과 손실 적용 구간을 일치시킨다(2026-08-11).
            # desirable_mask()는 앞 preference_frames(8)프레임 다수결로 청크 하나에 이진 라벨을
            # 붙이는데, reward는 청크 전체(Tp=16)에 걸쳐 계산된다 — 두 구간이 어긋나 있다.
            # 실측(square_round1_new_merged_k10): rejected로 라벨된 청크에서 실제로 밀어내는
            # 프레임 중 43.8%가 INTV(사람 교정)다. 앞 8프레임으로 제한하면 8.3%로 떨어진다.
            # APO 원문은 샘플당 액션이 1개라 이 자유도 자체가 없어 어느 쪽도 원문 근거가 없다 —
            # 그래서 옵션으로 두고 단일 변수로 검정한다. "all"이 종래 동작.
            mask = mask.clone()
            mask[:, self.kto_loss.preference_frames:] = 0.0

        # z0(KL anchor) — z0_method="mismatch"면 KTO 원문 방식(state는 그대로 두고 action만
        # 다른 샘플 것으로 섞은 mismatched pair로 별도 추정, KTO paper Implementation 절
        # "matching inputs x′ with unrelated outputs yU′", APO 원문 코드의 `mismatch_label`
        # 확인, 2026-07-29). **매칭 방식은 APO 원문 코드(`dataset/collator.py`,
        # `PaddedCollatorForActionPrediction`)와 일치시킴 — 무작위 순열이 아니라
        # `indices = list(range(1,N)) + [0]` 형태의 **결정적 circular shift**(배치 내
        # 인덱스를 한 칸씩 미룸, [검증-코드, 2026-07-31 WebFetch로 직접 확인] — 이전엔
        # `torch.randperm`으로 잘못 구현돼 있었음, EXP-10.md 2026-07-31 절 참고).
        # "match"(2026-07-30)면 diffusion-kto 공식 코드(`train_kto_sd_v1.5.py`) 기본값
        # (`--bce_offset=none`)과 동일하게 셔플 없이 배치 reward 평균을 씀
        # (APOKTOLoss.compute의 mismatch_reward=None 분기) — 자세한 트레이드오프는
        # apo_system.py 모듈 docstring/EXP-10.md 2026-07-30 절 참고.
        # mismatch일 땐 KL 항 역전파 안 함(원문 "we do not back-propagate through the KL term").
        mismatch_reward = None
        mismatch_reward_reject = None
        # mismatch 통계(raw_model_mse_mismatch 등)는 "정책이 관측을 제대로 구분하는가"(조건
        # 판별력)를 재는 유일한 지표다 — EXP-10 (115)절에서 0% 붕괴를 진단한 게 이 값이었다.
        # 그런데 z0_method="mismatch"일 때만 계산돼서, match/zero로 돌린 런에는 아예 기록이
        # 없었다. 나중에 그 시점 데이터로 재구성이 안 되므로 **z0_method와 무관하게** 남긴다
        # (2026-08-11). U-Net forward 2회가 추가되므로 매 스텝이 아니라 diag_mismatch_every
        # 스텝마다만 — 0이면 완전히 끔. z0에 쓰지 않을 땐 계산 후 None으로 되돌려 손실 경로에
        # 영향을 주지 않는다. 이 블록은 no_grad이고 새 난수를 뽑지 않아(noise/timesteps/cond를
        # 재사용) 학습 결과에 영향이 없다 — 켠 런과 끈 런을 그대로 비교할 수 있다.
        # 계산 안 한 스텝에 직전 값이 그대로 다시 찍히지 않도록 매 스텝 비운다(2026-08-11).
        # 매 스텝 계산하던 때(z0_method=mismatch 전용)는 항상 최신이라 문제가 없었는데,
        # 주기 계산으로 바꾸면서 나머지 99스텝이 stale 값을 새 값처럼 로깅하게 됐다.
        self._diag_mismatch = None
        self._diag_mismatch_by_donor = None
        self._diag_step += 1
        _diag_due = (self.diag_mismatch_every > 0
                     and self._diag_step % self.diag_mismatch_every == 0)
        _z0_needs_mismatch = self.z0_method == "mismatch"
        if _z0_needs_mismatch or _diag_due:
            with torch.no_grad():
                if B > 1:
                    mismatch_action = torch.roll(action, shifts=-1, dims=0)
                    mismatch_mask = mask * torch.roll(mask, shifts=-1, dims=0)
                    mismatch_noisy = scheduler.add_noise(mismatch_action, noise, timesteps)
                    mismatch_pred = policy.unet(mismatch_noisy, timesteps, cond)
                    mismatch_sq_err = F.mse_loss(mismatch_pred, noise, reduction="none")
                    mismatch_model_mse = mismatch_sq_err.mean(dim=-1)
                    mismatch_ref_pred = self.reference.unet(mismatch_noisy, timesteps, ref_cond)
                    mismatch_ref_sq_err = F.mse_loss(mismatch_ref_pred, noise, reduction="none")
                    mismatch_ref_mse = mismatch_ref_sq_err.mean(dim=-1)
                    # reward_reduction="mean"이면 앵커도 같은 단위로 집계해야 한다
                    # (chosen reward만 평균내고 z0는 합으로 두면 margin이 단위가 다른
                    # 두 양의 차가 된다 — (88)절 gripper 단위 불일치와 같은 계열의 함정).
                    _mm_agg = mismatch_mask.sum(dim=1).clamp(min=1.0) \
                        if self.kto_loss.reward_reduction == "mean" else 1.0
                    mismatch_reward = ((-mismatch_model_mse * mismatch_mask).sum(dim=1) - (
                        -mismatch_ref_mse * mismatch_mask
                    ).sum(dim=1)) / _mm_agg
                    # margin_rejected 전용 앵커 — gripper 제외(rejected_reward와 같은 단위,
                    # apo_loss.py compute()의 mismatch_reward_reject 참고, 2026-08-08 발견한
                    # 단위 불일치 수정). U-Net 재forward 없이 위에서 이미 계산한
                    # mismatch_sq_err/mismatch_ref_sq_err에서 gripper 차원만 빼고 다시
                    # 평균낸 것 — 추가 비용 없음.
                    mismatch_model_mse_reject = mismatch_sq_err[..., gripper_free].mean(dim=-1)
                    mismatch_ref_mse_reject = mismatch_ref_sq_err[..., gripper_free].mean(dim=-1)
                    mismatch_reward_reject = ((-mismatch_model_mse_reject * mismatch_mask).sum(dim=1) - (
                        -mismatch_ref_mse_reject * mismatch_mask
                    ).sum(dim=1)) / _mm_agg
                    # z0(=mismatch_reward 평균)가 reward_chosen보다 4~5배 큰 현상의 원인을
                    # 분해하기 위한 진단(2026-08-08 추가). z0가 큰 게 (a) 모델이 mismatch에서
                    # 정말 많이 개선돼서인지 (b) reference가 mismatch에서 워낙 나빠 개선
                    # 여지가 컸던 것인지는 reward(=차이)만으로는 구분이 안 된다 —
                    # raw 값 양쪽을 다 남겨야 판별 가능(EXP-10.md 2026-08-08 절).
                    _mm_denom = mismatch_mask.sum(dim=1).clamp(min=1.0)
                    self._diag_mismatch = (
                        float(((mismatch_model_mse * mismatch_mask).sum(dim=1) / _mm_denom).mean().item()),
                        float(((mismatch_ref_mse * mismatch_mask).sum(dim=1) / _mm_denom).mean().item()),
                    )
                    # z0 추정량(mismatch reward)이 rejected 억제에 오염되는지 확인하기 위한
                    # 진단(2026-08-08 추가). torch.roll(action, -1)이 만드는 mismatch pair의
                    # ~25%는 undesirable(rejected) action을 그대로 가져온다 — 그 action이
                    # loss로 억제되면 mismatch_model_mse도 같이 올라가 z0를 끌어내릴 수 있다.
                    # KTO가 mismatch 앵커를 쓰는 이유는 "chosen/rejected reward와 독립인
                    # 기준선"이 필요해서인데, 배치 내 roll이 그 독립성을 깨고 있을 가능성 —
                    # desirable-donor/undesirable-donor로 나눠 평균을 분리해야 판별 가능
                    # (EXP-10.md 2026-08-08 절). 여기 없으면 나중에 이 로그 자체가 없어서
                    # 그 시점 데이터로는 재구성이 안 된다(raw_ref_mse_mismatch 사례와 동일 이유).
                    donor_desirable = _desirable_mask(
                        action_mode, preference_frames=self.kto_loss.preference_frames
                    )
                    donor_desirable_rolled = torch.roll(donor_desirable, shifts=-1)
                    _per_model_mse = (mismatch_model_mse * mismatch_mask).sum(dim=1) / _mm_denom
                    _des_n = donor_desirable_rolled.float().sum().clamp(min=1.0)
                    _undes_n = (~donor_desirable_rolled).float().sum().clamp(min=1.0)
                    self._diag_mismatch_by_donor = (
                        float((_per_model_mse * donor_desirable_rolled.float()).sum().item() / _des_n),
                        float((_per_model_mse * (~donor_desirable_rolled).float()).sum().item() / _undes_n),
                    )
            if not _z0_needs_mismatch:
                mismatch_reward = None          # 진단으로만 계산했으므로 z0 경로엔 안 넘긴다
                mismatch_reward_reject = None
        if self.z0_method == "zero":
            # 배치에서 z0를 추정하지 않고 상수 0으로 고정(TRL apo_zero_unpaired와 동일 —
            # 모듈 docstring 참고). APOKTOLoss.compute는 mismatch_reward가 주어지면 그
            # detach().mean()을 z0_raw로 쓰므로, 전부-0 텐서를 넘기면 z0_raw=0이 되고
            # z0_clamp_min<=0<=z0_clamp_max인 한 clamp 후에도 그대로 0 — mismatch pair용
            # U-Net 추가 forward(2회) 없이 같은 코드 경로를 그대로 재사용한다.
            mismatch_reward = torch.zeros(B, device=device)

        if self.weighting is not None:
            if self.weight_ref_timestep is not None:
                # 가중치 전용 고정 기준 timestep(2026-07-31 추가).
                # 문제: adaptive weight는 배치 안 샘플들의 오차를 서로 비교해 순위를 매기는데
                # (w_i = e_i/sum(e)), 샘플마다 t가 랜덤이라 오차 크기가 t에 따라 100배 넘게
                # 달라진다(실측: base MSE가 t=0-9에서 0.1485 vs t=90-99에서 0.0012). 그러면
                # 순위가 "이 샘플이 어려운가"가 아니라 "어떤 t를 뽑았는가"를 반영한다. APO
                # 원문은 토큰 디코딩이라 timestep 개념이 없어 이 문제가 없다 — diffusion 이식에서
                # 새로 생긴 구조적 문제(EXP-10.md 2026-07-31 절).
                # 해결: 학습 손실은 표준 DDPM대로 샘플별 랜덤 t를 그대로 쓰고(gradient 분산
                # 이점 유지), **가중치 계산만** 모든 샘플에 공통인 고정 t에서 별도 forward로
                # 구한다. 역전파 없음(가중치는 원래 detach 대상).
                with torch.no_grad():
                    t_ref = torch.full((B,), int(self.weight_ref_timestep), device=device, dtype=torch.long)
                    noisy_ref = scheduler.add_noise(action, noise, t_ref)
                    pred_ref = policy.unet(noisy_ref, t_ref, cond)
                    if self.action_error_source == "x0_l1":
                        abar_r = scheduler.alphas_cumprod.to(device)[t_ref].view(-1, 1, 1)
                        x0_r = ((noisy_ref - (1 - abar_r).sqrt() * pred_ref) / abar_r.sqrt()).clamp(-1.0, 1.0)
                        err_bt = (x0_r - action).abs().mean(dim=-1)
                    else:
                        err_bt = F.mse_loss(pred_ref, noise, reduction="none").mean(dim=-1)
                    error = (err_bt * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B,)
            elif self.action_error_source == "x0_l1":
                # APO 원문은 adaptive weight를 reward와 **다른 양**으로 계산한다: reward는
                # 토큰 log확률 비지만, weight는 토큰을 연속 액션으로 디코딩한 뒤의 L1 오차다
                # (`ddp_robotic_trainer.py:276-295`의 decode_token_ids_to_actions →
                # true/false_action_l1_loss). 논문 4.2절이 이유를 명시 — "to bridge the gap
                # between token classification and continuous action regression", 즉 reward에
                # 쓰는 양이 실제 액션 오차를 반영하지 못하기 때문이다[검증-원문, 2026-07-31].
                # diffusion에서 이에 대응하는 연산은 노이즈 예측으로부터 깨끗한 액션을 복원하는
                # 것: x0 = (x_t - sqrt(1-abar)*eps_pred)/sqrt(abar). 추가 forward 없이 이미
                # 계산된 값으로 구해진다.
                abar = scheduler.alphas_cumprod.to(device)[timesteps].view(-1, 1, 1)
                x0_pred = (noisy - (1 - abar).sqrt() * pred.detach()) / abar.sqrt()
                # 고-노이즈 timestep에서는 sqrt(abar)가 0에 가까워 위 복원이 폭주한다(스모크
                # 테스트 실측: clamp 없이 평균 오차 112 vs noise_mse의 1.19). 그러면 가중치가
                # "어떤 샘플이 어려운가"가 아니라 "어떤 timestep이 뽑혔나"에 좌우돼 무의미해진다.
                # 정책의 inference_scheduler가 clip_sample=True로 매 스텝 x0을 [-1,1]로 자르므로
                # (액션은 MinMax로 [-1,1] 정규화됨) 동일한 clamp를 적용해 범위를 맞춘다.
                x0_pred = x0_pred.clamp(-1.0, 1.0)
                x0_l1 = (x0_pred - action).abs().mean(dim=-1)  # (B, Tp)
                error = (x0_l1 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B,)
            else:
                error = (model_mse.detach() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B,)
            weight = self.weighting.compute_weights(action_mode, error=error)  # (B, Tp)
            self.last_action_error = float(error.mean().item())
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
            mismatch_reward_reject=mismatch_reward_reject, frame_mask=mask,
        )
        # 배치의 랜덤 diffusion timestep 분포(2026-08-08 추가) — grad_norm 스파이크가
        # reward/mse 집계 지표와는 무관하다는 게 실측으로 반증돼서(EXP-10.md 2026-08-07 밤
        # 절), 지금까지 한 번도 로깅 안 한 남은 후보(작은 t=고-SNR=가파른 loss landscape일
        # 수 있음)를 확인하기 위함.
        if getattr(self, "_diag_mismatch", None) is not None:
            metrics["raw_model_mse_mismatch"], metrics["raw_ref_mse_mismatch"] = self._diag_mismatch
        if getattr(self, "_diag_mismatch_by_donor", None) is not None:
            metrics["raw_model_mse_mismatch_desirable_donor"], metrics["raw_model_mse_mismatch_undesirable_donor"] = (
                self._diag_mismatch_by_donor
            )
        metrics["t_min"] = int(timesteps.min().item())
        metrics["t_mean"] = float(timesteps.float().mean().item())
        metrics["t_max"] = int(timesteps.max().item())

        des = _desirable_mask(action_mode, preference_frames=self.kto_loss.preference_frames)

        if self.bc_aux_weight > 0:
            per_sample_error = (model_mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B,)
            # mean(desirable 샘플 수 기준, 2026-08-07 수정) — kto_loss가 sum에서 mean으로
            # 바뀐 것과 같은 이유(apo_loss.py 해당 라인 주석 참고)로 일관성 있게 맞춤. 이전엔
            # sum이라 desirable 샘플 수가 많은 배치일수록 이 항의 영향력이 커지는 부작용도
            # 있었음 — mean이면 그 문제도 같이 없어짐.
            n_des = des.float().sum().clamp(min=1.0)
            bc_loss = (per_sample_error * des.float()).sum() / n_des
            total_loss = total_loss + self.bc_aux_weight * bc_loss
            metrics["bc_aux_loss"] = bc_loss.item()

        if self.hinge_weight > 0:
            # DPOP hinge(모듈 docstring 참고) — chosen이 reference보다 절대적으로 나빠질
            # 때만(model_mse > ref_mse) 벌점. bc_aux와 같은 (B,Tp)->per-sample->그룹평균
            # 패턴을 재사용하되, ref_mse도 같은 방식으로 마스킹·평균낸 뒤 차이를 clamp.
            # 적용 대상은 desirable 전체가 아니라 old data(=INTV가 없는 desirable)뿐이다
            # (2026-08-09). hinge는 "이미 하던 것을 잃지 마라"는 하한이므로, 새로 배워야 하는
            # INTV correction에 걸면 reference 수준이 학습의 하한이자 사실상 목표가 되어
            # Acquisition을 방해한다. Cell B(2026-08-08, desirable 전체 적용)와 다른 점.
            window = action_mode[:, : self.kto_loss.preference_frames]
            old_mask = (des & ~(window == LABEL_INTV).any(dim=1)).float()
            n_old = old_mask.sum().clamp(min=1.0)
            per_sample_model_err = (model_mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            per_sample_ref_err = (ref_mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            hinge_per_sample = (per_sample_model_err - per_sample_ref_err).clamp(min=0.0)
            hinge_loss = (hinge_per_sample * old_mask).sum() / n_old
            total_loss = total_loss + self.hinge_weight * hinge_loss
            metrics["hinge_loss"] = hinge_loss.item()
            metrics["hinge_num_old"] = float(n_old.item())
            metrics["hinge_active_frac"] = float(
                ((hinge_per_sample > 0).float() * old_mask).sum().item() / n_old.item()
            )

        if self.kl_retain_weight > 0:
            # FDPP(arXiv:2501.08259) Eq.19/20의 KL 정규화를 오프라인으로 이식.
            # 원문은 두 조건부의 진짜 가우시안 KL을 denoising step마다 합한다:
            #   D_KL[p_theta(A^{k-1}|A^k,S) || p_pre(A^{k-1}|A^k,S)]
            #        = ||mu_theta - mu_pre||^2 / (2 sigma_k^2)
            #        = c_k * ||eps_theta - eps_ref||^2,
            #   c_k = beta_k / (2 alpha_k (1 - alphabar_{k-1}))     [sigma_k^2 = betatilde_k]
            # k=1(t=0)에서는 DDPM 관례 sigma_1^2 = beta_1을 따라 분모를 (1-alphabar_1)로 둔다.
            # 원문과 다른 점(불가피): 원문의 S_t는 정책 롤아웃 상태, A^k는 생성 체인의 중간
            # 샘플이다(PPO 온라인). 우리는 오프라인이라 데이터 상태·데이터 액션에 노이즈를
            # 씌운 지점에서 잰다. design_deviations #22.
            # ref_pred는 no_grad라 상수 타깃 — gradient는 policy로만 흐른다.
            kl_per_frame = F.mse_loss(pred, ref_pred, reduction="none").mean(dim=-1)  # (B, Tp)
            kl_per_sample = (kl_per_frame * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            if self.kl_retain_ck:
                ac = scheduler.alphas_cumprod.to(device)
                a_t = scheduler.alphas.to(device)[timesteps]
                b_t = scheduler.betas.to(device)[timesteps]
                ab_prev = ac[(timesteps - 1).clamp(min=0)]
                denom = torch.where(timesteps > 0, 1.0 - ab_prev, 1.0 - ac[timesteps])
                c_k = b_t / (2.0 * a_t * denom.clamp(min=1e-8))
                kl_per_sample = kl_per_sample * c_k
                metrics["kl_ck_mean"] = float(c_k.mean().item())
                metrics["kl_ck_max"] = float(c_k.max().item())
            if self.kl_retain_scope == "old":
                window_kl = action_mode[:, : self.kto_loss.preference_frames]
                sel = (des & ~(window_kl == LABEL_INTV).any(dim=1)).float()
            else:
                sel = torch.ones_like(kl_per_sample)
            n_sel = sel.sum().clamp(min=1.0)
            kl_loss = (kl_per_sample * sel).sum() / n_sel
            total_loss = total_loss + self.kl_retain_weight * kl_loss
            metrics["kl_retain_loss"] = kl_loss.item()
            metrics["kl_retain_n"] = float(n_sel.item())

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
        hinge_weight=sys_cfg.get("hinge_weight", 0.0),
        kl_retain_weight=sys_cfg.get("kl_retain_weight", 0.0),
        kl_retain_scope=sys_cfg.get("kl_retain_scope", "all"),
        kl_retain_ck=sys_cfg.get("kl_retain_ck", True),
        reward_reduction=sys_cfg.get("reward_reduction", "sum"),
        z0_gripper_free=sys_cfg.get("z0_gripper_free", True),
        loss_frames=sys_cfg.get("loss_frames", "all"),
        diag_mismatch_every=sys_cfg.get("diag_mismatch_every", 0),
        z0_method=sys_cfg.get("z0_method", "match"),
        ars_enabled=sys_cfg.get("ars_enabled", False),
        ars_k1=sys_cfg.get("ars_k1", 1.0),
        ars_eps=sys_cfg.get("ars_eps", 1e-4),
        ars_ratio_clip=sys_cfg.get("ars_ratio_clip", 10.0),
        action_error_source=sys_cfg.get("action_error_source", "noise_mse"),
        weight_ref_timestep=sys_cfg.get("weight_ref_timestep", None),
        encoder_lr_scale=sys_cfg.get("encoder_lr_scale", 0.0),
        encoder_lora=sys_cfg.get("encoder_lora", False),
    )
