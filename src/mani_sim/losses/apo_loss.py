"""시스템 축 — apo (reference 모델 + KTO 앵커 필요).

원문 대조: GeWu-Lab/Action-Preference-Optimization trainer/ddp_robotic_trainer.py
APOTrainer.loss()/get_batch_metrics() (L173-364) [검증-원문, 2026-07-16]. z0(anchor) 추정
방식은 apo_system.py의 z0_method로 토글(match=배치 reward 평균 | mismatch=교차매칭) — 근거는
apo_system.py 모듈 docstring 및 EXP-10.md 2026-07-30 절 참고. clamp 범위는 APO 원문 방식
(대칭, 음수 허용) — 상세 근거는 EXP-10.md "z0 clamp" 절 참고.

undesirable(reject) reward는 chosen/z0와 다른 입력(log_probs_reject/ref_log_probs_reject,
gripper 제외)을 받을 수 있다 — APO Appendix C의 gripper 마스킹, apo_system.py에서 계산해
넘겨준다.

**ARS(Adaptive Rejection Scaling, 2026-07-30 추가, [적응])**: arXiv:2511.19049("Beyond Reward
Margin: Rethinking and Resolving Likelihood Displacement in Diffusion Models", ICLR 2026)가
보고하는 현상 — DPO류 loss가 rejected를 밀어낼 때, chosen과 NTK 유사도가 높은(=구별하기
어려운) rejected 샘플일수록 그 gradient가 chosen 쪽으로 "번져서"(spillover) chosen 품질
자체를 깎아먹는다(그 논문 자체는 diffusion 전용 현상이 아니라 LLM DPO에서도 보고된 일반
현상이라고 명시함).

원문 수식(PG-DPO, Eq.7, [검증-원문 2026-07-30 WebFetch로 직접 확인]):
`α(x_w,x_l) = σ[K1·(r_w-r_l)/(r_l+ε)]` — **짝지어진(paired)** chosen reward r_w와 rejected
reward r_l의 차이를, r_l 크기로 정규화한 비율에 sigmoid. Eq.9의 DPO 단일 sigmoid 안에서
rejected 항에 곱해짐.

**우리 KTO(unpaired)로의 이식, 두 가지 필연적 이탈 — 둘 다 원문을 그대로 못 쓰는 이유가
분명함(문헌 검색 확인, 2026-07-30) — KTO(unpaired) + 이 안전장치를 결합한 선행 사례
자체가 없음(ELBO-KTO는 diffusion LLM 텍스트 전용이고 이 안전장치를 안 다룸, ACPO도
DPO/paired 기반)**:
1. **r_w 대체**: pair가 없어 "이 rejected에 대응하는 chosen"이 없다 — 배치 앵커 z0를
   대신 씀: `margin_i = z0 - reward_rejected_i`.
2. **정규화 부호 안전장치**: 원문은 `r_l+ε`로 나누는데(원문 도메인은 r_l이 대체로 양수인
   듯), 우리 reward_rejected는 부호가 자주 바뀐다(round1 실측: -2.2~+0.66대) — `r_l+ε`
   그대로 쓰면 0 근처에서 비율이 튀거나 부호가 뒤집힌다. `|r_l|+ε`(절대값)로 바꿔 이 문제를
   피함 — 이 지점만 원문과 다른 우리 쪽 안전장치.

최종: `ars_scale_i = sigmoid(K1 · margin_i.detach() / (|reward_rejected_i.detach()|+ε))`을
KTO의 (원문과 달리 이미 분리된) rejected sigmoid 항 바깥에 곱한다 — DPO의 단일 공유
sigmoid 구조가 아니라 KTO의 독립된 두 항 구조라 적용 위치도 원문과 다름. 원문의 γ(모드
전환) 항은 이번엔 구현 안 함.

**동기**: 우리 undesirable(PREINTV) 샘플은 정의상 desirable(DEMO/ROLLOUT) 샘플 바로 직전
프레임이라 시각적·운동학적으로 가장 가깝다 — 이 논문이 말하는 "spillover 최악 조건"에
가깝다는 게 round1 mismatch 붕괴(사각너트 아닌 곳으로 접근) 관측 이후 유력 가설로 지목됨.

SIRIUS와의 결정적 차이: **reference 모델(정책의 이전 상태, 고정)이 필요**하다 — reward를
"GT action을 policy가 reference보다 얼마나 더 그럴듯하다고 보는가"(log-ratio)로 정의하고,
z0를 기준점 삼아 KTO의 sigmoid 손실을 쓴다.

**가중치는 이 클래스가 계산하지 않는다** — 외부(weighting/class_based.py 또는
weighting/action_error.py)에서 계산한 (B,T) weight를 받아 곱하기만 한다. 원문 APO는
adaptive weight를 자체 내장하지만(=action_error 축), 우리는 이걸 분리해서 class_based
가중치를 KTO 구조에 꽂는 조합도 실험할 수 있게 했다(2026-07-16(4) 사용자 지시 — 세 축
독립 조합).

preference(chosen/rejected) 판정은 `datasets/labels.desirable_mask`로 weighting/action_error.py
와 동일 기준 공유.

**beta는 확정 전 기본값**(Diffusion-KTO 공식값 그대로) — APO(0.1)는 토큰 log확률 합
스케일용이라 우리(MSE 차이 스케일)엔 안 맞는다. 배포 데이터로 실측 후
`|beta*(reward-z0)|`의 중앙값이 0.5~2에 오도록 재조정할 것.
"""

from collections import OrderedDict

import torch

from mani_sim.datasets.labels import desirable_mask as _desirable_mask


class APOKTOLoss:
    def __init__(
        self,
        beta=1000.0,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        preference_frames=8,
        z0_clamp_min=-50.0,
        z0_clamp_max=50.0,
        ars_enabled=False,
        ars_k1=1.0,
        ars_eps=1e-4,
    ):
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.preference_frames = preference_frames
        self.z0_clamp_min = z0_clamp_min
        self.z0_clamp_max = z0_clamp_max
        self.ars_enabled = ars_enabled
        self.ars_k1 = ars_k1
        self.ars_eps = ars_eps

    def compute(self, log_probs, ref_log_probs, weight, action_mode_window, mismatch_reward=None,
                log_probs_reject=None, ref_log_probs_reject=None):
        """log_probs, ref_log_probs, weight: (B, T) — 패딩 프레임은 호출부가 미리 0으로
        마스킹해서 넘긴다(합산에서 자동 제외). action_mode_window: (B, W), W>=preference_frames.
        mismatch_reward: (B,) 또는 None — z0_method="mismatch"일 때 호출부(apo_system.py)가
        계산해서 넘김. 주어지면 z0를 여기서 추정, None이면 z0_method="match"로 배치 reward
        평균을 씀(diffusion-kto 공식 코드 기본값과 동일 — 근거는 apo_system.py 참고, 더 이상
        버그가 아니라 명시적으로 선택 가능한 옵션이다, 2026-07-30).
        log_probs_reject/ref_log_probs_reject: (B, T) 또는 None — undesirable(reject) reward
        전용 입력(APO Appendix C, gripper 제외). None이면 log_probs/ref_log_probs를 그대로 씀.

        반환: (scalar loss, dict 진단 지표).
        """
        reward = log_probs.sum(dim=1) - ref_log_probs.sum(dim=1)  # (B,)
        if log_probs_reject is not None:
            reward_reject = log_probs_reject.sum(dim=1) - ref_log_probs_reject.sum(dim=1)
        else:
            reward_reject = reward
        z0_source = mismatch_reward if mismatch_reward is not None else reward
        # 대칭 clamp(APO 원문 방식) — 근거: EXP-10.md "z0 clamp" 절.
        z0 = z0_source.detach().mean().clamp(min=self.z0_clamp_min, max=self.z0_clamp_max)

        mask = _desirable_mask(action_mode_window, preference_frames=self.preference_frames)
        sample_weight = weight.mean(dim=1)  # (B,T) -> (B,) — class_based처럼 T별로 다르면 평균

        n_chosen = int(mask.sum().item())
        n_rejected = int((~mask).sum().item())

        chosen_reward = reward[mask]
        margin_chosen = chosen_reward - z0  # 클수록(양수) chosen이 z0보다 명확히 나음=안전
        chosen_util = torch.sigmoid(self.beta * margin_chosen)  # KTO/Diffusion-KTO의 utility 항 그 자체
        chosen_losses = self.desirable_weight * (1 - chosen_util)
        chosen_weight_raw = sample_weight[mask]
        chosen_weight_sum = float(chosen_weight_raw.sum().item()) if n_chosen else 0.0
        # 그룹 내 평균을 1.0으로 재정규화 — action_error weight의 그룹합이 exp(-x) 곡률 때문에
        # 그룹 크기에 비대칭적으로 반응해(desirable은 beta_d 근처로 포화, undesirable은 n에
        # 비례) 의도치 않게 그룹 간 loss 기여도를 왜곡시킨다(EXP-10.md "7차 시도" 절 참고).
        # 재정규화 후엔 그룹별 총 기여도가 desirable_weight/undesirable_weight·n_chosen/
        # n_rejected로만 결정된다.
        chosen_weight = chosen_weight_raw / max(chosen_weight_sum, 1e-8) * n_chosen if n_chosen else chosen_weight_raw

        rejected_reward = reward_reject[~mask]
        margin_rejected = z0 - rejected_reward  # 클수록(양수) 그 rejected 샘플이 z0보다 명확히 나쁨=안전
        margin = margin_rejected  # 기존 이름(ARS 계산에서 그대로 사용) 유지
        rejected_util = torch.sigmoid(self.beta * margin_rejected)  # ars 적용 전 utility 항
        if self.ars_enabled:
            # PG-DPO 원문 Eq.7: sigmoid(K1·(r_w-r_l)/(r_l+eps)) — r_w를 z0로 대체(unpaired라
            # 페어가 없음), 분모는 |r_l|+eps로 절대값(원문은 r_l+eps인데 우리 reward는 부호가
            # 자주 바뀌어서 그대로 쓰면 0 근처에서 비율이 튀거나 부호가 뒤집힘 — 모듈
            # docstring 참고). stop-gradient(순수 스케일 팩터, z0/margin/분모 전부 학습
            # 신호로 안 씀) — chosen과 구별하기 어려운(margin 작음/음수) 샘플일수록 밀어내는
            # 힘을 줄인다.
            ars_scale = torch.sigmoid(
                self.ars_k1 * margin.detach() / (rejected_reward.detach().abs() + self.ars_eps)
            )
        else:
            ars_scale = torch.ones_like(rejected_reward)
        rejected_losses = self.undesirable_weight * ars_scale * (1 - torch.sigmoid(self.beta * margin))
        rejected_weight_raw = sample_weight[~mask]
        rejected_weight_sum = float(rejected_weight_raw.sum().item()) if n_rejected else 0.0
        rejected_weight = (
            rejected_weight_raw / max(rejected_weight_sum, 1e-8) * n_rejected if n_rejected else rejected_weight_raw
        )

        losses = torch.cat([chosen_losses, rejected_losses])
        weights = torch.cat([chosen_weight, rejected_weight])
        total_loss = (losses * weights).sum()

        # 그룹별 실제 loss 기여도(weight까지 반영된 후) — desirable_weight/undesirable_weight·
        # n_chosen/n_rejected로 "이론상" 비율을 계산했던 것(EXP-10.md 2026-07-30 정정 절)을
        # 매 스텝 실측으로 직접 검증할 수 있게 로깅한다. chosen_contrib+rejected_contrib=total_loss.
        chosen_contrib = float((chosen_losses.detach() * chosen_weight.detach()).sum().item()) if n_chosen else 0.0
        rejected_contrib = (
            float((rejected_losses.detach() * rejected_weight.detach()).sum().item()) if n_rejected else 0.0
        )

        metrics = OrderedDict(
            num_chosen=n_chosen,
            num_rejected=n_rejected,
            reward_chosen_mean=float(chosen_reward.detach().mean().item()) if n_chosen else float("nan"),
            reward_rejected_mean=float(rejected_reward.detach().mean().item()) if n_rejected else float("nan"),
            z0=float(z0.item()),
            reward_min=float(reward.detach().min().item()),
            reward_max=float(reward.detach().max().item()),
            reward_std=float(reward.detach().std().item()) if reward.numel() > 1 else 0.0,
            raw_weight_sum_chosen=chosen_weight_sum,
            raw_weight_sum_rejected=rejected_weight_sum,
            ars_scale_mean=float(ars_scale.detach().mean().item()) if n_rejected else float("nan"),
            # utility 항(KTO/Diffusion-KTO Eq.8의 U) 자체 — reward 값이 아니라 "loss가 실제로
            # 얼마나 만족됐는가"를 직접 보여준다. 1에 가까울수록 그 그룹의 loss가 만족(=낮음),
            # 0에 가까울수록 불만족(=높음). mean(sigmoid(x)) != sigmoid(mean(x))이라 reward_mean/z0로부터
            # 역산 불가 — 반드시 per-sample로 따로 평균내야 한다.
            margin_chosen_mean=float(margin_chosen.detach().mean().item()) if n_chosen else float("nan"),
            margin_rejected_mean=float(margin_rejected.detach().mean().item()) if n_rejected else float("nan"),
            chosen_util_mean=float(chosen_util.detach().mean().item()) if n_chosen else float("nan"),
            rejected_util_mean=float(rejected_util.detach().mean().item()) if n_rejected else float("nan"),
            chosen_loss_contrib=chosen_contrib,
            rejected_loss_contrib=rejected_contrib,
        )
        return total_loss, metrics
