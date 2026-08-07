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
2. **정규화 부호 안전장치**: 원문은 `r_l+ε`로 나누는데, 이게 안전한 이유(왜 `r_l`이 음수가
   안 되는지)를 원문은 전혀 논하지 않는다(2026-08-06 WebFetch로 원문 재확인 — r_l의 부호에
   대한 서술 자체가 없음, "원문 도메인은 대체로 양수"라는 이전 기록은 근거 없는 추측이었음,
   정정). 오히려 논리적으로는 DPO/KTO가 rejected를 제대로 밀어낼수록 r_l은 음수로 수렴해야
   정상이고(실측: round1 -2.2~+0.66대(학습 초반, 부호 요동) → round2-new -18~-29대(학습
   후반, 확실히 음전)), ARS가 겨냥하는 "구별 어려운" 샘플(margin 작음)은 정의상 r_l이 앵커에
   가까운(=0 근처일 수 있는) 구간이라 원문 도메인에서도 같은 불안정성이 잠재했을 가능성이
   있다. 즉 이건 "우리 도메인이 원문과 달라서 이탈"이 아니라 **"원문이 다루지 않은 지점을
   우리가 실측(부호 요동)으로 발견해 방어적으로 절대값을 씀"**이 정확한 서술. `|r_l|+ε`로
   바꿔 이 문제를 피함.

최종: `ars_scale_i = sigmoid(K1 · margin_i.detach() / (|reward_rejected_i.detach()|+ε))`을
KTO의 (원문과 달리 이미 분리된) rejected sigmoid 항 바깥에 곱한다 — DPO의 단일 공유
sigmoid 구조가 아니라 KTO의 독립된 두 항 구조라 적용 위치도 원문과 다름. 원문의 γ(모드
전환) 항은 이번엔 구현 안 함.

**분모 불안정성 + ratio clip 안전장치(2026-08-05 추가, [적응])**: `|reward_rejected_i|`가 0에
가까운 샘플(=chosen과 가장 구별하기 어려운, 안전장치가 가장 필요한 바로 그 경우)에서는
margin의 부호와 무관하게 비율 `margin/(|r_l|+ε)`의 크기가 폭주한다. **주의**: 이 자체가
NaN/inf 같은 수치 오류를 일으키진 않는다 — `torch.sigmoid`는 fp32에서 큰 입력에도 안전하게
0.0/1.0으로 saturate하고(직접 검증, 2026-08-05), 이 코드베이스는 mixed-precision(autocast)도
안 씀. 실제 효과는: clamp 없이 ratio가 크게 튀면 `ars_scale`이 정확히 0.0 또는 1.0으로
반올림돼 그 샘플의 rejected loss 기여가 완전히 사라지거나(gradient 완전 차단) 전혀 감쇠
안 되는 극단이 됨 — clamp(±10, 직접 검증)을 걸면 `sigmoid(±10)≈0.999955/0.0000454`로 정확히
0/1이 아닌 잔여 신호가 남는다. 즉 이 fix는 **지금 관측된 활성 버그를 고친다기보단, gradient가
완전히 죽는 극단을 막고 K1 재조정·향후 mixed-precision 도입 등에 대비하는 방어적 장치**다.
PG-DPO 원문(arXiv:2511.19049 Eq.7/Eq.8)은 "ε는 수치 안정용 작은 상수"라고만 하고 이 지점을
다루지 않음([검증-원문], WebFetch로 확인). MADPO(arXiv:2510.05342)가 제안하는 piecewise
cap 방식을 따라, sigmoid 적용 전에 비율 자체를 `ars_ratio_clip`(기본 10.0)으로 clamp한다:
`ars_ratio = clamp(K1·margin/(|r_l|+ε), -ars_ratio_clip, ars_ratio_clip)`. round2에서 반복
관측된 foreign-policy 개입 데이터(reference와 크게 어긋난 데이터)의 "confusable minority"
샘플이 이 구간에 해당할 수 있다는 게 계기 — 이런 데이터가 앞으로도 반복될 것이므로 근본적으로
학습 방식 자체가 견고해야 한다는 사용자 판단([사용자 지시], 2026-08-05).

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
        ars_ratio_clip=10.0,
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
        self.ars_ratio_clip = ars_ratio_clip

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
            ars_ratio = self.ars_k1 * margin.detach() / (rejected_reward.detach().abs() + self.ars_eps)
            ars_ratio = ars_ratio.clamp(min=-self.ars_ratio_clip, max=self.ars_ratio_clip)
            ars_scale = torch.sigmoid(ars_ratio)
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
        # sum이 아니라 mean(2026-08-07 수정) — Diffusion-KTO 공식 코드(`loss = l_kto.mean()`,
        # train_kto_sd_v1.5.py, 2026-08-07 WebFetch로 재확인)는 mean을 쓰는데 우리는 APO
        # 원문 관행(sum, `APO_analysis.md` "sum vs mean: 논문 E(mean), 코드 sum → LR로 보정")을
        # 따라가고 있었음 — mean은 batch size에 무관하지만 sum은 batch size에 비례해서
        # gradient가 커진다. 우리 batch_size=64는 두 원문(KTO=4, APO=8)보다 훨씬 큰데 LR
        # 재조정은 reward 스케일 차이만 반영했지 이 batch size 차이는 반영한 적이 없었음 —
        # 우리 diffusion 다리 자체가 APO가 아니라 Diffusion-KTO를 따라 만든 것이므로, 두
        # 원문이 갈리는 지점에선 Diffusion-KTO 쪽을 따르는 게 더 일관됨. grad_norm이
        # 비정상적으로 컸던(수만 단위) 현상의 유력한 원인 후보.
        total_loss = (losses * weights).mean()

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
