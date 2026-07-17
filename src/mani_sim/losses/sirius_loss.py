"""시스템 축 — sirius (reference 모델 없는 단순 가중 손실).

원문 대조: UT-Austin-RPL/sirius robomimic/algo/bc.py BC_RNN_GMM._compute_losses()
(L662-664: `action_loss *= weights; action_loss.mean()`). [검증-원문, 2026-07-16]

reference 모델·KTO sigmoid·z0 앵커 전부 없음 — per-step loss(NLL 또는 MSE)에 가중치를
곱하고 그냥 평균낸다. 정규화(가중치 합=1 맞추기)도 없음(SIRIUS 원문과 동일, RA-BC의
`sum(w)/batch_size` 정규화와 다름).

weight 축(class_based/action_error) 어느 쪽을 꽂아도 이 함수는 그대로 — "시스템"과
"가중치"가 독립 축이라는 게 이 분리의 요점.
"""


def sirius_loss(per_step_loss, weight):
    """per_step_loss, weight: 둘 다 (B, T). 반환: scalar."""
    return (per_step_loss * weight).mean()
