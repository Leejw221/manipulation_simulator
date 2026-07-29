"""시스템 축 — sirius (reference 모델 없는 단순 가중 손실).

per-step loss에 가중치를 곱하고 평균낸다. 배치별 가중치 정규화(호출부에서 수행,
근거: EXP-10.md "SIRIUS loss 정규화" 절)와 결합하면 moai_policy `WeightedDiffusionPolicy`
와 동치.

weight 축(class_based/action_error) 어느 쪽을 꽂아도 이 함수는 그대로 — "시스템"과
"가중치"가 독립 축이라는 게 이 분리의 요점.
"""


def sirius_loss(per_step_loss, weight):
    """per_step_loss, weight: 둘 다 (B, T). 반환: scalar."""
    return (per_step_loss * weight).mean()
