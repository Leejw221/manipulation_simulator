"""openvla_policy.py 를 factory registry에 등록. transformers/peft 선택적 의존성이라
factory.py가 이 모듈 import를 try/except로 감싼다(미설치 시 나머지 정책은 정상 동작)."""

from mani_sim.factory import registry
from mani_sim.policies.openvla.openvla_policy import OpenVLAPolicy


@registry.register_policy("openvla")
def _build_openvla(task_cfg, policy_cfg):
    return OpenVLAPolicy(
        model_id=policy_cfg.get("model_id", "openvla/openvla-7b"),
        lora_rank=policy_cfg.get("lora_rank", 32),
        lora_alpha=policy_cfg.get("lora_alpha", 16),
        action_dim=task_cfg.action_dim,
        load_in_4bit=policy_cfg.get("load_in_4bit", False),
        policy_adapter_path=policy_cfg.get("policy_adapter_path", None),
    )
