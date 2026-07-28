"""Policy/Runner registry — moai_policy(flare)의 factory.py 패턴을 mani_sim 규모에 맞게 적용.

flare와의 차이: policy 생성자가 서로 이질적(DiffusionPolicyImage=obs_keys/dims,
OpenVLAPolicy=lora_rank 등 vla 하이퍼) — 클래스를 직접 등록하는 대신 **빌더 함수**를 등록해
`create_policy(name, task_cfg, policy_cfg)`라는 균일한 진입점만 factory가 보장한다
(각 정책은 자기 생성자 형태를 유지). runner는 생성자 형태가 충분히 균일해(config/policy/device/
dataloader) 클래스를 그대로 등록한다.

사용:
    @registry.register_policy("diffusion_image")
    def _build(task_cfg, policy_cfg): ...

    @registry.register_runner("diffusion_trainer")
    class DiffusionTrainer: ...

    policy = registry.create_policy("diffusion_image", task_cfg, policy_cfg)
    runner_cls = registry.get_runner_class("diffusion_trainer")
"""

import logging

logger = logging.getLogger(__name__)


class Registry:
    def __init__(self):
        self._policy_builders = {}
        self._runners = {}
        self._system_builders = {}

    def register_policy(self, name):
        def _reg(builder_fn):
            self._policy_builders[name] = builder_fn
            return builder_fn
        return _reg

    def register_runner(self, name):
        def _reg(cls):
            self._runners[name] = cls
            return cls
        return _reg

    def register_system(self, name):
        """system 축(sirius|apo) — apo만 실제 빌더가 필요(reference 모델+KTO). sirius는
        기존 weighted-BC 경로(diffusion_trainer.py의 weighting-only 분기) 그대로라 등록 대상 아님."""
        def _reg(builder_fn):
            self._system_builders[name] = builder_fn
            return builder_fn
        return _reg

    def get_policy_builder(self, name):
        if name not in self._policy_builders:
            raise ValueError(f"Unknown policy: {name!r} (registered: {sorted(self._policy_builders)})")
        return self._policy_builders[name]

    def get_runner_class(self, name):
        if name not in self._runners:
            raise ValueError(f"Unknown runner: {name!r} (registered: {sorted(self._runners)})")
        return self._runners[name]

    def get_system_builder(self, name):
        if name not in self._system_builders:
            raise ValueError(f"Unknown system: {name!r} (registered: {sorted(self._system_builders)})")
        return self._system_builders[name]

    def create_policy(self, name, task_cfg, policy_cfg):
        return self.get_policy_builder(name)(task_cfg, policy_cfg)

    def create_system(self, name, cfg, policy, weighting, device, init_state_dict):
        return self.get_system_builder(name)(cfg, policy, weighting, device, init_state_dict)

    def list_policies(self):
        return sorted(self._policy_builders)

    def list_runners(self):
        return sorted(self._runners)

    def list_systems(self):
        return sorted(self._system_builders)


registry = Registry()

# 등록 부수효과를 위해 import(각 모듈이 자기 자신을 @registry.register_*로 등록).
# openvla는 transformers/peft가 선택적 의존성이라 실패해도 나머지가 죽지 않게 개별 try.
from mani_sim.policies.diffusion import diffusion_policy_registrations  # noqa: E402,F401
from mani_sim.policies.diffusion import apo_system  # noqa: E402,F401
from mani_sim.policies.bc import bc_registrations  # noqa: E402,F401
from mani_sim.runners import diffusion_trainer  # noqa: E402,F401

try:
    from mani_sim.policies.openvla import openvla_registrations  # noqa: F401
    from mani_sim.runners import openvla_trainer  # noqa: F401
except ImportError as e:
    logger.info(f"OpenVLA policy/runner 미등록(선택적 의존성 없음): {e}")
