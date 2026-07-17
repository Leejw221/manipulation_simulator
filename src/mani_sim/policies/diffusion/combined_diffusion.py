"""Diffusion Policy(LowDim/Image 공통) 위에서 system×weight 두 축을 조합.

`outputs/train.py`(`--policy bc`)의 `CombinedBCRNNGMM`과 정확히 대응하는 diffusion판.
다리(diffusion엔 log_prob이 intractable → denoising MSE 차이가 그 자리)는
`papers/Diffusion-KTO_analysis.md` 참고(원문+코드 검증 기록).

system=sirius: reference 모델 없이 그냥 `sirius_loss(-model_mse, weight)`.
system=apo: dual-forward(+mismatch)로 reward/z0를 만들고, 가중치는 weighting/*.py
  (class_based 또는 action_error)가 계산해서 넘겨받는다 — 어느 쪽이든 이 클래스는 그대로.
"""

import copy

import torch
import torch.nn.functional as F

from mani_sim.losses.apo_loss import APOKTOLoss
from mani_sim.losses.sirius_loss import sirius_loss


class DiffusionCombined(torch.nn.Module):
    def __init__(self, policy, system, weight_module, apo_loss=None, init_state_dict=None):
        """policy: DiffusionPolicyLowDim | DiffusionPolicyImage.
        system: "sirius" | "apo". weight_module: ClassBasedWeight | ActionErrorWeight.
        apo_loss: APOKTOLoss(system="apo"일 때만 필요)."""
        super().__init__()
        assert system in ("sirius", "apo")
        if system == "apo":
            assert apo_loss is not None

        if init_state_dict is not None:
            policy.load_state_dict(init_state_dict)
        self.policy = policy
        self.system = system
        self.weight_module = weight_module
        self.apo_loss = apo_loss

        if system == "apo":
            reference = copy.deepcopy(policy)
            for p in reference.parameters():
                p.requires_grad_(False)
            self.reference = reference
        else:
            self.reference = None

    def _step_mse(self, action_target, cond, ref_cond, t, scheduler):
        noise = torch.randn_like(action_target)
        noisy = scheduler.add_noise(action_target, noise, t)
        pred = self.policy.unet(noisy, t, cond)
        model_mse = F.mse_loss(pred, noise, reduction="none").mean(dim=-1)  # (B, Tp)
        if self.reference is None:
            return model_mse, None
        with torch.no_grad():
            ref_pred = self.reference.unet(noisy, t, ref_cond)
        ref_mse = F.mse_loss(ref_pred, noise, reduction="none").mean(dim=-1)
        return model_mse, ref_mse

    def compute_loss(self, batch):
        """batch: {'obs':..., 'action':(B,Tp,Da), 'action_mask':(B,Tp), 'action_mode':(B,Tp)}.
        반환: (scalar loss, dict 진단 지표(system=apo일 때만 비어있지 않음))."""
        obs, action = batch["obs"], batch["action"]
        action_mode = batch["action_mode"]
        device = action.device
        B = action.shape[0]

        cond = self.policy.get_global_cond(obs)
        scheduler = self.policy.train_scheduler
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()

        ref_cond = None
        if self.reference is not None:
            with torch.no_grad():
                ref_cond = self.reference.get_global_cond(obs)

        model_mse, ref_mse = self._step_mse(action, cond, ref_cond, timesteps, scheduler)

        if self.system == "sirius":
            weight = self.weight_module.compute_weights(action_mode=action_mode, error=model_mse.detach().mean(dim=1))
            loss = sirius_loss(model_mse, weight)
            return loss, {}

        # system == "apo": mismatch까지 계산 + 외부 weight 모듈
        shift = torch.roll(torch.arange(B, device=device), shifts=-1)
        mismatch_action = action[shift]
        mismatch_model_mse, mismatch_ref_mse = self._step_mse(mismatch_action, cond, ref_cond, timesteps, scheduler)

        error = model_mse.detach().mean(dim=1)  # (B,) — action_error 축이면 이걸 씀, class_based면 무시
        weight = self.weight_module.compute_weights(action_mode=action_mode, error=error)

        loss, metrics = self.apo_loss.compute(
            log_probs=-model_mse,
            ref_log_probs=-ref_mse,
            mismatch_log_probs=-mismatch_model_mse,
            ref_mismatch_log_probs=-mismatch_ref_mse,
            weight=weight,
            action_mode_window=action_mode,
        )
        return loss, metrics
