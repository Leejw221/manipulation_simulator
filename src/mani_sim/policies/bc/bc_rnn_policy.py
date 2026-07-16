"""BC-RNN — SIRIUS/robomimic 기준 기본 정책 (low-dim). Diffusion 없는 단순 회귀 BC.

robomimic 표준 BC-RNN은 매 스텝 하나씩 action을 내지만, 이 구현은 mani_sim의 나머지
인프라(RobomimicSequenceDataset·rollout.py의 receding-horizon 소비·RA-BC의 chunk 단위 진행도)가
전부 "obs_horizon 이력 → pred_horizon짜리 action chunk" 형태를 전제하므로, LSTM으로 이력을
인코딩한 뒤 한 번에 chunk를 회귀 출력한다(diffusion 대신 MSE). obs_keys·obs_dims·predict_action_chunk
인터페이스를 DiffusionPolicyLowDim과 동일하게 맞춰 rollout.py를 그대로 재사용한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCRNNPolicyLowDim(nn.Module):
    def __init__(
        self,
        obs_keys,
        obs_dims,
        obs_horizon,
        action_dim,
        pred_horizon,
        hidden_dim=1000,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        self.obs_keys = list(obs_keys)
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

        obs_dim = sum(obs_dims[k] for k in self.obs_keys)
        self.rnn = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, pred_horizon * action_dim)

    def _predict(self, obs):
        """obs: {key: (B,To,Dk)} → (B, Tp, Da) action chunk (정규화된 값 기준)."""
        x = torch.cat([obs[k] for k in self.obs_keys], dim=-1)  # (B, To, obs_dim)
        _out, (h_n, _c_n) = self.rnn(x)
        h_last = h_n[-1]  # (B, hidden_dim) 마지막 레이어의 최종 hidden state
        chunk = self.head(h_last)
        return chunk.view(-1, self.pred_horizon, self.action_dim)

    def compute_loss(self, batch, reduction="mean"):
        """batch: {'obs':.., 'action':(B,Tp,Da), 'action_mask':(B,Tp)}.

        reduction="mean" → 스칼라(DiffusionPolicyLowDim.compute_loss와 동일 관례).
        reduction="none" → 샘플별 스칼라 (B,) — RA-BC 등 외부 가중치 결합용.
        """
        pred = self._predict(batch["obs"])
        action = batch["action"]
        mask = batch["action_mask"].float()  # (B, Tp)

        per_step = F.mse_loss(pred, action, reduction="none").mean(dim=-1)  # (B, Tp)
        per_sample = (per_step * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B,)

        if reduction == "none":
            return per_sample
        return per_sample.mean()

    @torch.no_grad()
    def predict_action_chunk(self, obs):
        """DiffusionPolicyLowDim과 동일 인터페이스 — rollout.py 재사용."""
        return self._predict(obs)
