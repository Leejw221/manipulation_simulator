"""bc_rnn_policy.py 를 factory registry에 등록."""

from mani_sim.factory import registry
from mani_sim.policies.bc.bc_rnn_policy import BCRNNPolicyLowDim


@registry.register_policy("bc_rnn_lowdim")
def _build_bc_rnn_lowdim(task_cfg, policy_cfg):
    return BCRNNPolicyLowDim(
        obs_keys=task_cfg.obs_keys,
        obs_dims=task_cfg.obs_dims,
        obs_horizon=policy_cfg.obs_horizon,
        action_dim=task_cfg.action_dim,
        pred_horizon=policy_cfg.pred_horizon,
        hidden_dim=policy_cfg.get("hidden_dim", 1000),
        num_layers=policy_cfg.get("num_layers", 2),
        dropout=policy_cfg.get("dropout", 0.0),
    )
