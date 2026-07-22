"""SARM(arXiv:2509.25358) StageTransformer/SubtaskTransformer — xdofai/opensarm(저자 공식 repo,
models/stage_estimator.py·subtask_estimator.py) 원문 그대로 포팅[검증-원문, 2026-07-22].
lerobot 포크(huggingface/lerobot src/lerobot/policies/sarm)도 대조했으나 구조가 완전히
동일해서(teacher-forcing 비율만 0.75로 다름) 저자 공식 repo 쪽(50/50)을 따른다.

dense(세분화 annotation) 브랜치는 사용자 지시로 제외 — sparse만 구현. 두 논문(SARM2 등) 통합
버전이 아니라 SARM 단독 아키텍처만 포팅했다.
"""

import torch
import torch.nn as nn


class StageTransformer(nn.Module):
    """시퀀스 윈도우(T프레임) -> stage 분류 logits(B,T,C). causal transformer."""

    def __init__(self, d_model=512, vis_emb_dim=512, text_emb_dim=512, state_dim=18,
                 n_layers=6, n_heads=8, dropout=0.1, num_cameras=1, num_classes_sparse=5):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        fused_in = d_model * (num_cameras + 2)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in), nn.Linear(fused_in, d_model), nn.ReLU(),
        )
        self.head = nn.Linear(d_model, num_classes_sparse)

    def _prep_lang(self, lang_emb, B, T, D):
        if lang_emb.dim() == 3:
            return self.lang_proj(lang_emb).unsqueeze(1)
        return self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)

    def forward(self, img_seq, lang_emb, state, lengths):
        """img_seq: (B,N,T,vis_emb_dim). lang_emb: (B,E) or (B,T,E). state: (B,T,state_dim).
        lengths: (B,). -> stage logits (B,T,num_classes_sparse)."""
        B, N, T, _ = img_seq.shape
        D = self.d_model
        device = img_seq.device

        vis_proj = self.visual_proj(img_seq)
        state_proj = self.state_proj(state).unsqueeze(1)
        lang_proj = self._prep_lang(lang_emb, B, T, D)

        x = torch.cat([vis_proj, lang_proj, state_proj], dim=1)
        x[:, :N, 0, :] = x[:, :N, 0, :] + self.first_pos

        x_tokens = x.view(B, (N + 2) * T, D)
        L = x_tokens.size(1)
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        mask = base_mask.unsqueeze(1).expand(B, N + 2, T).reshape(B, (N + 2) * T)
        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

        h = self.transformer(x_tokens, mask=causal_mask, src_key_padding_mask=mask, is_causal=True)
        h = h.view(B, N + 2, T, D).permute(0, 2, 1, 3).reshape(B, T, (N + 2) * D)
        fused = self.fusion_backbone(h)
        return self.head(fused)


class SubtaskTransformer(nn.Module):
    """시퀀스 윈도우 + stage prior(one-hot) -> tau 회귀(B,T) in [0,1]. causal transformer."""

    def __init__(self, d_model=512, vis_emb_dim=512, text_emb_dim=512, state_dim=18,
                 n_layers=6, n_heads=8, dropout=0.1, num_cameras=1):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)

        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        fused_in = d_model * (num_cameras + 3)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in), nn.Linear(fused_in, d_model), nn.ReLU(),
        )
        self.head = nn.Linear(d_model, 1)

    def _prep_lang(self, lang_emb, B, T, D):
        if lang_emb.dim() == 3:
            return self.lang_proj(lang_emb).unsqueeze(1)
        return self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)

    def _stage_to_dmodel(self, stage_prior):
        B, one, T, C = stage_prior.shape
        D = self.d_model
        if D == C:
            return stage_prior
        if D > C:
            pad = torch.zeros(B, one, T, D - C, device=stage_prior.device, dtype=stage_prior.dtype)
            return torch.cat([stage_prior, pad], dim=-1)
        return stage_prior[..., :D]

    def forward(self, img_seq, lang_emb, state, lengths, stage_prior):
        """stage_prior: (B,1,T,C) one-hot. -> tau (B,T) in [0,1]."""
        B, N, T, _ = img_seq.shape
        D = self.d_model
        device = img_seq.device

        vis_proj = self.visual_proj(img_seq)
        state_proj = self.state_proj(state).unsqueeze(1)
        lang_proj = self._prep_lang(lang_emb, B, T, D)
        stage_emb = self._stage_to_dmodel(stage_prior)

        x = torch.cat([vis_proj, lang_proj, state_proj, stage_emb], dim=1)
        x[:, :N, 0, :] = x[:, :N, 0, :] + self.first_pos

        x_tokens = x.view(B, (N + 3) * T, D)
        L = x_tokens.size(1)
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        mask = base_mask.unsqueeze(1).expand(B, N + 3, T).reshape(B, (N + 3) * T)
        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

        h = self.transformer(x_tokens, mask=causal_mask, src_key_padding_mask=mask, is_causal=True)
        h = h.view(B, N + 3, T, D).permute(0, 2, 1, 3).reshape(B, T, (N + 3) * D)
        fused = self.fusion_backbone(h)
        return torch.sigmoid(self.head(fused)).squeeze(-1)


def gen_stage_emb(num_classes, targets):
    """targets(B,T) float(stage.tau) -> one-hot(B,1,T,num_classes)(정수부=stage)."""
    idx = targets.long().clamp(min=0, max=num_classes - 1)
    onehot = torch.eye(num_classes, device=targets.device)[idx]
    return onehot.unsqueeze(1)


class SARMFullModel(nn.Module):
    """stage_model+subtask_model 컨테이너 — 체크포인트 하나로 저장/로드하기 위한 래퍼
    (utils/checkpoints.py의 `<model>.state_dict()` 컨벤션을 그대로 쓰기 위함, 학습 루프
    자체는 train_sarm.py에서 두 모델을 독립 optimizer로 따로 학습시킨다[원문 그대로]).

    alpha_bar(SARM Eq.1, 데이터셋 전체 stage별 평균 소요비율)를 buffer로 같이 저장 —
    predict_progress()가 Eq.3-4(P̂_{k-1}+ᾱ_k·τ̂)로 최종 global progress를 합성할 때
    학습 시점과 같은 값을 쓰도록 보장(따로 파일 경로로 넘기다 어긋나는 걸 방지)."""

    def __init__(self, stage_model, subtask_model, alpha_bar=None):
        super().__init__()
        self.stage_model = stage_model
        self.subtask_model = subtask_model
        num_classes = stage_model.head.out_features
        if alpha_bar is None:
            alpha_bar = torch.full((num_classes,), 1.0 / num_classes)
        else:
            alpha_bar = torch.as_tensor(alpha_bar, dtype=torch.float32)
        self.register_buffer("alpha_bar", alpha_bar)

    @torch.no_grad()
    def predict_progress(self, img_seq, lang_emb, state, lengths):
        """추론 시엔 GT가 없으므로 stage_pred의 argmax를 subtask_model 조건으로 씀
        (xdofai sarm_ws.py의 eval 경로와 동일). SARM Eq.3-4로 합성: P̂_{k-1}+ᾱ_k·τ̂.
        -> progress(B,T) float[0,1)."""
        num_classes = self.stage_model.head.out_features
        stage_logits = self.stage_model(img_seq, lang_emb, state, lengths)
        stage_idx = stage_logits.argmax(dim=-1)
        stage_onehot = torch.eye(num_classes, device=stage_idx.device)[stage_idx].unsqueeze(1)
        tau = self.subtask_model(img_seq, lang_emb, state, lengths, stage_onehot)
        cum = torch.cat([stage_idx.new_zeros(1, dtype=self.alpha_bar.dtype), torch.cumsum(self.alpha_bar, dim=0)])
        return cum[stage_idx] + self.alpha_bar[stage_idx] * tau
