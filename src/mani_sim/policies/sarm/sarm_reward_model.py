"""SARM(arXiv:2509.25358) reward model — 경량 버전(언어 조건화 없음, datasets/sarm_dataset.py
docstring 참고: task가 Transport 하나뿐이라 텍스트로 task를 구분할 필요가 없음[사용자 판단]).
단, **비전 인코더는 CLIP(고정, 사전학습 그대로)을 씀** — reward model이 하는 일(어느
stage인지·얼마나 진행됐는지 같은 semantic 판단)은 policy의 정밀 모터제어와 달리 CLIP류
대규모 사전학습 semantic feature가 유리할 것으로 판단(2026-07-19 사용자 지적: "CLIP으로 feature
추출하면 semantic 정보를 더 잘 뽑는다" — DiffusionPolicyImage.VisionEncoder(ResNet18, 랜덤
초기화, 우리 200 demo만으로 학습)와 달리 CLIP은 4억 쌍으로 사전학습돼 있어 좁은 시뮬레이션
데이터만으로 시각 특징을 처음부터 배울 필요가 없음). 언어 타워는 안 씀(비전 인코더만 분리해서
사용) — "CLIP=언어조건화 필요"라는 초기 판단이 틀렸음, semantic feature 자체가 이점이라 언어
없이도 비전 인코더만 떼어쓰는 게 맞음.

원문 구조(stage 분류 + tau 회귀 동시 예측)는 그대로 유지: CLIP 비전 인코더(고정) + proprio
state를 이어붙여 공유 trunk에 넣고, stage 분류 head(K-way)와 tau 회귀 head(sigmoid, [0,1])로
분기한다.

predict_progress()는 lerobot 문서의 Formula(2)를 이 경량 버전에 맞게 단순화한 것 —
원문은 dataset-level temporal proportion(α̅_k)으로 stage별 길이를 보정하지만, 우리
stage_progress()가 이미 학습 라벨 생성 시 그 역할(구간별 상대위치)을 하므로 추론 시엔
argmax(stage)+tau를 그대로 (stage+tau)/K로 정규화하면 학습 타겟과 동일한 척도가 된다.
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipVisionEncoder(nn.Module):
    """CLIP(openai/clip-vit-base-patch32) 비전 인코더만 떼어써서(언어 타워 없음) 고정.
    입력 (B,3,H,W) float[0,1](우리 파이프라인 관례) -> CLIP 224x224 리사이즈+정규화 -> image_embeds(512)."""

    def __init__(self, model_id="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_id)
        self.out_dim = self.model.config.projection_dim  # ViT-B/32 = 512
        self.freeze = freeze
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image):
        x = nn.functional.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.model(pixel_values=x)
        return out.image_embeds

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()  # 고정 CLIP은 BatchNorm/Dropout 없지만(ViT+LayerNorm) 관례상 eval 유지
        return self


class SARMRewardModel(nn.Module):
    def __init__(self, num_stages, state_dim, hidden_dim=256, freeze_encoder=True):
        super().__init__()
        self.num_stages = num_stages
        self.encoder = ClipVisionEncoder(freeze=freeze_encoder)
        feat_dim = self.encoder.out_dim + state_dim
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.stage_head = nn.Linear(hidden_dim, num_stages)
        self.tau_head = nn.Linear(hidden_dim, 1)

    def forward(self, image, state):
        """image: (B,3,H,W) float[0,1]. state: (B,state_dim). -> (stage_logits(B,K), tau(B,) in [0,1])."""
        feat = torch.cat([self.encoder(image), state], dim=-1)
        h = self.trunk(feat)
        stage_logits = self.stage_head(h)
        tau = torch.sigmoid(self.tau_head(h)).squeeze(-1)
        return stage_logits, tau

    def compute_loss(self, batch):
        stage_logits, tau_pred = self.forward(batch["image"], batch["state"])
        stage_loss = nn.functional.cross_entropy(stage_logits, batch["stage"])
        tau_loss = nn.functional.mse_loss(tau_pred, batch["tau"])
        loss = stage_loss + tau_loss
        return loss, {"stage_loss": stage_loss.item(), "tau_loss": tau_loss.item()}

    @torch.no_grad()
    def predict_progress(self, image, state):
        """(B,3,H,W), (B,state_dim) -> progress(B,) float[0,1) — argmax(stage)+tau를 K로 정규화."""
        stage_logits, tau = self.forward(image, state)
        stage_idx = stage_logits.argmax(dim=-1).float()
        return (stage_idx + tau) / self.num_stages
