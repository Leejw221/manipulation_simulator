"""CLIP 텍스트 인코더 — SARM 원문(xdofai/opensarm models/clip_encoder.py)의 FrozenCLIPEncoder.
encode_text()에 대응. 비전 쪽은 기존 sarm_reward_model.ClipVisionEncoder 재사용(같은
openai/clip-vit-base-patch32 백본, 프로젝션 512dim이라 호환)."""

import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


class ClipTextEncoder(nn.Module):
    def __init__(self, model_id="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        self.model = CLIPTextModelWithProjection.from_pretrained(model_id)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.out_dim = self.model.config.projection_dim
        self.freeze = freeze
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts, device):
        """texts: list[str] 길이 B -> (B, out_dim) text embeds."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        return self.model(**inputs).text_embeds
