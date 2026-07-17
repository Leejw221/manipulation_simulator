"""OpenVLA 원문의 ActionTokenizer 이식 — 연속 action을 vocabulary 끝 N개 토큰에 매핑.

원문: openvla/openvla `prismatic/vla/action_tokenizer.py`(균등 256-bin, 각 action 차원을
독립적으로 discretize, vocab 뒤쪽 n_bins개 토큰 재사용 — 새 토큰 추가 없이 기존
tokenizer만으로 행동을 "언어처럼" 표현). APO_analysis.md에도 이미 확인해둔 방식
("action_tokenizer=256-bin 균등양자화") — 여기서는 그 방식을 그대로 재구현한다
(원본 저장소를 의존성으로 끌어오지 않고, 우리 쪽엔 이 파일 하나로 자립).

⚠ 미검증(가중치 미로딩 상태) — OpenVLA 실제 tokenizer(vocab_size)로 스모크 테스트 안 됨.
"""

import numpy as np


class ActionTokenizer:
    def __init__(self, tokenizer, n_bins=256, min_action=-1.0, max_action=1.0):
        """tokenizer: HuggingFace tokenizer(vocab_size 필요, 뒤쪽 n_bins개 토큰을 action용으로 재사용)."""
        self.tokenizer = tokenizer
        self.n_bins = n_bins
        self.min_action, self.max_action = min_action, max_action

        self.bins = np.linspace(min_action, max_action, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = tokenizer.vocab_size - n_bins - 1

    def encode(self, action):
        """action: (..., Da) float array in [min_action, max_action] -> (..., Da) token id array."""
        action = np.clip(action, a_min=float(self.bins[0]), a_max=float(self.bins[-1]))
        discretized = np.digitize(action, self.bins)
        return self.tokenizer.vocab_size - discretized

    def decode(self, token_ids):
        """token_ids: (..., Da) -> (..., Da) float array(각 bin의 중심값)."""
        discretized = self.tokenizer.vocab_size - np.asarray(token_ids)
        discretized = np.clip(discretized - 1, a_min=0, a_max=len(self.bin_centers) - 1)
        return self.bin_centers[discretized]

    @property
    def vocab_size(self):
        return self.n_bins
