"""OpenVLA(7B) 정책 래퍼 — policy 축의 세 번째 백본.

**[검증] 2026-07-16(4) 밤 — 실제 가중치로 로딩·forward·log-prob까지 확인됨.**
`verify_smoke()`로 policy==reference(diff=0.0, LoRA 초기화가 정확히 0 delta라는 이론과
일치) 확인. 이 과정에서 잡은 버그 3개(환경 버전 충돌): transformers 최신판엔
`AutoModelForVision2Seq`가 없어져 `4.40.1`로 다운그레이드 → peft 최신판이 그 구버전
transformers와 안 맞아 `0.11.1`로 다운그레이드 → timm 최신판이 OpenVLA 자체 버전가드
(`>=0.9.10,<1.0.0`)에 걸려 `0.9.16`으로 다운그레이드. 코드 버그 1개: `pixel_values`를
모델과 같은 dtype(bfloat16)으로 안 캐스팅해서 `RuntimeError` — 고침(`_build_inputs`).

**reference 모델 = 메모리 절약 설계(2026-07-16(4) 사용자와 합의)**: APO 원문처럼 policy와
reference를 통째로 두 벌 로드하면(각 ~14GB) 이 카드(24GB)에도 안 들어간다
(14+14=28GB > 24GB). 대신 **base 모델 하나만 로드하고 LoRA adapter를 껐다 켜서** reference를
만든다(HuggingFace peft의 `disable_adapter()` 컨텍스트 — LoRA+DPO/KTO에서 표준적으로 쓰는
패턴이자, LoRA로 파인튜닝하는 이상 policy와 reference의 base 가중치는 애초에 동일하므로
방식으로도 정확하다). 총 메모리 ≈ base 1벌(~14GB, 4bit면 ~4GB) + LoRA adapter(수백 MB) —
policy/diffusion 학습과 동시에는 못 돌리지만 단독으로는 24GB에 들어간다.

policy(백본) 축의 다른 두 구현(bc: robomimic BC_RNN_GMM, diffusion: DiffusionPolicy)과
같은 최소 인터페이스를 맞춘다: log_probs 계열 계산 + adaptive-weight용 error 계산.
"""

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from mani_sim.policies.openvla.action_tokenizer import ActionTokenizer


class OpenVLAPolicy(nn.Module):
    def __init__(
        self,
        model_id="openvla/openvla-7b",
        lora_rank=32,
        lora_alpha=16,  # 공식 finetune.py 그대로: min(lora_rank, 16) — lora_rank=32면 16
        action_dim=7,
        load_in_4bit=False,
        device="cuda",
        policy_adapter_path=None,
    ):
        super().__init__()
        # 지연 import — 가중치 안 쓰는 bc/diffusion 전용 실행에선 transformers/peft가
        # 없어도 mani_sim 나머지가 동작해야 하므로(선택적 의존성).
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from peft import LoraConfig, PeftModel, get_peft_model

        self.action_dim = action_dim
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            load_in_4bit=load_in_4bit,
        )

        if policy_adapter_path is not None:
            # 이전 라운드(예: round0 base) 학습으로 저장된 LoRA를 그대로 로드 — 새 LoRA
            # 초기화(get_peft_model) 대신 학습된 가중치를 씀. rollout·다음 라운드 이어붙이기용.
            # is_trainable=True 필수: PeftModel.from_pretrained 기본값은 False(추론 전용
            # 가정) — 이대로 두면 로드된 LoRA 파라미터가 전부 requires_grad=False가 돼서
            # 이어서 학습(resume)할 때 "optimizer got an empty parameter list"로 즉시
            # 실패한다(2026-07-17 발견 — resume을 실제로 처음 밟아본 경로).
            self.model = PeftModel.from_pretrained(
                base_model, policy_adapter_path, adapter_name="policy", is_trainable=True
            )
        else:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            self.model = get_peft_model(base_model, lora_config, adapter_name="policy")
        self.model.to(device)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()  # gradient checkpointing + LoRA(입력 freeze) 조합에 필요

        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, n_bins=256)
        self._reference_adapter_loaded = False

    def load_reference_adapter(self, adapter_path):
        """이전 라운드 저장된 LoRA adapter를 "reference"란 이름으로 추가 로드 — round0에서
        시작할 땐 호출 안 함(reference_context()가 자동으로 disable_adapter=pretrained base로
        대체)."""
        self.model.load_adapter(adapter_path, adapter_name="reference")
        self._reference_adapter_loaded = True

    @contextmanager
    def reference_context(self):
        """이 안에서의 forward는 reference 정책(고정) 기준. load_reference_adapter를 안
        불렀으면 pretrained base(adapter 전체 비활성)를 reference로 쓴다(round0 관례 —
        "이전 라운드 정책"이 없으니 사전학습 그대로가 유일하게 말이 되는 기준점)."""
        if self._reference_adapter_loaded:
            self.model.set_adapter("reference")
            try:
                yield
            finally:
                self.model.set_adapter("policy")
        else:
            with self.model.disable_adapter():
                yield

    def _build_inputs(self, images, instructions, actions):
        """images: list[PIL.Image] 길이 B. instructions: list[str] 길이 B.
        actions: (B, Da) float(정규화됨, [-1,1]) -> processor 입력 + action 토큰을 이어붙인
        labels. OpenVLA 공식 프롬프트 포맷(`"In: what action ...\\nOut:"`)을 따른다."""
        prompts = [f"In: What action should the robot take to {instr}?\nOut:" for instr in instructions]
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        # pixel_values는 모델과 같은 dtype(bfloat16)이어야 함 — input_ids/attention_mask는
        # long/bool 그대로 두고 pixel_values만 캐스팅(공식 README의 `.to(device, dtype=bf16)`를
        # 그대로 하면 정수 텐서까지 bf16이 돼버려 여기선 분리).
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        action_tokens = self.action_tokenizer.encode(actions)  # (B, Da) int
        action_tokens = torch.as_tensor(action_tokens, dtype=torch.long, device=self.device)
        input_ids = torch.cat([inputs["input_ids"], action_tokens], dim=1)
        attention_mask = torch.cat(
            [inputs["attention_mask"], torch.ones_like(action_tokens)], dim=1
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": inputs["pixel_values"],
            "prompt_len": inputs["input_ids"].shape[1],
        }

    def _action_token_logps(self, model_inputs, model=None):
        """input_ids 중 action 토큰 구간(prompt_len 이후)의 per-token log-prob. (B, Da)."""
        model = model or self.model
        out = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pixel_values=model_inputs["pixel_values"],
        )
        logits = out.logits[:, :-1]  # next-token 예측이라 마지막 하나 밀림
        labels = model_inputs["input_ids"][:, 1:]
        prompt_len = model_inputs["prompt_len"]

        action_logits = logits[:, prompt_len - 1 :]  # action 토큰 위치의 예측
        action_labels = labels[:, prompt_len - 1 :]

        log_probs_all = action_logits.float().log_softmax(-1)
        token_logps = torch.gather(log_probs_all, dim=2, index=action_labels.unsqueeze(-1)).squeeze(-1)
        return token_logps  # (B, Da)

    def compute_log_probs(self, images, instructions, actions):
        """(B, Da) per-액션차원-토큰 log-prob. BC의 (B,T) log_probs·diffusion의 -MSE와
        같은 자리에 대응(T축 = action_dim 토큰들)."""
        model_inputs = self._build_inputs(images, instructions, actions)
        return self._action_token_logps(model_inputs)

    def compute_error(self, images, instructions, actions):
        """adaptive-weight(action_error 축)용 오차 신호 — action 토큰의 평균 cross-entropy
        (perplexity 계열). ⚠ 원 논문 APO는 토큰을 역토큰화한 연속행동 L1을 쓴다(디코딩 필요,
        π0-FAST에서 디코드 실패 시 weight=1 fallback을 만든 원인) — 우리는 diffusion 쪽과
        동일하게 "이미 계산되는 예측오차 자체"를 그대로 써서 그 마찰을 피한다(2026-07-16(4)
        사용자 지시의 diffusion 쪽 결정과 동일 원칙 적용, [적응]). L1이 필요하면
        `self.action_tokenizer.decode(...)`로 별도 구현."""
        log_probs = self.compute_log_probs(images, instructions, actions)
        return (-log_probs).mean(dim=1).detach()  # (B,)

    def compute_base_loss(self, images, instructions, actions):
        """round0(비가중 base) 학습용 — HF 표준 causal LM loss(모델이 labels로 자체 계산).
        prompt 구간은 -100(IGNORE_INDEX)으로 마스킹해 action 토큰에서만 loss 계산."""
        model_inputs = self._build_inputs(images, instructions, actions)
        labels = model_inputs["input_ids"].clone()
        labels[:, : model_inputs["prompt_len"]] = -100
        out = self.model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pixel_values=model_inputs["pixel_values"],
            labels=labels,
        )
        return out.loss

    @torch.no_grad()
    def predict_action(self, image, instruction):
        """단일 (image, instruction) -> (action_dim,) 연속 action. 그리디(탐욕적) 자기회귀
        생성 — `.generate()`(HF 표준 API) 대신 한 토큰씩 직접 forward하며 argmax로 뽑는다.
        이유: 이 모델은 커스텀 멀티모달 구조(trust_remote_code)라 `.generate()`의 KV-cache·
        pixel_values 처리 호환이 불확실 — 이미 학습/평가에서 검증된 단일 forward
        (`_action_token_logps`와 같은 경로)를 반복하는 쪽이 확실하다. action_dim(7)번만
        반복이라 비용도 작음."""
        self.model.eval()
        prompts = [f"In: What action should the robot take to {instruction}?\nOut:"]
        inputs = self.processor(text=prompts, images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        generated_tokens = []
        for _ in range(self.action_dim):
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask, pixel_values=inputs["pixel_values"]
            )
            next_token = out.logits[0, -1].argmax().item()
            generated_tokens.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=self.device, dtype=torch.long)], dim=1
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)], dim=1
            )

        action = self.action_tokenizer.decode(np.array(generated_tokens)[None, :])
        return action[0]


def verify_smoke(model_id="openvla/openvla-7b", device="cuda"):
    """가중치가 실제로 로딩·forward 되는지 최소 확인(더미 1 batch). 실행 전 GPU 여유 확인 필수
    (docstring 상단 메모리 계산 참고). 지금까지 실행된 적 없음 — 여유 생기면 이걸로 먼저 검증."""
    from PIL import Image

    policy = OpenVLAPolicy(model_id=model_id, device=device)
    images = [Image.new("RGB", (224, 224))] * 2
    instructions = ["pick up the square nut"] * 2
    actions = np.random.uniform(-1, 1, size=(2, 7)).astype(np.float32)

    log_probs = policy.compute_log_probs(images, instructions, actions)
    print("log_probs shape:", log_probs.shape)

    with policy.reference_context():
        ref_log_probs = policy.compute_log_probs(images, instructions, actions)
    print("reference log_probs shape:", ref_log_probs.shape)
    print("policy==reference 초기 동일해야: diff =", (log_probs - ref_log_probs).abs().max().item())
