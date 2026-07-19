"""체크포인트 저장/재개(resume) 유틸.

오늘(2026-07-17) PC 리부트로 OpenVLA·stage 학습이 중간에 죽었을 때 "어디서부터 이어가나"를
매번 손으로 찾아야 했던 문제 — 이 유틸로 표준화한다.

두 컨벤션을 그대로 따른다(파일 포맷을 새로 발명하지 않음 — 이미 학습해둔 체크포인트와의
호환을 깨지 않기 위함):
  - DP/BC 계열: `<output_dir>/policy_epoch<N>.pt` (epoch 번호로 정렬)
  - OpenVLA 계열: `<output_dir>/lora_latest/policy`, `<output_dir>/lora_step<N>/policy`
    (PeftModel.save_pretrained의 어댑터별 서브디렉토리 규칙, train_openvla_base.py 그대로)
"""

import glob
import os
import re

import torch


def get_latest_epoch_checkpoint(output_dir, prefix="policy_epoch"):
    """`<prefix><N>.pt` 중 가장 큰 N의 경로. 없으면 (None, 0)."""
    pattern = os.path.join(output_dir, f"{prefix}*.pt")
    candidates = []
    for path in glob.glob(pattern):
        m = re.search(rf"{re.escape(prefix)}(\d+)\.pt$", os.path.basename(path))
        if m:
            candidates.append((int(m.group(1)), path))
    if not candidates:
        return None, 0
    epoch, path = max(candidates, key=lambda x: x[0])
    return path, epoch


def save_epoch_checkpoint(output_dir, epoch, model, extra=None):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"policy_epoch{epoch}.pt")
    payload = {"model": model.state_dict(), "epoch": epoch}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_epoch_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt


def get_openvla_resume_adapter(output_dir, prefer="lora_latest"):
    """OpenVLA LoRA 재개용 어댑터 경로. `<output_dir>/<prefer>/policy`가 있으면 그걸,
    없으면 `lora_step<N>/policy` 중 가장 큰 N. 없으면 None."""
    preferred = os.path.join(output_dir, prefer, "policy")
    if os.path.isdir(preferred):
        return preferred

    candidates = []
    for path in glob.glob(os.path.join(output_dir, "lora_step*", "policy")):
        m = re.search(r"lora_step(\d+)", path)
        if m:
            candidates.append((int(m.group(1)), path))
    if not candidates:
        return None
    _, path = max(candidates, key=lambda x: x[0])
    return path
