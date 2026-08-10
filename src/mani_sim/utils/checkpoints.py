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
from omegaconf import OmegaConf

RUN_CONFIG_FILENAME = "run_config.yaml"


def save_run_config(output_dir, task_cfg, policy_cfg, policy_name):
    """학습이 실제로 쓴 task/policy 설정을 체크포인트 폴더에 사본으로 남긴다 — eval.py가
    이걸 읽어 task=/policy=를 사람이 다시 맞춰줄 필요 없이 자동으로 맞춘다(2026-07-25).
    hydra 내부(.hydra/config.yaml)에 기대지 않는 이유: 우리가 형식을 소유해야 hydra
    버전이 바뀌어도 안 깨진다."""
    os.makedirs(output_dir, exist_ok=True)
    payload = OmegaConf.create({"task": task_cfg, "policy": policy_cfg, "policy_name": policy_name})
    OmegaConf.save(payload, os.path.join(output_dir, RUN_CONFIG_FILENAME))


def load_run_config(checkpoint_path):
    """checkpoint_path 옆의 run_config.yaml을 읽는다. 없으면 None(구 체크포인트·openvla 등 —
    호출부에서 기존 CLI 값을 그대로 쓰도록 처리)."""
    path = os.path.join(os.path.dirname(checkpoint_path), RUN_CONFIG_FILENAME)
    if not os.path.isfile(path):
        return None
    return OmegaConf.load(path)


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


RESUME_STATE_FILENAME = "resume_state.pt"


def save_resume_state(output_dir, epoch, model, optimizer, lr_scheduler, ema=None):
    """재개(resume) 전용 상태 — 고정 파일명으로 매번 덮어쓴다(2026-07-25). AdamW
    optimizer state가 모델 파라미터의 2배 크기라(exp_avg, exp_avg_sq), epoch별
    마일스톤 체크포인트(`policy_epoch<N>.pt`, eval/비교용)에 매번 같이 넣으면 저장이
    N배로 불어난다(실측: 1.1GB→3.3GB/개). resume엔 항상 가장 최근 것 하나만
    필요하므로 별도 파일 하나에만 둔다.

    ema를 넘기면 EMA 상태(averaged_model + optimization_step)도 같이 저장한다
    (2026-08-11 추가). 없으면 resume 시 averaged_model이 init_from 시점에 머문 채
    decay가 0부터 재램프돼, 저장되는 policy_epoch*.pt의 평균 구간이 의도와 달라진다."""
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "model": model.state_dict(), "epoch": epoch,
        "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict(),
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, os.path.join(output_dir, RESUME_STATE_FILENAME))


def load_resume_state(output_dir, device):
    """resume_state.pt가 있으면 그 dict, 없으면 None(구 학습 — 옛 policy_epoch<N>.pt로
    fallback해야 함, 호출부 책임)."""
    path = os.path.join(output_dir, RESUME_STATE_FILENAME)
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location=device, weights_only=False)


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
