"""OpenVLA round0(비가중 base) 학습 루프 — outputs/train_openvla_base.py를 registry로 이식.

DP와 학습 단위가 다르다: DP는 epoch(전체 데이터 1순회) 기준, OpenVLA는 **gradient step**
기준(청크 없이 프레임 단위 + gradient accumulation, 원 논문·공식 finetune.py 관례).
그래서 DiffusionTrainer와 다른 러너 클래스로 분리하되, checkpoint resume 인터페이스
(prepare_for_resume, staticmethod)는 scripts/train.py가 두 러너를 균일하게 다룰 수 있게 맞춘다.

resume 특이사항: DP는 "policy 만든 뒤 state_dict 로드"지만, OpenVLA는 LoRA를
`PeftModel.from_pretrained(adapter_path)`로 **생성 시점**에 넣어야 한다 — 그래서
`prepare_for_resume`가 policy 생성 *전*에 policy_cfg.policy_adapter_path를 채워 넣는다
(scripts/train.py가 이 순서를 지킨다).
"""

import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from mani_sim.datasets.normalization import MinMaxNormalizer, compute_minmax_stats, save_stats
from mani_sim.datasets.openvla_dataset import OpenVLAFrameDataset, collate_openvla
from mani_sim.factory import registry
from mani_sim.utils.checkpoints import get_openvla_resume_adapter

logger = logging.getLogger(__name__)


@registry.register_runner("openvla_trainer")
class OpenVLATrainer:
    @staticmethod
    def prepare_for_resume(cfg):
        """policy 생성 전에 호출 — 이어받을 LoRA adapter가 있으면 policy_cfg에 채워 넣는다."""
        adapter = get_openvla_resume_adapter(cfg.output_dir)
        if adapter:
            cfg.policy.policy_adapter_path = adapter
            logger.info(f"resume: found adapter at {adapter}")
        else:
            logger.info("resume: no existing adapter found, starting from base")
        return adapter is not None

    def __init__(self, cfg, policy, device):
        self.cfg = cfg
        self.policy = policy
        self.device = device
        os.makedirs(cfg.output_dir, exist_ok=True)

        stats = compute_minmax_stats(cfg.task.hdf5_path, obs_keys=[])
        save_stats(stats, os.path.join(cfg.output_dir, "normalization_stats.json"))
        self.normalizer = MinMaxNormalizer(stats)

        self.dataset = OpenVLAFrameDataset(cfg.task.hdf5_path, image_key=cfg.task.rgb_keys[0])
        self.dataloader = DataLoader(
            self.dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
            num_workers=cfg.num_workers, collate_fn=collate_openvla,
        )
        logger.info(f"dataset len={len(self.dataset)} (frame 단위, 청크 없음)")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.model.parameters() if p.requires_grad], lr=cfg.lr
        )

    def resume_start_step(self):
        """OpenVLA는 adapter 자체에 학습 진행분이 이미 들어있음(construction-time 로드) —
        진행 gradient step 수는 파일명에서 못 뽑으므로(lora_latest는 매번 덮어씀) 0부터
        세되, 가중치는 이미 이어받은 상태로 시작(=fine-tune 계속의 의미로 충분)."""
        return 0

    def train(self, max_steps, start_step=0, save_every=250, log_every=10, use_wandb=False):
        if use_wandb:
            import wandb

        grad_step = start_step
        phys_step = 0
        accum_loss = 0.0
        t0 = time.time()
        self.optimizer.zero_grad()
        accumulation_steps = self.cfg.get("grad_accumulation_steps", 8)

        def save(tag):
            save_dir = os.path.join(self.cfg.output_dir, tag)
            self.policy.model.save_pretrained(save_dir)
            logger.info(f"saved LoRA adapter -> {save_dir} (로드 경로: {save_dir}/policy)")

        done = grad_step >= max_steps
        while not done:
            for batch in self.dataloader:
                actions_norm = self.normalizer.normalize_action(
                    torch.as_tensor(batch["actions"], dtype=torch.float32)
                ).numpy()

                loss = self.policy.compute_base_loss(batch["images"], batch["instructions"], actions_norm)
                (loss / accumulation_steps).backward()
                accum_loss += loss.item()
                phys_step += 1

                if phys_step % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.policy.model.parameters() if p.requires_grad], max_norm=1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    grad_step += 1

                    if grad_step % log_every == 0:
                        avg_loss = accum_loss / (log_every * accumulation_steps)
                        elapsed = time.time() - t0
                        logger.info(f"grad_step {grad_step}/{max_steps} loss {avg_loss:.4f} "
                                    f"({elapsed / max(grad_step - start_step, 1):.2f}s/step)")
                        if use_wandb:
                            wandb.log({"loss": avg_loss}, step=grad_step)
                        accum_loss = 0.0

                    if grad_step % save_every == 0:
                        save("lora_latest")

                    if grad_step >= max_steps:
                        done = True
                        break

        save("lora_final")
        logger.info(f"DONE. total grad_steps={grad_step}, wall={time.time() - t0:.0f}s")
