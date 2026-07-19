"""SIRIUS/APO 스타일 round-based HITL 오케스트레이션.

매 라운드: (1) 현재 체크포인트를 collect.py로 배포해 사람 개입 데이터 수집 → (2) round0..현재까지
누적(merge_rounds.py) → (3) 누적 데이터로 train.py 재학습(옵션: SIRIUS 스타일 가중치,
train.weighting_kind) → (4) eval.py로 새 체크포인트 성공률 측정. train/eval/collect 각각의 실제
로직은 건드리지 않고 서브프로세스로 그대로 호출한다 — 세 스크립트 다 `@hydra.main`이라 프로세스당
config 하나를 전제하는데, 라운드마다 다른 override로 여러 번 부르려면 서브프로세스가 가장 안전하다
(eval_openvla.py에서 hydra.compose를 in-process로 반복 호출할 때 겪었던 전역상태 문제를 회피).

round0은 이미 학습된 base 체크포인트(순수 데모로 train.py를 미리 돌려 만든 것)를 배포해 시작한다
— round0_checkpoint=null이면 정책 없이(zero-action stub) 순수 개입으로만 첫 라운드를 모은다.

**주의**: task=<이름>/policy=<이름>(config 그룹 선택)만 collect/train/eval 하위 프로세스로
전달된다. `task.xxx=yyy`처럼 이 스크립트 호출부에 준 개별 필드 오버라이드는 하위 프로세스에
전파되지 않는다(각 단계가 독립 hydra 프로세스라서) — 그런 오버라이드가 필요하면 전용 task/policy
yaml을 새로 만드는 걸 권장.

**scope**: policy_name은 diffusion_lowdim/diffusion(image)/bc_rnn_lowdim만 지원(체크포인트가
`policy_epoch<N>.pt` 관례를 따르는 것들). OpenVLA(별도 어댑터 디렉토리 관례)는 아직 미지원.

사용:
    python -m mani_sim.scripts.round task=door_cabinet_low_dim policy=diffusion_unet_lowdim \
        policy_name=diffusion_lowdim \
        round0_checkpoint=outputs/door_cabinet_low_dim/diffusion_unet_lowdim/policy_epoch300.pt \
        num_rounds=3 train.weighting_kind=class_based
"""

import ast
import logging
import os
import subprocess
import sys

import hydra
from omegaconf import DictConfig

from mani_sim.utils.checkpoints import get_latest_epoch_checkpoint

logger = logging.getLogger(__name__)


def _run(module, overrides, capture=False):
    cmd = [sys.executable, "-m", f"mani_sim.scripts.{module}"] + overrides
    logger.info("$ " + " ".join(cmd))
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return result.stdout
    subprocess.run(cmd, check=True)  # collect는 대화형(PICO/키보드) — stdio를 그대로 상속
    return None


def _run_merge(output_path, input_paths):
    cmd = [sys.executable, "-m", "mani_sim.scripts.merge_rounds", output_path] + input_paths
    logger.info("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_eval_metrics(stdout):
    """eval.py 마지막 줄(print(metrics))을 파싱."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return ast.literal_eval(line)
    raise ValueError(f"eval 출력에서 metrics dict를 못 찾음:\n{stdout}")


@hydra.main(config_path="../configs", config_name="round", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    task_policy = [f"task={cfg.task.name}", f"policy={cfg.policy.name}", f"policy_name={cfg.policy_name}"]

    checkpoint = cfg.round0_checkpoint
    round_hdf5s = []
    round_metrics = []

    for r in range(cfg.num_rounds):
        round_dir = os.path.join(cfg.output_dir, f"round{r}")
        os.makedirs(round_dir, exist_ok=True)
        logger.info(f"===== round {r} 시작 (배포 체크포인트: {checkpoint}) =====")

        # 1) collect: 현재 체크포인트를 배포해 사람 개입 데이터 수집(대화형).
        collect_hdf5 = os.path.join(round_dir, "collect.hdf5")
        _run("collect", task_policy + [
            f"checkpoint_path={checkpoint if checkpoint is not None else 'null'}",
            f"output_path={collect_hdf5}",
            f"num_episodes={cfg.collect.num_episodes}",
            f"intervention_device={cfg.collect.intervention_device}",
            f"render={cfg.collect.render}",
            f"preintv_length={cfg.collect.preintv_length}",
            f"control_fps={cfg.collect.control_fps}",
        ])
        round_hdf5s.append(collect_hdf5)

        # 2) merge: round0..r 누적(SIRIUS/APO 관례 — 매 라운드 "그 시점까지 전체"로 재학습).
        cumulative_hdf5 = os.path.join(round_dir, "cumulative.hdf5")
        _run_merge(cumulative_hdf5, round_hdf5s)

        # 3) train: 누적 데이터로 처음부터 재학습(옵션: SIRIUS 스타일 가중치).
        train_output_dir = os.path.join(round_dir, "policy")
        train_overrides = task_policy + [
            f"task.hdf5_path={cumulative_hdf5}",
            f"output_dir={train_output_dir}",
            f"num_epochs={cfg.train.num_epochs}",
            f"eval_every_epochs={cfg.train.eval_every_epochs}",
            f"batch_size={cfg.train.batch_size}",
            f"use_wandb={cfg.train.use_wandb}",
        ]
        if cfg.train.weighting_kind:
            train_overrides.append(f"weighting.kind={cfg.train.weighting_kind}")
        _run("train", train_overrides)

        checkpoint, epoch = get_latest_epoch_checkpoint(train_output_dir)
        if checkpoint is None:
            raise RuntimeError(f"round {r}: train 후 체크포인트를 못 찾음({train_output_dir})")
        logger.info(f"round {r}: 새 체크포인트 = {checkpoint} (epoch {epoch})")

        # 4) eval: 새 체크포인트 성공률 측정(헤드리스).
        eval_stdout = _run("eval", task_policy + [
            f"checkpoint_path={checkpoint}",
            f"num_episodes={cfg.eval.num_episodes}",
            f"max_steps={cfg.eval.max_steps}",
            f"eval_seed={cfg.eval.eval_seed}",
        ], capture=True)
        metrics = _parse_eval_metrics(eval_stdout)
        round_metrics.append(metrics)
        logger.info(f"round {r}: success_rate={metrics['success_rate']:.3f} (n={metrics['num_episodes']})")

    logger.info(f"전체 {cfg.num_rounds}라운드 완료. 최종 체크포인트: {checkpoint}")
    for r, m in enumerate(round_metrics):
        logger.info(f"  round {r}: success_rate={m['success_rate']:.3f}")
    return {"final_checkpoint": checkpoint, "round_metrics": round_metrics}


if __name__ == "__main__":
    main()
