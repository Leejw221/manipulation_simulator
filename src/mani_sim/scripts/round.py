"""SIRIUS/APO 스타일 round-based HITL 오케스트레이션.

매 라운드: (1) 현재 체크포인트를 collect.py로 배포해 사람 개입 데이터 수집 → (2) round0..현재까지
누적(merge_rounds.py) → (3) 누적 데이터로 train.py 재학습(옵션: SIRIUS 스타일 가중치,
train.weighting_kind / APO 시스템, train.system_kind — 2026-07-27 추가) → (4) eval.py로 새
체크포인트 성공률 측정. train/eval/collect 각각의 실제 로직은 건드리지 않고 서브프로세스로 그대로
호출한다 — 세 스크립트 다 `@hydra.main`이라 프로세스당 config 하나를 전제하는데, 라운드마다 다른
override로 여러 번 부르려면 서브프로세스가 가장 안전하다(eval_openvla.py에서 hydra.compose를
in-process로 반복 호출할 때 겪었던 전역상태 문제를 회피).

**system_kind=apo일 때만** 이 라운드를 배포한 체크포인트를 다음 train.py 호출의 `init_from`으로
넘긴다 — SIRIUS 조건(system_kind=null)은 지금처럼 매 라운드 scratch 재학습 그대로다. APO는
reference(=직전 정책 freeze)가 필수라 warm-start를 강제하지만, SIRIUS는 그럴 필요가 없어서
기존 라운드 실험 설계(round0..r 누적 데이터로 매번 처음부터)를 건드리지 않았다.

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
    # -u(unbuffered) 필수 - 파이프로 연결되면(capture=True든, 상위 프로세스가 우리 stdout을
    # 또 파이프로 감싸든) 파이썬이 터미널이 아니라고 판단해 자동으로 줄단위가 아니라
    # 블록단위 버퍼링으로 바꾼다(표준 파이썬 동작). 그러면 로그가 한동안 안 나오다가
    # 한꺼번에 쏟아지는 형태가 되는데, 이걸 "출력이 없다=멈췄다"로 오판하는 감시가 걸려있으면
    # 실제로는 멀쩡히 도는 프로세스가 죽는다(2026-07-28 새벽 실측 - SIRIUS 학습을 tee로
    # 파이프에 물렸더니 3번 연속 정확히 같은 지점에서 죽었고, -u 없이는 round.py의 eval
    # capture=True 경로도 똑같은 위험에 노출돼 있었음).
    cmd = [sys.executable, "-u", "-m", f"mani_sim.scripts.{module}"] + overrides
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
    # round_hdf5s를 데모로 미리 채워서 시작 - merge_rounds.py 자기 docstring의 "round0(데모)"
    # 컨벤션과 맞추기 위함(2026-07-27 발견한 gap: 예전엔 여기가 항상 빈 리스트로 시작해서
    # 누적 재학습에 원본 데모가 한 번도 안 들어갔음). round0_demo_hdf5는 extract_demo_subset.py
    # 산출물(예: data/intervention/transport_demo20.hdf5) - null이면 예전처럼 라운드
    # 수집분만으로 시작(데모 없는 task, 혹은 의도적으로 뺄 때).
    round_hdf5s = [cfg.round0_demo_hdf5] if cfg.get("round0_demo_hdf5", None) else []
    round_metrics = []

    for r in range(cfg.num_rounds):
        round_dir = os.path.join(cfg.output_dir, f"round{r}")
        os.makedirs(round_dir, exist_ok=True)
        logger.info(f"===== round {r} 시작 (배포 체크포인트: {checkpoint}) =====")

        # 1) collect: 현재 체크포인트를 배포해 사람 개입 데이터 수집(대화형).
        collect_hdf5 = os.path.join(round_dir, "collect.hdf5")
        collect_overrides = task_policy + [
            f"checkpoint_path={checkpoint if checkpoint is not None else 'null'}",
            f"output_path={collect_hdf5}",
            f"num_episodes={cfg.collect.num_episodes}",
            f"intervention_device={cfg.collect.intervention_device}",
            f"render={cfg.collect.render}",
            f"preintv_length={cfg.collect.preintv_length}",
            f"control_fps={cfg.collect.control_fps}",
        ]
        # round_size_ratio: SIRIUS/APO 두 조건 다 같은 라운드당 데이터량으로 통제(2026-07-27
        # 가중치 공식이 요구하는 demo 비율 역산 — collect.yaml 설명 참고). 없으면(null) 예전처럼
        # num_episodes 상한만으로 종료.
        if cfg.collect.get("round_size_ratio", None):
            collect_overrides.append(f"round_size_ratio={cfg.collect.round_size_ratio}")
        _run("collect", collect_overrides)
        round_hdf5s.append(collect_hdf5)

        # 2) merge: round0..r 누적(SIRIUS/APO 관례 — 매 라운드 "그 시점까지 전체"로 재학습).
        # round_hdf5s가 round0_demo_hdf5로 이미 채워져 있으면(위 참고) 데모까지 포함된 진짜 전체.
        cumulative_hdf5 = os.path.join(round_dir, "cumulative.hdf5")
        _run_merge(cumulative_hdf5, round_hdf5s)

        # 3) train: 누적 데이터로 처음부터 재학습(옵션: SIRIUS 스타일 가중치).
        train_output_dir = os.path.join(round_dir, "policy")
        train_overrides = task_policy + [
            f"task.hdf5_path={cumulative_hdf5}",
            # cumulative_hdf5는 merge_rounds.py가 만든 것 - mask 그룹 자체가 없어서 task
            # yaml에 filter_key(예: demo20)가 박혀있으면 robomimic SequenceDataset이
            # `mask/<key>` 조회 시 KeyError로 죽는다(2026-07-27 실측 확인) - 여기서 명시적으로
            # 꺼야 함(이미 이 파일 자체가 그 subset만 담고 있어 필터링이 더 필요 없기도 함).
            "task.filter_key=null",
            f"output_dir={train_output_dir}",
            f"num_epochs={cfg.train.num_epochs}",
            f"eval_every_epochs={cfg.train.eval_every_epochs}",
            f"batch_size={cfg.train.batch_size}",
            f"use_wandb={cfg.train.use_wandb}",
        ]
        if cfg.train.weighting_kind:
            train_overrides.append(f"weighting.kind={cfg.train.weighting_kind}")
        if cfg.train.get("system_kind", None):
            train_overrides.append(f"system.kind={cfg.train.system_kind}")
            if cfg.train.get("apo_lr", None):
                train_overrides.append(f"lr={cfg.train.apo_lr}")
            if cfg.train.get("apo_weight_decay", None):
                train_overrides.append(f"weight_decay={cfg.train.apo_weight_decay}")
            if cfg.train.get("apo_max_grad_norm", None):
                train_overrides.append(f"max_grad_norm={cfg.train.apo_max_grad_norm}")
            if checkpoint is not None:
                # 이 라운드를 "배포한" 체크포인트 == APO 원문의 vla_path(policy/reference가
                # 같은 지점에서 출발). round0는 round0_checkpoint가 이미 이 역할.
                train_overrides.append(f"init_from={checkpoint}")
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
