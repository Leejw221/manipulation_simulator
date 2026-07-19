# mani_sim

robomimic 벤치마크 위에서 Diffusion Policy를 학습·평가하는 시뮬레이션 실험 코드베이스.
Data_Efficient_Improvement 연구(Sirius + APO + Diffusion Policy 통합)의 검증 실험 기반.

설계 배경·의존성 원칙·마일스톤별 상세 기록은 [`docs/plan.md`](docs/plan.md) 참고 (이 문서는 "어떻게 쓰는지", plan.md는 "왜 이렇게 만들었는지").

## 용어

같은 "task"라는 말이 상황에 따라 다른 걸 가리켜서 헷갈리기 쉽다. 다섯 개로 나눠서 본다.

| 용어 | 정의 | 이 프로젝트에서 |
|---|---|---|
| **Simulator (시뮬레이터)** | 물리 계산 + 렌더링을 실제로 수행하는 엔진 | MuJoCo |
| **Task (태스크)** | "무슨 문제를 푸는가"의 명세 — 로봇 종류, 물체, 성공 조건, action/obs 정의. 그 자체로 실행되는 게 아니라 규칙 | robosuite가 정의 (`Lift`, `Can`, `Square`, ...) |
| **Environment/env (환경)** | 그 task 규칙을 따라 지금 메모리에서 실제로 돌아가는 인스턴스 — `.reset()`, `.step()` 호출 가능한 살아있는 객체 | `EnvRobosuite` (robomimic이 robosuite를 감싼 것) |
| **Dataset/data (데이터)** | 과거에 그 task를 수행한 기록을 저장해둔 것 — 실행 중이 아니라 숫자(HDF5 파일) | `data/robomimic/<task>/ph/low_dim_v15.hdf5` |
| **Benchmark (벤치마크)** | {task + 특정 dataset + 평가 방식(에피소드 수, 성공률 기준)}을 표준으로 묶어 여러 논문이 같은 기준으로 비교하게 한 것 | "robomimic Lift PH 벤치마크" — Sirius·APO가 결과를 낸 기준 |

관계:
```
Task(규칙 정의) ──┬── 지금 실행하면 → Environment(살아있는 인스턴스) → rollout 평가에 씀
                  └── 과거 수행 기록 → Dataset(HDF5) → 학습에 씀

Task + Dataset + 평가방식 = Benchmark
```

**스택 레이어** (아래로 갈수록 구체적):
```
mani_sim (이 repo: Diffusion Policy, 학습 루프)
   ↓
robomimic (데이터셋 포맷, env 인터페이스 표준화 — robosuite 외 Gym/iGibson도 지원하는 추상 인터페이스)
   ↓
robosuite (로봇 모델, task 정의, 컨트롤러(OSC), teleop 장치)
   ↓
MuJoCo (물리 엔진 + 렌더링)
```
robomimic이 robosuite에 종속되지 않고 `EnvBase` 추상 인터페이스로 시뮬레이터를 갈아 끼울 수 있게 설계된 것과 같은 이유로, 이 repo도 robomimic 관련 코드를 `envs/robomimic/`·`datasets/`에만 격리해뒀다 (`docs/plan.md` 설계원칙 참고).

## 환경 셋업

```bash
conda create -n mani_sim python=3.10
conda activate mani_sim
pip install -r requirements.txt
pip install -e .
```

버전 고정 근거(robomimic v0.4.0, mujoco==3.2.3 등 실제로 부딪힌 문제들)는 `docs/plan.md`의 "M0 확정 결과" 참고.

## 데이터셋 다운로드

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="robomimic/robomimic_datasets",
    filename="v1.5/lift/ph/low_dim_v15.hdf5",   # task 이름만 바꾸면 다른 task
    repo_type="dataset",
    local_dir="data/robomimic/lift/ph",
)
```

## 구조 (moai_policy/flare 컨벤션)

`policy_name`/`runner_name`(factory registry 키)로 정책·학습루프를 갈아 끼운다 — 알고리즘마다
스크립트를 복붙하지 않는다(`docs/plan.md` §9 백로그, 실제로 처리함).

```
factory.py                  # registry: create_policy(name, task_cfg, policy_cfg), get_runner_class(name)
policies/<algo>/*_registrations.py   # 각 정책을 @registry.register_policy(...)로 등록
runners/{diffusion_trainer,openvla_trainer}.py   # 실제 학습 루프(정책 종류 모름, config로 분기)
scripts/{train,eval}.py     # 얇은 hydra 진입점 — policy_name/runner_name만 바꾸면 재사용
utils/{checkpoints,task_utils}.py    # resume·low_dim/image 판별 공용 유틸
```

등록된 policy: `bc_rnn_lowdim` · `diffusion_lowdim`(구, low_dim 전용) · `diffusion`(image, 기본값) · `openvla`.
등록된 runner: `diffusion_trainer` · `openvla_trainer`.

## 사용법

### 학습
```bash
python -m mani_sim.scripts.train                              # configs/train.yaml 기본값(square, diffusion, image)
python -m mani_sim.scripts.train task=square_stage             # stage conditioning 켜기
python -m mani_sim.scripts.train task=square_low_dim policy=diffusion_unet_lowdim policy_name=diffusion_lowdim  # 구 low_dim 경로
python -m mani_sim.scripts.train policy=openvla policy_name=openvla runner_name=openvla_trainer batch_size=2 max_steps=3000
python -m mani_sim.scripts.train resume=true                   # 같은 output_dir의 최신 체크포인트/adapter에서 이어받기
```
체크포인트·정규화 통계는 `outputs/train/<task>_<policy>/`에 저장됨(OpenVLA는 `lora_latest/policy`·`lora_step<N>/policy`). wandb는 `use_wandb=true`로 켜기(`~/.netrc`에 로그인 필요, 기본은 꺼짐).

### 평가 (rollout, 화면 렌더링 없이)
```bash
python -m mani_sim.scripts.eval checkpoint_path=outputs/train/square_diffusion/policy_epoch300.pt num_episodes=50
python -m mani_sim.scripts.eval task=square_stage checkpoint_path=... num_episodes=50   # stage 체크포인트(온라인 stage tracker 자동)
python -m mani_sim.scripts.eval task=square policy_name=openvla policy=openvla \
    checkpoint_path=outputs/train/square_openvla/lora_final/policy \
    stats_path=outputs/train/square_openvla/normalization_stats.json num_episodes=10
```

### 실시간 시각화 (`render=true`)
```bash
# low_dim: MuJoCo 네이티브 뷰어(mjviewer) — DISPLAY 있는 화면에서, MUJOCO_GL unset 필수
unset MUJOCO_GL
python -m mani_sim.scripts.eval task=square_low_dim checkpoint_path=... num_episodes=3 render=true

# image: 오프스크린(egl) 렌더 + OpenCV 창(mjviewer 온스크린과 동시 사용 시 GL 충돌로 세그폴트 — 실측 지뢰)
MUJOCO_GL=egl DISPLAY=:1 python -m mani_sim.scripts.eval checkpoint_path=... num_episodes=3 render=true
```
DISPLAY가 가리키는 화면에 뜸 — 원격 접속 중이고 X forwarding 없으면 안 보일 수 있음.
OpenVLA는 이 PC 터미널 환경에서 cv2 라이브 뷰어(`render=true`)가 원인 불명으로 멈추는 문제가
있어(2026-07-19) `render=true` 대신 `save_gif=outputs/openvla_rollout.gif`로 헤드리스 캡처 권장.

### 개입 데이터 수집 (사람이 정책을 배포하고 개입)
```bash
# PICO(기본) — 화면에서 실행, PICO 연결 필요
python -m mani_sim.scripts.collect checkpoint_path=outputs/train/.../policy_epochN.pt

# 키보드 개입, policy_name으로 BC/DiffusionPolicy(low_dim·image) 아무거나 배포 가능
python -m mani_sim.scripts.collect policy_name=bc_rnn_lowdim intervention_device=keyboard
```

### 가중치 학습 (SIRIUS 스타일, 옵션)
사람 개입으로 모은 데이터(action_mode 라벨 포함)를 학습할 때 SIRIUS 원문의 class_based
고정 가중치나 action_error 적응 가중치를 켤 수 있다(reference 모델 없는 단순 가중 손실 —
APO의 reference+KTO는 아직 미구현):
```bash
python -m mani_sim.scripts.train task.hdf5_path=outputs/intervention/round.hdf5 weighting.kind=class_based
```

### round.py — SIRIUS/APO round-based HITL 오케스트레이션
매 라운드 "배포(collect) → 누적(merge_rounds) → 재학습(train, 옵션: 가중치) → 평가(eval)"를
자동 반복한다. collect/train/eval을 각각 서브프로세스로 그대로 호출하므로(개별 스크립트가
그대로 검증 대상) round.py 자체는 오케스트레이션만 담당:
```bash
python -m mani_sim.scripts.round task=door_cabinet_low_dim policy=diffusion_unet_lowdim \
    policy_name=diffusion_lowdim \
    round0_checkpoint=outputs/door_cabinet_low_dim/diffusion_unet_lowdim/policy_epoch300.pt \
    num_rounds=3 train.weighting_kind=class_based
```
`task=<이름>`/`policy=<이름>`(config 그룹 선택)만 하위 프로세스로 전달된다 — `task.xxx=yyy`
같은 개별 필드 오버라이드는 round.py 호출부에 줘도 전파되지 않는다(각 단계가 독립 hydra
프로세스). policy_name은 diffusion_lowdim/diffusion(image)/bc_rnn_lowdim만 지원(OpenVLA
체크포인트 관례가 달라 아직 미지원).

### 진단 도구
```bash
python -m mani_sim.scripts.label_stages --hdf5 data/robomimic/square/ph/v1.5/square/ph/square_image_v15.hdf5
python -m mani_sim.scripts.stage_counterfactual --checkpoint outputs/train/square_stage_diffusion/policy_epoch300.pt
```

## 지원 task

**image(기본, 접미사 없음)**: `square`(=agentview+wrist 84×84), `square_stage`(+stage_onehot 7-dim).
**low_dim(레거시, `_low_dim` 접미사로 구분)**: `lift_low_dim`, `can_low_dim`, `square_low_dim`, `square_low_dim_stage`.

robomimic이 공식 지원하는 다른 task (아직 config 미작성):

| task | dataset_type | 비고 |
|---|---|---|
| `lift` | ph, mh, mg | 가장 쉬움. 구현됨 |
| `can` | ph, mh, mg, paired | Sirius/APO에서도 사용. 구현됨 |
| `square` | ph, mh | |
| `transport` | ph, mh | 양팔(bimanual) |
| `tool_hang` | ph | 가장 어려움 |

### 새 task 추가하는 법 (Can으로 실제 확인한 절차)

**① 데이터셋 다운로드** — `filename`의 task 이름만 바꾼다:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="robomimic/robomimic_datasets",
    filename="v1.5/can/ph/low_dim_v15.hdf5",
    repo_type="dataset",
    local_dir="data/robomimic/can/ph",
)
```

**② HDF5 열어서 실제 obs 키·차원, 그리고 `env_name` 확인** — 데이터셋 이름("can")과 실제 robosuite `env_name`이 다를 수 있으니(Can → `PickPlaceCan`) 반드시 `env_args`로 확인한다. obs 키는 보통 동일(`robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `object`)하지만 `object` 차원은 task마다 다르다(Lift=10, Can=14 — 물체 개수·종류가 다르므로):
```python
import h5py, json
f = h5py.File("data/robomimic/can/ph/v1.5/can/ph/low_dim_v15.hdf5", "r")
demo0 = f["data"][list(f["data"].keys())[0]]
print(demo0["obs"]["object"].shape)                       # (T, 14)
print(json.loads(f["data"].attrs["env_args"])["env_name"]) # "PickPlaceCan"
```

**③ `configs/task/can_low_dim.yaml` 작성** (②에서 확인한 값 그대로):
```yaml
name: can_low_dim
env_name: PickPlaceCan
robots: Panda
hdf5_path: data/robomimic/can/ph/v1.5/can/ph/low_dim_v15.hdf5
obs_keys: [robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object]
obs_dims: {robot0_eef_pos: 3, robot0_eef_quat: 4, robot0_gripper_qpos: 2, object: 14}
action_dim: 7
```

**④ 학습·평가는 `task=<이름>` 오버라이드만으로 재사용** (모델·스크립트 코드는 그대로):
```bash
python -m mani_sim.scripts.train task=can_low_dim num_epochs=50
python -m mani_sim.scripts.eval task=can_low_dim \
    checkpoint_path=outputs/train/can_low_dim_diffusion_unet_lowdim/policy_epoch50.pt \
    num_episodes=20
python -m mani_sim.scripts.eval task=can_low_dim \
    checkpoint_path=outputs/train/can_low_dim_diffusion_unet_lowdim/policy_epoch50.pt render=true
```

위 ①~③은 실제로 실행해 확인했음(데이터 로드·env 생성 성공). ④(학습 실행)는 아직 안 돌려봄 — `lift_low_dim`과 동일한 코드 경로라 동작할 것으로 보이나 [추정].

## 현재 상태

M2(Diffusion Policy low-dim, lift 태스크)까지 완료 — `docs/plan.md` 마일스톤 표 참고.
