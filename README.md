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

## 사용법

### 학습
```bash
python -m mani_sim.scripts.train                    # configs/train.yaml 기본값
python -m mani_sim.scripts.train num_epochs=50       # hydra 오버라이드
```
체크포인트·정규화 통계는 `outputs/train/<task>_<policy>/`에 저장됨. wandb 로깅은 `use_wandb: true`(기본값)로 켜져 있음 — `~/.netrc`에 wandb 로그인 필요.

### 평가 (rollout, 화면 렌더링 없이)
```bash
python -m mani_sim.scripts.eval \
    checkpoint_path=outputs/train/lift_low_dim_diffusion_unet_lowdim/policy_epoch50.pt \
    num_episodes=20
```

### 실시간 시각화 (MuJoCo 뷰어 창)
```bash
unset MUJOCO_GL   # 반드시 unset — egl로 두면 오프스크린 강제라 창이 안 뜸
python -m mani_sim.scripts.live_rollout \
    checkpoint_path=outputs/train/lift_low_dim_diffusion_unet_lowdim/policy_epoch50.pt \
    num_episodes=3 max_steps=200
```
DISPLAY가 가리키는 화면에 뜸 — 원격 접속 중이고 X forwarding 없으면 안 보일 수 있음. 마우스로 카메라 회전·줌 가능. 에피소드가 바뀔 때(`env.reset()`) 창이 한 번씩 깜빡이는 건 robosuite 정상 동작(공식 teleop 스크립트도 동일).

## 지원 task

현재 config가 준비된 것: **`lift_low_dim`**, **`can_low_dim`** (`configs/task/`).

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
python -m mani_sim.scripts.live_rollout task=can_low_dim \
    checkpoint_path=outputs/train/can_low_dim_diffusion_unet_lowdim/policy_epoch50.pt
```

위 ①~③은 실제로 실행해 확인했음(데이터 로드·env 생성 성공). ④(학습 실행)는 아직 안 돌려봄 — `lift_low_dim`과 동일한 코드 경로라 동작할 것으로 보이나 [추정].

## 현재 상태

M2(Diffusion Policy low-dim, lift 태스크)까지 완료 — `docs/plan.md` 마일스톤 표 참고.
