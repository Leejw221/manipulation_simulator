# mani_sim 구현 계획

> 작성: 2026-07-03 (ljw_workspace 계획 세션). 코드 작업 세션(Sonnet)이 이 문서를 읽고 시작한다.
> 연구 맥락 원본: `ljw_workspace/Data_Efficient_Improvement/research_design.md` (판단의 single source of truth는 그쪽).
> 신뢰도 태그: [검증] 코드·실험 확인됨 · [논문] 논문 주장 · [추정] 가설/미확인.

## 1. 목표

**Diffusion Policy를 robomimic 벤치마크에서 학습·평가할 수 있는 자체 시뮬레이션 코드베이스.**

- 연구 배경: Sirius + APO + Diffusion Policy 통합 연구(HITL 루프에서 샘플 단위 적응 가중치)의 검증 실험 기반.
- **robomimic을 첫 벤치마크로 택한 이유**: 핵심 레퍼런스인 Sirius와 APO가 robomimic 기반으로 실험함 → 비교 가능성 확보.
- **이번 범위**: 데이터 로딩 → DP 학습 → rollout 평가 → 공개 벤치마크 수치 재현까지.
- **범위 밖 (이후 단계, 자리만 준비)**: HITL 루프(deploy–개입–relabel–재학습), 샘플 가중치 loss, intervention 수집 방식(teleop vs synthetic — 미정, 추후 결정).

## 2. 설계 원칙

1. **구조는 manipulation_pipeline(flare) 컨벤션을 따른다** — src-layout, hydra config, policies/envs/datasets 분리, pyproject optional-dependencies.
2. **의존성 금지: lerobot, flare(manipulation_pipeline)** — 코드 이식도 하지 않는다. DP는 별도 참조 구현에서 출발 (아래 §5).
3. **벤치마크 확장은 "격리"로만 준비** — robomimic 전용 코드(import 포함)는 `envs/robomimic/`과 datasets 어댑터 안에만. 공통 인터페이스(추상 클래스)는 두 번째 벤치마크 추가 시점에 확정한다. 조기 추상화 금지.
4. **policy는 벤치마크와 독립** — policy 코드에서 robomimic을 import하지 않는다. 관찰/행동은 공통 batch dict로만 주고받는다.
5. 데이터·산출물(HDF5, ckpt, wandb, 비디오)은 git 밖 (`data/`, `outputs/` → .gitignore).

## 3. 디렉토리 구조

```
mani_sim/
├── pyproject.toml            # 패키지: mani_sim, optional-deps: train/eval
├── environment.yml           # conda env (버전 고정은 M0에서 확정)
├── README.md
├── docs/
│   └── plan.md               # 이 문서
├── src/mani_sim/
│   ├── configs/              # hydra: task/, policy/, train.yaml, eval.yaml
│   ├── datasets/             # robomimic HDF5 → 공통 batch dict (obs, action chunk, 정규화)
│   ├── envs/
│   │   └── robomimic/        # env factory + wrapper (robomimic/robosuite import 격리)
│   ├── policies/
│   │   └── diffusion/        # DP 구현 (diffusers 스케줄러 기반)
│   ├── networks/             # vision encoder, U-Net/transformer backbone
│   ├── runners/              # train loop, rollout evaluator
│   ├── scripts/              # train.py, eval.py, download_datasets.py
│   └── utils/
├── tests/
├── data/                     # .gitignore — robomimic 데이터셋
└── outputs/                  # .gitignore — ckpt, 로그, wandb, 평가 비디오
```

## 4. 의존성 방침

- **벤치마크**: `robomimic` + `robosuite` (+ mujoco). **버전 고정이 M0의 핵심 과제** — robomimic 데이터셋은 특정 robosuite 버전에서 수집되어 버전이 어긋나면 환경 동작이 달라짐 [추정, M0에서 확인].
- **DP**: `diffusers`(noise scheduler), `einops`, torch.
- **인프라**: `hydra-core`, `wandb`, `imageio`/`av`(평가 비디오).
- Python 버전: robomimic/robosuite 호환 범위에서 결정 (flare는 3.12지만 우리는 독립 — 3.10이 안전할 수 있음 [추정, M0에서 확인]).

## 5. Diffusion Policy 구현 소스

- **lerobot식 DP를 자체 포팅한다 — flare가 한 방식과 같은 패턴** [검증: `flare/policies/diffusion/diffusion_policy.py`는 "Adapted from Chi et al., 2023 and LeRobot implementation", lerobot import 없이 자기완결(외부 의존은 diffusers뿐)].
  구성: ResNet18+SpatialSoftmax 인코더, ConditionalUnet1d(FiLM), MinMax 정규화, DDPM/DDIM 스케줄러(diffusers).
- flare 구현은 **구조 참조용** (의존·import 금지는 유지). 재현 접지 기준은 Chi et al. 공개 성공률 수치 (robomimic lift/can/square 등).
- manipulation_pipeline의 `sirius/` 폴더(weighted_diffusion_policy 등)는 **미정리 상태** — 이후 가중치 단계에서 참고하더라도 검증 없이 신뢰하지 말 것 (사용자 고지, 2026-07-03).
- 구현 순서: U-Net 1D conditional (CNN 기반) 먼저, transformer 변형은 필요 시.

### 시뮬레이터 의존 체인 (참고)
`mani_sim → robomimic(데이터셋·env wrapper) → robosuite(태스크·로봇) → MuJoCo(물리엔진)`
robomimic 자체는 시뮬레이터가 아니라 데모 학습 프레임워크. Sirius도 robosuite 기반 태스크 사용.

## 6. 마일스톤 (각각 완료 기준 포함)

| # | 내용 | 완료 기준 |
|---|---|---|
| M0 | ✅ 완료 (2026-07-03) 환경 셋업: conda env, robomimic·robosuite·mujoco 버전 고정, 데이터셋 다운로드(lift PH) | env 생성 + robomimic env 렌더 + HDF5 로드 smoke test 통과. 확정 버전을 environment.yml과 이 문서에 기록 |
| M1 | ✅ 완료 (2026-07-03) 데이터 파이프라인: HDF5 → 공통 batch (obs dict, action chunk, 정규화, To/Tp 시퀀스 샘플링) | shape·정규화 테스트 + 샘플 시각화로 육안 검증 |
| M2 | ✅ 완료 (2026-07-03) DP low-dim: lift PH state 기반 학습(50 epoch) + rollout 평가 + mjviewer 시각화 | success_rate=0.9(20 에피소드) — Chi et al. 공개 수치와의 정식 대조는 아직 [추정 재확인 필요] |
| M3 | DP image: vision encoder + lift/can/square PH image 학습 | 공개 수치(image) 대비 합리적 범위 — 미달 시 원인 분석 후 진행 여부 판단 |
| M4 | 정리: config만으로 태스크 교체, eval 스크립트·비디오 저장, README | 새 세션이 README만 보고 학습~평가 재실행 가능 |

- M2가 첫 번째 진짜 검증점: 여기서 수치가 안 나오면 M3로 넘어가지 말고 원인(데이터 파이프라인 vs 모델 vs 평가)을 분리해서 잡는다.

### M2 완료 [검증, 2026-07-03 실제 학습·평가 실행 확인]

**50 epoch 실제 학습 실행 → rollout 평가 success_rate=0.9 (20 에피소드 중 18개 성공).** 데이터→모델→학습→rollout 전체 파이프라인이 실제로 동작함을 확인. wandb 로깅 연결(`leeju0917` 계정, project=`mani_sim`) — run: https://wandb.ai/leeju0917-seoul-national-university-ofscience-and-technology/mani_sim/runs/chz119qv . Loss는 1.13(step 0) → 0.05 근처(epoch 10)로 빠르게 감소 후 0.03~0.07 사이 평탄화.

**M2 완료 기준("Chi et al. 공개 수치 재현") 관련**: 0.9라는 수치 자체는 이 태스크(Lift, robomimic 벤치마크 중 가장 쉬움)에서 높은 축에 속하지만, **공개 논문 수치와 직접 대조는 아직 안 함** [추정 — 남은 일]. 20 에피소드는 "빠른 검증"용 소규모 표본이라 정식 비교엔 더 많은 에피소드(예: 50)로 재확인 필요.

**시각화 (mjviewer)**: `envs/robomimic/factory.py`의 `make_lowdim_env(render=True, renderer="mjviewer")`로 MuJoCo 네이티브 패시브 뷰어(`mujoco.viewer.launch_passive`) 연결 확인. robomimic `EnvRobosuite`는 `render=True`일 때 내부적으로 `renderer="mujoco"`(OpenCV 창)를 강제 덮어쓰므로, `render=False`로 생성한 뒤 raw robosuite env(`env.env`)의 `has_renderer`/`renderer` 속성을 직접 설정해 우회. `scripts/live_rollout.py`(신규) — receding horizon rollout을 그대로 수행하며 매 step마다 `env.render()` 호출. 끝에 `env.env.close()` 필요(안 하면 종료 시 `GLXBadWindow`).
- 에피소드가 바뀔 때(`env.reset()`)마다 창이 닫혔다 다시 열리는 현상 확인 — robosuite `reset()`이 `renderer=="mjviewer"`일 때 `hard_reset` 설정과 무관하게 무조건 뷰어를 파괴하는 하드코딩(`# always terminate mjviewer` 주석) 때문. 공식 `collect_human_demonstrations.py`의 `collect_human_trajectory()`도 동일 패턴(매 데모마다 `env.reset(); env.render()`)이라 **로봇수트 표준 동작**으로 확인. 에피소드 내부는 창이 계속 유지되므로 **HITL 개입(Sirius/APO 방식)은 이 현상과 무관하게 구현 가능** — SpaceMouse/Keyboard teleop 장치는 렌더러 선택과 무관하게 별도 하드웨어 폴링으로 동작함(공식 스크립트로 확인, `mjgui` 모드만 mjviewer 전용).
- UI 패널(Simulation/Rendering 등) 없는 것도 정상 — `MjviewerRenderer`가 `show_left_ui=False, show_right_ui=False`로 명시적으로 끔.

이전 배선 확인 기록(아래)은 실제 학습 이전 단계 기록으로 보존.

**구현 위치**:
- `src/mani_sim/networks/conditional_unet1d.py` — 1D conditional U-Net (FiLM), Chi et al./LeRobot의 공개된 표준 아키텍처를 구조 참고해 자체 구현(flare 코드 import 없음).
- `src/mani_sim/policies/diffusion/diffusion_policy.py` — `DiffusionPolicyLowDim`. low_dim obs만 지원(vision encoder는 M3). obs_horizon(To) 프레임을 이어붙여 global conditioning 벡터로 사용. 학습 noise scheduler=DDPM, 추론 sampling=DDIM (diffusers).
- `src/mani_sim/envs/robomimic/factory.py` — `make_lowdim_env()`. robomimic import는 이 파일과 datasets 어댑터에만 존재(원칙 유지).
- `src/mani_sim/runners/rollout.py` — receding horizon(Ta<=Tp) closed-loop rollout. `env.is_success()['task']`로 성공 판정.
- `src/mani_sim/configs/{train,eval}.yaml` + `configs/{task,policy}/` — hydra 설정 분리. task=lift_low_dim, policy=diffusion_unet_lowdim.
- `src/mani_sim/scripts/{train,eval}.py` — hydra 진입점.

**검증한 것 (전부 실행해 확인)**:
1. 모델 forward/backward smoke test — 파라미터 16.6M, loss finite, 전 파라미터 gradient 흐름 확인, `predict_action_chunk` 출력 shape 정상.
2. hydra config 합성(`train.yaml`+`task/lift_low_dim`+`policy/diffusion_unet_lowdim`) 정상.
3. **배선 확인 (실제 학습 아님, batch 2개만)**: config → dataset → policy → optimizer → `loss.backward()` → `optimizer.step()` → checkpoint 저장 → 재로드까지 크래시 없이 동작.
4. **rollout 배선 확인 (2 에피소드, max_steps=20)**: env 생성 → obs 이력 유지 → `predict_action_chunk`(DDIM 5-step) → 정규화 해제 → env.step 반복 실행 → `is_success` 체크 → 최종 metrics dict 반환까지 크래시 없이 동작. `success_rate=0.0`은 예상대로(거의 학습 안 된 정책이므로 무의미한 수치, 배선 확인용).

**의도적으로 미확인 상태로 남긴 것 (실제 학습 규모 키울 때 확인 필요)**:
- `num_workers>0`에서 robomimic `SequenceDataset`의 내부 h5py 핸들이 멀티프로세스 워커로 안전하게 피클링되는지 — 현재 `num_workers=0`으로 고정해둠 (config에 이유 주석).
- `down_dims=(256,512)`는 U-Net 다운샘플 1회(stride 2) → `pred_horizon`은 짝수여야 함(현재 16, 문제없음). down_dims를 3단 이상으로 늘리면 `pred_horizon`이 그만큼 더 큰 2의 거듭제곱 배수여야 함 — 아직 실측 검증은 안 함.
- 실제 학습 시 하이퍼파라미터(lr, num_epochs, batch_size 등)는 Chi et al. 논문/공식 config 값과 대조 안 함 — 지금 값(`lr=1e-4, num_train_timesteps=100`)은 관행적 기본값이며 [추정].

### M0 확정 결과 [검증, 2026-07-03 실행 확인]

**conda 환경**: `mani_sim` (python 3.10). `requirements.txt`가 버전 단일 출처, `environment.yml`은 그걸 참조.

**핵심 버전 고정**:
- torch==2.5.1+cu121 / torchvision==0.20.1+cu121 (이 머신 GPU: RTX 4060 Laptop 8GB, driver 550.163.01/CUDA 12.4 — cu121 wheel 정상 동작 확인. `rise` conda env에서 이미 검증된 조합 재사용)
- robosuite==1.5.1 (robomimic 공식 문서가 released dataset 재현에 명시 권장하는 브랜치와 동일 버전)
- robomimic: PyPI 0.3.0 아닌 **git v0.4.0 태그** 고정. 이유: v0.4.0은 이미 v1.5 데이터셋 레지스트리(HF 호스팅)를 포함하면서도, master/v0.5.0에서 추가된 `lang_utils.py`(import 시점에 CLIP 텍스트 인코더 자동 다운로드 — dataset.py·env_robosuite.py가 둘 다 이걸 import해서 우회 불가)와 `diffusion_policy.py`(diffusers==0.11.1 요구, 우리 DP 구현과 충돌)가 없음. 실제 소스 대조로 확인(GitHub raw 파일 비교), 실행으로 재확인.
- **mujoco==3.2.3 고정 (핵심 발견)**: robosuite 1.5.1의 `controller.py`가 `mujoco.mj_fullM(model, dst, qM_sparse)`로 호출하는데, 이는 구 API. mujoco 3.10.0(당시 latest)은 `mj_fullM(model, data, dst)`로 시그니처가 바뀌어 `TypeError` 발생 — 실행해서 실제로 재현·확인. 3.2.3(robosuite 최소요구버전)으로 내리니 해결. **3.2.4~3.9.x 사이 정확히 어디서 바뀌었는지는 미확인** — 필요하면 나중에 좁혀볼 것.
  - 부작용: `mink`(GR1 로봇 whole-body IK용) 패키지가 mujoco>=3.8.1을 요구해 경고 뜸. Panda 로봇만 쓰는 한 무해.
- diffusers==0.30.3 + huggingface_hub 1.21.0(robomimic이 최신으로 끌어옴) 조합 — flare(manipulation_pipeline)가 이미 이 조합으로 실사용 중임을 확인해 신뢰. 실제 import·동작도 문제없음.

**환경 격리 이슈 (별도 발견, 고쳐둠)**:
- `~/.local/lib/python3.10/site-packages`에 구버전 diffusers(0.27.2)·einops(0.7.0)·numpy(1.26.4)·opencv(4.12.0)가 있었고, conda 환경의 user-site가 기본 활성화라 이게 conda env 패키지보다 **import 우선순위가 높았음** (재현성 붕괴 위험). `.bashrc`가 ROS2(`/opt/ros/humble`, `~/ros2_ws`)를 무조건 소싱해 PYTHONPATH도 전역 오염.
- 조치: `.bashrc`의 ROS2 source 2줄만 주석 처리(삭제 아님, 백업 `~/.bashrc.bak_20260703_092935`) + `mani_sim` conda 환경 전용 activate.d/deactivate.d 훅으로 `PYTHONNOUSERSITE=1` 설정·PYTHONPATH 격리. 다른 conda 환경(rise, openvla 등)엔 영향 없음.

**robomimic 사용 시 필수 초기화 (버그 아님, 표준 절차)**:
- env·dataset을 만들기 *전에* 반드시 `robomimic.utils.obs_utils.initialize_obs_utils_with_obs_specs({...})` 호출 필요 (어떤 obs 키가 low_dim/rgb/depth인지 등록). 안 하면 `TypeError: argument of type 'NoneType' is not iterable`.
- `EnvRobosuite.__init__`은 카메라 등 kwargs를 **robosuite 네이티브 이름 그대로** 전달함(자동 변환 없음) — `camera_heights`/`camera_widths`(복수형) 등 robosuite 1.5.1식 이름을 직접 써야 함. (`camera_height`/`camera_width` 단수형 변환은 `EnvRobosuite.create_for_data_processing` classmethod에만 있음 — 데이터셋 이미지 추출 전용 경로.)

**데이터셋 다운로드**: robomimic 자체 유틸(`file_utils.py`) 대신 `huggingface_hub.hf_hub_download(repo_id="robomimic/robomimic_datasets", repo_type="dataset", filename="v1.5/{task}/{ph|mh}/low_dim_v15.hdf5")`로 직접 받음 (v0.4.0의 file_utils는 안전하지만, 우리 스크립트를 robomimic 내부에 의존시키지 않기 위해). lift PH low_dim 다운로드·로드 확인 (200 demos).

**Smoke test 결과** (`MUJOCO_GL=egl` 필요 — 이 머신 DISPLAY=:0 + libEGL 확인됨):
- robosuite 단독: Lift/Panda env 생성 → reset → step → 정상. `agentview_image` (84,84,3) uint8 렌더 확인.
- robomimic `EnvRobosuite` 래퍼: 위 초기화 절차 포함 시 정상 동작.
- robomimic `SequenceDataset`: lift PH low_dim HDF5 로드, `seq_length=16` 시퀀스 배치 shape 정상 (`actions`: (16,7), `obs['robot0_eef_pos']`: (16,3)).

**§7 체크리스트 관련 실측치** (Sirius/APO 비교용 참고, 판단은 아직 안 함):
- lift PH 데이터셋 기록 시 env_kwargs: action space = **OSC_POSE** (delta pose, 6 pose + 1 gripper = 7-dim), `control_freq=20Hz`, `camera_heights/widths=84`. robosuite 1.5.1 실제 기록값이므로 이후 비교 실험 설정 맞출 때 이 값이 기준점.

### M1 확정 결과 [검증, 2026-07-03 실행 확인]

**구현 위치**: `src/mani_sim/datasets/normalization.py`(MinMax 통계·정규화), `src/mani_sim/datasets/robomimic_dataset.py`(robomimic 격리 어댑터). robomimic import는 이 두 파일에만 존재.

**robomimic `SequenceDataset`의 실제 시맨틱 (M0에서 가정했던 것과 다름, 실행해서 확인)**:
- `frame_stack`(=To)과 `seq_length`(=Tp)는 obs·action에 **동일한 하나의 윈도우**(길이 `To-1+Tp`)를 만든다. obs와 action이 별도 길이로 안 나뉘어 나옴.
- 윈도우는 timestep `[t-To+1, t+Tp-1]`을 담음 → **앞 To개 = 관측 이력, 뒤 Tp개 = 예측할 행동 청크**로 슬라이싱하는 건 호출자(우리) 책임. `RobomimicSequenceDataset.__getitem__`이 이 슬라이싱을 수행.
- `get_pad_mask=True`로 패딩 마스크를 받아 에피소드 경계 근처(짧은 residual)에서 반복-패딩된 action을 구분 가능하게 함 — 실제로 데모 길이(59) 근처 인덱스(56~58)에서 마스크가 정확히 False로 전환되는 것 확인.

**obs 키 구성**: `robomimic/exps/templates/bc.json`(공식 BC 예제 config)의 low_dim 키와 동일하게 확정 — `robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object`. M0에서 임의로 골랐던 값이 공식값과 일치함을 재확인.

**정규화**: MinMax → `[-1, 1]`, 공식 `(x-min)/(max-min+eps)*2-1` / 역변환 `(x+1)/2*(max-min)+min` (Chi et al. Diffusion Policy 표준 컨벤션. flare의 `NormalizeMinMax`가 동일 공식 사용하는 걸 구조 참고용으로만 대조, import는 안 함). 통계는 `compute_minmax_stats()`가 HDF5를 h5py로 직접 스캔해 계산(robomimic 유틸 안 거침).

**검증**: `tests/test_dataset.py` 4개 pytest 통과 (batch shape, 정규화 범위, normalize↔unnormalize 라운드트립, 에피소드 경계 패딩 마스크). `scripts/visualize_dataset_sample.py`로 obs/action 시퀀스 플롯 육안 확인(`outputs/m1_dataset_check/sample_traj.png`) — gripper로 추정되는 한 차원이 -1→1로 급전환(그립 이벤트), 나머지 pose 델타는 0 근처 진동 — 물리적으로 타당함.

**아직 안 한 것 (M2에서)**: image obs(camera) 경로는 M1에서 안 건드림 (low_dim만). train/valid split(`filter_by_attribute`)도 아직 안 씀 — robomimic 데이터셋에 기본으로 안 들어있어 M2에서 별도 마스크 생성 필요할 수 있음 [미확인].

## 7. 비교 정합성 체크리스트 (Sirius·APO와 수치 비교 대비)

> 이후 HITL·가중치 실험에서 레퍼런스와 비교하려면 실험 설정을 맞춰야 한다. 코드 작업 전 확인:

- [ ] Sirius가 쓴 태스크·데이터셋(robomimic 표준 태스크인지, 자체 수집인지) — 원문 `ljw_workspace/Data_Efficient_Improvement/papers/_txt/` + 공식 코드
- [ ] APO가 쓴 태스크·데이터셋·관찰 설정 — 상동
- [ ] 관찰 모달리티(image 구성: 카메라 뷰·해상도), action space(OSC pose vs joint), chunk 길이
- [ ] `research_design.md` 비교표의 "Sirius base model = Diffusion Policy [검증]" 재확인 — 원논문은 BC-RNN 기반이라는 상충 기억 있음(Claude, [추정]). Sirius 공식 코드로 확정할 것

## 8. 미결정 사항 (계획 시점에 의도적으로 보류)

- intervention 수집 방식 (teleop vs synthetic expert) — HITL 단계 진입 시 결정
- 두 번째 벤치마크 후보 (LIBERO 등) — 격리 원칙만 지키고 후보 선정은 보류
- 가중치 loss(Sirius weighted BC / APO KTO-style)의 구현 위치 — policy 내부 vs runner, 해당 단계에서 결정
