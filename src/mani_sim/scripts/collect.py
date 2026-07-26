"""사람 개입 rollout으로 개입 데이터셋 수집. 입력 장치는 키보드 또는 PICO VR(robosuite
task) / 키보드+Placo IK(Piper task, env_backend=piper_mujoco로 task.yaml이 결정).
(구 collect_intervention.py — train.py/eval.py와 동사 기준 이름 통일을 위해 리네임.)

사용:
    # PICO(기본, robosuite task) — 화면에서 실행, PICO 연결 필요
    python -m mani_sim.scripts.collect checkpoint_path=outputs/train/.../policy_epochN.pt

    # 키보드 개입으로 전환(robosuite task)
    python -m mani_sim.scripts.collect intervention_device=keyboard

    # Piper task(env_backend=piper_mujoco) — piper_collect conda env에서 실행 필요
    # (flare/lerobot/placo 의존, mani_sim 기본 env엔 없음). 저장 포맷도 이 task만 lerobot
    # (LeRobotDataset)로 다르다 - 연구실 표준 포맷과 맞추기 위한 선택(2026-07-26).
    <piper_collect>/bin/python -m mani_sim.scripts.collect task=piper_sort_return

    # 정책 없이 조작·저장 경로만 먼저 테스트
    python -m mani_sim.scripts.collect num_episodes=1

조작(PICO): B=개입 on/off · grip=클러치(잡은 동안만 팔 이동) · trigger=그리퍼 ·
            A=시점 전환(정면→각진→top→측면 순환, 깊이 판단용) · Y=에피소드 종료.
조작(키보드, robosuite): toggle_key(기본 Ctrl)=개입 on/off · 화살표/회전키=이동 ·
            space=그리퍼 · Enter=종료.
조작(키보드, Piper): wasdqe=이동 uoikjl=회전 f=그리퍼 토글 Enter=종료(항상 사람이 조작,
            토글 없음 - round0 수집이 주 용도라서, PiperKeyboardIntervention 참고).
저장물: robosuite task는 robomimic 형식 HDF5(+action_mode), Piper task는 LeRobotDataset
(parquet+video) - 학습 시 convert_lerobot_to_zarr.py로 zarr 변환 필요.
"""

import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mani_sim.datasets.intervention_writer import write_intervention_hdf5
from mani_sim.datasets.labels import LABEL_INTV, LABEL_PREINTV
from mani_sim.datasets.normalization import MinMaxNormalizer, compute_minmax_stats
from mani_sim.factory import registry
from mani_sim.runners.intervention_rollout import KeyboardIntervention, collect_episode
from mani_sim.utils.task_utils import is_image_task, is_piper_task, make_eval_env, task_lowdim_keys, task_obs_keys

logger = logging.getLogger(__name__)


def _is_piper_task(cfg):
    return is_piper_task(cfg.task)


def _attach_piper_mujoco_viewer(env):
    """자유 시점 3D 씬 뷰어(mujoco 온스크린, GLFW) - eval 때 보던 것과 같은 종류.
    PiperMujocoEnv.apply_action()이 매 스텝 알아서 self._sync_viewer()를 호출하므로
    (원본 코드, 손 안 댐) env._env.viewer에 핸들만 꽂아두면 그 뒤론 자동 갱신된다.
    아래 rerun(카메라+joint 대시보드)과 별개 용도라 둘 다 띄운다(2026-07-26, 사용자 요청 -
    rerun만으로는 "eval 때 보던 자유 시점 3D 뷰"가 없어서 부족하다는 피드백).

    ⚠ 실측 확인(2026-07-26, 실제 랩 PC "moai-pobi"): 이 GLFW 창 생성이 이 머신에서 GLX
    에러로 실패함(개발 중 의심했던 "원격 샌드박스라 그럴 것"이라는 추정이 틀렸음 - 실제
    PC에서도 동일 에러, 근본 원인은 머신 전체의 GLX 드라이버 문제로 보임). rerun(Vulkan)은
    같은 머신에서 정상 동작 확인됨. 그래서 실패해도 전체 수집이 죽지 않게 try/except로
    감싸고 경고만 남긴다 - 고쳐질 때까지 rerun만으로 시각화."""
    import mujoco.viewer

    try:
        env._env.viewer = mujoco.viewer.launch_passive(env._env.model, env._env.data)
    except Exception as e:
        logger.warning(
            f"mujoco 온스크린 뷰어 생성 실패({type(e).__name__}: {e}) - GLX 드라이버 문제로 "
            "추정됨(실측: moai-pobi에서도 재현). rerun 시각화만으로 계속 진행합니다."
        )


def _make_piper_render_fn():
    """카메라+joint 값을 보여주는 대시보드 - lerobot 자체 시각화(rerun, `--display_data=true`
    가 record.py에서 쓰는 것과 동일 저수준 함수)를 재사용한다(2026-07-26). 실물 하드웨어
    텔레옵과 같은 방식이라 더 일관되고, rerun은 완전히 별도 프로세스/창이라 front_cam/
    wrist_cam 오프스크린 렌더와 GL 컨텍스트를 공유하지 않는다 - 위 mujoco 뷰어와 같이 띄워도
    서로 안 부딪힌다.

    collect_episode()의 render_fn(obs_raw) 콜백 하나만 채워주면 되고(action은 이 콜백엔
    안 넘어와서 로그 안 함 - 관측만으로도 조작엔 충분, 필요해지면 나중에 추가), 에피소드
    흐름 제어(Enter=종료)는 이미 PiperKeyboardIntervention이 담당하므로 항상 True 반환."""
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    init_rerun(session_name="piper_collect")

    def render_fn(obs_raw):
        log_rerun_data(observation=obs_raw)
        return True

    return render_fn


def _write_piper_lerobot(cfg, episodes, obs_keys):
    """Piper task 전용 저장 - robomimic HDF5(write_intervention_hdf5) 대신 LeRobotDataset
    (연구실 표준 포맷 - 2026-07-26 사용자 결정). obs_keys는 이미 collect_episode()가
    front_image/wrist_image/state 형태로 모아뒀으니 observation.images.<cam>/
    observation.state로 이름만 맞춰 옮긴다. finalize() 필수(실측 확인: 안 하면
    meta/episodes/*.parquet가 안 써져 다시 못 읽음, 2026-07-26 실측으로 발견한 버그)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    camera_keys = [k[: -len("_image")] for k in obs_keys if k.endswith("_image")]
    lowdim_keys = [k for k in obs_keys if not k.endswith("_image")]
    action_dim = episodes[0]["actions"].shape[-1] if episodes else cfg.task.action_dim
    # PiperMujocoEnv(robot_mode=single) 규약: [joint1..6, gripper] - state_feature_names() 참고.
    joint_names = [f"joint{i}.pos" for i in range(1, 7)] + ["gripper.pos"]

    features = {
        "action": {"dtype": "float32", "shape": (action_dim,), "names": joint_names},
        "observation.state": {"dtype": "float32", "shape": (action_dim,), "names": joint_names},
    }
    for key in camera_keys:
        sample_shape = episodes[0]["obs"][0][f"{key}_image"].shape if episodes else (84, 84, 3)
        features[f"observation.images.{key}"] = {
            "dtype": "video", "shape": tuple(sample_shape), "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id, fps=cfg.control_fps, root=cfg.output_root,
        robot_type="piper_single_mujoco", features=features, use_videos=True,
    )
    try:
        for ep in episodes:
            for t in range(len(ep["actions"])):
                frame = {
                    "action": ep["actions"][t].astype("float32"),
                    "observation.state": ep["obs"][t][lowdim_keys[0]].astype("float32"),
                    "task": cfg.task_description,
                }
                for key in camera_keys:
                    # collect_episode()가 obs_seq를 전부 np.float32로 캐스팅해서(intervention_rollout.py)
                    # 이미지도 [0,255] 범위의 float32가 돼 있음(uint8 아님) - lerobot의 이미지 writer는
                    # uint8 [0,255] 또는 float [0,1]만 받아서, 여기서 다시 uint8로 되돌려야 함
                    # (실측 확인: 2026-07-26, 안 하면 PNG 저장 단계에서 조용히 실패 후 나중에
                    # save_episode()에서 FileNotFoundError로 터짐).
                    img = np.clip(ep["obs"][t][f"{key}_image"], 0, 255).astype(np.uint8)
                    frame[f"observation.images.{key}"] = img
                dataset.add_frame(frame)
            dataset.save_episode()
    finally:
        dataset.finalize()
    return cfg.output_root


def _build_policy_predict_fn(cfg, device, action_horizon):
    """registry 기반(diffusion_lowdim/diffusion(image)/bc_rnn_lowdim 등 cfg.policy_name 아무거나)
    + MinMaxNormalizer, 청크(action_horizon>1) 재사용. (구 _build_diffusion_predict_fn —
    DiffusionPolicyLowDim 하드코딩을 registry.create_policy로 일반화 — round.py가
    policy_name=bc_rnn_lowdim으로도 라운드를 돌릴 수 있어야 해서, 2026-07-19.)

    checkpoint_path=None + 데이터(hdf5 또는 Piper의 zarr)도 아직 없으면(새 task의 첫 수집,
    round0) 신경망 자체를 안 만들고 제자리(zero-action) stub을 쓴다 - 어차피 개입이 매 스텝
    override하니 무작위 초기화 정책을 굳이 돌려 로봇이 개입 전에 제멋대로 움직이게 둘 이유가
    없다.
    """
    import os

    from mani_sim.runners.intervention_rollout import _predict_chunk

    data_path = cfg.task.zarr_path if _is_piper_task(cfg) else cfg.task.hdf5_path
    if cfg.checkpoint_path is None and not os.path.exists(data_path):
        print(f"데이터 없음({data_path}) + checkpoint 없음 -> 정책 없이 제자리(zero-action) "
              "stub 사용 (round0 첫 수집 - 개입 켜기 전엔 로봇이 가만히 있음)")
        zero_chunk = np.zeros((action_horizon, cfg.task.action_dim), dtype=np.float32)
        return None, lambda history: zero_chunk

    if _is_piper_task(cfg):
        from mani_sim.datasets.normalization import compute_minmax_stats_zarr

        stats = compute_minmax_stats_zarr(cfg.task.zarr_path, task_lowdim_keys(cfg.task))
    else:
        stats = compute_minmax_stats(cfg.task.hdf5_path, task_lowdim_keys(cfg.task))
    normalizer = MinMaxNormalizer(stats)
    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy).to(device)

    if cfg.checkpoint_path:
        ckpt = torch.load(cfg.checkpoint_path, map_location=device)
        policy.load_state_dict(ckpt["model"])
        print("loaded checkpoint:", cfg.checkpoint_path)
    else:
        print("no checkpoint — 무작위 초기화 정책 (조작/저장 경로 테스트용)")

    obs_keys = task_obs_keys(cfg.task)
    rgb_keys = cfg.task.rgb_keys if is_image_task(cfg.task) else ()
    predict_fn = lambda history: _predict_chunk(policy, normalizer, history, obs_keys, device, rgb_keys=rgb_keys)
    return policy, predict_fn


def _build_robomimic_policy_env(cfg, device):
    """SIRIUS/APO 배포 라운드용: robomimic 체크포인트(BC_RNN_GMM 계열, 우리 걸로 학습된
    baseline/RA-BC/SIRIUS/APO 전부 이 경로 — low_dim이든 image든 동일) — 자체 정규화·
    RNN 은닉상태를 갖고 있어 매 스텝 재계획(action_horizon=1)한다. algo_name이
    "bc_rabc"/"bc_sirius"/"bc_apo"면 로드 전에 해당 train_robomimic_*.py를 import해
    REGISTERED_CONFIGS/ALGO_REGISTRY에 등록해야 한다(eval_checkpoint.py와 동일한 이유) —
    여기서는 셋 다 미리 import해 둔다.

    env도 체크포인트 메타데이터로 직접 만든다(`FileUtils.env_from_checkpoint`) — task
    yaml의 obs_keys를 참고하지 않는다. 체크포인트가 image를 쓰면 robomimic이 자동으로
    render_offscreen=True·카메라 설정을 맞춰준다(env_from_checkpoint 자체 로직) — 우리가
    카메라 이름·해상도를 따로 지정할 필요 없음. 화면(사람이 보고 개입) 표시는 이후
    env.env.has_renderer=True로 별도 켠다(make_lowdim_env와 같은 트릭).
    """
    import robomimic.utils.file_utils as RMFileUtils

    import train  # noqa: F401  ("bc_combined" 등록 — SIRIUS/APO 축 조합, 2026-07-16(4) 통합)
    import train_robomimic_rabc  # noqa: F401  ("bc_rabc" 등록)

    policy, ckpt_dict = RMFileUtils.policy_from_checkpoint(
        ckpt_path=cfg.checkpoint_path, device=device, verbose=False
    )
    env, _ = RMFileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, env_name=None, render=False, render_offscreen=False, verbose=False
    )
    if cfg.render:
        env.env.has_renderer = True
        env.env.renderer = "mjviewer"

    obs_keys = list(policy.policy.global_config.all_obs_keys)
    policy.start_episode()
    print("loaded robomimic checkpoint:", cfg.checkpoint_path, "| obs_keys:", obs_keys)

    def predict_fn(obs_history):
        ob = obs_history[-1]  # robomimic 정책은 자체 RNN 은닉상태로 이력 관리 — 최신 관측 1개만
        action = policy(ob=ob)  # (Da,) ndarray, 이미 unnormalize 완료
        return action[None, :]  # (T=1, Da) — collect_episode의 청크 인터페이스에 맞춤

    return policy, env, predict_fn, obs_keys


@hydra.main(config_path="../configs", config_name="collect", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    policy_kind = cfg.get("policy_kind", "diffusion")
    if policy_kind == "robomimic":
        policy, env, predict_fn, obs_keys = _build_robomimic_policy_env(cfg, device)
        action_horizon = 1  # robomimic 정책은 매 스텝 재계획(청크 없음)
    else:
        # 2026-07-25 수정: 예전엔 여기서 env_kwargs를 빈 dict로 새로 만들어서 cfg.task.env_kwargs
        # (env_configuration 등, hdf5에서 derive됨)를 아예 무시했음 — eval.py/make_eval_env는
        # 이 필드를 읽는데 collect.py만 따로 놀았던 버그. Transport처럼 env_kwargs가 실제로
        # 의미 있는 task로 개입 라운드를 수집하면 로봇이 학습 데이터와 다른 위치에 스폰됨
        # (7/21 학습 쪽 사고와 같은 종류, 지금까지 door_cabinet(단일팔)만 써서 안 드러났을 뿐).
        env_kwargs = OmegaConf.to_container(cfg.task.env_kwargs, resolve=True) if cfg.task.get("env_kwargs") else {}
        if "outside_color" in cfg.task:
            env_kwargs["outside_color"] = cfg.task.outside_color
        if is_image_task(cfg.task):
            # render=True(사람이 보는 mjviewer 창)와 오프스크린 image obs 렌더는 서로 다른 GL
            # 컨텍스트 요구라 함께 못 씀(mani_sim의 기존 landmine) — PICO 수집 중엔 사람이
            # 화면을 봐야 하므로 render=cfg.render 그대로 두고, image obs는 make_image_env가
            # 알아서 오프스크린으로 렌더한다(EnvRobosuite가 내부적으로 둘 다 처리). make_eval_env는
            # image+render를 여기서 처리 안 하므로(문서화된 지뢰) 생성 후 그대로 직접 패치한다.
            env = make_eval_env(cfg.task, env_kwargs_override=env_kwargs)
            if cfg.render and not _is_piper_task(cfg):
                env.env.has_renderer = True
                env.env.renderer = "mjviewer"
            # Piper의 시각화(render=true)는 mjviewer가 아니라 rerun(아래 intervention
            # 생성부에서 render_fn으로 연결) - env 생성 시점엔 할 게 없음.
            obs_keys = task_obs_keys(cfg.task)
        else:
            env = make_eval_env(cfg.task, render=cfg.render, env_kwargs_override=env_kwargs)
            obs_keys = list(cfg.task.obs_keys)
        policy, predict_fn = _build_policy_predict_fn(cfg, device, cfg.policy.action_horizon)
        action_horizon = cfg.policy.action_horizon

    cycler = None
    render_fn = None
    if _is_piper_task(cfg):
        from mani_sim.runners.piper_intervention import PiperKeyboardIntervention

        intervention = PiperKeyboardIntervention(control_fps=cfg.control_fps)
        if cfg.render:
            _attach_piper_mujoco_viewer(env)
            render_fn = _make_piper_render_fn()
    elif cfg.intervention_device == "pico":
        from mani_sim.runners.pico_intervention import PICOIntervention

        intervention = PICOIntervention(
            env.env,
            side=cfg.pico.side,
            pos_scale=cfg.pico.pos_scale,
            rot_scale=cfg.pico.rot_scale,
            gripper_sign=cfg.pico.gripper_sign,
            toggle_button=cfg.pico.toggle_button,
            end_button=cfg.pico.end_button,
            R_headset_world=cfg.pico.R_headset_world,
            grip_threshold=cfg.pico.grip_threshold,
            control_hz=cfg.control_fps,
            euro_min_cutoff=cfg.pico.euro_min_cutoff,
            euro_beta=cfg.pico.euro_beta,
            euro_d_cutoff=cfg.pico.euro_d_cutoff,
        )
        print(
            f"[개입/PICO] {cfg.pico.toggle_button}=개입 on/off · grip=클러치(잡은 동안만 이동) · "
            f"trigger=그리퍼 · A=시점전환 · {cfg.pico.end_button}=에피소드 종료 (성공/종료까지 진행)"
        )

        if cfg.render:
            from mani_sim.runners.sim_viewer import CameraCycler

            cycler = CameraCycler(env)
            prev_a = [False]
            # image task면 손목뷰(robot0_eye_in_hand_image)를 별도 cv2 창으로 동시에 띄운다 -
            # 사람이 조작하면서 "정책이 실제로 볼 화면"을 실시간으로 같이 확인할 수 있게
            # (2026-07-18 요청: 제3자뷰 + 손목뷰 동시 표시, 제3자뷰는 기존처럼 A로 시점전환).
            wrist_key = "robot0_eye_in_hand_image"
            show_wrist = is_image_task(cfg.task) and wrist_key in cfg.task.rgb_keys
            if show_wrist:
                import cv2

            # 특정 지점의 정확한 world 좌표가 필요할 때(예: "여기가 목표 지점이어야 한다") -
            # 그리퍼를 그 자리로 가져가면 1초에 한 번 eef_pos를 콘솔에 찍는다. 스크린샷으로
            # 좌표를 추측하는 게 계속 안 맞아서 추가함(2026-07-18).
            step_counter = [0]

            def render_fn(obs_raw):
                a_btn = bool(intervention.xrt.get_A_button())
                if a_btn and not prev_a[0]:
                    print(f"\n[시점] {cycler.cycle()}")
                prev_a[0] = a_btn
                step_counter[0] += 1
                if step_counter[0] % cfg.control_fps == 0:
                    print(f"[eef_pos] {obs_raw['robot0_eef_pos']}")
                if show_wrist:
                    img_hwc = np.transpose(np.asarray(obs_raw[wrist_key]), (1, 2, 0))
                    img_bgr = cv2.cvtColor(
                        np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                    )
                    # 저장(학습용)은 84x84 그대로, 화면 표시만 확대(사람이 보기 편하게).
                    img_bgr = cv2.resize(img_bgr, (420, 420), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("wrist view (robot0_eye_in_hand)", img_bgr)
                    cv2.waitKey(1)
                return cycler.render()
    else:
        intervention = KeyboardIntervention(env.env, toggle_key=cfg.toggle_key)
        print(
            f"[개입/키보드] {cfg.toggle_key}=개입 on/off · 화살표/회전키=이동 · space=그리퍼 · "
            "Enter=에피소드 종료 (max_steps 없이 성공/종료까지 진행)"
        )

    episodes = []
    try:
        for i in range(cfg.num_episodes):
            intervention.reset()
            if policy_kind == "robomimic":
                policy.start_episode()  # 매 에피소드 RNN 은닉상태 리셋
            ep = collect_episode(
                env,
                policy,
                None,  # normalizer: predict_fn이 자체 처리(robomimic) 또는 내부 클로저에 이미 바인딩(diffusion)
                obs_keys,
                cfg.policy.obs_horizon,
                action_horizon,
                device,
                intervention,
                max_steps=cfg.max_steps,
                should_end_fn=intervention.should_end,
                preintv_length=cfg.preintv_length,
                # Piper는 render_fn(rerun 로깅, _make_piper_render_fn)이 시각화를 담당하므로
                # 여기서 env.render(mode="human")를 또 호출할 필요 없음(불필요한 오프스크린
                # 렌더 인스턴스 하나가 더 생기는 걸 막음).
                render=cfg.render and not _is_piper_task(cfg),
                render_fn=render_fn,
                control_fps=cfg.control_fps,
                predict_fn=predict_fn,
            )
            modes = ep["action_modes"]
            print(
                f"ep {i}: {len(modes)} frames · "
                f"intv={int((modes == LABEL_INTV).sum())} "
                f"preintv={int((modes == LABEL_PREINTV).sum())} "
                f"success={ep['success']}"
            )
            episodes.append(ep)
            if cycler is not None and not cycler.is_running():
                print("뷰어 창이 닫혀 수집을 종료합니다.")
                break
    finally:
        intervention.close()
        if _is_piper_task(cfg) and env._env.viewer is not None:
            env._env.viewer.close()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass

    if _is_piper_task(cfg):
        out = _write_piper_lerobot(cfg, episodes, obs_keys)
    else:
        out = write_intervention_hdf5(cfg.output_path, episodes, obs_keys)
    print("저장:", out, "| 총 에피소드:", len(episodes))


if __name__ == "__main__":
    main()
