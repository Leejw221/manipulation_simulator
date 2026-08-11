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
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mani_sim.datasets.intervention_writer import read_env_args, read_intervention_hdf5, write_intervention_hdf5
from mani_sim.datasets.labels import LABEL_INTV, LABEL_PREINTV
from mani_sim.datasets.normalization import MinMaxNormalizer, compute_minmax_stats
from mani_sim.factory import registry
from mani_sim.runners.intervention_rollout import KeyboardIntervention, collect_episode
from mani_sim.runners.rollout import maybe_speed_up_inference
from mani_sim.utils.task_utils import is_image_task, is_piper_task, make_eval_env, task_lowdim_keys, task_obs_keys

logger = logging.getLogger(__name__)


def _is_piper_task(cfg):
    return is_piper_task(cfg.task)


def _demo_frame_count(cfg):
    """round_size_ratio 계산용 — 현재 task(task.filter_key가 있으면 그 부분집합)의 총
    프레임 수. moai_policy(실물 Piper) rollout_intervention.py의 동일 계산과 같은 개념."""
    if _is_piper_task(cfg):
        import zarr

        z = zarr.open(cfg.task.zarr_path, mode="r")
        return int(z["data"]["action"].shape[0])

    import h5py

    with h5py.File(cfg.task.hdf5_path, "r") as f:
        filter_key = cfg.task.get("filter_key", None)
        demo_names = (
            [d.decode() if isinstance(d, bytes) else d for d in f["mask"][filter_key][:]]
            if filter_key else list(f["data"].keys())
        )
        return sum(f["data"][d]["actions"].shape[0] for d in demo_names)


def _write_piper_lerobot(cfg, episodes, obs_keys):
    """Piper task 전용 저장 - robomimic HDF5(write_intervention_hdf5) 대신 LeRobotDataset
    (연구실 표준 포맷 - 2026-07-26 사용자 결정). obs_keys는 이미 collect_episode()가
    front_image/wrist_image/state 형태로 모아뒀으니 observation.images.<cam>/
    observation.state로 이름만 맞춰 옮긴다. finalize() 필수(실측 확인: 안 하면
    meta/episodes/*.parquet가 안 써져 다시 못 읽음, 2026-07-26 실측으로 발견한 버그)."""
    import os

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    camera_keys = [k[: -len("_image")] for k in obs_keys if k.endswith("_image")]
    lowdim_keys = [k for k in obs_keys if not k.endswith("_image")]
    action_dim = episodes[0]["actions"].shape[-1] if episodes else cfg.task.action_dim
    # PiperMujocoEnv(robot_mode=single) 규약: [joint1..6, gripper] - state_feature_names() 참고.
    joint_names = [f"joint{i}.pos" for i in range(1, 7)] + ["gripper.pos"]

    # 실측 확인(2026-07-26): output_root가 이미 있으면 create()는 exist_ok=False라
    # FileExistsError로 죽는다 - 같은 task로 여러 번 나눠 수집하는 게 정상 워크플로우라
    # (하루에 다 못 모음, round.py처럼 누적) 기존 데이터셋이면 resume()으로 이어서 추가한다.
    if os.path.isdir(cfg.output_root) and os.path.exists(os.path.join(cfg.output_root, "meta", "info.json")):
        dataset = LeRobotDataset.resume(repo_id=cfg.repo_id, root=cfg.output_root)
        logger.info(f"기존 데이터셋에 이어서 저장: {cfg.output_root}(기존 {dataset.num_episodes}개 에피소드)")
    else:
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

    maybe_speed_up_inference(policy, cfg.get("deploy_num_inference_steps", None))

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
    # env_seed(2026-08-11 추가): env reset(너트 초기 위치) 시퀀스를 고정한다. 여태 torch만
    # 고정하고 np.random은 안 걸어서 **매 실행마다 초기 조건이 달랐고**, 그래서 배포로 잰
    # 개입률을 eval(eval_seed의 고정 초기조건)의 성공률과 같은 상황에서 비교할 수 없었다.
    # 2026-08-10 팀미팅 교수님 지적("성공률이 올랐는데 개입률이 늘어날 수가 있나 / 초기
    # 세팅이나 seed나 다 같아야 되는데")이 바로 이 문제다.
    #   - 개입률 측정용 배포: eval의 eval_seed와 **같은 값**을 준다 -> 같은 초기조건 집합
    #   - 학습 데이터 수집: null로 두면 매번 다른 초기조건(데이터 다양성 확보)
    # eval.py의 eval_seed와 같은 구조(None이면 안 건다).
    if cfg.get("env_seed", None) is not None:
        np.random.seed(cfg.env_seed)
        logger.info(f"env_seed={cfg.env_seed} — env reset(초기 조건) 시퀀스 고정")

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
        from mani_sim.runners.piper import make_camera_toggle_fn, make_piper_intervention

        intervention = make_piper_intervention(cfg)
        # 카메라 라이브 프리뷰는 안 띄운다 - cv2 창이 MuJoCo GL 컨텍스트와 같이 뜨면 이 PC에서
        # 멈추는 기존 지뢰(eval.py에 2026-07-20 기록됨)를 그대로 재현함(2026-07-26 실측 재확인).
        # mujoco 온스크린 뷰어(아래)가 실시간 시각 피드백을 이미 주므로 카메라는 저장만 한다.
        if cfg.render:
            env.attach_viewer()
            render_fn = make_camera_toggle_fn(env, intervention)
    elif cfg.intervention_device == "pico":
        from mani_sim.runners.pico_intervention import PICOIntervention

        intervention = PICOIntervention(
            env.env,
            side_robot_index=dict(cfg.pico.side_robot_index) if cfg.pico.side_robot_index else None,
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
            # image task면 손목뷰(robot0_eye_in_hand_image)를 별도 cv2 창으로 동시에 띄우려
            # 했었는데(2026-07-18 요청) - cv2 라이브 창이 MuJoCo GL 컨텍스트와 같이 뜨면 이 PC
            # 에서 멈추고 SIGINT도 안 먹히는 지뢰(eval.py에 2026-07-20 기록, 2026-07-26 Transport
            # PICO 개입 테스트 중 재확인 - collect.py도 처음 밟은 것) - 고쳐질 때까지 꺼둔다.
            wrist_key = "robot0_eye_in_hand_image"
            show_wrist = False
            if show_wrist:
                import cv2

            def render_fn(obs_raw):
                a_btn = bool(intervention.xrt.get_A_button())
                if a_btn and not prev_a[0]:
                    print(f"\n[시점] {cycler.cycle()}")
                prev_a[0] = a_btn
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

    round_frame_threshold = None
    total_round_frames = 0
    total_intv_frames = 0
    if cfg.get("round_size_ratio", None):
        demo_frame_count = _demo_frame_count(cfg)
        round_frame_threshold = int(demo_frame_count * cfg.round_size_ratio)
        demo_ratio = 1.0 / (1.0 + cfg.round_size_ratio)
        print(
            f"[라운드 목표] {round_frame_threshold} 프레임 "
            f"({cfg.round_size_ratio}x {demo_frame_count} demo 프레임 -> 합친 데이터셋 demo 비율 {demo_ratio:.0%})"
        )

    episodes = []
    i = 0
    # 재개(resume) — output_path에 이전 수집분이 이미 있으면(같은 output_path로 재실행)
    # 그걸 먼저 불러와 이어서 모은다. Piper task는 LeRobotDataset.resume()으로 별도 처리되므로
    # (build_or_resume_dataset 경로) 여기선 robosuite(robomimic HDF5) task만 해당.
    if cfg.save and not _is_piper_task(cfg) and os.path.exists(cfg.output_path):
        episodes = read_intervention_hdf5(cfg.output_path, obs_keys)
        i = len(episodes)
        print(f"[재개] {cfg.output_path}에서 기존 에피소드 {i}개 로드")
        if round_frame_threshold is not None:
            for ep in episodes:
                total_round_frames += len(ep["action_modes"])
                total_intv_frames += int((ep["action_modes"] == LABEL_INTV).sum())
            print(
                f"  라운드 누적(재개분 포함): {total_round_frames}/{round_frame_threshold} 프레임 "
                f"(intv 누적 {total_intv_frames})"
            )

    try:
        while i < cfg.num_episodes:
            if round_frame_threshold is not None and total_round_frames >= round_frame_threshold:
                print(f"[라운드 종료] 재개 시점에 이미 목표 프레임 달성 ({total_round_frames} >= {round_frame_threshold})")
                break
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
                # robomimic은 이미 매 스텝(action_horizon=1) 재계획이라 비동기 추론이 의미
                # 없음 - cfg.async_infer와 무관하게 항상 끔.
                async_infer=cfg.get("async_infer", False) and policy_kind != "robomimic",
                merger_name=cfg.get("merger_name", "overwrite"),
                te_coeff=cfg.get("te_coeff", 0.01),
            )
            # should_delete_previous: 직전에 이미 저장된 에피소드까지 취소(PICOIntervention
            # 오른손 스틱 오른쪽, 2026-07-27) - 목록에서 pop하고 라운드 누적값도 되돌린다.
            # 없는 기기는 getattr 기본값으로 무시(기존 동작 유지).
            if getattr(intervention, "should_delete_previous", lambda: False)():
                if episodes:
                    removed = episodes.pop()
                    i -= 1
                    if round_frame_threshold is not None:
                        removed_modes = removed["action_modes"]
                        total_round_frames -= len(removed_modes)
                        total_intv_frames -= int((removed_modes == LABEL_INTV).sum())
                    print(f"ep {i}: 직전 저장 에피소드 삭제 요청 - 목록에서 제거")
                else:
                    print("삭제할 이전 에피소드가 없음(아직 저장된 게 없음)")

            # should_rerecord/should_stop은 PiperXRIntervention 전용(lerobot 관례: 스틱
            # 왼쪽=다시 녹화, end_button=세션 전체 중단, 2026-07-26) - 없는 기기는 getattr
            # 기본값으로 기존 동작(항상 저장하고 계속 진행) 그대로 유지.
            if getattr(intervention, "should_rerecord", lambda: False)():
                print(f"ep {i}: 다시 녹화 요청 - 버리고 재시도")
            else:
                modes = ep["action_modes"]
                print(
                    f"ep {i}: {len(modes)} frames · "
                    f"intv={int((modes == LABEL_INTV).sum())} "
                    f"preintv={int((modes == LABEL_PREINTV).sum())} "
                    f"success={ep['success']}"
                )
                episodes.append(ep)
                i += 1
                if round_frame_threshold is not None:
                    total_round_frames += len(modes)
                    total_intv_frames += int((modes == LABEL_INTV).sum())
                    print(
                        f"  라운드 누적: {total_round_frames}/{round_frame_threshold} 프레임 "
                        f"(intv 누적 {total_intv_frames})"
                    )
                    if total_round_frames >= round_frame_threshold:
                        print(f"[라운드 종료] 목표 프레임 달성 ({total_round_frames} >= {round_frame_threshold})")
                        break
            if cycler is not None and not cycler.is_running():
                print("뷰어 창이 닫혀 수집을 종료합니다.")
                break
            if getattr(intervention, "should_stop", lambda: False)():
                print("세션 종료 요청 - 남은 에피소드는 건너뜁니다.")
                break
    except KeyboardInterrupt:
        # Ctrl+C도 정상 종료처럼 처리 - 지금까지 모은 에피소드는 아래 저장 단계까지 그대로
        # 진행한다(그냥 raise하면 finally 이후 저장 코드에 도달 못 해 전부 유실됨). 다음에
        # 같은 output_path로 재실행하면 위 재개 로직이 이어서 수집한다.
        print("\n[중단] Ctrl+C 감지 - 지금까지 모은 에피소드 저장 후 종료합니다.")
    finally:
        intervention.close()
        if _is_piper_task(cfg):
            env.close_viewer()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass

    # 개입률 요약(2026-08-11 추가) — 교수님 지적(2026-08-10 팀미팅)대로 "같은 초기조건에서
    # 성공률 X, 개입률 Y"를 나란히 말하려면 이 수치가 필요하다. env_seed를 eval의 eval_seed와
    # 같은 값으로 주고 save=false로 돌리면, 데이터를 남기지 않고 개입률만 잴 수 있다.
    # 두 가지를 따로 낸다 — 어느 쪽을 쓸지는 주장에 따라 다르다:
    #   에피소드 개입률: 개입이 한 번이라도 필요했던 에피소드 비율 ("혼자 못 끝낸 비율")
    #   프레임 개입률  : 전체 프레임 중 사람이 조종한 비율 ("사람 노동량")
    if episodes:
        n_ep = len(episodes)
        ep_with_intv = sum(1 for ep in episodes if int((ep["action_modes"] == LABEL_INTV).sum()) > 0)
        n_frames = sum(len(ep["action_modes"]) for ep in episodes)
        n_intv = sum(int((ep["action_modes"] == LABEL_INTV).sum()) for ep in episodes)
        n_success = sum(1 for ep in episodes if ep.get("success"))
        intv_counts = sorted(int((ep["action_modes"] == LABEL_INTV).sum()) for ep in episodes)
        median_intv = intv_counts[len(intv_counts) // 2]
        print(
            "\n=== 개입률 요약 ===\n"
            f"  에피소드 {n_ep}개 · 성공 {n_success} ({n_success / n_ep:.1%})\n"
            f"  에피소드 개입률: {ep_with_intv}/{n_ep} ({ep_with_intv / n_ep:.1%})\n"
            f"  프레임 개입률  : {n_intv}/{n_frames} ({n_intv / max(n_frames, 1):.1%})\n"
            f"  에피소드당 개입 프레임 중앙값: {median_intv}\n"
            f"  env_seed={cfg.get('env_seed', None)} "
            f"({'eval과 같은 초기조건 집합' if cfg.get('env_seed', None) is not None else '⚠ 고정 안 됨 — eval 성공률과 나란히 비교 불가'})"
        )

    if not cfg.save:
        print(f"save=false - 저장 생략(정책+개입 동작 확인용) | 총 에피소드: {len(episodes)}")
    elif _is_piper_task(cfg):
        out = _write_piper_lerobot(cfg, episodes, obs_keys)
        print("저장:", out, "| 총 에피소드:", len(episodes))
    else:
        out = write_intervention_hdf5(cfg.output_path, episodes, obs_keys, read_env_args(cfg.task.hdf5_path))
        print("저장:", out, "| 총 에피소드:", len(episodes))


if __name__ == "__main__":
    main()
