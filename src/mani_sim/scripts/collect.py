"""사람 개입 rollout으로 개입 데이터셋 수집. 입력 장치는 키보드 또는 PICO VR.
(구 collect_intervention.py — train.py/eval.py와 동사 기준 이름 통일을 위해 리네임.)

사용:
    # PICO(기본) — 화면에서 실행, PICO 연결 필요
    python -m mani_sim.scripts.collect checkpoint_path=outputs/train/.../policy_epochN.pt

    # 키보드 개입으로 전환
    python -m mani_sim.scripts.collect intervention_device=keyboard

    # 정책 없이 조작·저장 경로만 먼저 테스트
    python -m mani_sim.scripts.collect num_episodes=1

조작(PICO): B=개입 on/off · grip=클러치(잡은 동안만 팔 이동) · trigger=그리퍼 ·
            A=시점 전환(정면→각진→top→측면 순환, 깊이 판단용) · Y=에피소드 종료.
조작(키보드): toggle_key(기본 Ctrl)=개입 on/off · 화살표/회전키=이동 · space=그리퍼 · Enter=종료.
저장물은 robomimic 형식 HDF5(+action_mode)라 학습에 바로 쓴다.
"""

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from mani_sim.datasets.intervention_writer import write_intervention_hdf5
from mani_sim.datasets.labels import LABEL_INTV, LABEL_PREINTV
from mani_sim.datasets.normalization import MinMaxNormalizer, compute_minmax_stats
from mani_sim.envs.robomimic.factory import make_image_env, make_lowdim_env
from mani_sim.factory import registry
from mani_sim.runners.intervention_rollout import KeyboardIntervention, collect_episode
from mani_sim.utils.task_utils import is_image_task, task_lowdim_keys, task_obs_keys


def _build_policy_predict_fn(cfg, device, action_horizon):
    """registry 기반(diffusion_lowdim/diffusion(image)/bc_rnn_lowdim 등 cfg.policy_name 아무거나)
    + MinMaxNormalizer, 청크(action_horizon>1) 재사용. (구 _build_diffusion_predict_fn —
    DiffusionPolicyLowDim 하드코딩을 registry.create_policy로 일반화 — round.py가
    policy_name=bc_rnn_lowdim으로도 라운드를 돌릴 수 있어야 해서, 2026-07-19.)

    checkpoint_path=None + hdf5_path도 아직 없으면(새 task의 첫 수집, round0) 신경망 자체를
    안 만들고 제자리(zero-action) stub을 쓴다 - 어차피 PICO 개입이 매 스텝 override하니
    무작위 초기화 정책을 굳이 돌려 로봇이 개입 전에 제멋대로 움직이게 둘 이유가 없다.
    """
    import os

    from mani_sim.runners.intervention_rollout import _predict_chunk

    if cfg.checkpoint_path is None and not os.path.exists(cfg.task.hdf5_path):
        print(f"hdf5 없음({cfg.task.hdf5_path}) + checkpoint 없음 -> 정책 없이 제자리(zero-action) "
              "stub 사용 (round0 첫 수집 - 개입 켜기 전엔 로봇이 가만히 있음)")
        zero_chunk = np.zeros((action_horizon, cfg.task.action_dim), dtype=np.float32)
        return None, lambda history: zero_chunk

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
        gripper_types = cfg.task.get("gripper_types", None)
        env_kwargs = {}
        if "outside_color" in cfg.task:
            env_kwargs["outside_color"] = cfg.task.outside_color
        if is_image_task(cfg.task):
            # render=True(사람이 보는 mjviewer 창)와 오프스크린 image obs 렌더는 서로 다른 GL
            # 컨텍스트 요구라 함께 못 씀(mani_sim의 기존 landmine) — PICO 수집 중엔 사람이
            # 화면을 봐야 하므로 render=cfg.render 그대로 두고, image obs는 make_image_env가
            # 알아서 오프스크린으로 렌더한다(EnvRobosuite가 내부적으로 둘 다 처리).
            env = make_image_env(
                cfg.task.env_name, cfg.task.robots,
                list(cfg.task.lowdim_keys), list(cfg.task.rgb_keys),
                list(cfg.task.camera_names), image_size=cfg.task.image_size,
                gripper_types=gripper_types, env_kwargs=env_kwargs,
            )
            if cfg.render:
                env.env.has_renderer = True
                env.env.renderer = "mjviewer"
            obs_keys = task_obs_keys(cfg.task)
        else:
            env = make_lowdim_env(
                cfg.task.env_name, cfg.task.robots, cfg.task.obs_keys, render=cfg.render,
                gripper_types=gripper_types, env_kwargs=env_kwargs,
            )
            obs_keys = list(cfg.task.obs_keys)
        policy, predict_fn = _build_policy_predict_fn(cfg, device, cfg.policy.action_horizon)
        action_horizon = cfg.policy.action_horizon

    cycler = None
    render_fn = None
    if cfg.intervention_device == "pico":
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
                render=cfg.render,
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
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass

    out = write_intervention_hdf5(cfg.output_path, episodes, obs_keys)
    print("저장:", out, "| 총 에피소드:", len(episodes))


if __name__ == "__main__":
    main()
