"""사람 개입 rollout으로 개입 데이터셋 수집. 입력 장치는 키보드 또는 PICO VR.

사용:
    # PICO(기본) — 화면에서 실행, PICO 연결 필요
    python -m mani_sim.scripts.collect_intervention checkpoint_path=outputs/train/.../policy_epochN.pt

    # 키보드 개입으로 전환
    python -m mani_sim.scripts.collect_intervention intervention_device=keyboard

    # 정책 없이 조작·저장 경로만 먼저 테스트
    python -m mani_sim.scripts.collect_intervention num_episodes=1

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
from mani_sim.envs.robomimic.factory import make_lowdim_env
from mani_sim.policies.diffusion.diffusion_policy import DiffusionPolicyLowDim
from mani_sim.runners.intervention_rollout import KeyboardIntervention, collect_episode


def _build_diffusion_predict_fn(cfg, device, action_horizon):
    """기존 경로: 우리 자체 DiffusionPolicyLowDim + MinMaxNormalizer, 청크(action_horizon>1) 재사용."""
    from mani_sim.runners.intervention_rollout import _predict_chunk

    normalizer = MinMaxNormalizer(compute_minmax_stats(cfg.task.hdf5_path, cfg.task.obs_keys))
    policy = DiffusionPolicyLowDim(
        obs_keys=cfg.task.obs_keys,
        obs_dims=cfg.task.obs_dims,
        obs_horizon=cfg.policy.obs_horizon,
        action_dim=cfg.task.action_dim,
        pred_horizon=cfg.policy.pred_horizon,
        down_dims=cfg.policy.down_dims,
        kernel_size=cfg.policy.kernel_size,
        n_groups=cfg.policy.n_groups,
        diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
        num_train_timesteps=cfg.policy.num_train_timesteps,
        beta_schedule=cfg.policy.beta_schedule,
        num_inference_steps=cfg.policy.num_inference_steps,
    ).to(device)

    if cfg.checkpoint_path:
        ckpt = torch.load(cfg.checkpoint_path, map_location=device)
        policy.load_state_dict(ckpt["model"])
        print("loaded checkpoint:", cfg.checkpoint_path)
    else:
        print("no checkpoint — 무작위 초기화 정책 (조작/저장 경로 테스트용)")

    predict_fn = lambda history: _predict_chunk(policy, normalizer, history, cfg.task.obs_keys, device)
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
        env = make_lowdim_env(cfg.task.env_name, cfg.task.robots, cfg.task.obs_keys, render=cfg.render)
        obs_keys = list(cfg.task.obs_keys)
        policy, predict_fn = _build_diffusion_predict_fn(cfg, device, cfg.policy.action_horizon)
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

            def render_fn():
                a_btn = bool(intervention.xrt.get_A_button())
                if a_btn and not prev_a[0]:
                    print(f"\n[시점] {cycler.cycle()}")
                prev_a[0] = a_btn
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

    out = write_intervention_hdf5(cfg.output_path, episodes, obs_keys)
    print("저장:", out, "| 총 에피소드:", len(episodes))


if __name__ == "__main__":
    main()
