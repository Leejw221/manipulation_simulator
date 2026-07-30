"""통합 rollout 평가(DP: low_dim·image 공용) — outputs/final_eval.py를 대체.
render=true면 화면에 띄운다 — low_dim/image 둘 다 MuJoCo 자체 온스크린 뷰어(mjviewer)를
쓴다(2026-07-27부터, collect.py와 동일 기법: 오프스크린 렌더 env에 `env.env.has_renderer=True`
로 뷰어를 얹음). image task는 한때 cv2 라이브 창으로 그리려다 이 PC에서 오프스크린 GL
컨텍스트와 충돌해 막아뒀었는데(live_rollout.py 지뢰), collect.py가 cv2 없이 실사용 검증한
방식으로 교체해 해소함 — Piper task(env_backend=piper_mujoco)만 예외(뷰어 방식이 rerun/
attach_viewer로 달라 여기 대상 아님, save_gif로 대체). OpenVLA는 predict_action_chunk가
아니라 매 스텝 단일 이미지 재계획이라 인터페이스가 달라 policy_name=="openvla"일 때 별도
루프로 분기한다.

사용:
    python -m mani_sim.scripts.eval checkpoint_path=outputs/train/square_diffusion_unet/policy_epoch300.pt
    python -m mani_sim.scripts.eval task=square_stage checkpoint_path=... render=true
    unset MUJOCO_GL  # low_dim 화면(mjviewer)일 때만 필요
    python -m mani_sim.scripts.eval task=square policy_name=openvla policy=openvla \
        checkpoint_path=outputs/train/square_openvla/lora_final/policy \
        stats_path=outputs/train/square_openvla/normalization_stats.json save_gif=outputs/openvla_rollout.gif
"""

import json
import logging
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.factory import registry
from mani_sim.runners.rollout import maybe_speed_up_inference, rollout_policy, rollout_policy_async
from mani_sim.utils.checkpoints import load_run_config
from mani_sim.utils.task_utils import is_image_task, is_piper_task, make_eval_env, task_obs_keys

logger = logging.getLogger(__name__)


def _patch_mjviewer_esc_quit():
    """robosuite의 MjviewerRenderer.update()는 mujoco.viewer.launch_passive()에 key_callback을
    안 넘겨서, 이 렌더러로는 ESC로 창을 닫을 방법이 없다(add_keypress_callback도 이 렌더러에선
    저장만 하고 아무데도 안 불림 - opencv_renderer 전용으로만 실제 연결돼 있음, 2026-07-29
    확인). site-packages는 안 건드리고 mani_sim 쪽에서 update()만 ESC 콜백 추가해 패치한다.

    key_callback은 launch_passive가 띄운 렌더 스레드에서 불린다 - 처음엔 거기서 바로
    self.viewer.close()(=GL 윈도우 파괴)를 했는데, 다른 스레드가 만든 GL 컨텍스트를 그
    스레드가 아닌 곳에서 부르는 거라 창이 멈춘 채 안 닫히는 증상으로 이어졌다(2026-07-29
    실측: ESC 누르면 렌더는 멈추는데 프로세스가 안 끝남). 그래서 콜백은 플래그만 세우고,
    실제 close()는 메인 스레드(_mjviewer_is_running이 감지한 뒤, rollout_policy가 루프를
    끝내고 나서 _run_dp_eval의 env.env.close())가 하도록 옮겼다."""
    from mujoco import viewer as mj_viewer
    from robosuite.renderers.mjviewer.mjviewer_renderer import MjviewerRenderer

    if getattr(MjviewerRenderer, "_esc_patched", False):
        return

    def _update_with_esc(self):
        if self.viewer is None:
            self._esc_pressed = False
            self._enter_pressed = False

            def _key_callback(keycode):
                if keycode == 256:  # GLFW_KEY_ESCAPE
                    self._esc_pressed = True
                elif keycode in (257, 335):  # GLFW_KEY_ENTER, GLFW_KEY_KP_ENTER
                    self._enter_pressed = True

            self.viewer = mj_viewer.launch_passive(
                self.env.sim.model._model, self.env.sim.data._data,
                show_left_ui=False, show_right_ui=False, key_callback=_key_callback,
            )
            self.viewer.opt.geomgroup[0] = 0
            if self.camera_config is not None:
                self.viewer.cam.lookat = self.camera_config["lookat"]
                self.viewer.cam.distance = self.camera_config["distance"]
                self.viewer.cam.azimuth = self.camera_config["azimuth"]
                self.viewer.cam.elevation = self.camera_config["elevation"]
            if self.camera_id is not None:
                if self.camera_id >= 0:
                    self.viewer.cam.type = 2
                    self.viewer.cam.fixedcamid = self.camera_id
                else:
                    self.viewer.cam.type = 0
        self.viewer.sync()

    MjviewerRenderer.update = _update_with_esc
    MjviewerRenderer._esc_patched = True


def _mjviewer_is_running(env) -> bool:
    """render=true 루프에서 매 스텝(메인 스레드에서) 호출 -- ESC 플래그가 섰거나 창을 직접
    닫아(X 버튼, mujoco.viewer 자체 지원) is_running()이 False가 됐으면 False. 뷰어가 아직
    없거나 mjviewer가 아니면 True(중단 신호 아님). 실제 close()는 여기서 호출하지 않는다 -
    호출부가 False를 받아 rollout을 끝낸 뒤 env.env.close()가 메인 스레드에서 처리한다."""
    renderer = getattr(getattr(env, "env", env), "viewer", None)
    handle = getattr(renderer, "viewer", None)
    if handle is None:
        return True
    if getattr(renderer, "_esc_pressed", False):
        return False
    return handle.is_running()


def _mjviewer_skip_requested(env) -> bool:
    """render=true 루프에서 매 스텝(메인 스레드에서) 호출 -- Enter 키가 눌렸으면 True를
    반환하고 플래그를 바로 소비(리셋)한다(다음 에피소드까지 계속 True로 남지 않게). 사람이
    보다가 이미 실패라고 판단했을 때 max_steps 타임아웃까지 안 기다리고 다음 에피소드로
    넘어가기 위한 용도(2026-07-30)."""
    renderer = getattr(getattr(env, "env", env), "viewer", None)
    if getattr(renderer, "_enter_pressed", False):
        renderer._enter_pressed = False
        return True
    return False


def _apply_run_config(cfg):
    """checkpoint_path 옆 run_config.yaml(학습 시 자동 저장)이 있으면 task/policy/policy_name을
    거기서 통째로 교체해 학습·추론 설정이 어긋나는 걸 막는다(2026-07-25). 없으면(구 체크포인트,
    openvla는 아직 저장 안 함) 아무것도 안 하고 기존 CLI 값 그대로 진행.

    ⚠ OmegaConf.merge가 아니라 직접 대입으로 교체한다 — merge는 하위 dict까지 깊은 병합이라,
    eval.yaml 기본 task(square)에만 있고 실제 학습 task(예: transport)엔 없는 필드가
    살아남는 걸 실측으로 확인함(예: transport 기본값일 때 square 체크포인트를 평가하면
    transport의 env_kwargs가 안 지워지고 섞여 들어감 — 정확히 7/21에 사고 낸 그 종류의
    필드). 직접 대입은 완전 교체라 이 문제가 없음(마찬가지로 실측 확인)."""
    if not cfg.get("use_run_config", True) or not cfg.checkpoint_path:
        return cfg
    saved = load_run_config(cfg.checkpoint_path)
    if saved is None:
        return cfg
    cfg.task = saved.task
    cfg.policy = saved.policy
    cfg.policy_name = saved.policy_name
    logger.info(f"run_config.yaml에서 자동 적용: task={saved.task.name} policy_name={saved.policy_name}")
    return cfg


def _to_pil(img_chw01):
    """make_image_env가 주는 CHW float[0,1](robomimic ObsUtils 규약) -> PIL Image(HWC uint8).
    hdf5 raw(HWC uint8)와 형식이 달라 변환 필요(EXP-01에 기록된 지뢰, OpenVLA rollout에서
    처음 실제로 밟음)."""
    from PIL import Image
    img_hwc = np.transpose(np.asarray(img_chw01), (1, 2, 0))
    img_uint8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


def _run_dp_eval(cfg, device):
    """DiffusionPolicy/BC 등 predict_action_chunk 인터페이스 공용 경로(receding-horizon)."""
    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy).to(device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    stats_path = cfg.stats_path or os.path.join(os.path.dirname(cfg.checkpoint_path), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))

    if cfg.render or cfg.get("async_infer", False):
        # render(화면 표시)든 async_infer(연속 재계획)든 DDPM 100-step(청크당 ~500ms)으로는
        # 감당이 안 돼 DDIM으로 가속해야 한다 - render는 눈으로 보기엔 너무 느려서 끊기고,
        # async_infer는 재계획이 못 따라와 merger가 청크 뒷부분(action_horizon을 넘는 구간)
        # 까지 억지로 쓰게 돼 오히려 성공률이 떨어진다(2026-07-28 실측: DDPM으로 async 돌리면
        # anchor간격 8스텝·0/3, DDIM이면 anchor간격 1스텝) - collect.py 배포와 같은 이유.
        maybe_speed_up_inference(policy, cfg.get("deploy_num_inference_steps", None))

    extra_obs_fn = extra_reset_fn = None
    if cfg.task.get("use_online_stage_tracker", False):
        from mani_sim.datasets.stage_labeler import (
            ShiftedStageTracker, make_online_tracker, num_stages_for_task, onehot as stage_onehot,
        )
        tracker = make_online_tracker(cfg.task.name)
        n_stages = num_stages_for_task(cfg.task.name)
        if cfg.get("stage_shift_target", None) is not None:
            tracker = ShiftedStageTracker(
                tracker, target_stage=cfg.stage_shift_target, window=cfg.stage_shift_window,
                num_stages=n_stages,
            )
            logger.info(
                f"stage shift 강건성 체크: target_stage={cfg.stage_shift_target} "
                f"window={cfg.stage_shift_window}"
            )

        def extra_obs_fn(obs_raw):
            return stage_onehot(np.array([tracker.step(obs_raw)]), num=n_stages)[0]

        extra_reset_fn = tracker.reset

    on_episode_step = None
    gif_frames = None
    if cfg.render and is_image_task(cfg.task) and not is_piper_task(cfg.task):
        # 예전엔 cv2.namedWindow로 라이브 프리뷰를 그리려다 이 PC에서 MuJoCo 오프스크린 GL
        # 컨텍스트와 충돌해 멈췄음(2026-07-20 실측, cv2 창 생성 시점부터 CPU 100% 무한
        # 대기·SIGINT도 안 먹힘, 원인 미해결) — 그래서 그동안 image+render는 아예 막고
        # save_gif로만 우회했음. collect.py가 cv2 대신 MuJoCo 자체 온스크린 뷰어(mjviewer)를
        # 오프스크린 렌더 env에 그대로 얹는 방식으로 이 조합을 실사용(PICO 배포, 수십
        # 에피소드) 검증했으니 여기도 같은 기법을 쓴다(2026-07-27) — 온/오프스크린 둘 다
        # MuJoCo 자체 렌더링 스택이라 서로 다른 GUI 라이브러리 간 충돌이 없다.
        env = make_eval_env(cfg.task)
        env.env.has_renderer = True
        env.env.renderer = "mjviewer"
        # on_episode_step의 sleep은 화면 페이싱일 뿐 - 실제 끊김은 rollout_policy() 내부의
        # 동기 청크 계산(DDPM 100-step, ~500ms/청크) 때문이라 여기서 정책 자체를 가속해야
        # 한다(collect.py의 배포 가속과 동일 함수, 2026-07-27). 학습/성공률 측정(render=false
        # 호출)은 이 분기 자체를 안 타므로 전혀 영향 없음.
        maybe_speed_up_inference(policy, cfg.get("deploy_num_inference_steps", None))
        _patch_mjviewer_esc_quit()

        def on_episode_step(env_, ep, step):
            env_.render()
            time.sleep(0.03)  # 사람 눈으로 따라갈 수 있게 살짝 속도 조절
            if _mjviewer_skip_requested(env_):
                return "skip"
            return _mjviewer_is_running(env_)
    elif cfg.save_gif and is_image_task(cfg.task):
        env = make_eval_env(cfg.task)
        gif_frames = []
        # Piper task는 3인칭 카메라 이름이 다르다(camera_names.front="front_cam" vs
        # robosuite의 "agentview") - env_backend로 분기해서 고른다(2026-07-26).
        gif_camera = cfg.task.camera_names.front if is_piper_task(cfg.task) else "agentview"

        def on_episode_step(env_, ep, step):
            if ep == 0 and step % 4 == 0:
                frame = env_.render(mode="rgb_array", height=256, width=256, camera_name=gif_camera)
                from PIL import Image
                gif_frames.append(Image.fromarray(frame))
    elif cfg.render:
        # low_dim: MuJoCo 네이티브 mjviewer(온스크린).
        env = make_eval_env(cfg.task, render=True)
        _patch_mjviewer_esc_quit()

        def on_episode_step(env_, ep, step):
            env_.render()
            time.sleep(0.03)  # 사람 눈으로 따라갈 수 있게 살짝 속도 조절
            if _mjviewer_skip_requested(env_):
                return "skip"
            return _mjviewer_is_running(env_)
    else:
        env = make_eval_env(cfg.task)

    if cfg.eval_seed is not None:
        np.random.seed(cfg.eval_seed)  # env reset(너트 초기 위치) 시퀀스 고정 — 공정 비교용
        # DDPM/DDIM 샘플링 노이즈(x_T 초기값 포함)도 torch 전역 RNG를 쓰는데 여기가 안
        # 고정돼 있으면 같은 checkpoint+eval_seed로도 매 실행마다 policy가 다른 action을
        # 생성함(2026-07-28 실측: render=true로 같은 checkpoint 반복 관찰 중 초반 행동이
        # 실행마다 달라짐) — train.py/collect.py는 이미 seed 고정하는데 여기만 빠져있었음.
        torch.manual_seed(cfg.eval_seed)

    if cfg.get("async_infer", False):
        # extra_obs_fn(stage tracker) 등은 async 경로 미지원 - 켜져 있으면 사람이 바로
        # 알아채게 명시적으로 막는다(조용히 무시하지 않음).
        if extra_obs_fn is not None:
            raise ValueError("async_infer=true는 아직 use_online_stage_tracker와 병행 미지원.")
        metrics = rollout_policy_async(
            env=env, policy=policy, normalizer=normalizer,
            obs_keys=task_obs_keys(cfg.task), obs_horizon=cfg.policy.obs_horizon,
            action_horizon=cfg.policy.action_horizon, max_steps=cfg.max_steps, num_episodes=cfg.num_episodes,
            device=device, rgb_keys=cfg.task.rgb_keys if is_image_task(cfg.task) else (),
            verbose=True, merger_name=cfg.get("merger_name", "overwrite"), te_coeff=cfg.get("te_coeff", 0.01),
            # render=true면 위 분기에서 이미 env에 mjviewer를 얹어놨다 - 여기선 collect.py와
            # 같은 방식(collect_episode의 render=True 내부 env.render(mode="human"))으로
            # 화면에 띄우고, control_fps로 사람이 보기 좋게 실시간 페이싱(누락돼 있었음,
            # 2026-07-28 - render=true+async_infer=true 조합이 지금까지 안 이어져 있었음).
            render=cfg.render, control_fps=cfg.get("control_fps", 20.0) if cfg.render else 0.0,
        )
    else:
        metrics = rollout_policy(
            env=env, policy=policy, normalizer=normalizer,
            obs_keys=task_obs_keys(cfg.task), obs_horizon=cfg.policy.obs_horizon,
            action_horizon=cfg.policy.action_horizon, max_steps=cfg.max_steps, num_episodes=cfg.num_episodes,
            device=device, rgb_keys=cfg.task.rgb_keys if is_image_task(cfg.task) else (),
            extra_obs_fn=extra_obs_fn, extra_obs_reset_fn=extra_reset_fn, on_episode_step=on_episode_step,
            verbose=True,
        )
    if gif_frames:
        gif_frames[0].save(cfg.save_gif, save_all=True, append_images=gif_frames[1:],
                            duration=60, loop=0, optimize=True)
        logger.info(f"GIF 저장: {cfg.save_gif} ({len(gif_frames)} 프레임)")
    env.env.close()
    return metrics


def _run_openvla_eval(cfg):
    """OpenVLA는 predict_action_chunk가 아니라 매 스텝 단일 이미지로 재계획(원 논문 방식) —
    rollout_policy()의 청크 인터페이스와 안 맞아 별도 루프."""
    from mani_sim.datasets.openvla_dataset import SQUARE_INSTRUCTION

    if cfg.stats_path is None:
        raise ValueError(
            "policy_name=openvla는 stats_path를 명시해야 함 — checkpoint_path가 "
            "{output_dir}/lora_final/policy라 dirname 추정(DP 관례)이 안 맞음"
            "(stats는 {output_dir}/normalization_stats.json)."
        )

    cfg.policy.policy_adapter_path = cfg.checkpoint_path
    normalizer = MinMaxNormalizer(load_stats(cfg.stats_path))
    policy = registry.create_policy(cfg.policy_name, cfg.task, cfg.policy)
    policy.model.eval()
    logger.info(f"loaded adapter: {cfg.checkpoint_path}")

    # OpenVLA는 224 사전학습 해상도를 기대(task.image_size가 84여도 여기선 224로 별도 렌더).
    # (2026-07-25: env_kwargs를 여기서 아예 안 넘기던 버그도 make_eval_env 경유로 같이 해결됨
    # — Transport 등 env_kwargs가 실제로 의미 있는 task에서 OpenVLA를 평가하면 로봇이
    # 잘못된 위치에 스폰됐을 것, collect.py와 같은 종류.)
    env = make_eval_env(cfg.task, image_size_override=224)

    if cfg.eval_seed is not None:
        np.random.seed(cfg.eval_seed)
        torch.manual_seed(cfg.eval_seed)

    image_key = cfg.task.rgb_keys[0]
    instruction = cfg.openvla_instruction or SQUARE_INSTRUCTION

    successes = []
    t0 = time.time()
    for ep in range(cfg.num_episodes):
        obs = env.reset()
        success = False
        steps = 0
        gif_frames = [] if (cfg.save_gif and ep == 0) else None
        while steps < cfg.max_steps:
            img = _to_pil(obs[image_key])
            action_norm = policy.predict_action(img, instruction)
            action = normalizer.unnormalize_action(torch.as_tensor(action_norm, dtype=torch.float32)).numpy()
            obs, _r, done, _i = env.step(action)
            steps += 1
            if gif_frames is not None and steps % 2 == 0:  # 매 2프레임마다(용량 절반)
                frame = env.render(mode="rgb_array", height=224, width=224, camera_name="agentview")
                from PIL import Image
                gif_frames.append(Image.fromarray(frame))
            if env.is_success()["task"]:
                success = True
            if success or done or steps >= cfg.max_steps:
                break
        if gif_frames:
            gif_frames[0].save(cfg.save_gif, save_all=True, append_images=gif_frames[1:],
                                duration=60, loop=0, optimize=True)
            logger.info(f"GIF 저장: {cfg.save_gif} ({len(gif_frames)} 프레임)")
        successes.append(success)
        logger.info(f"episode {ep}: steps={steps} success={success}")

    env.env.close()
    logger.info(f"elapsed={time.time() - t0:.0f}s")
    return {
        "success_rate": float(np.mean(successes)),
        "num_episodes": cfg.num_episodes,
        "num_successes": int(np.sum(successes)),
    }


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    # robosuite의 "Loading controller configuration..." 등 INFO 로그가 매 env.reset()마다
    # 두 줄씩 찍힌다(robosuite 자체 콘솔 핸들러 + "robosuite_logs" 로거가 propagate=False를
    # 안 걸어놔서 Hydra root logger로도 전파돼 다시 한 번 포맷팅됨) - render로 지켜볼 때
    # "ep N: steps=... success=..." 줄이 이 노이즈에 묻혀서 WARNING 이상만 남긴다(2026-07-30).
    logging.getLogger("robosuite_logs").setLevel(logging.WARNING)

    cfg = _apply_run_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if cfg.policy_name == "openvla":
        metrics = _run_openvla_eval(cfg)
    else:
        metrics = _run_dp_eval(cfg, device)

    logger.info(f"checkpoint={cfg.checkpoint_path} {metrics}")
    print(metrics)

    output_dir = cfg.output_dir or os.path.join(
        os.path.dirname(cfg.checkpoint_path), "eval_results", time.strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(output_dir, exist_ok=True)
    # episodes(에피소드별 success/steps, rollout_policy의 verbose=True와 짝 - 2026-07-27)는
    # metrics.json엔 안 넣는다 - 그 파일은 원래 집계(success_rate 등)만 담는 관례라, 대신
    # episodes.csv로 따로 저장해 나중에 화면 안 보고도 통계/에피소드별 리뷰가 가능하게 한다.
    episodes = metrics.pop("episodes", None)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"checkpoint_path": cfg.checkpoint_path, **metrics}, f, indent=2)
    if episodes:
        import csv

        episodes_path = os.path.join(output_dir, "episodes.csv")
        with open(episodes_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "success", "steps"])
            writer.writeheader()
            writer.writerows(episodes)
        logger.info(f"에피소드별 기록 저장: {episodes_path} ({len(episodes)}개)")
    return metrics


if __name__ == "__main__":
    main()
