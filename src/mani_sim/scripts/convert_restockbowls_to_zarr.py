"""RoboCasa RestockBowls LeRobotDataset -> mani_sim zarr(ReplayBuffer), stage 라벨 포함.

convert_lerobot_to_zarr.py(Piper)와 같은 목적이지만 두 가지가 다르다:
  - 이미지는 `lerobot.datasets.lerobot_dataset.LeRobotDataset`을 안 쓰고 각 에피소드의
    mp4를 PyAV로 직접 순차 디코딩한다. LeRobotDataset.__getitem__으로 프레임을 하나씩
    임의 접근하면 매 호출마다 재탐색(seek)이 발생해 에피소드 하나(1758프레임)에 9분+
    걸림(실측, 2026-08-04) — 반면 컨테이너를 한 번 열고 순차 디코딩하면 같은 에피소드가
    1초 미만(실측). state/action은 매우 가벼워서 그냥 parquet를 pandas로 직접 읽는다.
  - stage 라벨은 robocasa/scripts/label_stages.py(정본, robocasa/research_design.md 참고)의
    raw state replay 로직으로 프레임 단위로 계산해서 같이 저장한다.

robosuite/robocasa가 필요해서 `robocasa` conda env에서만 실행 가능하다(mani_sim의 기본
학습 env인 robomimic엔 robocasa 없음 - 변환 산출물인 zarr는 numpy+zarr만 있으면 읽히므로
학습은 그대로 robomimic env에서 한다. zarr/numcodecs 버전은 robomimic env와 맞춤:
zarr==2.18.3, numcodecs<0.16 - zarr 3.x는 온디스크 포맷이 달라 호환 안 됨, 실측 확인).

사용(robocasa env에서, 아무 디렉토리에서나):
    python convert_restockbowls_to_zarr.py
"""
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
import robosuite
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, "/home/moai/jungwook_ws/robocasa")
import robocasa.utils.lerobot_utils as LU  # noqa: E402

sys.path.insert(0, "/home/moai/jungwook_ws/ljw_workspace/robocasa/scripts")
import label_stages as LS  # noqa: E402

_PIPER_CAPSTONE_DIR = Path(__file__).resolve().parents[3] / "mani_sim_external" / "piper_capstone"
sys.path.insert(0, str(_PIPER_CAPSTONE_DIR))
from replay_buffer import ReplayBuffer  # noqa: E402

DATASET = Path(
    "/home/moai/jungwook_ws/robocasa/datasets/v1.0/pretrain/composite/RestockBowls/20250725/lerobot"
)
ZARR_PATH = "/home/moai/jungwook_ws/ljw_workspace/robocasa/data/restockbowls.zarr"
CAMERA_KEYS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
IMAGE_HW = (84, 84)  # mani_sim VisionEncoder 기본값에 맞춤
NUM_EPISODES = 105


def make_replay_env():
    env_meta = LU.get_env_metadata(DATASET)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["renderer"] = "mjviewer"
    return robosuite.make(**env_kwargs)


def decode_episode_video(cam, ep, hw):
    """episode_{ep:06d}.mp4를 순차 디코딩해서 (T,H,W,3) uint8로 반환(리사이즈 포함)."""
    path = DATASET / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
    container = av.open(str(path))
    frames = []
    for frame in container.decode(video=0):
        img_hwc_u8 = frame.to_ndarray(format="rgb24")  # (H,W,3) uint8
        img_chw = torch.from_numpy(img_hwc_u8).permute(2, 0, 1)
        img_small = TF.resize(img_chw, list(hw), antialias=True)
        frames.append(img_small.permute(1, 2, 0).numpy())
    container.close()
    return np.stack(frames)


def read_episode_table(ep):
    df = pd.read_parquet(DATASET / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    return state, action


def main():
    env = make_replay_env()
    buffer = ReplayBuffer.create_from_path(ZARR_PATH, mode="a")

    ok, fail = 0, 0
    for ep in range(NUM_EPISODES):
        try:
            _, _, stages = LS.label_episode(env, ep)  # (T,) int, 1..6
        except ValueError as e:
            print(f"ep{ep:03d} 라벨링 실패, 건너뜀: {e}")
            fail += 1
            continue

        state, action = read_episode_table(ep)
        assert len(state) == len(stages), f"ep{ep}: 프레임 수 불일치 parquet={len(state)} vs replay={len(stages)}"

        data = {
            "state": state,
            "action": action,
            "stage": (stages - 1).astype(np.int64),  # 0-indexed for nn.Embedding
        }
        for cam in CAMERA_KEYS:
            imgs = decode_episode_video(cam, ep, IMAGE_HW)
            assert len(imgs) == len(stages), f"ep{ep} {cam}: 비디오 프레임 수 불일치 {len(imgs)} vs {len(stages)}"
            data[f"{cam}_image"] = imgs

        buffer.add_episode(data)
        ok += 1
        print(f"ep{ep:03d}: {len(stages)} frames -> zarr (누적 {buffer.n_steps} steps)", flush=True)

    print(f"변환 완료: {ok}/{NUM_EPISODES} episodes 성공({fail} 실패), {buffer.n_steps} steps -> {ZARR_PATH}")


if __name__ == "__main__":
    main()
