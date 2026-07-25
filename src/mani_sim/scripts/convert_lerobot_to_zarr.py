"""collect.py(Piper task, env_backend=piper_mujoco)가 저장한 LeRobotDataset(parquet+video)을
mani_sim 학습이 쓰는 zarr(ReplayBuffer)로 변환.

⚠ collect.py의 Piper 경로와 마찬가지로 `piper_collect` conda env(lerobot+ffmpeg 설치됨,
비디오 디코딩 필요)에서 실행한다 - mani_sim 학습 환경(robomimic)엔 lerobot이 없다. 변환
산출물인 zarr는 numpy+zarr만으로 읽을 수 있어서(ZarrSequenceDataset, robomimic env)
그쪽엔 lerobot이 전혀 필요 없다.

mani_sim 패키지의 다른 모듈을 import하지 않는 독립 스크립트(hydra만 공유) - collect.py는
이제 mani_sim 패키지에 의존하므로 piper_collect env에도 mani_sim을 설치해뒀지만
(2026-07-26), 이 변환 스크립트는 굳이 그럴 필요가 없어 계속 독립으로 둔다.

사용(piper_collect env에서, mani_sim 저장소 루트 기준):
    /path/to/envs/piper_collect/bin/python src/mani_sim/scripts/convert_lerobot_to_zarr.py \
        lerobot_root=data/piper_sort_return/lerobot_raw \
        zarr_path=data/piper_sort_return/piper_sort_return.zarr
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 순수 로컬 데이터셋이라 허브 조회 자체가 불필요(실측: 안 하면 401)

import hydra
import numpy as np
from omegaconf import DictConfig


def _episode_arrays(dataset, camera_keys, ep_from, ep_to):
    states, actions, images = [], [], {k: [] for k in camera_keys}
    for idx in range(ep_from, ep_to):
        item = dataset[idx]
        states.append(item["observation.state"].numpy())
        actions.append(item["action"].numpy())
        for k in camera_keys:
            img_chw01 = item[f"observation.images.{k}"].numpy()  # (3,H,W) float[0,1]
            img_hwc_u8 = np.clip(img_chw01 * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            images[k].append(img_hwc_u8)

    data = {
        "state": np.stack(states).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
    }
    for k in camera_keys:
        data[f"{k}_image"] = np.stack(images[k])
    return data


@hydra.main(config_path="../configs", config_name="convert_lerobot_to_zarr", version_base=None)
def main(cfg: DictConfig):
    from flare.datasets.replay_buffer import ReplayBuffer
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.lerobot_root)
    camera_keys = list(cfg.camera_names)

    buffer = ReplayBuffer.create_from_path(cfg.zarr_path, mode="a")
    for ep_idx in range(dataset.num_episodes):
        row = dataset.meta.episodes[ep_idx]
        ep_from, ep_to = row["dataset_from_index"], row["dataset_to_index"]
        data = _episode_arrays(dataset, camera_keys, ep_from, ep_to)
        buffer.add_episode(data)
        print(f"episode {ep_idx}: {ep_to - ep_from} frames -> zarr (누적 {buffer.n_steps} steps)")

    print(f"변환 완료: {dataset.num_episodes} episodes, {buffer.n_steps} steps -> {cfg.zarr_path}")


if __name__ == "__main__":
    main()
