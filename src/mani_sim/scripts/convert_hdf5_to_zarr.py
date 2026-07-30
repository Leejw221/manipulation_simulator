"""robomimic 형식 HDF5(collect.py/merge_rounds.py 산출물) -> Zarr(ReplayBuffer) 변환.

collect.py(사람 개입 수집)·merge_rounds.py(라운드 누적)는 그대로 hdf5를 쓴다 - 둘 다 robomimic
관례(마스크 필터키, demo_i 그룹)에 맞물려 있어 굳이 바꿀 이유가 없다. 이 스크립트는 그 결과물을
**학습 직전에** Zarr로 변환하는 별도 단계다.

바꾸는 이유(2026-07-27 밤): robomimic SequenceDataset은 h5py의 fork-불안정성 때문에
num_workers>=1이면 이미지 전체를 메모리에 캐싱해야 하는데(diffusion_trainer.py의 cache_mode
분기 주석 참고), 라운드가 쌓여 데이터가 커질수록 이게 OOM으로 이어진다(round0 44demo/23815
프레임에서 실측). Zarr는 이 제약이 없어 멀티워커 + 저메모리 로딩을 동시에 가능케 한다.

이미지는 collect.py가 이미 raw(HWC, uint8) 포맷으로 저장하므로(2026-07-27 이미지 포맷 버그
수정) 별도 변환 없이 그대로 복사한다.

학습 파이프라인 연결: task.yaml에 `dataset_backend=zarr`+`zarr_path=...`를 얹으면(로봇수트
task도 CLI에서 `+task.dataset_backend=zarr +task.zarr_path=...`로 추가 가능) 시뮬레이터
(env_backend, robosuite 그대로 유지)는 안 건드리고 학습 데이터 저장 포맷만 hdf5->Zarr로
바뀐다(task_utils.uses_zarr_dataset이 이 둘을 독립 축으로 분리— 2026-07-27 밤). "저장은
Zarr, 시뮬레이션은 robosuite"로 Square 200개 데모 학습을 실전 검증함(2026-07-29,
num_workers=4로 OOM 없이 정상 진행 — mani_sim_status.md 참고).

사용: python -m mani_sim.scripts.convert_hdf5_to_zarr task=transport_demo20 \
    hdf5_path=data/intervention/transport_round0_cumulative.hdf5 \
    zarr_path=data/intervention/transport_round0_cumulative.zarr
(hdf5_path을 생략하면 task.hdf5_path을 그대로 쓴다 - 원본 robomimic 데이터셋 변환용.)
"""

import sys
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig

from mani_sim.utils.task_utils import is_image_task, task_lowdim_keys

_PIPER_CAPSTONE_DIR = Path(__file__).resolve().parents[3] / "mani_sim_external" / "piper_capstone"
if str(_PIPER_CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPER_CAPSTONE_DIR))


def convert_hdf5_to_zarr(hdf5_path, zarr_path, lowdim_keys, rgb_keys, filter_key=None):
    from replay_buffer import ReplayBuffer

    with h5py.File(hdf5_path, "r") as fin:
        if filter_key:
            demo_names = [d.decode() if isinstance(d, bytes) else d for d in fin["mask"][filter_key][:]]
        else:
            demo_names = list(fin["data"].keys())
        demo_names = sorted(demo_names, key=lambda k: int(k.split("_")[1]))

        buffer = ReplayBuffer.create_from_path(zarr_path, mode="a")
        for name in demo_names:
            demo = fin[f"data/{name}"]
            data = {key: demo["obs"][key][()].astype(np.float32) for key in lowdim_keys}
            for key in rgb_keys:
                data[key] = demo["obs"][key][()]  # 이미 raw(HWC, uint8) - 변환 불필요
            data["action"] = demo["actions"][()].astype(np.float32)
            if "action_mode" in demo:
                data["action_mode"] = demo["action_mode"][()].astype(np.int64)
            buffer.add_episode(data)
            print(f"{name}: {len(data['action'])} frames -> zarr (누적 {buffer.n_steps} steps)")

    print(f"변환 완료: {len(demo_names)} demos, {buffer.n_steps} steps -> {zarr_path}")
    return zarr_path


@hydra.main(config_path="../configs", config_name="convert_hdf5_to_zarr", version_base=None)
def main(cfg: DictConfig):
    rgb_keys = list(cfg.task.rgb_keys) if is_image_task(cfg.task) else []
    lowdim_keys = task_lowdim_keys(cfg.task)
    hdf5_path = cfg.get("hdf5_path", None) or cfg.task.hdf5_path
    convert_hdf5_to_zarr(hdf5_path, cfg.zarr_path, lowdim_keys, rgb_keys, filter_key=cfg.get("filter_key", None))


if __name__ == "__main__":
    main()
