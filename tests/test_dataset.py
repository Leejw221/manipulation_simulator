"""M1 데이터 파이프라인 검증: shape·정규화 정합성.

lift PH low_dim 데이터셋(M0에서 다운로드)이 로컬에 있어야 통과한다.
"""

import os

import pytest

from mani_sim.datasets.normalization import MinMaxNormalizer, compute_minmax_stats
from mani_sim.datasets.robomimic_dataset import RobomimicSequenceDataset

HDF5_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "robomimic", "lift", "ph", "v1.5", "lift", "ph", "low_dim_v15.hdf5"
)
OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]

pytestmark = pytest.mark.skipif(
    not os.path.exists(HDF5_PATH), reason="lift PH low_dim 데이터셋 없음 (M0 다운로드 단계 참고)"
)


@pytest.fixture(scope="module")
def stats():
    return compute_minmax_stats(HDF5_PATH, OBS_KEYS)


@pytest.fixture(scope="module")
def normalizer(stats):
    return MinMaxNormalizer(stats)


@pytest.fixture(scope="module")
def dataset(normalizer):
    return RobomimicSequenceDataset(
        hdf5_path=HDF5_PATH,
        obs_keys=OBS_KEYS,
        obs_horizon=2,
        pred_horizon=16,
        normalizer=normalizer,
    )


def test_batch_shapes(dataset):
    batch = dataset[0]
    assert set(batch["obs"].keys()) == set(OBS_KEYS)
    assert batch["obs"]["robot0_eef_pos"].shape == (2, 3)
    assert batch["obs"]["robot0_eef_quat"].shape == (2, 4)
    assert batch["obs"]["robot0_gripper_qpos"].shape == (2, 2)
    assert batch["obs"]["object"].shape == (2, 10)
    assert batch["action"].shape == (16, 7)
    assert batch["action_mask"].shape == (16,)
    assert batch["action_mask"].dtype.is_floating_point is False


def test_normalized_values_within_range(dataset):
    batch = dataset[0]
    for tensor in batch["obs"].values():
        assert tensor.min() >= -1.0 - 1e-4
        assert tensor.max() <= 1.0 + 1e-4
    assert batch["action"].min() >= -1.0 - 1e-4
    assert batch["action"].max() <= 1.0 + 1e-4


def test_normalize_unnormalize_roundtrip(stats, normalizer, dataset):
    raw_dataset = RobomimicSequenceDataset(
        hdf5_path=HDF5_PATH,
        obs_keys=OBS_KEYS,
        obs_horizon=2,
        pred_horizon=16,
        normalizer=None,
    )
    raw_batch = raw_dataset[100]
    normalized = normalizer.normalize_action(raw_batch["action"])
    recovered = normalizer.unnormalize_action(normalized)
    assert torch_allclose(recovered, raw_batch["action"])


def test_action_mask_marks_padding_at_episode_end(dataset):
    # M0에서 확인: 첫 demo 길이=59, 마지막 실제 timestep index=58 → 윈도우 대부분이 패딩.
    batch = dataset[58]
    assert batch["action_mask"][0].item() is True
    assert batch["action_mask"][1:].any().item() is False


def torch_allclose(a, b, atol=1e-4):
    return (a - b).abs().max().item() < atol
