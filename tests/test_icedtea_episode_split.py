import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from jepa_policy.datasets.icedtea_cache import create_source
from jepa_policy.datasets.icedtea_dataset import (
    ACTION_NAMES,
    IcedTeaDataset,
    IcedTeaDatasetConfig,
)


@pytest.fixture
def episode_dataset_root(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data" / "chunk-000"
    episode_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True)
    episode_dir.mkdir(parents=True)
    features = {key: {"names": ACTION_NAMES} for key in ("action", "observation.state")}
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"features": features}))
    tasks = [
        {"task_index": 0, "task": "grasp the red iced tea."},
        {"task_index": 1, "task": "other task"},
    ]
    pq.write_table(pa.Table.from_pylist(tasks), tmp_path / "meta" / "tasks.parquet")

    frames = []
    episodes = []
    for episode_id in range(4):
        task_index = int(episode_id == 3)
        start = len(frames)
        for frame in range(12):
            baseline = episode_id * 10 + frame
            action = [float(baseline + joint) for joint in range(len(ACTION_NAMES))]
            state = [float(2 * baseline + joint) for joint in range(len(ACTION_NAMES))]
            if episode_id == 2:
                action = [value + 10000.0 for value in action]
                state = [value - 10000.0 for value in state]
            frames.append(
                {
                    "action": action,
                    "observation.state": state,
                    "timestamp": frame / 15.0,
                    "episode_index": episode_id,
                    "task_index": task_index,
                }
            )
        episodes.append(
            {
                "episode_index": episode_id,
                "tasks": [tasks[task_index]["task"]],
                "dataset_from_index": start,
                "dataset_to_index": len(frames),
            }
        )
    pq.write_table(pa.Table.from_pylist(frames), data_dir / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(episodes), episode_dir / "file-000.parquet")
    return tmp_path


def test_training_statistics_exclude_held_out_extremes(episode_dataset_root: Path):
    selected = IcedTeaDataset(
        IcedTeaDatasetConfig(episode_dataset_root, episode_indices=(0, 1))
    )
    all_episodes = IcedTeaDataset(IcedTeaDatasetConfig(episode_dataset_root))
    joint_offsets = np.arange(len(ACTION_NAMES), dtype=np.float32)
    np.testing.assert_array_equal(selected.action_min, joint_offsets)
    np.testing.assert_array_equal(selected.state_min, joint_offsets)
    np.testing.assert_array_equal(selected.action_range, np.full(8, 21.0))
    np.testing.assert_array_equal(selected.state_range, np.full(8, 42.0))
    assert np.all(all_episodes.action_range > 10000)
    assert np.all(all_episodes.state_min < -9000)
    training_rows = np.isin(selected.episode_ids, (0, 1))
    for values in (selected.actions[training_rows], selected.states[training_rows]):
        np.testing.assert_array_equal(values.min(axis=0), -np.ones(8))
        np.testing.assert_array_equal(values.max(axis=0), np.ones(8))


def test_selected_windows_stay_inside_their_episode(episode_dataset_root: Path):
    config = IcedTeaDatasetConfig(episode_dataset_root, episode_indices=(0, 1))
    dataset = IcedTeaDataset(config)
    assert dataset.selected_episode_indices == (0, 1)
    assert tuple(dataset.episodes) == (0, 1)
    assert dataset.samples == [(0, 0), (1, 0), (2, 0), (12, 1), (13, 1), (14, 1)]
    for start, episode_id in dataset.samples:
        assert set(dataset.episode_ids[start : start + config.horizon]) == {episode_id}
        assert set(dataset.episode_ids[start : start + config.obs_steps]) == {
            episode_id
        }
        future_row = start + config.obs_steps - 1 + config.future_steps
        assert dataset.episode_ids[future_row] == episode_id


@pytest.mark.parametrize(
    ("episode_indices", "message"),
    [
        ((0, 0), "duplicate"),
        ((99,), "not found"),
        ((), "at least one"),
    ],
)
def test_invalid_episode_selection_fails(
    episode_dataset_root: Path, episode_indices: tuple[int, ...], message: str
):
    config = IcedTeaDatasetConfig(episode_dataset_root, episode_indices=episode_indices)
    with pytest.raises(ValueError, match=message):
        IcedTeaDataset(config)


def test_default_dataset_uses_all_labels(episode_dataset_root: Path):
    dataset = IcedTeaDataset(IcedTeaDatasetConfig(episode_dataset_root))
    assert tuple(dataset.episodes) == (0, 1, 2, 3)
    assert len(dataset) == 12
    assert {episode_id for _, episode_id in dataset.samples} == {0, 1, 2, 3}
    assert set(dataset.task_indices) == {0, 1}


def test_episode_indices_can_cross_text_labels(episode_dataset_root: Path):
    dataset = IcedTeaDataset(
        IcedTeaDatasetConfig(episode_dataset_root, episode_indices=(0, 3))
    )
    assert tuple(dataset.episodes) == (0, 3)
    assert len(dataset) == 6
    assert {episode_id for _, episode_id in dataset.samples} == {0, 3}
    joint_offsets = np.arange(len(ACTION_NAMES), dtype=np.float32)
    np.testing.assert_array_equal(dataset.action_min, joint_offsets)
    np.testing.assert_array_equal(dataset.action_range, np.full(8, 41.0))
    np.testing.assert_array_equal(dataset.state_range, np.full(8, 82.0))


def test_cache_factory_keeps_cross_label_episode_selection(episode_dataset_root: Path):
    config = IcedTeaDatasetConfig(episode_dataset_root, episode_indices=(0, 3))
    dataset = IcedTeaDataset(config)
    source = create_source(config)
    assert tuple(source.dataset.episodes) == (0, 3)
    assert source.num_steps == len(dataset) == 6
    assert source.dataset.samples == dataset.samples
    np.testing.assert_array_equal(source.dataset.action_range, dataset.action_range)
    np.testing.assert_array_equal(source.dataset.state_range, dataset.state_range)
