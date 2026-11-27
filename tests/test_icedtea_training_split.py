import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from jepa_policy.datasets.icedtea_split import (
    EpisodeSplitOptions,
    prepare_episode_split,
)


def make_metadata(root: Path, count: int = 200) -> Path:
    metadata = root / "meta/episodes/chunk-000"
    metadata.mkdir(parents=True)
    (root / "meta/info.json").write_text('{"fps": 15}', encoding="utf-8")
    records = [
        {
            "episode_index": index,
            "tasks": ["fetch iced tea." if index % 2 else "grasp the red iced tea."],
            "dataset_from_index": index * 20,
            "dataset_to_index": (index + 1) * 20,
        }
        for index in range(count)
    ]
    pq.write_table(pa.Table.from_pylist(records), metadata / "file-000.parquet")
    return root


def write_indices(path: Path, values: object) -> Path:
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("count", "train_count", "eval_count"), [(200, 190, 10), (228, 216, 12), (2, 1, 1)]
)
def test_complete_episode_partition_and_expected_counts(
    tmp_path, count, train_count, eval_count
):
    root = make_metadata(tmp_path / "dataset", count)
    output = tmp_path / "run"
    split = prepare_episode_split(root, EpisodeSplitOptions(output))
    assert len(split.train) == train_count
    assert len(split.evaluation) == eval_count
    assert not set(split.train) & set(split.evaluation)
    assert set(split.train) | set(split.evaluation) == set(range(count))
    assert split.candidates == tuple(range(count))
    assert json.loads((output / "train_episode_indices.json").read_text()) == list(
        split.train
    )
    assert json.loads((output / "eval_episode_indices.json").read_text()) == list(
        split.evaluation
    )
    payload = json.loads((output / "episode_split.json").read_text())
    assert payload["metadata_sha256"] == split.metadata_sha256
    assert payload["dataset"] == str(root.resolve())


def test_split_is_reproducible_without_changing_global_random_state(tmp_path):
    root = make_metadata(tmp_path / "dataset")
    np.random.seed(123)
    before = np.random.get_state()
    first = prepare_episode_split(root, EpisodeSplitOptions(tmp_path / "first"))
    second = prepare_episode_split(root, EpisodeSplitOptions(tmp_path / "second"))
    after = np.random.get_state()
    assert first == second
    np.testing.assert_array_equal(before[1], after[1])
    different = prepare_episode_split(
        root, EpisodeSplitOptions(tmp_path / "third", seed=1)
    )
    assert first.evaluation != different.evaluation


def test_candidate_indices_select_pool_without_task_matching(tmp_path):
    root = make_metadata(tmp_path / "dataset", 228)
    pool = list(reversed(range(10, 210)))
    path = write_indices(tmp_path / "pool.json", pool)
    split = prepare_episode_split(root, EpisodeSplitOptions(tmp_path / "run", path))
    assert len(split.train) == 190
    assert len(split.evaluation) == 10
    assert set(split.train) | set(split.evaluation) == set(pool)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "at least two"),
        ([1], "at least two"),
        ([1, 1], "duplicate"),
        ([0, 999], "not found"),
        ([True, 1], "integer"),
        ([0, 1.0], "integer"),
        ([0, "1"], "integer"),
        ([-1, 0], "integer"),
        ({"episodes": [0, 1]}, "list"),
    ],
)
def test_invalid_candidates_fail_without_writing_outputs(tmp_path, values, message):
    root = make_metadata(tmp_path / "dataset")
    path = write_indices(tmp_path / "pool.json", values)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match=message):
        prepare_episode_split(root, EpisodeSplitOptions(output, path))
    assert not output.exists()


def test_existing_output_is_reusable_but_conflicts_do_not_overwrite(tmp_path):
    root = make_metadata(tmp_path / "dataset")
    options = EpisodeSplitOptions(tmp_path / "run")
    split = prepare_episode_split(root, options)
    snapshots = {path.name: path.read_bytes() for path in options.output.iterdir()}
    assert prepare_episode_split(root, options) == split
    with pytest.raises(FileExistsError, match="different episode split"):
        prepare_episode_split(root, EpisodeSplitOptions(options.output, seed=1))
    assert snapshots == {
        path.name: path.read_bytes() for path in options.output.iterdir()
    }


def test_output_preflight_prevents_partial_split_files(tmp_path):
    root = make_metadata(tmp_path / "dataset")
    output = tmp_path / "run"
    output.mkdir()
    write_indices(output / "eval_episode_indices.json", [999])
    with pytest.raises(FileExistsError):
        prepare_episode_split(root, EpisodeSplitOptions(output))
    assert not (output / "episode_split.json").exists()
    assert not (output / "train_episode_indices.json").exists()


def make_checkpoint(output: Path) -> Path:
    models = output / "models"
    models.mkdir()
    checkpoint = models / "model_step_001000.pt"
    checkpoint.touch()
    return checkpoint


def test_resume_reuses_saved_partition_and_checks_identity(tmp_path):
    root = make_metadata(tmp_path / "dataset")
    previous = tmp_path / "previous"
    original = prepare_episode_split(root, EpisodeSplitOptions(previous))
    checkpoint = make_checkpoint(previous)
    resumed = prepare_episode_split(
        root, EpisodeSplitOptions(tmp_path / "resumed", resume_checkpoint=checkpoint)
    )
    assert resumed == original
    with pytest.raises(ValueError, match="seed"):
        prepare_episode_split(
            root,
            EpisodeSplitOptions(
                tmp_path / "wrong_seed", resume_checkpoint=checkpoint, seed=1
            ),
        )
    pool = write_indices(tmp_path / "pool.json", list(range(180)))
    with pytest.raises(ValueError, match="candidate_episode_indices"):
        prepare_episode_split(
            root, EpisodeSplitOptions(tmp_path / "wrong_pool", pool, checkpoint)
        )


def test_resume_rejects_dataset_metadata_drift(tmp_path):
    root = make_metadata(tmp_path / "dataset")
    previous = tmp_path / "previous"
    prepare_episode_split(root, EpisodeSplitOptions(previous))
    checkpoint = make_checkpoint(previous)
    (root / "meta/info.json").write_text('{"fps": 30}', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata_sha256"):
        prepare_episode_split(
            root,
            EpisodeSplitOptions(tmp_path / "resumed", resume_checkpoint=checkpoint),
        )
    assert not (tmp_path / "resumed").exists()


def test_resume_rejects_overlapping_or_incomplete_partition(tmp_path):
    root = make_metadata(tmp_path / "dataset")
    previous = tmp_path / "previous"
    prepare_episode_split(root, EpisodeSplitOptions(previous))
    checkpoint = make_checkpoint(previous)
    path = previous / "episode_split.json"
    payload = json.loads(path.read_text())
    payload["train_episode_indices"][0] = payload["eval_episode_indices"][0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid saved"):
        prepare_episode_split(
            root,
            EpisodeSplitOptions(tmp_path / "resumed", resume_checkpoint=checkpoint),
        )


def test_duplicate_metadata_ids_are_rejected(tmp_path):
    root = make_metadata(tmp_path / "dataset")
    path = root / "meta/episodes/chunk-000/file-001.parquet"
    pq.write_table(pa.table({"episode_index": [0]}), path)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_episode_split(root, EpisodeSplitOptions(tmp_path / "run"))


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_invalid_seed_fails(tmp_path, seed):
    root = make_metadata(tmp_path / "dataset")
    with pytest.raises(ValueError, match="nonnegative integer"):
        prepare_episode_split(root, EpisodeSplitOptions(tmp_path / "run", seed=seed))
