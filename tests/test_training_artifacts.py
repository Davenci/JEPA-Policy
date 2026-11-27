"""Check that resumed weights retain the original training normalization."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from jepa_policy.training_artifacts import prepare_run_directory, validate_run_output


@pytest.fixture
def previous_training(tmp_path):
    dataset = SimpleNamespace(
        root=tmp_path / "dataset",
        episodes={3: {}, 7: {}},
        samples=[(0, 3), (12, 7)],
        action_min=np.arange(8, dtype=np.float32),
        action_range=np.full(8, 2.0, dtype=np.float32),
        state_min=np.arange(8, dtype=np.float32) - 10.0,
        state_range=np.full(8, 4.0, dtype=np.float32),
        image_size=128,
        image_interpolation=1,
    )
    previous = tmp_path / "previous"
    models, _ = prepare_run_directory(previous, dataset, None)
    checkpoint = models / "model_step_001000.pt"
    checkpoint.touch()
    return SimpleNamespace(dataset=dataset, output=previous, checkpoint=checkpoint)


def test_resume_preserves_equal_saved_training_statistics(tmp_path, previous_training):
    output = tmp_path / "resumed"
    original = (previous_training.output / "normalization_stats.json").read_bytes()
    models, metrics = prepare_run_directory(
        output, previous_training.dataset, previous_training.checkpoint
    )
    assert models == output / "models"
    assert metrics == output / "metrics.jsonl"
    assert models.is_dir()
    assert (output / "normalization_stats.json").read_bytes() == original
    assert (
        previous_training.output / "normalization_stats.json"
    ).read_bytes() == original
    metadata = json.loads((output / "dataset_metadata.json").read_text())
    assert metadata["episode_indices"] == [3, 7]


@pytest.mark.parametrize(
    "field", ["action_min", "action_range", "state_min", "state_range"]
)
def test_resume_rejects_changed_statistics_before_writing_new_metadata(
    tmp_path, previous_training, field
):
    output = tmp_path / "resumed"
    original = (previous_training.output / "normalization_stats.json").read_bytes()
    getattr(previous_training.dataset, field)[0] += 0.25
    with pytest.raises(ValueError, match="normalization differs"):
        prepare_run_directory(
            output, previous_training.dataset, previous_training.checkpoint
        )
    assert not output.exists()
    assert (
        previous_training.output / "normalization_stats.json"
    ).read_bytes() == original


def test_resume_rejects_different_training_episode_set(tmp_path, previous_training):
    output = tmp_path / "resumed"
    previous_training.dataset.episodes = {3: {}, 8: {}}
    with pytest.raises(ValueError, match="different training episodes"):
        prepare_run_directory(
            output, previous_training.dataset, previous_training.checkpoint
        )
    assert not output.exists()


def test_resume_requires_statistics_next_to_original_checkpoint(
    tmp_path, previous_training
):
    output = tmp_path / "resumed"
    checkpoint = tmp_path / "missing_statistics/models/model_step_001000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    with pytest.raises(FileNotFoundError, match="normalization_stats.json"):
        prepare_run_directory(output, previous_training.dataset, checkpoint)
    assert not output.exists()


def test_nonempty_output_is_rejected_without_overwriting_existing_work(tmp_path):
    output = tmp_path / "run"
    validate_run_output(output)
    assert not output.exists()
    output.mkdir()
    validate_run_output(output)
    existing = output / "metrics.jsonl"
    existing.write_text('{"step": 10, "loss": 0.5}\n')
    original = existing.read_bytes()
    with pytest.raises(FileExistsError, match="new, empty training output"):
        validate_run_output(output)
    assert existing.read_bytes() == original
    assert list(output.iterdir()) == [existing]
