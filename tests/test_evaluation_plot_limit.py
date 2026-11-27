"""Keep full shadow evaluation while rendering only a small episode preview."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples import evaluate_x1_icedtea as entry


class ZeroActionPolicy:
    horizon = 10
    action_start = 1
    chunk_size = 4
    temporal_ensembler = None
    temporal_ensemble_k = 0.01

    def __init__(self):
        self.calls = []

    def predict(self, observation):
        self.calls.append(observation)
        return np.zeros((self.chunk_size, 8), dtype=np.float32)


def make_dataset(count):
    episodes = []
    for index in range(count):
        length = index + 1
        value = 10.0 if index == count - 1 and count > 3 else 0.0
        episodes.append(
            entry.Episode(
                f"episode_{index:06d}",
                length,
                np.full((length, 8), value, dtype=np.float32),
                lambda frame, episode_id=index: (episode_id, frame),
            )
        )
    return entry.Dataset(episodes, name="synthetic-shadow-evaluation")


@pytest.fixture
def evaluation_context(tmp_path, monkeypatch):
    training = tmp_path / "training"
    training.mkdir()
    (training / "normalization_stats.json").write_text("{}\n")
    monkeypatch.setattr(entry, "TRAIN_OUTPUT", training)
    monkeypatch.setattr(entry, "CHECKPOINT", training / "models/model.pt")
    monkeypatch.setattr(entry, "TEST_DATASET", tmp_path / "dataset")
    monkeypatch.setattr(entry, "configure", lambda arguments: None)
    model = ZeroActionPolicy()
    monkeypatch.setattr(entry, "build_model", lambda *arguments: model)
    rendered = []
    monkeypatch.setattr(
        entry,
        "render_episode",
        lambda *arguments, **keywords: rendered.append((arguments, keywords)),
    )
    metrics = []
    monkeypatch.setattr(entry, "print_evaluation_mse", metrics.append)
    args = SimpleNamespace(
        max_episodes=None,
        chunk_size=4,
        temporal_ensemble=False,
        temporal_ensemble_k=0.01,
        output=tmp_path / "evaluation",
    )
    return SimpleNamespace(args=args, model=model, rendered=rendered, metrics=metrics)


@pytest.mark.parametrize("case", [(1, None), (2, None), (3, None), (10, None), (10, 2)])
def test_all_selected_episodes_contribute_to_error_but_only_three_are_plotted(
    case, evaluation_context, monkeypatch
):
    count, explicit_limit = case
    context = evaluation_context
    context.args.max_episodes = explicit_limit
    dataset = make_dataset(count)
    monkeypatch.setattr(entry, "build_dataset", lambda: dataset)
    output = entry.run_evaluation(context.args)
    selected = dataset.episodes[:explicit_limit]
    expected_ids = [episode.episode_id for episode in selected]
    assert len(context.model.calls) == sum(
        (episode.length + 3) // 4 for episode in selected
    )
    assert len(list((output / "artifacts").glob("*.npz"))) == len(selected)
    assert context.metrics[0]["frames"] == sum(episode.length for episode in selected)
    for episode in selected:
        with np.load(output / "artifacts" / f"{episode.episode_id}.npz") as artifact:
            np.testing.assert_array_equal(
                artifact["ground_truth"], episode.ground_truth
            )
            np.testing.assert_array_equal(
                artifact["prediction"], np.zeros_like(episode.ground_truth)
            )
    error_sum = sum(
        float(np.square(episode.ground_truth).sum()) for episode in selected
    )
    expected_mse = error_sum / (8 * context.metrics[0]["frames"])
    assert context.metrics[0]["joint_mse_sum"] == pytest.approx(7 * expected_mse)
    assert context.metrics[0]["gripper_mse"] == pytest.approx(expected_mse)
    assert len(context.rendered) == 2 * min(len(selected), 3)
    plotted_ids = [
        item[0][0].split("/")[-1].removesuffix(".npz") for item in context.rendered
    ]
    assert plotted_ids == [
        episode_id for episode_id in expected_ids[:3] for _ in range(2)
    ]
    metadata = json.loads((output / "evaluation_config.json").read_text())
    assert metadata["episode_ids"] == expected_ids
    assert metadata["plotted_episode_ids"] == expected_ids[:3]
