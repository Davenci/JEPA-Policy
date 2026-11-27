"""Check ACT temporal aggregation through the evaluation adapter on CPU."""

import sys

import numpy as np
import pytest
import torch
from model_eval import Dataset, Episode, evaluate, load_episode

from examples import evaluate_x1_icedtea as entry


class MarkerAgent:
    history_value = None
    tail_value = None

    def __init__(self, config):
        self.config = config
        self.sample_calls = []

    def load(self, path, load_optimizer=False):
        pass

    def eval(self):
        pass

    def sample(self, **arguments):
        self.sample_calls.append(arguments)
        marker = arguments["obs"]["agent_pos"][0, -1, 0]
        offsets = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
        prediction = (marker + offsets).expand(1, 10, 8).clone()
        if self.history_value is not None:
            prediction[:, 0, :] = self.history_value
        if self.tail_value is not None:
            prediction[:, 9, :] = self.tail_value
        return prediction, None


@pytest.fixture
def policy_factory(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "TRAIN_OUTPUT", tmp_path)
    monkeypatch.setattr(entry, "TEST_DATASET", tmp_path)
    monkeypatch.setattr(entry, "CHECKPOINT", tmp_path / "model.pt")
    monkeypatch.setattr(entry, "DEVICE", "cpu")
    monkeypatch.setattr(
        entry,
        "TRAINING_STATS",
        {"features": {"action": {"min": [-1.0] * 8, "range": [2.0] * 8}}},
    )
    monkeypatch.setattr(entry, "TrainingAgent", MarkerAgent)
    return entry.IcedTeaPolicy


def observation(marker):
    return {"agent_pos": torch.full((2, 8), float(marker))}


def parse_cli(monkeypatch, options):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_x1_icedtea.py",
            "--dataset",
            "/prepared",
            "--checkpoint",
            "/run/models/model.pt",
            *options,
        ],
    )
    return entry.parse_arguments()


def make_episode(name, marker, ground_truth):
    def observe(frame):
        return observation(marker + frame * 10)

    return Episode(name, len(ground_truth), ground_truth, observe)


def test_temporal_ensemble_cli_defaults_and_explicit_coefficient(monkeypatch):
    defaults = parse_cli(monkeypatch, [])
    assert defaults.temporal_ensemble is False
    assert defaults.chunk_size == 4
    assert defaults.temporal_ensemble_k == 0.01

    options = parse_cli(
        monkeypatch,
        ["--temporal-ensemble", "--chunk-size", "1", "--temporal-ensemble-k", "0.1"],
    )
    assert options.temporal_ensemble is True
    assert options.chunk_size == 1
    assert options.temporal_ensemble_k == 0.1


@pytest.mark.parametrize(
    "options",
    [
        ["--temporal-ensemble"],
        ["--temporal-ensemble", "--chunk-size", "4"],
        ["--temporal-ensemble", "--chunk-size", "8"],
        ["--temporal-ensemble", "--chunk-size", "9"],
        ["--temporal-ensemble", "--chunk-size", "1", "--temporal-ensemble-k", "-0.1"],
        ["--temporal-ensemble", "--chunk-size", "1", "--temporal-ensemble-k", "nan"],
        ["--temporal-ensemble", "--chunk-size", "1", "--temporal-ensemble-k", "inf"],
    ],
)
def test_temporal_ensemble_cli_rejects_invalid_settings(monkeypatch, options):
    with pytest.raises(SystemExit) as error:
        parse_cli(monkeypatch, options)
    assert error.value.code == 2


def test_temporal_ensemble_aligns_all_still_valid_chunks(policy_factory):
    policy = policy_factory(1, temporal_ensemble=True, temporal_ensemble_k=0.01)
    first = policy.predict(observation(0))
    second = policy.predict(observation(10))
    third = policy.predict(observation(20))

    weight = np.exp(-0.01)
    np.testing.assert_array_equal(first, np.ones((1, 8), dtype=np.float32))
    np.testing.assert_allclose(second, (2.0 + weight * 11.0) / (1.0 + weight))
    np.testing.assert_allclose(
        third,
        (3.0 + weight * 12.0 + weight**2 * 21.0) / (1.0 + weight + weight**2),
    )
    assert first.shape == second.shape == third.shape == (1, 8)
    assert all(call["act_0"].shape == (1, 10, 8) for call in policy.agent.sample_calls)
    assert all(call["return_future"] is True for call in policy.agent.sample_calls)


def test_full_nine_step_prediction_keeps_last_future_action(
    policy_factory, monkeypatch
):
    monkeypatch.setattr(MarkerAgent, "history_value", -1e8)
    monkeypatch.setattr(MarkerAgent, "tail_value", 9000.0)
    policy = policy_factory(1, temporal_ensemble=True)
    actions = [policy.predict(observation(frame * 10)) for frame in range(9)]

    candidates = np.asarray([9000.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0, 81.0])
    weights = np.exp(-0.01 * np.arange(9))
    expected = np.dot(candidates, weights) / weights.sum()
    np.testing.assert_array_equal(actions[0], np.ones((1, 8), dtype=np.float32))
    np.testing.assert_allclose(actions[-1], expected, rtol=1e-6)
    assert policy.temporal_ensembler.prediction_horizon == 9


def test_temporal_ensemble_retains_valid_zero_candidates(policy_factory):
    policy = policy_factory(1, temporal_ensemble=True)
    policy.predict(observation(-2))
    current = policy.predict(observation(10))

    expected = np.exp(-0.01) * 11.0 / (1.0 + np.exp(-0.01))
    np.testing.assert_allclose(current, expected, rtol=1e-6)


@pytest.mark.parametrize("chunk_size", [4, 8, 9])
def test_temporal_ensemble_requires_one_executed_action(policy_factory, chunk_size):
    with pytest.raises(ValueError, match="chunk"):
        policy_factory(chunk_size, temporal_ensemble=True)


@pytest.mark.parametrize("chunk_size", [4, 8])
def test_disabled_temporal_ensemble_preserves_chunk_output(policy_factory, chunk_size):
    policy = policy_factory(chunk_size)
    assert policy.temporal_ensembler is None

    expected = np.repeat(
        np.arange(11, 11 + chunk_size, dtype=np.float32).reshape(-1, 1), 8, axis=1
    )
    np.testing.assert_array_equal(policy.predict(observation(10)), expected)
    policy.reset_episode()
    np.testing.assert_array_equal(policy.predict(observation(10)), expected)


def test_evaluator_resets_aggregation_between_episodes(policy_factory, tmp_path):
    policy = policy_factory(1, temporal_ensemble=True)
    first = make_episode("first", 0, np.zeros((3, 8), dtype=np.float32))
    second = make_episode("second", 100, np.full((2, 8), -500.0, dtype=np.float32))

    records = evaluate(policy, Dataset([first, second]), tmp_path / "artifacts")
    first_artifact = load_episode(records[0]["path"])
    second_artifact = load_episode(records[1]["path"])

    assert first_artifact.metadata["inference_points"] == [0, 1, 2]
    assert second_artifact.metadata["inference_points"] == [0, 1]
    np.testing.assert_array_equal(second_artifact.prediction[0], np.full(8, 101.0))
    np.testing.assert_allclose(
        second_artifact.prediction[1],
        (102.0 + np.exp(-0.01) * 111.0) / (1.0 + np.exp(-0.01)),
    )
    np.testing.assert_array_equal(second_artifact.ground_truth, second.ground_truth)


def test_ground_truth_never_participates_in_aggregation(policy_factory, tmp_path):
    policy = policy_factory(1, temporal_ensemble=True)
    episode_a = make_episode("a", 2, np.zeros((3, 8), dtype=np.float32))
    episode_b = make_episode("b", 2, np.full((3, 8), 5000.0, dtype=np.float32))

    records = evaluate(policy, Dataset([episode_a, episode_b]), tmp_path / "artifacts")
    prediction_a = load_episode(records[0]["path"]).prediction
    prediction_b = load_episode(records[1]["path"]).prediction

    np.testing.assert_array_equal(prediction_a, prediction_b)
