"""Check configurable executed chunks without loading checkpoint weights."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from model_eval import Dataset, Episode, evaluate, load_episode

from examples import evaluate_x1_icedtea as entry


class FakeTrainingAgent:
    def __init__(self, config):
        self.config = config
        self.sample_calls = []
        self.load_calls = []
        self.evaluating = False

    def load(self, path, load_optimizer=False):
        self.load_calls.append((path, load_optimizer))

    def eval(self):
        self.evaluating = True

    def sample(self, **arguments):
        self.sample_calls.append(arguments)
        current_frame = arguments["obs"]["agent_pos"][0, -1, 0]
        frame_offsets = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
        prediction = (frame_offsets + current_frame - 1).expand(1, 10, 8)
        return prediction, None


@pytest.fixture
def configured_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "TRAIN_OUTPUT", tmp_path)
    monkeypatch.setattr(entry, "TEST_DATASET", tmp_path)
    monkeypatch.setattr(entry, "CHECKPOINT", tmp_path / "checkpoint.pt")
    monkeypatch.setattr(entry, "DEVICE", "cpu")
    monkeypatch.setattr(
        entry,
        "TRAINING_STATS",
        {"features": {"action": {"min": [-1.0] * 8, "range": [2.0] * 8}}},
    )
    monkeypatch.setattr(entry, "TrainingAgent", FakeTrainingAgent)
    return entry.IcedTeaPolicy


@pytest.mark.parametrize(("argument", "expected"), [(None, 4), ("8", 8), ("9", 9)])
def test_chunk_size_command_line(monkeypatch, argument, expected):
    arguments = [
        "evaluate_x1_icedtea.py",
        "--dataset",
        "/prepared-dataset",
        "--checkpoint",
        "/trained-model.pt",
    ]
    if argument is not None:
        arguments.extend(["--chunk-size", argument])
    monkeypatch.setattr(sys, "argv", arguments)

    assert entry.parse_arguments().chunk_size == expected


@pytest.mark.parametrize("argument", ["0", "10"])
def test_invalid_chunk_size_command_line(monkeypatch, argument):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_x1_icedtea.py",
            "--dataset",
            "/prepared-dataset",
            "--checkpoint",
            "/trained-model.pt",
            "--chunk-size",
            argument,
        ],
    )

    with pytest.raises(SystemExit) as error:
        entry.parse_arguments()

    assert error.value.code == 2


@pytest.mark.parametrize("chunk_size", [1, 4, 8, 9])
def test_executed_chunk_changes_without_shortening_model_horizon(
    configured_policy, chunk_size
):
    policy = configured_policy(chunk_size)
    observation = {"agent_pos": torch.tensor([[7.0] * 8, [8.0] * 8])}

    prediction = policy.predict(observation)

    expected = np.repeat(
        np.arange(8, 8 + chunk_size, dtype=np.float32).reshape(-1, 1), 8, axis=1
    )
    np.testing.assert_array_equal(prediction, expected)
    call = policy.agent.sample_calls[0]
    assert call["act_0"].shape == (1, 10, 8)
    assert torch.count_nonzero(call["act_0"]) == 0
    assert call["obs"]["agent_pos"].shape == (1, 2, 8)
    assert call["num_steps"] == 1
    assert call["use_ema"] is True
    assert call["return_future"] is True
    assert policy.agent.load_calls == [(str(Path(entry.CHECKPOINT)), False)]
    assert policy.agent.evaluating is True


@pytest.mark.parametrize("chunk_size", [0, 10])
def test_direct_policy_rejects_unavailable_future_actions(
    configured_policy, chunk_size
):
    with pytest.raises(ValueError, match="between 1 and 9"):
        configured_policy(chunk_size)


def test_four_step_execution_replans_and_covers_final_partial_chunk(
    configured_policy, tmp_path
):
    policy = configured_policy(4)
    observed_frames = []

    def observe(frame):
        observed_frames.append(frame)
        return {"agent_pos": torch.tensor([[frame - 1.0] * 8, [float(frame)] * 8])}

    ground_truth = np.repeat(np.arange(11, dtype=np.float32).reshape(-1, 1), 8, axis=1)
    dataset = Dataset([Episode("partial_tail", 11, ground_truth, observe)])

    results = evaluate(policy, dataset, tmp_path / "artifacts")
    artifact = load_episode(results[0]["path"])

    assert observed_frames == [0, 4, 8]
    assert artifact.chunk_size == 4
    assert artifact.metadata["inference_points"] == [0, 4, 8]
    np.testing.assert_array_equal(artifact.decision_points, np.arange(11))
    np.testing.assert_array_equal(artifact.prediction, ground_truth)
    assert entry.calculate_evaluation_mse(results)["joint_mse_sum"] == 0.0
