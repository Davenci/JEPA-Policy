"""Exercise checkpoint scheduling and real MLflow histories without a GPU job."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from mlflow import MlflowClient

from examples import train_x1_icedtea as training
from jepa_policy import training_monitor
from jepa_policy.checkpoint_evaluation import (
    CheckpointEvaluationOptions,
    CheckpointEvaluator,
)
from jepa_policy.datasets.icedtea_dataset import ACTION_NAMES
from jepa_policy.datasets.icedtea_split import (
    EpisodeSplitOptions,
    prepare_episode_split,
)
from jepa_policy.mlflow_tracking import MlflowOptions
from jepa_policy.training_artifacts import prepare_run_directory
from jepa_policy.training_monitor import TrainingMonitor, TrainingMonitoringOptions


class EpisodeCollection:
    def __init__(self, ids):
        self.episodes = dict.fromkeys(ids)

    def __len__(self):
        return len(self.episodes) * 3


def make_config(output, steps=4):
    optimization = training.create_optimization_config(steps, 2, 0)
    optimization.device = "cpu"
    return training.create_config(output / "dataset", output, optimization)


@pytest.fixture
def local_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "isolated tracking").as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("training-monitor-integration")
    options = MlflowOptions(
        tracking_uri=uri,
        experiment_name="training-monitor-integration",
        run_name="synthetic-monitor-verification",
    )
    return SimpleNamespace(options=options, client=client, experiment_id=experiment_id)


def assert_monitor_artifacts_and_history(output, run_id, tracking):
    records = [
        json.loads(line)
        for line in (output / "evaluation_metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2]
    assert [record["loss"] for record in records] == [2.0, 1.0]
    assert (output / "loss_curve.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for name in ("train/loss", "eval/loss", "eval/shadow_action_mse"):
        assert [
            point.step for point in tracking.client.get_metric_history(run_id, name)
        ] == [1, 2]
    assert tracking.client.get_run(run_id).info.status == "FINISHED"
    assert len(tracking.client.search_runs([tracking.experiment_id])) == 1
    assert tracking.client.list_artifacts(run_id) == []


def test_checkpoint_monitor_saves_then_evaluates_and_records_histories(
    tmp_path, monkeypatch, local_tracking
):
    output = tmp_path / "run"
    (output / "models").mkdir(parents=True)
    events = []

    def save_checkpoint(path, training_state):
        step = training_state["n_gradient_step"] + 1
        events.append(("save", step))
        path.write_text(json.dumps({"step": step}))

    def evaluate_checkpoint(agent, path, step):
        assert json.loads(path.read_text())["step"] == step
        events.append(("evaluate", step))
        return {
            "loss": 2.0 / step,
            "shadow_action_mse": 0.25 / step,
            "joint_mse_sum": 0.1 / step,
            "gripper_mse": 0.2 / step,
            "windows": 9,
            "frames": 20,
            "episodes": 1,
        }

    monkeypatch.setattr(
        training_monitor,
        "CheckpointEvaluator",
        lambda *arguments: SimpleNamespace(evaluate=evaluate_checkpoint),
    )
    options = TrainingMonitoringOptions(output, local_tracking.options)
    with TrainingMonitor(
        make_config(output),
        (EpisodeCollection(range(19)), EpisodeCollection([19])),
        options,
    ) as monitor:
        for step in (1, 2):
            record = {"step": step, "loss": 1.0 / step, "learning_rate": 1e-4}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            monitor.log_training(record)
            monitor.checkpoint(SimpleNamespace(save=save_checkpoint), step - 1)
        run_id = monitor.tracker.run_id
    assert events == [("save", 1), ("evaluate", 1), ("save", 2), ("evaluate", 2)]
    assert_monitor_artifacts_and_history(output, run_id, local_tracking)


@pytest.mark.parametrize(
    "case",
    [(4, 2, [2, 4]), (5, 2, [2, 4, 5]), (5, 0, [5]), (10, 1, list(range(1, 11)))],
)
def test_training_loop_saves_each_due_checkpoint_and_final_only_once(
    tmp_path, monkeypatch, case
):
    steps, save_every, checkpoint_steps = case
    config = make_config(tmp_path, steps)
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=config.optimization.lr)
    agent = SimpleNamespace(optimizer=optimizer, train=lambda: None)
    calls = SimpleNamespace(checkpoints=[], records=[])
    monkeypatch.setattr(training, "TrainingAgent", lambda config: agent)
    monkeypatch.setattr(training, "create_dataloader", lambda *arguments: [{}] * steps)
    monkeypatch.setattr(training, "record_model_info", lambda *arguments: None)
    monkeypatch.setattr(training, "record_first_batch", lambda *arguments: None)
    monkeypatch.setattr(
        training,
        "prepare_run_directory",
        lambda *arguments: (tmp_path, tmp_path / "metrics.jsonl"),
    )
    monkeypatch.setattr(training.torch.cuda, "max_memory_allocated", lambda: 0)

    def update(*arguments):
        optimizer.step()
        return {"loss": 1.0}

    def checkpoint(agent, step):
        calls.checkpoints.append(step + 1)
        return tmp_path / f"model_step_{step + 1:06d}.pt"

    monkeypatch.setattr(training, "update_agent", update)
    monitor = SimpleNamespace(log_training=calls.records.append, checkpoint=checkpoint)
    options = training.TrainingRunOptions(tmp_path, save_every, None, log_every=3)
    training.train(config, object(), options, monitor)
    assert calls.checkpoints == checkpoint_steps
    local_records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [point["step"] for point in local_records] == list(range(1, steps + 1))
    logged_steps = [point["step"] for point in calls.records]
    expected = sorted({1, *range(3, steps + 1, 3), *checkpoint_steps})
    assert logged_steps == expected
    assert len(logged_steps) == len(set(logged_steps))


def create_frame_dataset(root: Path, output: Path):
    metadata_dir = root / "meta/episodes/chunk-000"
    data_dir = root / "data/chunk-000"
    metadata_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    features = {key: {"names": ACTION_NAMES} for key in ("action", "observation.state")}
    (root / "meta/info.json").write_text(json.dumps({"features": features}))
    records = [
        {
            "episode_index": index,
            "dataset_from_index": index * 12,
            "dataset_to_index": (index + 1) * 12,
        }
        for index in range(20)
    ]
    pq.write_table(pa.Table.from_pylist(records), metadata_dir / "file-000.parquet")
    split = prepare_episode_split(root, EpisodeSplitOptions(output))
    frames = []
    for episode_id in range(20):
        for frame in range(12):
            value = float(episode_id * 12 + frame)
            if episode_id in split.evaluation:
                value += 1_000_000.0
            frames.append(
                {
                    "action": [value] * 8,
                    "observation.state": [value * 2] * 8,
                    "timestamp": frame / 15.0,
                    "episode_index": episode_id,
                    "task_index": episode_id % 2,
                }
            )
    pq.write_table(pa.Table.from_pylist(frames), data_dir / "file-000.parquet")
    return split


def test_actual_training_loaders_exclude_holdout_extremes_and_save_training_stats(
    tmp_path,
):
    root, output = tmp_path / "dataset", tmp_path / "run"
    split = create_frame_dataset(root, output)
    args = SimpleNamespace(
        dataset=root, output=output, episode_indices=None, resume=None, split_seed=0
    )
    config = make_config(output)
    train_dataset, eval_dataset = training.load_training_datasets(args, config)
    assert tuple(train_dataset.episodes) == split.train
    assert tuple(eval_dataset.episodes) == split.evaluation
    assert not set(train_dataset.episodes) & set(eval_dataset.episodes)
    assert train_dataset.normalize is True
    assert eval_dataset.normalize is False
    assert not hasattr(eval_dataset, "state_min")
    assert train_dataset.action_min.max() < 1_000
    assert train_dataset.action_range.max() < 1_000
    np.testing.assert_array_equal(eval_dataset.actions, eval_dataset.raw_actions)
    prepare_run_directory(output, train_dataset, None)
    stats = json.loads((output / "normalization_stats.json").read_text())
    assert stats["episode_indices"] == list(split.train)
    np.testing.assert_array_equal(
        stats["features"]["action"]["min"], train_dataset.action_min
    )
    evaluator = CheckpointEvaluator(
        config, (train_dataset, eval_dataset), CheckpointEvaluationOptions(output)
    )
    heldout = np.isin(eval_dataset.episode_ids, split.evaluation)
    expected = (
        eval_dataset.raw_states - train_dataset.state_min
    ) / train_dataset.state_range * 2 - 1
    np.testing.assert_array_equal(evaluator.states, expected)
    assert evaluator.states[heldout].min() > 1_000
    np.testing.assert_array_equal(eval_dataset.states, eval_dataset.raw_states)
