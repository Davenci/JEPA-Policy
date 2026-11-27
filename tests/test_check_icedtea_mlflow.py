"""Keep the training-shell inspection strictly read-only."""

from types import SimpleNamespace

import pytest

from tools.check_icedtea_mlflow import (
    TrackingInspectionOptions,
    inspect_tracking,
    summarize_metric_history,
)


class ReadOnlyTrackingClient:
    def __init__(self):
        self.reads = []
        self.experiment = SimpleNamespace(experiment_id="11", lifecycle_stage="active")
        self.run = SimpleNamespace(
            info=SimpleNamespace(experiment_id="11", status="FINISHED")
        )
        self.history = [
            SimpleNamespace(step=1000, value=0.2),
            SimpleNamespace(step=2000, value=0.1),
        ]

    def get_experiment_by_name(self, name):
        self.reads.append(("experiment", name))
        return self.experiment

    def get_run(self, run_id):
        self.reads.append(("run", run_id))
        return self.run

    def get_metric_history(self, run_id, name):
        self.reads.append(("history", run_id, name))
        return self.history


@pytest.fixture
def inspection_client(monkeypatch):
    client = ReadOnlyTrackingClient()
    monkeypatch.setattr(
        "tools.check_icedtea_mlflow.MlflowClient", lambda **kwargs: client
    )
    return client


def test_experiment_inspection_only_reads_existing_experiment(inspection_client):
    options = TrackingInspectionOptions(tracking_uri="file:///tmp/isolated-inspection")
    result = inspect_tracking(options)
    assert result["ok"] is True
    assert result["run_history_checked"] is False
    assert result["experiment_id"] == "11"
    assert result["server"] == {"version": None, "version_checked": False}
    assert all(
        isinstance(value, bool) for value in result["authentication_present"].values()
    )
    assert inspection_client.reads == [("experiment", "pretrain-fm")]


def test_run_inspection_counts_complete_metric_histories(inspection_client):
    options = TrackingInspectionOptions(
        tracking_uri="file:///tmp/isolated-inspection", run_id="synthetic-existing-run"
    )
    result = inspect_tracking(options)
    assert result["run_history_checked"] is True
    assert result["run_status"] == "FINISHED"
    for name in ("train/loss", "eval/loss"):
        assert result["metrics"][name] == {
            "points": 2,
            "distinct_steps": 2,
            "min_step": 1000,
            "max_step": 2000,
            "finite_values": True,
        }
    assert inspection_client.reads == [
        ("experiment", "pretrain-fm"),
        ("run", "synthetic-existing-run"),
        ("history", "synthetic-existing-run", "train/loss"),
        ("history", "synthetic-existing-run", "eval/loss"),
    ]


@pytest.mark.parametrize("lifecycle_stage", [None, "deleted"])
def test_inspection_rejects_missing_or_inactive_experiment(
    inspection_client, lifecycle_stage
):
    inspection_client.experiment = (
        None
        if lifecycle_stage is None
        else SimpleNamespace(experiment_id="11", lifecycle_stage=lifecycle_stage)
    )
    with pytest.raises(ValueError, match="missing or inactive"):
        inspect_tracking(TrackingInspectionOptions(tracking_uri="file:///tmp/isolated"))
    assert len(inspection_client.reads) == 1


def test_run_from_another_experiment_is_rejected(inspection_client):
    inspection_client.run.info.experiment_id = "12"
    with pytest.raises(ValueError, match="does not belong"):
        inspect_tracking(
            TrackingInspectionOptions(
                tracking_uri="file:///tmp/isolated", run_id="other-run"
            )
        )
    assert not any(read[0] == "history" for read in inspection_client.reads)


@pytest.mark.parametrize(
    "history",
    [
        [],
        [SimpleNamespace(step=1000, value=0.2)],
        [SimpleNamespace(step=1000, value=0.2), SimpleNamespace(step=1000, value=0.1)],
        [
            SimpleNamespace(step=1000, value=float("nan")),
            SimpleNamespace(step=2000, value=0.1),
        ],
        [SimpleNamespace(step=0, value=0.2), SimpleNamespace(step=2000, value=0.1)],
    ],
)
def test_invalid_or_single_step_history_is_not_reported_as_verified(
    inspection_client, history
):
    inspection_client.history = history
    with pytest.raises(ValueError):
        summarize_metric_history(inspection_client, "existing-run", "eval/loss")


def test_repeated_steps_are_counted_separately_from_distinct_steps(inspection_client):
    inspection_client.history.append(SimpleNamespace(step=2000, value=0.09))
    result = summarize_metric_history(inspection_client, "existing-run", "eval/loss")
    assert result["points"] == 3
    assert result["distinct_steps"] == 2
