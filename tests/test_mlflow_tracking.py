"""Verify actual SDK histories against an isolated local MLflow store."""

import json

import pytest
from mlflow import MlflowClient

from jepa_policy.mlflow_tracking import (
    MlflowOptions,
    TrainingTracker,
    select_metrics,
    validate_http_authentication,
    validate_tracking_uri,
)


@pytest.fixture
def isolated_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "isolated tracking").as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("checkpoint-evaluation-test")
    options = MlflowOptions(
        tracking_uri=uri,
        experiment_name="checkpoint-evaluation-test",
        run_name="synthetic-isolated-verification",
    )
    return options, client, experiment_id


def test_training_and_evaluation_histories_share_run(isolated_tracking, tmp_path):
    options, client, experiment_id = isolated_tracking
    output = tmp_path / "training-output"
    with TrainingTracker(
        options, output, {"batch_size": 4, "loss_type": "mip"}
    ) as tracker:
        for step in (10, 20):
            tracker.log_train(
                {
                    "step": step,
                    "loss": 1 / step,
                    "learning_rate": 0.0001,
                    "grad_norm": 2.0,
                },
                step,
            )
            tracker.log_evaluation(
                {
                    "loss": 2 / step,
                    "loss_action": 1.5 / step,
                    "shadow_action_mse": 0.25,
                },
                step,
            )
        tracker.flush()
        run_id = tracker.run_id
    run = client.get_run(run_id)
    assert run.info.experiment_id == experiment_id
    assert run.info.status == "FINISHED"
    assert run.data.params == {"batch_size": "4", "loss_type": "mip"}
    for key in ("train/loss", "train/learning_rate", "train/grad_norm", "eval/loss"):
        assert [point.step for point in client.get_metric_history(run_id, key)] == [
            10,
            20,
        ]
    assert [
        point.value for point in client.get_metric_history(run_id, "eval/loss")
    ] == [0.2, 0.1]
    assert "train/step" not in run.data.metrics
    assert client.list_artifacts(run_id) == []
    identity = json.loads((output / "mlflow_run.json").read_text())
    assert identity["run_id"] == run_id
    assert identity["status"] == "FINISHED"


def test_training_failure_marks_run_failed(isolated_tracking, tmp_path):
    options, client, _ = isolated_tracking
    with pytest.raises(RuntimeError, match="training failure"):
        with TrainingTracker(options, tmp_path / "failed", {}) as tracker:
            tracker.log_train({"loss": 1.0}, 1)
            run_id = tracker.run_id
            raise RuntimeError("training failure")
    assert client.get_run(run_id).info.status == "FAILED"


def test_nonfinite_metric_fails_run_without_silent_substitution(
    isolated_tracking, tmp_path
):
    options, client, _ = isolated_tracking
    with pytest.raises(ValueError, match="finite"):
        with TrainingTracker(options, tmp_path / "nonfinite", {}) as tracker:
            run_id = tracker.run_id
            tracker.log_evaluation({"loss": float("nan")}, 10)
    assert client.get_run(run_id).info.status == "FAILED"
    assert client.get_metric_history(run_id, "eval/loss") == []


def test_missing_experiment_is_not_created(isolated_tracking, tmp_path):
    original, client, _ = isolated_tracking
    options = MlflowOptions(
        tracking_uri=original.tracking_uri, experiment_name="missing"
    )
    before = {experiment.experiment_id for experiment in client.search_experiments()}
    with pytest.raises(ValueError, match="missing or inactive"):
        with TrainingTracker(options, tmp_path / "missing", {}):
            pytest.fail("Missing experiment must not start a run")
    assert client.get_experiment_by_name("missing") is None
    assert {
        experiment.experiment_id for experiment in client.search_experiments()
    } == before


def test_existing_output_cannot_be_attached_to_a_new_run(isolated_tracking, tmp_path):
    options, client, experiment_id = isolated_tracking
    output = tmp_path / "existing"
    with TrainingTracker(options, output, {}):
        pass
    with pytest.raises(FileExistsError, match="run identity"):
        with TrainingTracker(options, output, {}):
            pytest.fail("Existing run identity must not be replaced")
    assert len(client.search_runs([experiment_id])) == 1


def test_disabled_tracker_never_contacts_sdk(tmp_path, monkeypatch):
    def unexpected_client(*arguments, **keywords):
        pytest.fail("Disabled tracking must not construct an SDK client")

    monkeypatch.setattr("mlflow.MlflowClient", unexpected_client)
    with TrainingTracker(MlflowOptions(enabled=False), tmp_path, {}) as tracker:
        tracker.log_train({"loss": 1.0}, 1)
        tracker.log_evaluation({"loss": 2.0}, 1)
        tracker.flush()
    assert not (tmp_path / "mlflow_run.json").exists()


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "./mlruns",
        "https:///",
        "https://host/path?auth=value",
        "https://host/#fragment",
    ],
)
def test_tracking_uri_rejects_ambiguous_or_query_configuration(uri):
    with pytest.raises(ValueError):
        validate_tracking_uri(uri)


@pytest.mark.parametrize(
    "uri",
    [
        "https://redaction-marker@host\uff0fpath",
        "https://redaction-marker@[invalid-ipv6",
    ],
)
def test_invalid_uri_parser_errors_do_not_expose_userinfo(uri):
    with pytest.raises(ValueError, match="^Invalid MLflow tracking URI$") as caught:
        validate_tracking_uri(uri)
    assert "redaction-marker" not in str(caught.value)
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize("prefix", ["train", "eval"])
def test_metrics_select_known_scalars_and_reject_nonfinite_values(prefix):
    assert select_metrics({"loss": 0.5, "step": 10, "path": "unused"}, prefix) == {
        f"{prefix}/loss": 0.5
    }
    with pytest.raises(ValueError, match="finite"):
        select_metrics({"loss": float("inf")}, prefix)
    with pytest.raises(TypeError, match="scalar"):
        select_metrics({"loss": "incorrect"}, prefix)
    with pytest.raises(ValueError, match="Missing"):
        select_metrics({"step": 10}, prefix)


def test_http_requires_complete_authentication_in_launch_environment(monkeypatch):
    for name in (
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_TRACKING_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="missing"):
        validate_http_authentication("https://mlflow.example")


def test_parameter_types_are_checked_before_run_creation(tmp_path):
    with pytest.raises(TypeError, match="scalar"):
        TrainingTracker(MlflowOptions(), tmp_path, {"batch_size": [16]})
    with pytest.raises(ValueError, match="finite"):
        TrainingTracker(MlflowOptions(), tmp_path, {"learning_rate": float("nan")})


def test_logging_requires_optimizer_step_consistency(isolated_tracking, tmp_path):
    options, _, _ = isolated_tracking
    with TrainingTracker(options, tmp_path / "step-check", {}) as tracker:
        with pytest.raises(ValueError, match="differs"):
            tracker.log_train({"loss": 1.0, "step": 9}, 10)
        with pytest.raises(ValueError, match="positive"):
            tracker.log_evaluation({"loss": 1.0}, 0)


def test_finalization_keeps_original_training_exception(
    isolated_tracking, tmp_path, monkeypatch
):
    options, _, _ = isolated_tracking
    with pytest.raises(RuntimeError, match="original") as caught:
        with TrainingTracker(options, tmp_path / "finalization", {}) as tracker:

            def cannot_finish(*arguments, **keywords):
                raise OSError("synthetic tracking failure")

            monkeypatch.setattr(tracker.client, "set_terminated", cannot_finish)
            raise RuntimeError("original")
    assert "OSError" in caught.value.__notes__[0]
