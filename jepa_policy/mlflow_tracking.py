"""Record existing training and checkpoint evaluation scalars in one MLflow run."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class MlflowOptions:
    enabled: bool = True
    tracking_uri: str = "https://mlflow.seer-robotics.cn"
    experiment_name: str = "pretrain-fm"
    run_name: str | None = None


def validate_tracking_uri(tracking_uri: str) -> None:
    try:
        parsed = urlsplit(tracking_uri)
    except ValueError:
        raise ValueError("Invalid MLflow tracking URI") from None
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("MLflow tracking URI must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("MLflow tracking URI must not contain a query or fragment")
    if parsed.scheme not in {"http", "https", "file", "sqlite"}:
        raise ValueError("An explicit HTTP(S), file or SQLite tracking URI is required")
    if parsed.scheme in {"http", "https"} and not parsed.hostname:
        raise ValueError("MLflow tracking URI requires a hostname")
    if parsed.scheme == "file" and (parsed.netloc or not parsed.path.startswith("/")):
        raise ValueError("File tracking requires an absolute local file URI")


def validate_http_authentication(tracking_uri: str) -> None:
    if urlsplit(tracking_uri).scheme not in {"http", "https"}:
        return
    username = bool(os.environ.get("MLFLOW_TRACKING_USERNAME"))
    password = bool(os.environ.get("MLFLOW_TRACKING_PASSWORD"))
    token = bool(os.environ.get("MLFLOW_TRACKING_TOKEN"))
    if username != password:
        raise ValueError("MLflow Basic authentication needs both injected fields")
    if token and username:
        raise ValueError("MLflow Basic and Bearer authentication must not be mixed")
    if not (username or token):
        raise ValueError(
            "MLflow authentication is missing in this launch environment; "
            "inject MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD "
            "or the service-approved MLFLOW_TRACKING_TOKEN"
        )


def validate_parameters(parameters: dict) -> None:
    for key, value in parameters.items():
        if any(
            part in key.lower()
            for part in ("password", "token", "secret", "credential")
        ):
            raise ValueError("Sensitive fields must not be logged as MLflow parameters")
        if not isinstance(value, (str, Real, bool)):
            raise TypeError(f"MLflow parameter must be an explicit scalar: {key}")
        if isinstance(value, Real) and not math.isfinite(float(value)):
            raise ValueError(f"MLflow parameter must be finite: {key}")


def select_metrics(record: dict, prefix: str) -> dict[str, float]:
    scalar_names = {
        "learning_rate",
        "grad_norm",
        "data_wait_seconds",
        "update_seconds",
        "gpu_peak_memory_gib",
        "weighted_future_loss",
        "shadow_action_mse",
        "joint_mse_sum",
        "joint_mse_mean",
        "gripper_mse",
        "windows",
        "frames",
        "episodes",
        "seconds",
        "future_head_only_stopgrad",
    }
    selected = {}
    for key, value in record.items():
        if not (key == "loss" or key.startswith("loss_") or key in scalar_names):
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"MLflow metric must be an existing scalar: {key}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"MLflow metric must be finite: {key}")
        selected[f"{prefix}/{key}"] = number
    if f"{prefix}/loss" not in selected:
        raise ValueError(f"Missing required {prefix}/loss metric")
    return selected


class TrainingTracker:
    """Create one run without changing global fluent state or uploading artifacts."""

    def __init__(self, options: MlflowOptions, output: Path, parameters: dict):
        self.options = options
        self.output = Path(output)
        self.parameters = parameters
        self.client = None
        self.run_id = None
        self.experiment_id = None
        self.entered = False
        if options.enabled:
            validate_tracking_uri(options.tracking_uri)
            validate_parameters(parameters)
            if not options.experiment_name:
                raise ValueError("MLflow experiment name must not be empty")

    def __enter__(self):
        if self.entered:
            raise RuntimeError("MLflow tracker cannot be entered twice")
        self.entered = True
        if not self.options.enabled:
            return self
        from mlflow import MlflowClient
        from mlflow.entities import Param

        validate_http_authentication(self.options.tracking_uri)
        if (self.output / "mlflow_run.json").exists():
            raise FileExistsError("MLflow output already contains a run identity")
        self.client = MlflowClient(tracking_uri=self.options.tracking_uri)
        experiment = self.client.get_experiment_by_name(self.options.experiment_name)
        if experiment is None or experiment.lifecycle_stage != "active":
            raise ValueError("The specified MLflow experiment is missing or inactive")
        self.experiment_id = experiment.experiment_id
        name = self.options.run_name or self.output.name
        run = self.client.create_run(self.experiment_id, tags={"mlflow.runName": name})
        self.run_id = run.info.run_id
        try:
            self.write_identity("RUNNING")
            self.client.log_batch(
                self.run_id,
                params=[
                    Param(key, str(value)) for key, value in self.parameters.items()
                ],
                synchronous=True,
            )
        except BaseException as error:
            self.finish("FAILED", error)
            raise
        return self

    def __exit__(self, exception_type, exception, traceback):
        if self.options.enabled and self.run_id is not None:
            self.finish(
                "FAILED" if exception_type is not None else "FINISHED", exception
            )
        self.entered = False
        return False

    def write_identity(self, status: str) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        identity = {
            "tracking_uri": self.options.tracking_uri,
            "experiment_name": self.options.experiment_name,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "run_name": self.options.run_name or self.output.name,
            "status": status,
            "step_axis": "one_based_optimizer_step",
            "uploads_artifacts": False,
        }
        (self.output / "mlflow_run.json").write_text(
            json.dumps(identity, indent=2) + "\n", encoding="utf-8"
        )

    def finish(self, status: str, original_error: BaseException | None) -> None:
        try:
            self.client.set_terminated(self.run_id, status=status)
            self.write_identity(status)
        except Exception as tracking_error:
            if original_error is None:
                raise
            original_error.add_note(
                f"MLflow finalization also failed: {type(tracking_error).__name__}"
            )

    def log_train(self, record: dict, step: int) -> None:
        if not self.options.enabled:
            return
        if "step" in record and record["step"] != step:
            raise ValueError(
                "MLflow training step differs from the existing log record"
            )
        self.log_metrics(select_metrics(record, "train"), step)

    def log_evaluation(self, metrics: dict, step: int) -> None:
        if self.options.enabled:
            self.log_metrics(select_metrics(metrics, "eval"), step)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not self.entered or self.run_id is None:
            raise RuntimeError("Enter the MLflow tracker before logging metrics")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("MLflow requires a positive one-based optimizer step")
        from mlflow.entities import Metric

        timestamp = int(time.time() * 1000)
        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(key, value, timestamp, step) for key, value in metrics.items()
            ],
            synchronous=True,
        )

    def flush(self) -> None:
        """Metrics are synchronous; no pending queue remains after each log call."""
        if self.options.enabled and (not self.entered or self.run_id is None):
            raise RuntimeError("Enter the MLflow tracker before flushing metrics")
