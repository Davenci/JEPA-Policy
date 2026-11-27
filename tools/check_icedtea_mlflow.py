"""Inspect MLflow access and existing metric histories from the training shell."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow import MlflowClient

from jepa_policy.mlflow_tracking import (
    validate_http_authentication,
    validate_tracking_uri,
)


@dataclass(frozen=True)
class TrackingInspectionOptions:
    tracking_uri: str = "https://mlflow.seer-robotics.cn"
    experiment_name: str = "pretrain-fm"
    run_id: str | None = None


def authentication_presence() -> dict[str, bool]:
    return {
        name: bool(os.environ.get(name))
        for name in (
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD",
            "MLFLOW_TRACKING_TOKEN",
        )
    }


def summarize_metric_history(client, run_id: str, metric_name: str) -> dict:
    history = client.get_metric_history(run_id, metric_name)
    if not history:
        raise ValueError(f"Requested metric has no recorded history: {metric_name}")
    if not all(math.isfinite(point.value) for point in history):
        raise ValueError(f"Requested metric has non-finite values: {metric_name}")
    steps = {point.step for point in history}
    if len(steps) < 2:
        raise ValueError(
            f"Metric requires at least two distinct optimizer steps: {metric_name}"
        )
    if min(steps) <= 0:
        raise ValueError(f"Metric uses an invalid optimizer step: {metric_name}")
    return {
        "points": len(history),
        "distinct_steps": len(steps),
        "min_step": min(steps),
        "max_step": max(steps),
        "finite_values": True,
    }


def inspect_tracking(options: TrackingInspectionOptions) -> dict:
    validate_tracking_uri(options.tracking_uri)
    validate_http_authentication(options.tracking_uri)
    client = MlflowClient(tracking_uri=options.tracking_uri)
    experiment = client.get_experiment_by_name(options.experiment_name)
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError("The specified MLflow experiment is missing or inactive")
    result = {
        "ok": True,
        "mode": "read_only",
        "client_version": mlflow.__version__,
        "server": {"version": None, "version_checked": False},
        "authentication_present": authentication_presence(),
        "experiment_name": options.experiment_name,
        "experiment_id": experiment.experiment_id,
        "experiment_status": experiment.lifecycle_stage,
        "run_history_checked": False,
    }
    if options.run_id is None:
        return result
    run = client.get_run(options.run_id)
    if run.info.experiment_id != experiment.experiment_id:
        raise ValueError("Requested run does not belong to the specified experiment")
    result.update(
        run_id=options.run_id,
        run_status=run.info.status,
        run_history_checked=True,
        metrics={
            name: summarize_metric_history(client, options.run_id, name)
            for name in ("train/loss", "eval/loss")
        },
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="https://mlflow.seer-robotics.cn")
    parser.add_argument("--experiment", default="pretrain-fm")
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    # 1. 从当前训练终端读取非敏感连接参数，认证仍由原进程环境注入。
    args = parse_arguments()
    options = TrackingInspectionOptions(args.tracking_uri, args.experiment, args.run_id)
    # 2. 仅读取已有实验与可选 Run，输出不含认证值和原始服务异常。
    try:
        result = inspect_tracking(options)
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "authentication_present": authentication_presence(),
                }
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
