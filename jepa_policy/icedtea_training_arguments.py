"""Parse the single-GPU training, holdout and MLflow launch configuration."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from jepa_policy.mlflow_tracking import (
    validate_http_authentication,
    validate_tracking_uri,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/mnt/data/EIC01/vla/x1_pro/train/waterbar_2task_228eps_20261126_smoothed"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs")
        / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--episode-indices", type=Path)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-chunk-size", type=int, choices=range(1, 10), default=4)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mlflow-tracking-uri", default="https://mlflow.seer-robotics.cn"
    )
    parser.add_argument("--mlflow-experiment", default="pretrain-fm")
    parser.add_argument("--mlflow-run-name")
    parser.add_argument(
        "--cache-root", type=Path, default=Path("/tmp/jepa_icedtea_cache")
    )
    parser.add_argument("--cache-build-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_arguments(args) -> None:
    validate_tracking_uri(args.mlflow_tracking_uri)
    if args.mlflow:
        validate_http_authentication(args.mlflow_tracking_uri)
    if (
        args.steps <= 0
        or args.batch_size <= 0
        or args.workers < 0
        or args.save_every < 0
        or args.cache_build_workers <= 0
        or args.log_every <= 0
        or args.eval_batch_size <= 0
        or args.eval_seed < 0
        or args.split_seed < 0
    ):
        raise ValueError(
            "steps/batch-size/log-every/eval-batch-size/cache-build-workers must be positive; "
            "workers/save-every/split-seed/eval-seed must be nonnegative"
        )
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("Training requires one available CUDA device at cuda:0")
