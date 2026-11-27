"""Render training events with one console and file layout."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from loguru import logger


def format_log_record(record: dict) -> str:
    if record["extra"].get("compact", False):
        return "{message}\n{exception}"
    record["extra"]["body"] = "\n".join(
        "  " + line for line in record["message"].splitlines()
    )
    return (
        "{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {extra[section]}\n"
        "{extra[body]}\n{exception}"
    )


def setup_training_logging(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.configure(extra={"section": "模型初始化"})
    logger.add(
        sys.stdout,
        format=format_log_record,
        colorize=False,
        backtrace=False,
        diagnose=False,
        filter=lambda record: (
            record["extra"]["section"] != "模型初始化" or record["level"].no >= 30
        ),
    )
    logger.add(
        output_path / "training.log",
        format=format_log_record,
        colorize=False,
        backtrace=False,
        diagnose=False,
    )


def log_section(title: str, lines: list[str]) -> None:
    logger.bind(section=title).info("\n".join(lines))


class CacheLogStream:
    """Forward existing parent-process cache messages into the shared layout."""

    def __init__(self):
        self.pending = ""

    def write(self, message: str) -> int:
        self.pending += message
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip():
                log_section("缓存构建", [line.strip().removeprefix("[rolling_cache] ")])
        return len(message)

    def flush(self) -> None:
        if self.pending.strip():
            log_section("缓存构建", [self.pending.strip()])
        self.pending = ""


def log_training_progress(record: dict, elapsed_seconds: float) -> None:
    message = (
        f"step={record['step']} loss={record['loss']:.6f} "
        f"data={record['data_wait_seconds']:.2f}s "
        f"update={record['update_seconds']:.2f}s "
        f"elapsed={elapsed_seconds:.2f}s "
        f"gpu_peak={record['gpu_peak_memory_gib']:.1f}GiB"
    )
    logger.bind(section="训练进度", compact=True).info(message)


def describe_batch_tensors(batch: dict, prefix: str = "") -> list[dict]:
    entries = []
    for key, value in batch.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            entries.extend(describe_batch_tensors(value, name))
            continue
        entries.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "min": float(torch.amin(value)),
                "max": float(torch.amax(value)),
            }
        )
    return entries


def log_checkpoint(path: Path, step: int, elapsed_seconds: float) -> None:
    log_section(
        "权重保存",
        [
            f"训练步数={step:,} | 保存耗时={elapsed_seconds:.2f}s",
            f"路径={path}",
            f"文件大小={path.stat().st_size / 1024**2:.1f}MiB",
        ],
    )


def save_training_checkpoint(agent, model_dir: Path, step: int) -> Path:
    path = model_dir / f"model_step_{step + 1:06d}.pt"
    started = time.perf_counter()
    agent.save(path, training_state={"n_gradient_step": step})
    log_checkpoint(path, step + 1, time.perf_counter() - started)
    return path


def run_logged_training(entrypoint: Callable[[], None]) -> None:
    try:
        entrypoint()
    except Exception:
        logger.bind(section="训练失败").exception("训练已退出，错误堆栈如下")
        raise SystemExit(1) from None
