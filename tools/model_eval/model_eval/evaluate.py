"""Run episode predictions and save frame-aligned trajectory artifacts."""

from __future__ import annotations

import pathlib
import time
import traceback

import numpy as np

from .builders import build_from_arrays

__all__ = ["Episode", "Dataset", "evaluate"]


class Episode:
    """A dataset episode with frame observations and aligned ground truth."""

    __slots__ = ("episode_id", "length", "ground_truth", "observe", "prompt")

    def __init__(self, episode_id, length, ground_truth, observe, prompt=""):
        self.episode_id = str(episode_id)
        self.length = int(length)
        self.ground_truth = np.asarray(ground_truth, dtype=np.float32)
        self.observe = observe
        self.prompt = str(prompt)
        if self.ground_truth.ndim != 2:
            raise ValueError(
                f"ground_truth 必须是 [L, D]，实际 {self.ground_truth.shape}"
            )
        if self.ground_truth.shape[0] != self.length:
            raise ValueError(
                f"ground_truth 行数 {self.ground_truth.shape[0]} 与 length {self.length} 不一致"
            )
        if not callable(self.observe):
            raise ValueError("observe 必须是可调用对象")


class Dataset:
    """A named collection of episodes."""

    __slots__ = ("episodes", "name")

    def __init__(self, episodes, name=""):
        self.episodes = list(episodes)
        self.name = str(name)
        if not self.episodes:
            raise ValueError("数据集不能为空")
        for episode in self.episodes:
            if not isinstance(episode, Episode):
                raise TypeError(
                    f"episodes 里必须是 Episode，收到 {type(episode).__name__}"
                )


def _check_model(model):
    chunk = getattr(model, "chunk_size", None)
    if (
        not isinstance(chunk, (int, np.integer))
        or isinstance(chunk, bool)
        or chunk <= 0
    ):
        raise ValueError(f"model.chunk_size 必须是正整数，实际 {chunk!r}")
    if not callable(getattr(model, "predict", None)):
        raise TypeError("model 必须提供可调用的 predict(observation)")
    return int(chunk)


def evaluate(model, dataset, out_dir, *, stride=None, on_error="fail", log=print):
    """Evaluate each episode and save full-frame, stitched-chunk trajectories.

    Each call predicts ``[chunk_size, D]``. Predictions are copied onto the
    episode timeline; overlapping regions use the latest prediction chunk.
    A stride larger than the chunk is rejected because it leaves frame gaps.
    """
    chunk = _check_model(model)
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset 必须是 Dataset")
    step = int(stride) if stride else chunk
    if step <= 0:
        raise ValueError("stride 必须是正整数")
    if step > chunk:
        raise ValueError(
            f"完整逐帧曲线要求 stride <= chunk_size，实际 stride={step}, chunk_size={chunk}"
        )
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for episode in dataset.episodes:
        started = time.perf_counter()
        try:
            results.append(_run_one(model, episode, chunk, step, out_dir))
        except Exception as exc:  # noqa: BLE001
            if on_error == "skip":
                log(
                    f"  !! {episode.episode_id} 失败，已跳过：{type(exc).__name__}: {exc}"
                )
                log(traceback.format_exc(limit=3).rstrip())
                continue
            raise RuntimeError(
                f"回合 {episode.episode_id} 评估失败：{type(exc).__name__}: {exc}"
            ) from exc
        results[-1]["seconds"] = round(time.perf_counter() - started, 3)
        result = results[-1]
        log(
            f"  {episode.episode_id}: {result['decisions']} 个推理点 / "
            f"{episode.length} 帧 -> {result['path']} ({result['seconds']}s)"
        )
    return results


def _run_one(model, episode, chunk, stride, out_dir):
    """Stitch full model chunks and frame-aligned labels into one artifact."""
    length = episode.length
    if length <= 0:
        raise ValueError(f"{episode.episode_id}: length 必须为正整数")
    inference_points = np.arange(0, length, stride, dtype=np.int64)
    dims = episode.ground_truth.shape[1]
    predictions = np.empty((length, dims), dtype=np.float32)

    for point in inference_points:
        frame = int(point)
        output = np.asarray(model.predict(episode.observe(frame)), dtype=np.float32)
        if output.shape != (chunk, dims):
            raise ValueError(
                f"{episode.episode_id} 帧 {frame}: predict 返回形状 {output.shape}，"
                f"期望 ({chunk}, {dims})"
            )
        if not np.isfinite(output).all():
            raise ValueError(f"{episode.episode_id} 帧 {frame}: predict 返回了 NaN/Inf")
        end = min(frame + chunk, length)
        predictions[frame:end] = output[: end - frame]

    frames = np.arange(length, dtype=np.int64)
    state_base = np.empty_like(episode.ground_truth)
    state_base[0] = episode.ground_truth[0]
    state_base[1:] = episode.ground_truth[:-1]
    path = out_dir / f"{episode.episode_id}.npz"
    build_from_arrays(
        predictions,
        episode.ground_truth,
        path,
        decision_points=frames,
        chunk_size=chunk,
        episode_length=length,
        state_base=state_base,
        prompt=episode.prompt,
        episode_id=episode.episode_id,
        metadata={
            "curve_mode": "dense_chunk",
            "inference_points": inference_points.tolist(),
            "stride": stride,
            "overlap_policy": "latest_chunk_wins",
        },
    )
    return {
        "episode_id": episode.episode_id,
        "path": str(path),
        "decisions": int(inference_points.size),
        "frames": length,
        "chunk_size": chunk,
        "stride": stride,
    }
