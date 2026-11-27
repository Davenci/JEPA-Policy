"""LeRobot v3 reader for selected X1 episodes and right-arm joint actions."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

CAMERAS = {
    "head_image": "observation.images.head",
    "left_wrist_image": "observation.images.left_wrist",
    "right_wrist_image": "observation.images.right_wrist",
}
ACTION_NAMES = [*(f"right_{index}.pos" for index in range(1, 8)), "right_gripper.pos"]
MAX_OPEN_VIDEO_CONTAINERS = 24


@dataclass(frozen=True)
class IcedTeaDatasetConfig:
    root: Path
    obs_steps: int = 2
    horizon: int = 10
    future_steps: int = 4
    image_size: int = 128
    image_interpolation: int = cv2.INTER_LINEAR
    normalize: bool = True
    episode_indices: tuple[int, ...] | None = None


def initialize_icedtea_worker(worker_id: int) -> None:
    """Limit OpenCV and PyTorch CPU threads within this worker."""
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


class IcedTeaDataset(Dataset):
    """Read image and 8D action windows without filtering task labels."""

    def __init__(self, config: IcedTeaDatasetConfig):
        self.root = Path(config.root)
        self.obs_steps = config.obs_steps
        self.horizon = config.horizon
        self.future_steps = config.future_steps
        self.image_size = config.image_size
        self.image_interpolation = config.image_interpolation
        self.normalize = config.normalize
        self.selected_episode_indices = config.episode_indices
        self.episodes = self.load_episodes()
        self.load_frames()
        self.containers = OrderedDict()
        self.frame_cache = OrderedDict()
        self.samples = self.make_sample_indices()
        if not self.samples:
            raise ValueError(f"No valid windows in selected episodes of {self.root}")

    def load_frames(self) -> None:
        info = json.loads((self.root / "meta/info.json").read_text())
        action_names = info["features"]["action"]["names"]
        state_names = info["features"]["observation.state"]["names"]
        self.action_indices = [action_names.index(name) for name in ACTION_NAMES]
        self.state_indices = [state_names.index(name) for name in ACTION_NAMES]
        paths = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet data under {self.root / 'data'}")
        tables = [pq.read_table(path) for path in paths]
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        self.actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[
            :, self.action_indices
        ]
        self.states = np.asarray(
            table["observation.state"].to_pylist(), dtype=np.float32
        )[:, self.state_indices]
        self.timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        self.episode_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        self.task_indices = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
        self.raw_actions = self.actions.copy()
        self.raw_states = self.states.copy()
        if self.normalize:
            self.normalize_frames()
        else:
            self.actions = self.raw_actions.copy()
            self.states = self.raw_states.copy()

    def normalize_frames(self) -> None:
        selected_mask = np.isin(self.episode_ids, tuple(self.episodes))
        self.action_min = self.actions[selected_mask].min(axis=0)
        self.action_range = np.maximum(
            self.actions[selected_mask].max(axis=0) - self.action_min, 1e-6
        )
        self.state_min = self.states[selected_mask].min(axis=0)
        self.state_range = np.maximum(
            self.states[selected_mask].max(axis=0) - self.state_min, 1e-6
        )
        self.actions = (self.actions - self.action_min) / self.action_range * 2.0 - 1.0
        self.states = (self.states - self.state_min) / self.state_range * 2.0 - 1.0

    def load_episodes(self) -> dict[int, dict]:
        records = []
        paths = sorted((self.root / "meta/episodes").glob("chunk-*/*.parquet"))
        for path in paths:
            records.extend(pq.read_table(path).to_pylist())
        episodes = {int(record["episode_index"]): record for record in records}
        if self.selected_episode_indices is None:
            return episodes
        if not self.selected_episode_indices:
            raise ValueError("episode_indices must contain at least one episode")
        selected_ids = set(self.selected_episode_indices)
        if len(selected_ids) != len(self.selected_episode_indices):
            raise ValueError("episode_indices contains duplicate episode indices")
        missing_ids = sorted(selected_ids - episodes.keys())
        if missing_ids:
            raise ValueError(f"Episode indices not found in dataset: {missing_ids}")
        return {
            episode_id: episodes[episode_id]
            for episode_id in self.selected_episode_indices
        }

    def make_sample_indices(self) -> list[tuple[int, int]]:
        samples = []
        needed = max(self.horizon, self.obs_steps + self.future_steps)
        for episode_id, episode in self.episodes.items():
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            samples.extend(
                (index, episode_id) for index in range(start, end - needed + 1)
            )
        return samples

    def read_image(self, episode: dict, camera: str, row: int) -> torch.Tensor:
        video_key = CAMERAS[camera]
        prefix = f"videos/{video_key}"
        chunk = int(episode[f"{prefix}/chunk_index"])
        file_index = int(episode[f"{prefix}/file_index"])
        path = (
            self.root
            / "videos"
            / video_key
            / f"chunk-{chunk:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        timestamp = float(episode[f"{prefix}/from_timestamp"]) + float(
            self.timestamps[row]
        )
        cache_key = (str(path), round(timestamp * 15))
        if cache_key in self.frame_cache:
            self.frame_cache.move_to_end(cache_key)
            return self.frame_cache[cache_key]
        container_key = str(path)
        if container_key not in self.containers:
            container = av.open(container_key)
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            stream.codec_context.thread_type = "NONE"
            self.containers[container_key] = (container, stream)
            if len(self.containers) > MAX_OPEN_VIDEO_CONTAINERS:
                _, (old_container, _) = self.containers.popitem(last=False)
                old_container.close()
        else:
            self.containers.move_to_end(container_key)
        container, stream = self.containers[container_key]
        container.seek(
            int(timestamp / float(stream.time_base)), stream=stream, backward=True
        )
        selected = None
        for frame in container.decode(stream):
            selected = frame
            if float(frame.pts * stream.time_base) >= timestamp:
                break
        if selected is None:
            raise RuntimeError(f"No frame at {timestamp:.3f}s in {path}")
        image = cv2.resize(
            selected.to_ndarray(format="rgb24"),
            (self.image_size, self.image_size),
            interpolation=self.image_interpolation,
        )
        image = torch.from_numpy(np.moveaxis(image, -1, 0).copy())
        self.frame_cache[cache_key] = image
        if len(self.frame_cache) > 32:
            self.frame_cache.popitem(last=False)
        return image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        start, episode_id = self.samples[index]
        episode = self.episodes[episode_id]
        rows = range(start, start + self.obs_steps)
        observation = {
            camera: torch.stack([self.read_image(episode, camera, row) for row in rows])
            for camera in CAMERAS
        }
        observation["agent_pos"] = torch.from_numpy(
            self.states[start : start + self.obs_steps].copy()
        )
        action = torch.from_numpy(self.actions[start : start + self.horizon].copy())
        future_row = start + self.obs_steps - 1 + self.future_steps
        future = {
            camera: self.read_image(episode, camera, future_row) for camera in CAMERAS
        }
        future["agent_pos"] = torch.from_numpy(self.states[future_row].copy())
        return {"obs": observation, "action": action, "future_obs": future}
