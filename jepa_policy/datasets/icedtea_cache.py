from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from jepa_policy.datasets.icedtea_dataset import (
    CAMERAS,
    IcedTeaDataset,
    IcedTeaDatasetConfig,
)
from rolling_cache import DiskDecodedCacheReader, SampleSource, StepSampleCache

CACHE_FIELDS = {
    **{
        f"obs.{camera}": {"frame_shape": [2, 3, 128, 128], "dtype": "uint8"}
        for camera in CAMERAS
    },
    "obs.agent_pos": {"frame_shape": [2, 8], "dtype": "float32"},
    "action": {"frame_shape": [10, 8], "dtype": "float32"},
    **{
        f"future_obs.{camera}": {"frame_shape": [3, 128, 128], "dtype": "uint8"}
        for camera in CAMERAS
    },
    "future_obs.agent_pos": {"frame_shape": [8], "dtype": "float32"},
}


class IcedTeaSampleSource(SampleSource):
    def __init__(self, config: IcedTeaDatasetConfig):
        self.dataset = IcedTeaDataset(config)
        self._digest = hashlib.sha256(
            json.dumps(
                {
                    "samples": self.dataset.samples,
                    "obs_steps": config.obs_steps,
                    "horizon": config.horizon,
                    "future_steps": config.future_steps,
                    "task": self.dataset.task_name,
                },
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    @property
    def num_steps(self) -> int:
        return len(self.dataset)

    def field_specs(self) -> dict[str, dict]:
        return CACHE_FIELDS

    def sample_mapping_digest(self) -> str:
        return self._digest

    def episode_of(self, global_step: int) -> tuple[int, int]:
        start, episode_id = self.dataset.samples[global_step]
        episode = self.dataset.episodes[episode_id]
        return episode_id, start - int(episode["dataset_from_index"])

    def decode_step(self, global_step: int, *, fields=None) -> dict[str, np.ndarray]:
        item = self.dataset[global_step]
        sample = {
            **{f"obs.{key}": value.numpy() for key, value in item["obs"].items()},
            "action": item["action"].numpy(),
            **{
                f"future_obs.{key}": value.numpy()
                for key, value in item["future_obs"].items()
            },
        }
        return sample if fields is None else {key: sample[key] for key in fields}


def create_source(config: IcedTeaDatasetConfig):
    return IcedTeaSampleSource(config)


def build_icedtea_cache(
    dataset: IcedTeaDataset,
    cache_root: Path,
    cache_build_workers: int = 4,
) -> DiskDecodedCacheReader:
    config = {
        "dataset_path": str(dataset.root),
        "task": dataset.task_name,
        "obs_steps": dataset.obs_steps,
        "horizon": dataset.horizon,
        "future_steps": dataset.future_steps,
        "image_size": dataset.image_size,
        "fields": CACHE_FIELDS,
        "version": 1,
    }
    dataset_config = IcedTeaDatasetConfig(
        root=dataset.root,
        obs_steps=dataset.obs_steps,
        horizon=dataset.horizon,
        future_steps=dataset.future_steps,
        task_name=dataset.task_name,
        image_size=dataset.image_size,
        image_interpolation=dataset.image_interpolation,
        episode_indices=dataset.selected_episode_indices,
    )
    config["episode_indices"] = dataset.selected_episode_indices
    config["image_interpolation"] = dataset.image_interpolation
    source = IcedTeaSampleSource(dataset_config)
    cache = StepSampleCache(
        source,
        dataset_path=dataset.root,
        source_config=config,
        root=cache_root,
        frames_per_shard=1024,
        source_factory=partial(create_source, dataset_config),
        cache_build_workers=cache_build_workers,
        write_concurrency=cache_build_workers,
    )
    reader = cache.build()
    reader.validate_complete()
    print(f"Decoded cache ready: {cache.cache_dir}", flush=True)
    return reader


class CachedIcedTeaDataset(Dataset):
    def __init__(self, source: IcedTeaDataset, reader: DiskDecodedCacheReader):
        self.source = source
        self.cache_dir = reader.cache_dir
        self.reader = reader
        if len(source) != reader.num_frames:
            raise ValueError(
                f"Cache sample count {reader.num_frames} != dataset count {len(source)}"
            )

    def __len__(self) -> int:
        return len(self.source)

    def __getattr__(self, name: str):
        return getattr(self.source, name)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["reader"] = None
        source = copy.copy(self.source)
        source.containers = OrderedDict()
        source.frame_cache = OrderedDict()
        state["source"] = source
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.reader = DiskDecodedCacheReader.from_cache(
            self.cache_dir, open_shard_cache_size=2
        )

    def __getitem__(self, index: int) -> dict:
        sample = self.reader.get_item(index)
        return {
            "obs": {
                **{
                    camera: torch.from_numpy(sample[f"obs.{camera}"])
                    for camera in CAMERAS
                },
                "agent_pos": torch.from_numpy(sample["obs.agent_pos"]),
            },
            "action": torch.from_numpy(sample["action"]),
            "future_obs": {
                **{
                    camera: torch.from_numpy(sample[f"future_obs.{camera}"])
                    for camera in CAMERAS
                },
                "agent_pos": torch.from_numpy(sample["future_obs.agent_pos"]),
            },
        }
