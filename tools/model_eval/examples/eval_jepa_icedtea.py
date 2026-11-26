"""Adapt the X1 iced-tea checkpoint and LeRobot test data to model_eval."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from model_eval import Dataset, Episode

from examples.train_x1_icedtea import (
    create_config,
    create_optimization_config,
)
from jepa_policy.agent import TrainingAgent
from jepa_policy.datasets.icedtea_dataset import (
    CAMERAS,
    IcedTeaDataset,
    IcedTeaDatasetConfig,
)

TRAIN_OUTPUT = Path(
    os.environ.get(
        "JEPA_TRAIN_OUTPUT",
        "/home/x1/work/yuanyh/runs/jepa-icedtea-right-arm",
    )
)
TEST_DATASET = Path(
    os.environ.get(
        "JEPA_TEST_DATASET",
        "/home/x1/work/X1_Pro_data/waterbar_datasets/test/merged/"
        "CeMAT_merged_test_smoothed_060",
    )
)
CHECKPOINT = Path(
    os.environ.get(
        "JEPA_CHECKPOINT",
        str(TRAIN_OUTPUT / "models/model_latest.pt"),
    )
)
TEST_TASK_NAME = os.environ.get("JEPA_TEST_TASK_NAME", "grasp the iced tea.")
DEVICE = os.environ.get("JEPA_DEVICE", "cuda:0")


@dataclass(frozen=True)
class NormalizedTestData:
    actions_raw: np.ndarray
    states_normalized: np.ndarray


def restore_raw_actions(source: IcedTeaDataset) -> np.ndarray:
    return source.actions * (source.action_range / 2.0) + (
        source.action_min + source.action_range / 2.0
    )


def normalize_test_states(source: IcedTeaDataset, metadata: dict) -> np.ndarray:
    states_raw = source.states * (source.state_range / 2.0) + (
        source.state_min + source.state_range / 2.0
    )
    train_state_min = np.asarray(metadata["state_min"], dtype=np.float32)
    train_state_range = np.asarray(metadata["state_range"], dtype=np.float32)
    return (states_raw - train_state_min) / train_state_range * 2.0 - 1.0


def create_episode(
    source: IcedTeaDataset,
    episode_id: int,
    record: dict,
    frame_data: NormalizedTestData,
) -> Episode:
    start = int(record["dataset_from_index"])
    end = int(record["dataset_to_index"])
    episode_length = end - start

    def observe(
        frame: int,
        *,
        current_record=record,
        frame_start=start,
        length=episode_length,
    ):
        row = frame_start + frame
        next_row = frame_start + min(frame + 1, length - 1)
        observation = {
            camera: torch.stack(
                [
                    source.read_image(current_record, camera, row),
                    source.read_image(current_record, camera, next_row),
                ]
            )
            for camera in CAMERAS
        }
        observation["agent_pos"] = torch.from_numpy(
            frame_data.states_normalized[row : next_row + 1].copy()
        )
        return observation

    return Episode(
        episode_id=f"episode_{episode_id:06d}",
        length=episode_length,
        ground_truth=frame_data.actions_raw[start:end],
        observe=observe,
        prompt=TEST_TASK_NAME,
    )


def build_dataset() -> Dataset:
    """Load task-matched test episodes with normalization from training data."""
    metadata = json.loads((TRAIN_OUTPUT / "dataset_metadata.json").read_text())
    source = IcedTeaDataset(
        IcedTeaDatasetConfig(TEST_DATASET, task_name=TEST_TASK_NAME)
    )
    frame_data = NormalizedTestData(
        actions_raw=restore_raw_actions(source),
        states_normalized=normalize_test_states(source, metadata),
    )
    episodes = [
        create_episode(source, episode_id, record, frame_data)
        for episode_id, record in sorted(source.episodes.items())
        if TEST_TASK_NAME in record["tasks"]
    ]
    return Dataset(episodes, name=str(TEST_DATASET))


class IcedTeaPolicy:
    """Wrap the trained MIP agent in model_eval's chunk prediction interface."""

    chunk_size = 10

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("X1 checkpoint evaluation requires cuda:0")
        metadata = json.loads((TRAIN_OUTPUT / "dataset_metadata.json").read_text())
        optimization = create_optimization_config(steps=1, batch_size=1, workers=0)
        config = create_config(TEST_DATASET, TRAIN_OUTPUT, optimization)
        config.optimization.device = DEVICE
        self.action_min = torch.as_tensor(metadata["action_min"], device=DEVICE)
        self.action_range = torch.as_tensor(metadata["action_range"], device=DEVICE)
        self.agent = TrainingAgent(config)
        self.agent.load(str(CHECKPOINT), load_optimizer=False)
        self.agent.eval()

    def predict(self, observation: dict[str, torch.Tensor]) -> np.ndarray:
        inputs = {}
        for key, value in observation.items():
            value = value.unsqueeze(0).to(DEVICE)
            if value.dtype == torch.uint8:
                value = value.float().div_(255).mul_(2).sub_(1)
            else:
                value = value.float()
            inputs[key] = value
        initial_action = torch.zeros((1, self.chunk_size, 8), device=DEVICE)
        with torch.inference_mode():
            normalized, _ = self.agent.sample(
                act_0=initial_action,
                obs=inputs,
                num_steps=1,
                use_ema=True,
                return_future=True,
            )
        actions = (normalized[0] + 1.0) * self.action_range / 2.0 + self.action_min
        return actions.cpu().numpy()


def build_model() -> IcedTeaPolicy:
    return IcedTeaPolicy()
