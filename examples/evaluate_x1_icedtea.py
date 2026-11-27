"""Evaluate an X1 iced-tea JEPA checkpoint directly from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rolling_cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "model_eval"))

import cv2
import numpy as np
import torch
from model_eval import Dataset, Episode, evaluate, render_episode

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

TRAIN_OUTPUT: Path | None = None
TEST_DATASET: Path | None = None
CHECKPOINT: Path | None = None
TRAINING_STATS: dict | None = None
DEVICE = "cuda:0"
test_episode_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class NormalizedTestData:
    actions_raw: np.ndarray
    states_normalized: np.ndarray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--episode-indices", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def configure(args) -> None:
    global \
        TRAIN_OUTPUT, \
        TEST_DATASET, \
        CHECKPOINT, \
        TRAINING_STATS, \
        DEVICE, \
        test_episode_indices

    TEST_DATASET = Path(getattr(args, "dataset", None) or args.data_root)
    CHECKPOINT = Path(args.checkpoint)
    TRAIN_OUTPUT = CHECKPOINT.parent.parent
    DEVICE = getattr(args, "device", "cuda:0")
    indices_path = getattr(args, "episode_indices", None)
    test_episode_indices = (
        tuple(json.loads(Path(indices_path).read_text()))
        if indices_path is not None
        else None
    )

    stats_path = TRAIN_OUTPUT / "normalization_stats.json"
    for path in (stats_path, TEST_DATASET / "meta/info.json", CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)

    TRAINING_STATS = json.loads(stats_path.read_text())
    if TRAINING_STATS["schema_version"] != 1:
        raise ValueError(f"Unsupported normalization stats schema: {stats_path}")
    interpolation = TRAINING_STATS["image_preprocessing"]["resize_interpolation"]
    if interpolation != cv2.INTER_LINEAR:
        raise ValueError(
            "Evaluation image resize interpolation differs from the training "
            f"pipeline: {interpolation} != INTER_LINEAR ({cv2.INTER_LINEAR})"
        )


def restore_raw_actions(source: IcedTeaDataset) -> np.ndarray:
    return source.raw_actions.copy()


def normalize_test_states(source: IcedTeaDataset, stats: dict) -> np.ndarray:
    states_raw = source.raw_states
    state_stats = stats["features"]["observation.state"]
    train_state_min = np.asarray(state_stats["min"], dtype=np.float32)
    train_state_range = np.asarray(state_stats["range"], dtype=np.float32)
    normalized = (states_raw - train_state_min) / train_state_range * 2.0 - 1.0
    return normalized


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
    ):
        current_row = frame_start + frame
        previous_row = frame_start + max(frame - 1, 0)
        observation = {
            camera: torch.stack(
                [
                    source.read_image(current_record, camera, previous_row),
                    source.read_image(current_record, camera, current_row),
                ]
            )
            for camera in CAMERAS
        }
        observation["agent_pos"] = torch.from_numpy(
            np.stack(
                [
                    frame_data.states_normalized[previous_row],
                    frame_data.states_normalized[current_row],
                ]
            )
        )
        return observation

    return Episode(
        episode_id=f"episode_{episode_id:06d}",
        length=episode_length,
        ground_truth=frame_data.actions_raw[start:end],
        observe=observe,
        prompt="",
    )


def build_dataset() -> Dataset:
    if TRAIN_OUTPUT is None or TEST_DATASET is None or TRAINING_STATS is None:
        raise RuntimeError("Configure the evaluation paths before loading the dataset")
    source = IcedTeaDataset(
        IcedTeaDatasetConfig(
            TEST_DATASET,
            image_interpolation=TRAINING_STATS["image_preprocessing"][
                "resize_interpolation"
            ],
            normalize=False,
            episode_indices=test_episode_indices,
        )
    )
    frame_data = NormalizedTestData(
        actions_raw=restore_raw_actions(source),
        states_normalized=normalize_test_states(source, TRAINING_STATS),
    )
    episodes = [
        create_episode(source, episode_id, record, frame_data)
        for episode_id, record in sorted(source.episodes.items())
    ]
    if not episodes:
        raise ValueError(f"No episodes found in {TEST_DATASET}")
    return Dataset(episodes, name=str(TEST_DATASET))


class IcedTeaPolicy:
    """Wrap the trained policy in model_eval's chunk prediction interface."""

    chunk_size = 8

    def __init__(self):
        if (
            TRAIN_OUTPUT is None
            or TEST_DATASET is None
            or CHECKPOINT is None
            or TRAINING_STATS is None
        ):
            raise RuntimeError(
                "Configure the evaluation paths before loading the model"
            )
        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device {DEVICE}")
        optimization = create_optimization_config(steps=1, batch_size=1, workers=0)
        config = create_config(TEST_DATASET, TRAIN_OUTPUT, optimization)
        config.optimization.device = DEVICE
        action_stats = TRAINING_STATS["features"]["action"]
        self.action_min = torch.as_tensor(action_stats["min"], device=DEVICE)
        self.action_range = torch.as_tensor(action_stats["range"], device=DEVICE)
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
        initial_action = torch.zeros((1, 10, 8), device=DEVICE)
        with torch.inference_mode():
            normalized, _ = self.agent.sample(
                act_0=initial_action,
                obs=inputs,
                num_steps=1,
                use_ema=True,
                return_future=True,
            )
        executed_chunk = normalized[0, 1:9]
        actions = (executed_chunk + 1.0) * self.action_range / 2.0 + self.action_min
        return actions.cpu().numpy()


def build_model() -> IcedTeaPolicy:
    return IcedTeaPolicy()


def run_evaluation(args) -> Path:
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")

    configure(args)
    dataset = build_dataset()
    if args.max_episodes is not None:
        dataset = Dataset(dataset.episodes[: args.max_episodes], name=dataset.name)
    model = build_model()

    if args.output is None:
        output = (
            Path(__file__).resolve().parents[1]
            / "output"
            / "eval"
            / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M")
        )
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = args.output
    artifact_dir = output / "artifacts"
    figure_dir = output / "figures"
    results = evaluate(model, dataset, artifact_dir)
    for result in results:
        for kind in ("abs", "delta"):
            render_episode(
                result["path"],
                figure_dir / f"{result['episode_id']}_{kind}.png",
                kind=kind,
                data_root=dataset.name,
                checkpoint=str(CHECKPOINT),
            )
    print(f"Evaluated {len(results)} episodes; results: {output}", flush=True)
    return output


def main() -> None:
    args = parse_arguments()
    run_evaluation(args)


if __name__ == "__main__":
    main()
