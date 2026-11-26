"""Evaluate an X1 iced-tea JEPA checkpoint directly from the command line."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
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
DEVICE = "cuda:0"
TEST_TASK_NAME: str | None = None


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
    parser.add_argument("--task-name")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resolve_task_name(
    dataset_path: Path, trained_task: str, requested_task: str | None
) -> str:
    rows = pq.read_table(dataset_path / "meta/tasks.parquet").to_pylist()
    available_tasks = sorted({row["task"] for row in rows})
    if requested_task:
        if requested_task not in available_tasks:
            raise ValueError(
                f"Task {requested_task!r} is not present. Available tasks: {available_tasks}"
            )
        return requested_task
    if trained_task in available_tasks:
        return trained_task
    if len(available_tasks) == 1:
        return available_tasks[0]
    raise ValueError(
        f"Cannot infer task from checkpoint metadata. Available tasks: {available_tasks}. "
        "Pass --task-name to select one."
    )


def configure(args) -> None:
    global TRAIN_OUTPUT, TEST_DATASET, CHECKPOINT, DEVICE, TEST_TASK_NAME

    TEST_DATASET = Path(getattr(args, "dataset", None) or args.data_root)
    CHECKPOINT = Path(args.checkpoint)
    TRAIN_OUTPUT = CHECKPOINT.parent.parent
    DEVICE = getattr(args, "device", "cuda:0")

    metadata_path = TRAIN_OUTPUT / "dataset_metadata.json"
    for path in (metadata_path, TEST_DATASET / "meta/info.json", CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text())
    TEST_TASK_NAME = resolve_task_name(
        TEST_DATASET,
        metadata["task"],
        getattr(args, "task_name", None),
    )
    print(f"Task: {TEST_TASK_NAME}", flush=True)


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
        prompt=TEST_TASK_NAME or "",
    )


def build_dataset() -> Dataset:
    if TRAIN_OUTPUT is None or TEST_DATASET is None or TEST_TASK_NAME is None:
        raise RuntimeError("Configure the evaluation paths before loading the dataset")
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
    if not episodes:
        raise ValueError(f"No episodes found for task {TEST_TASK_NAME!r}")
    return Dataset(episodes, name=str(TEST_DATASET))


class IcedTeaPolicy:
    """Wrap the trained policy in model_eval's chunk prediction interface."""

    chunk_size = 10

    def __init__(self):
        if TRAIN_OUTPUT is None or TEST_DATASET is None or CHECKPOINT is None:
            raise RuntimeError(
                "Configure the evaluation paths before loading the model"
            )
        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device {DEVICE}")
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
            CHECKPOINT.parent.parent
            / "evaluation"
            / (f"{CHECKPOINT.stem}_{TEST_DATASET.name}")
        )
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
