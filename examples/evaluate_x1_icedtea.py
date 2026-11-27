"""Evaluate an X1 iced-tea JEPA checkpoint directly from the command line."""

from __future__ import annotations

import argparse
import hashlib
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
from jepa_policy.temporal_ensemble import ActTemporalEnsembler

TRAIN_OUTPUT: Path | None = None
TEST_DATASET: Path | None = None
CHECKPOINT: Path | None = None
TRAINING_STATS: dict | None = None
DEVICE = "cuda:0"
test_episode_indices: tuple[int, ...] | None = None
max_plot_episodes = 3


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
    parser.add_argument(
        "--chunk-size",
        type=int,
        choices=range(1, 10),
        default=4,
        help="Actions executed before the next inference; prediction horizon stays 10",
    )
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="ACT temporal ensembling; requires --chunk-size 1",
    )
    parser.add_argument(
        "--temporal-ensemble-k",
        type=float,
        default=0.01,
        help="ACT exponential weight coefficient; oldest prediction has rank zero",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.temporal_ensemble and args.chunk_size != 1:
        parser.error("--temporal-ensemble requires --chunk-size 1")
    if not np.isfinite(args.temporal_ensemble_k) or args.temporal_ensemble_k < 0:
        parser.error("--temporal-ensemble-k must be finite and nonnegative")
    return args


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

    def __init__(
        self,
        chunk_size: int = 4,
        temporal_ensemble: bool = False,
        temporal_ensemble_k: float = 0.01,
    ):
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
        self.horizon = config.task.horizon
        self.action_start = config.task.obs_steps - 1
        max_chunk_size = self.horizon - self.action_start
        if not 1 <= chunk_size <= max_chunk_size:
            raise ValueError(f"chunk-size must be between 1 and {max_chunk_size}")
        if temporal_ensemble and chunk_size != 1:
            raise ValueError("ACT temporal ensembling requires chunk-size 1")
        if not np.isfinite(temporal_ensemble_k) or temporal_ensemble_k < 0:
            raise ValueError("temporal-ensemble-k must be finite and nonnegative")
        self.chunk_size = chunk_size
        self.temporal_ensemble_k = temporal_ensemble_k
        self.temporal_ensembler = (
            ActTemporalEnsembler(max_chunk_size, 8, temporal_ensemble_k)
            if temporal_ensemble
            else None
        )
        action_stats = TRAINING_STATS["features"]["action"]
        self.action_min = torch.as_tensor(action_stats["min"], device=DEVICE)
        self.action_range = torch.as_tensor(action_stats["range"], device=DEVICE)
        self.agent = TrainingAgent(config)
        self.agent.load(str(CHECKPOINT), load_optimizer=False)
        self.agent.eval()

    def reset_episode(self) -> None:
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()

    def predict(self, observation: dict[str, torch.Tensor]) -> np.ndarray:
        inputs = {}
        for key, value in observation.items():
            value = value.unsqueeze(0).to(DEVICE)
            if value.dtype == torch.uint8:
                value = value.float().div_(255).mul_(2).sub_(1)
            else:
                value = value.float()
            inputs[key] = value
        initial_action = torch.zeros((1, self.horizon, 8), device=DEVICE)
        with torch.inference_mode():
            normalized, _ = self.agent.sample(
                act_0=initial_action,
                obs=inputs,
                num_steps=1,
                use_ema=True,
                return_future=True,
            )
        prediction_steps = (
            self.horizon - self.action_start
            if self.temporal_ensembler is not None
            else self.chunk_size
        )
        executed_chunk = normalized[
            0, self.action_start : self.action_start + prediction_steps
        ]
        actions = (executed_chunk + 1.0) * self.action_range / 2.0 + self.action_min
        raw_actions = actions.cpu().numpy()
        if self.temporal_ensembler is not None:
            return self.temporal_ensembler.update(raw_actions)[None, :]
        return raw_actions


def build_model(
    chunk_size: int = 4,
    temporal_ensemble: bool = False,
    temporal_ensemble_k: float = 0.01,
) -> IcedTeaPolicy:
    return IcedTeaPolicy(chunk_size, temporal_ensemble, temporal_ensemble_k)


def calculate_evaluation_mse(results: list[dict]) -> dict:
    squared_error_sum = np.zeros(8, dtype=np.float64)
    frame_count = 0
    for result in results:
        with np.load(result["path"], allow_pickle=False) as artifact:
            prediction = artifact["prediction"].astype(np.float64)
            ground_truth = artifact["ground_truth"].astype(np.float64)
        if (
            prediction.ndim != 2
            or prediction.shape != ground_truth.shape
            or prediction.shape[1] != 8
        ):
            raise ValueError(
                f"Expected aligned [frames, 8] arrays in {result['path']}; "
                f"got prediction={prediction.shape}, ground_truth={ground_truth.shape}"
            )
        squared_error_sum += np.square(prediction - ground_truth).sum(axis=0)
        frame_count += prediction.shape[0]
    if frame_count == 0:
        raise ValueError("Cannot calculate MSE without evaluated frames")
    per_dimension_mse = squared_error_sum / frame_count
    joint_mse_sum = float(per_dimension_mse[:7].sum())
    return {
        "frames": frame_count,
        "joint_mse_sum": joint_mse_sum,
        "joint_mse_mean": joint_mse_sum / 7.0,
        "gripper_mse": float(per_dimension_mse[7]),
        "per_dimension_mse": per_dimension_mse.tolist(),
    }


def print_evaluation_mse(metrics: dict) -> None:
    print(
        "逐帧加权均方误差（原始动作单位）\n"
        f"评估帧数={metrics['frames']}\n"
        f"七关节MSE之和={metrics['joint_mse_sum']:.8f}\n"
        f"七关节MSE均值={metrics['joint_mse_mean']:.8f}\n"
        f"夹爪MSE={metrics['gripper_mse']:.8f}",
        flush=True,
    )


def write_evaluation_config(
    output: Path, model: IcedTeaPolicy, dataset: Dataset
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    stats_path = TRAIN_OUTPUT / "normalization_stats.json"
    smoothing_enabled = model.temporal_ensembler is not None
    retained_steps = (
        model.horizon - model.action_start if smoothing_enabled else model.chunk_size
    )
    metadata = {
        "checkpoint": str(CHECKPOINT.resolve()),
        "dataset": str(TEST_DATASET.resolve()),
        "episode_ids": [episode.episode_id for episode in dataset.episodes],
        "plotted_episode_ids": [
            episode.episode_id for episode in dataset.episodes[:max_plot_episodes]
        ],
        "chunk_size": model.chunk_size,
        "prediction_horizon": model.horizon,
        "action_start": model.action_start,
        "raw_prediction_slice": [
            model.action_start,
            model.action_start + retained_steps,
        ],
        "normalization_file": str(stats_path),
        "normalization_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "temporal_ensemble": {
            "enabled": smoothing_enabled,
            "mode": "act_temporal_ensemble" if smoothing_enabled else "disabled",
            "execution_steps": model.chunk_size,
            "retained_prediction_steps": retained_steps,
            "candidate_chunks": "all_predictions_covering_current_timestep",
            "reference": "raw_predictions_aligned_to_current_timestep",
            "decay": model.temporal_ensemble_k,
            "weight_order": "oldest_to_newest",
            "weight_formula": "exp(-decay * prediction_rank)",
            "query_every_steps": model.chunk_size,
            "reset_each_episode": True,
            "valid_zero_actions": True,
        },
    }
    with (output / "evaluation_config.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def run_evaluation(args) -> Path:
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")

    configure(args)
    dataset = build_dataset()
    if args.max_episodes is not None:
        dataset = Dataset(dataset.episodes[: args.max_episodes], name=dataset.name)
    model = build_model(
        args.chunk_size, args.temporal_ensemble, args.temporal_ensemble_k
    )
    print(
        f"Evaluation execution steps={model.chunk_size}; prediction horizon={model.horizon}; "
        f"ACT temporal ensemble={args.temporal_ensemble}; k={args.temporal_ensemble_k}",
        flush=True,
    )

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
    write_evaluation_config(output, model, dataset)
    artifact_dir = output / "artifacts"
    figure_dir = output / "figures"
    results = evaluate(model, dataset, artifact_dir)
    for result in results[:max_plot_episodes]:
        for kind in ("abs", "delta"):
            render_episode(
                result["path"],
                figure_dir / f"{result['episode_id']}_{kind}.png",
                kind=kind,
                data_root=dataset.name,
                checkpoint=str(CHECKPOINT),
            )
    print_evaluation_mse(calculate_evaluation_mse(results))
    print(f"Evaluated {len(results)} episodes; results: {output}", flush=True)
    return output


def main() -> None:
    args = parse_arguments()
    run_evaluation(args)


if __name__ == "__main__":
    main()
