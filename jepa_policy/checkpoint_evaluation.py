"""Evaluate saved EMA weights without changing the running training agent."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jepa_policy.datasets.icedtea_dataset import CAMERAS
from jepa_policy.eval_rng import isolated_torch_rng
from tools.model_eval.model_eval import (
    Dataset as ShadowDataset,
)
from tools.model_eval.model_eval import (
    Episode,
    evaluate,
    render_episode,
)


@dataclass(frozen=True)
class CheckpointEvaluationOptions:
    output_path: Path
    batch_size: int = 64
    chunk_size: int = 4
    seed: int = 12345


def move_evaluation_batch(value, device: torch.device):
    if isinstance(value, dict):
        return {key: move_evaluation_batch(item, device) for key, item in value.items()}
    value = value.to(device=device, non_blocking=True)
    if value.dtype == torch.uint8:
        return value.float().div_(255).mul_(2).sub_(1)
    return value.float()


class NormalizedEvaluationWindows(Dataset):
    """Normalize held-out windows exclusively with the training ranges."""

    def __init__(self, source, training):
        if source.normalize:
            raise ValueError("Evaluation source must have normalize=False")
        self.source = source
        self.action_min = torch.as_tensor(training.action_min)
        self.action_range = torch.as_tensor(training.action_range)
        self.state_min = torch.as_tensor(training.state_min)
        self.state_range = torch.as_tensor(training.state_range)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        item = self.source[index]
        item["action"] = (
            item["action"] - self.action_min
        ) / self.action_range * 2.0 - 1.0
        for field in ("obs", "future_obs"):
            item[field]["agent_pos"] = (
                item[field]["agent_pos"] - self.state_min
            ) / self.state_range * 2.0 - 1.0
        return item


class CheckpointShadowPolicy:
    """Use the same two-pass, EMA inference and action alignment as standalone evaluation."""

    def __init__(self, agent, training, chunk_size):
        self.agent = agent
        self.device = torch.device(agent.config.optimization.device)
        self.horizon = agent.config.task.horizon
        self.action_start = agent.config.task.obs_steps - 1
        self.action_dim = agent.config.task.act_dim
        self.chunk_size = chunk_size
        self.action_min = torch.as_tensor(training.action_min, device=self.device)
        self.action_range = torch.as_tensor(training.action_range, device=self.device)

    def predict(self, observation):
        observation = {key: value.unsqueeze(0) for key, value in observation.items()}
        inputs = move_evaluation_batch(observation, self.device)
        initial = torch.zeros((1, self.horizon, self.action_dim), device=self.device)
        normalized, _ = self.agent.sample(
            initial, inputs, num_steps=1, use_ema=True, return_future=True
        )
        end = self.action_start + self.chunk_size
        selected = normalized[0, self.action_start : end]
        raw = (selected + 1.0) * self.action_range / 2.0 + self.action_min
        return raw.cpu().numpy()


class CheckpointEvaluator:
    def __init__(self, config, datasets: tuple, options):
        train_dataset, evaluation_dataset = datasets
        self.config = config
        self.training = train_dataset
        self.source = evaluation_dataset
        self.options = options
        self.device = torch.device(config.optimization.device)
        if options.batch_size <= 0 or options.seed < 0:
            raise ValueError(
                "Evaluation batch size must be positive and seed nonnegative"
            )
        if (
            not 1
            <= options.chunk_size
            <= config.task.horizon - config.task.obs_steps + 1
        ):
            raise ValueError("Evaluation chunk size exceeds the current/future horizon")
        if set(train_dataset.episodes) & set(evaluation_dataset.episodes):
            raise ValueError("Training and evaluation episodes must be disjoint")
        self.windows = NormalizedEvaluationWindows(evaluation_dataset, train_dataset)
        self.states = (
            self.source.raw_states - train_dataset.state_min
        ) / train_dataset.state_range * 2.0 - 1.0

    def load_saved_agent(self, agent, checkpoint_path: Path, step: int):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["training_state"]["n_gradient_step"] + 1 != step:
            raise ValueError(
                "Checkpoint optimizer step differs from the requested evaluation step"
            )
        saved = copy.copy(agent)
        saved.encoder = copy.deepcopy(agent.encoder_ema)
        saved.flow_map = copy.deepcopy(agent.flow_map_ema)
        saved.encoder.load_state_dict(state["encoder_ema"], strict=True)
        saved.flow_map.load_state_dict(state["flow_map_ema"], strict=True)
        saved.encoder_ema = saved.encoder
        saved.flow_map_ema = saved.flow_map
        saved.encoder.eval()
        saved.flow_map.eval()
        return saved

    def validation_loss(self, agent):
        loader = DataLoader(
            self.windows,
            batch_size=self.options.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            generator=torch.Generator().manual_seed(self.options.seed),
        )
        totals = {}
        sample_count = 0
        for batch_index, batch in enumerate(loader):
            batch = move_evaluation_batch(batch, self.device)
            count = batch["action"].shape[0]
            metrics = agent.validation_metrics(
                batch["action"],
                batch["obs"],
                torch.ones(count, device=self.device),
                batch["future_obs"],
                seed=self.options.seed + batch_index,
            )
            if not metrics or not all(
                math.isfinite(value) for value in metrics.values()
            ):
                raise ValueError(
                    "Validation loss must produce finite training-objective metrics"
                )
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * count
            sample_count += count
        if sample_count == 0:
            raise ValueError("No complete action windows in the evaluation split")
        result = {name: value / sample_count for name, value in totals.items()}
        result["loss"] = result["loss_total"]
        result["windows"] = sample_count
        return result

    def create_episode(self, episode_id, record):
        start = int(record["dataset_from_index"])
        end = int(record["dataset_to_index"])

        def observe(frame):
            rows = (start + max(frame - 1, 0), start + frame)
            observation = {
                camera: torch.stack(
                    [self.source.read_image(record, camera, row) for row in rows]
                )
                for camera in CAMERAS
            }
            observation["agent_pos"] = torch.from_numpy(self.states[list(rows)].copy())
            return observation

        return Episode(
            f"episode_{episode_id:06d}",
            end - start,
            self.source.raw_actions[start:end],
            observe,
        )

    def shadow_metrics(self, results):
        squared_error = np.zeros(self.config.task.act_dim, dtype=np.float64)
        frame_count = 0
        for result in results:
            with np.load(result["path"], allow_pickle=False) as artifact:
                error = artifact["prediction"].astype(np.float64) - artifact[
                    "ground_truth"
                ].astype(np.float64)
            squared_error += np.square(error).sum(axis=0)
            frame_count += error.shape[0]
        if frame_count == 0:
            raise ValueError("No shadow evaluation frames")
        mse = squared_error / frame_count
        return {
            "shadow_action_mse": float(
                np.mean(4.0 * mse / self.training.action_range**2)
            ),
            "joint_mse_sum": float(mse[:7].sum()),
            "gripper_mse": float(mse[7]),
            "frames": frame_count,
            "episodes": len(results),
        }

    def save_figures(self, results, output: Path, checkpoint_path: Path):
        for result in results:
            for kind in ("abs", "delta"):
                render_episode(
                    result["path"],
                    output / "figures" / f"{result['episode_id']}_{kind}.png",
                    kind=kind,
                    data_root=str(self.source.root),
                    checkpoint=str(checkpoint_path),
                )

    def normalization_provenance(self):
        path = self.options.output_path / "normalization_stats.json"
        payload = path.read_bytes()
        statistics = json.loads(payload)
        if statistics["schema_version"] != 1:
            raise ValueError("Unsupported saved training normalization schema")
        for feature, prefix in (("action", "action"), ("observation.state", "state")):
            for field in ("min", "range"):
                saved = np.asarray(
                    statistics["features"][feature][field], dtype=np.float32
                )
                actual = getattr(self.training, f"{prefix}_{field}")
                if not np.array_equal(saved, actual):
                    raise ValueError(
                        "Saved training normalization differs from evaluation inputs"
                    )
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def source_hashes(self):
        root = Path(__file__).resolve().parent
        names = (
            "checkpoint_evaluation.py",
            "agent.py",
            "samplers.py",
            "losses.py",
            "encoders.py",
            "network_utils.py",
            "networks/chitfm.py",
        )
        return {
            f"jepa_policy/{name}": hashlib.sha256(
                (root / name).read_bytes()
            ).hexdigest()
            for name in names
        }

    def evaluate(self, agent, checkpoint_path: Path, step: int) -> dict:
        checkpoint_path = Path(checkpoint_path)
        normalization = self.normalization_provenance()
        output = checkpoint_path.parent / f"{checkpoint_path.stem}_eval"
        output.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        with isolated_torch_rng(self.options.seed, self.device), torch.inference_mode():
            saved = self.load_saved_agent(agent, checkpoint_path, step)
            metrics = self.validation_loss(saved)
            episodes = [
                self.create_episode(episode_id, record)
                for episode_id, record in sorted(self.source.episodes.items())
            ]
            policy = CheckpointShadowPolicy(
                saved, self.training, self.options.chunk_size
            )
            results = evaluate(
                policy,
                ShadowDataset(episodes, name=str(self.source.root)),
                output / "artifacts",
            )
            metrics.update(self.shadow_metrics(results))
            self.save_figures(results, output, checkpoint_path)
        metrics["seconds"] = time.perf_counter() - started
        metadata = {
            "step": step,
            "checkpoint": str(checkpoint_path),
            "weights": "saved_ema",
            "dataset": str(self.source.root),
            "normalization": normalization,
            "source_hashes": self.source_hashes(),
            "evaluation_episode_indices": sorted(self.source.episodes),
            "training_episode_indices": sorted(self.training.episodes),
            "chunk_size": self.options.chunk_size,
            "temporal_ensemble": False,
            "seed": self.options.seed,
            "loss_definition": "training_joint_mip_two_step_with_future_auxiliary",
            "shadow_action_mse_definition": "frame_weighted_mse_in_training_normalized_space",
            "metrics": metrics,
        }
        (output / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metrics
