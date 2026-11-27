"""Persist the training-only normalization used by checkpoints and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jepa_policy.datasets.icedtea_dataset import ACTION_NAMES, IcedTeaDataset
from jepa_policy.training_startup import collect_normalization_info


def validate_run_output(output_path: Path) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Choose a new, empty training output: {output_path}")


def prepare_run_directory(
    output_path: Path, dataset: IcedTeaDataset, resume_checkpoint: Path | None
) -> tuple[Path, Path]:
    if resume_checkpoint is not None:
        validate_resume_normalization(dataset, resume_checkpoint)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "metrics.jsonl"
    if resume_checkpoint is not None and metrics_path.exists():
        raise FileExistsError(
            f"Resume output already has metrics; choose a new output directory: {metrics_path}"
        )
    model_dir = output_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "selection_mode": "all_episode_labels",
        "action_names": ACTION_NAMES,
        "action_min": dataset.action_min.tolist(),
        "action_range": dataset.action_range.tolist(),
        "state_min": dataset.state_min.tolist(),
        "state_range": dataset.state_range.tolist(),
        "dataset": str(dataset.root),
        "episode_indices": sorted(dataset.episodes),
        "window_episode_indices": sorted(
            {episode_id for _, episode_id in dataset.samples}
        ),
    }
    (output_path / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    normalization_stats = {
        "schema_version": 1,
        "selection_mode": "all_episode_labels",
        "dataset": str(dataset.root),
        "normalization": "selected_episode_min_max_to_minus1_plus1",
        "episode_indices": metadata["episode_indices"],
        "features": collect_normalization_info(dataset, output_path)["features"],
        "image_preprocessing": {
            "source_pixel_type": "uint8_rgb",
            "resize_shape": [dataset.image_size, dataset.image_size],
            "resize_interpolation": dataset.image_interpolation,
            "crop_shape": [112, 112],
            "training_crop_mode": "temporal_consistent_random",
            "evaluation_crop_mode": "center",
            "imagenet_normalization": False,
            "pixel_mapping": "uint8_to_0_1_then_minus1_to_plus1",
        },
    }
    (output_path / "normalization_stats.json").write_text(
        json.dumps(normalization_stats, indent=2) + "\n"
    )
    return model_dir, metrics_path


def validate_resume_normalization(dataset, checkpoint: Path) -> None:
    path = checkpoint.parent.parent / "normalization_stats.json"
    statistics = json.loads(path.read_text())
    if statistics["schema_version"] != 1:
        raise ValueError(f"Unsupported normalization schema: {path}")
    if statistics["episode_indices"] != sorted(dataset.episodes):
        raise ValueError("Resume normalization uses different training episodes")
    for feature, prefix in (("action", "action"), ("observation.state", "state")):
        for field in ("min", "range"):
            saved = np.asarray(statistics["features"][feature][field], dtype=np.float32)
            if not np.array_equal(saved, getattr(dataset, f"{prefix}_{field}")):
                raise ValueError(
                    f"Resume training normalization differs: {feature}.{field}"
                )
