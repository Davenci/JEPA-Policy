"""Persist a deterministic holdout of complete episodes before loading frames."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class EpisodeSplitOptions:
    output: Path
    episode_indices: Path | None = None
    resume_checkpoint: Path | None = None
    seed: int = 0


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[int, ...]
    evaluation: tuple[int, ...]
    candidates: tuple[int, ...]
    dataset_root: Path
    seed: int
    metadata_sha256: str


def validate_episode_indices(values: object, description: str) -> tuple[int, ...]:
    if not isinstance(values, list) or any(
        type(value) is not int or value < 0 for value in values
    ):
        raise ValueError(f"{description} must be a list of nonnegative integer IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{description} contains duplicate episode indices")
    return tuple(sorted(values))


def read_episode_metadata(dataset_root: Path) -> tuple[tuple[int, ...], str]:
    episode_paths = sorted((dataset_root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(
            f"No episode metadata in {dataset_root / 'meta/episodes'}"
        )
    info_path = dataset_root / "meta/info.json"
    digest = hashlib.sha256()
    indices = []
    for path in (info_path, *episode_paths):
        digest.update(path.relative_to(dataset_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    for path in episode_paths:
        indices.extend(
            pq.read_table(path, columns=["episode_index"])["episode_index"].to_pylist()
        )
    return validate_episode_indices(indices, "Episode metadata"), digest.hexdigest()


def split_payload(split: EpisodeSplit) -> dict:
    return {
        "schema_version": 1,
        "dataset": str(split.dataset_root),
        "metadata_sha256": split.metadata_sha256,
        "seed": split.seed,
        "evaluation_fraction": 0.05,
        "selection_mode": "all_episode_labels",
        "candidate_episode_indices": list(split.candidates),
        "train_episode_indices": list(split.train),
        "eval_episode_indices": list(split.evaluation),
        "candidate_episode_count": len(split.candidates),
        "train_episode_count": len(split.train),
        "eval_episode_count": len(split.evaluation),
    }


def restore_episode_split(split: EpisodeSplit, checkpoint: Path) -> EpisodeSplit:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    path = checkpoint.parent.parent / "episode_split.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = split_payload(split)
    identity_keys = (
        "schema_version",
        "dataset",
        "metadata_sha256",
        "seed",
        "evaluation_fraction",
        "selection_mode",
        "candidate_episode_indices",
    )
    for key in identity_keys:
        if payload[key] != expected[key]:
            raise ValueError(f"Resume episode split differs in {key}: {path}")
    train = validate_episode_indices(
        payload["train_episode_indices"], "Saved training IDs"
    )
    evaluation = validate_episode_indices(
        payload["eval_episode_indices"], "Saved evaluation IDs"
    )
    if (
        not train
        or not evaluation
        or set(train) & set(evaluation)
        or set(train) | set(evaluation) != set(split.candidates)
        or len(evaluation) != (len(split.candidates) + 19) // 20
    ):
        raise ValueError(f"Invalid saved training/evaluation partition: {path}")
    restored = EpisodeSplit(
        train,
        evaluation,
        split.candidates,
        split.dataset_root,
        split.seed,
        split.metadata_sha256,
    )
    if payload != split_payload(restored):
        raise ValueError(f"Saved episode split contains inconsistent metadata: {path}")
    return restored


def save_episode_split(split: EpisodeSplit, output: Path) -> None:
    files = {
        "episode_split.json": split_payload(split),
        "train_episode_indices.json": list(split.train),
        "eval_episode_indices.json": list(split.evaluation),
    }
    for name, payload in files.items():
        path = output / name
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
            raise FileExistsError(
                f"Output already contains a different episode split: {path}"
            )
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        path = output / name
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def prepare_episode_split(
    dataset_root: Path, options: EpisodeSplitOptions
) -> EpisodeSplit:
    if type(options.seed) is not int or options.seed < 0:
        raise ValueError("Episode split seed must be a nonnegative integer")
    dataset_root = dataset_root.resolve()
    available, fingerprint = read_episode_metadata(dataset_root)
    candidates = available
    if options.episode_indices is not None:
        candidates = validate_episode_indices(
            json.loads(options.episode_indices.read_text(encoding="utf-8")),
            "Candidate episode indices",
        )
        missing = sorted(set(candidates) - set(available))
        if missing:
            raise ValueError(f"Episode indices not found in dataset: {missing}")
    if len(candidates) < 2:
        raise ValueError("Training/evaluation split requires at least two episodes")
    evaluation_count = (len(candidates) + 19) // 20
    permutation = np.random.default_rng(options.seed).permutation(candidates)
    split = EpisodeSplit(
        train=tuple(sorted(int(value) for value in permutation[evaluation_count:])),
        evaluation=tuple(
            sorted(int(value) for value in permutation[:evaluation_count])
        ),
        candidates=candidates,
        dataset_root=dataset_root,
        seed=options.seed,
        metadata_sha256=fingerprint,
    )
    if options.resume_checkpoint is not None:
        split = restore_episode_split(split, options.resume_checkpoint)
    save_episode_split(split, options.output)
    return split
