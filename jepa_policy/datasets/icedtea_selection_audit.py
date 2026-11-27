"""Explain episode selection while retaining text labels as metadata only."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_episode_records(root: Path) -> dict[int, dict]:
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata under {root / 'meta/episodes'}")
    columns = [
        "episode_index",
        "tasks",
        "dataset_from_index",
        "dataset_to_index",
    ]
    records = {}
    for path in paths:
        for record in pq.read_table(path, columns=columns).to_pylist():
            episode_id = int(record["episode_index"])
            if episode_id in records:
                raise ValueError(f"Duplicate episode metadata: {episode_id}")
            records[episode_id] = record
    return records


def build_task_counts(dataset, episodes: dict[int, dict]) -> list[dict]:
    tasks = pq.read_table(
        dataset.root / "meta/tasks.parquet", columns=["task_index", "task"]
    ).to_pylist()
    return [
        {
            "task_index": int(task["task_index"]),
            "task_name": task["task"],
            "episodes": sum(
                task["task"] in episode["tasks"] for episode in episodes.values()
            ),
            "frames": int(np.count_nonzero(dataset.task_indices == task["task_index"])),
        }
        for task in sorted(tasks, key=lambda item: item["task_index"])
    ]


def build_episode_window_counts(dataset, selected: list[int]) -> list[dict]:
    counts = Counter(episode_id for _, episode_id in dataset.samples)
    if set(counts) - set(selected):
        raise ValueError("Training samples contain episodes outside the selection")
    needed = max(dataset.horizon, dataset.obs_steps + dataset.future_steps)
    results = []
    for episode_id in selected:
        record = dataset.episodes[episode_id]
        frames = int(record["dataset_to_index"]) - int(record["dataset_from_index"])
        windows = max(0, frames - needed + 1)
        if counts[episode_id] != windows:
            raise ValueError(
                f"Window count differs from metadata: episode {episode_id}"
            )
        results.append(
            {
                "episode_index": episode_id,
                "frames": frames,
                "windows": windows,
                "boundary_start_positions_excluded": frames - windows,
            }
        )
    return results


def build_normalization_scope(dataset) -> dict:
    mask = np.isin(dataset.episode_ids, tuple(dataset.episodes))
    episode_ids = np.unique(dataset.episode_ids[mask]).astype(int).tolist()
    return {
        "enabled": dataset.normalize,
        "scope": "selected_episode_frames",
        "episodes": len(episode_ids) if dataset.normalize else 0,
        "frames": int(mask.sum()) if dataset.normalize else 0,
        "episode_indices": episode_ids if dataset.normalize else [],
    }


def build_selection_counts(dataset, groups: dict, per_episode: list[dict]) -> dict:
    selected_mask = np.isin(dataset.episode_ids, groups["selected"])
    return {
        "all_episodes": len(groups["all"]),
        "all_frames": len(dataset.episode_ids),
        "index_selected_episodes": len(groups["selected"]),
        "index_selected_frames": int(selected_mask.sum()),
        "window_episodes": len(groups["window_used"]),
        "windows": len(dataset.samples),
        "excluded_by_index_episodes": len(groups["excluded_by_index"]),
        "short_episodes": len(groups["short"]),
        "boundary_start_positions_excluded": sum(
            record["boundary_start_positions_excluded"] for record in per_episode
        ),
    }


def build_selection_audit(dataset) -> dict:
    episodes = load_episode_records(dataset.root)
    selected = set(dataset.episodes)
    missing = selected - episodes.keys()
    if missing:
        raise ValueError(f"Selected episode metadata not found: {sorted(missing)}")
    groups = {
        "all": sorted(episodes),
        "selected": sorted(selected),
        "window_used": sorted({episode_id for _, episode_id in dataset.samples}),
        "excluded_by_index": sorted(episodes.keys() - selected),
    }
    per_episode = build_episode_window_counts(dataset, groups["selected"])
    groups["short"] = [
        row["episode_index"] for row in per_episode if row["windows"] == 0
    ]
    return {
        "dataset_root": str(dataset.root.resolve()),
        "label_usage": "metadata_only",
        "episode_index_filter": {
            "enabled": dataset.selected_episode_indices is not None,
            "requested": (
                list(dataset.selected_episode_indices)
                if dataset.selected_episode_indices is not None
                else None
            ),
        },
        "tasks": build_task_counts(dataset, episodes),
        "counts": build_selection_counts(dataset, groups, per_episode),
        "episode_indices": {key: ids for key, ids in groups.items() if key != "all"},
        "normalization": build_normalization_scope(dataset),
        "windows": {
            "needed_frames": max(
                dataset.horizon, dataset.obs_steps + dataset.future_steps
            ),
            "obs_steps": dataset.obs_steps,
            "horizon": dataset.horizon,
            "future_steps": dataset.future_steps,
        },
        "per_episode": per_episode,
    }


def format_episode_indices(indices: list[int]) -> str:
    if not indices:
        return "无"
    ranges = []
    start = previous = indices[0]
    for current in [*indices[1:], None]:
        if current is not None and current == previous + 1:
            previous = current
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = current
    return ", ".join(ranges)


def format_selection_audit(audit: dict) -> list[str]:
    counts = audit["counts"]
    groups = audit["episode_indices"]
    scope = audit["normalization"]
    lines = [
        "[数据筛选审计]",
        f"数据根目录: {audit['dataset_root']}",
        "标签仅作元数据统计，不参与筛选",
        "文本标签分布:",
    ]
    for task in audit["tasks"]:
        lines.append(
            f"  task_index={task['task_index']} {task['task_name']!r}: "
            f"{task['episodes']} 回合 / {task['frames']} 帧"
        )
    lines.extend(
        [
            f"回合筛选: 原始 {counts['all_episodes']} -> "
            f"索引选择 {counts['index_selected_episodes']} -> "
            f"可用窗口回合 {counts['window_episodes']}",
            f"排除原因: 未被索引选中 {counts['excluded_by_index_episodes']} 回合; "
            f"已选但长度不足 {counts['short_episodes']} 回合",
            f"帧数: 原始 {counts['all_frames']} / "
            f"索引选择 {counts['index_selected_frames']}",
            f"有效训练窗口: {counts['windows']}; 每窗至少 "
            f"{audit['windows']['needed_frames']} 帧，禁止跨回合",
            f"边界排除窗口起点: {counts['boundary_start_positions_excluded']} "
            "（不表示删除这些原始帧）",
            (
                f"归一化统计范围: 选中回合的全部帧，"
                f"{scope['episodes']} 回合 / {scope['frames']} 帧; "
                "短回合仍参与统计，未选择回合不参与"
                if scope["enabled"]
                else "归一化统计: 已禁用，读取原始值，不计算数据集统计量"
            ),
            f"实际选择回合索引: {format_episode_indices(groups['selected'])}",
            f"未选择回合索引: {format_episode_indices(groups['excluded_by_index'])}",
            f"长度不足回合索引: {format_episode_indices(groups['short'])}",
            "完整逐回合窗口数和排除索引见 dataset_selection.json",
        ]
    )
    return lines
