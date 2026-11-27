import json
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.train_x1_icedtea import (
    create_config,
    create_optimization_config,
    prepare_run_directory,
)
from jepa_policy import training_startup
from jepa_policy.datasets.icedtea_dataset import (
    ACTION_NAMES,
    IcedTeaDataset,
    IcedTeaDatasetConfig,
)
from jepa_policy.datasets.icedtea_selection_audit import (
    build_selection_audit,
    format_selection_audit,
)


def write_episode_dataset(root: Path, lengths: list[int], tasks: list[int]) -> Path:
    data_dir = root / "data" / "chunk-000"
    episode_dir = root / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True)
    episode_dir.mkdir(parents=True)
    features = {key: {"names": ACTION_NAMES} for key in ("action", "observation.state")}
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": 15, "features": features})
    )
    task_rows = [
        {"task_index": 0, "task": "grasp the red iced tea."},
        {"task_index": 1, "task": "fetch iced tea."},
    ]
    pq.write_table(pa.Table.from_pylist(task_rows), root / "meta" / "tasks.parquet")
    frames = []
    episodes = []
    for episode_id, (length, task_index) in enumerate(zip(lengths, tasks, strict=True)):
        start = len(frames)
        for frame in range(length):
            value = float(episode_id * 100 + frame)
            frames.append(
                {
                    "action": [value + joint for joint in range(8)],
                    "observation.state": [value * 2 + joint for joint in range(8)],
                    "timestamp": frame / 15.0,
                    "episode_index": episode_id,
                    "task_index": task_index,
                }
            )
        episodes.append(
            {
                "episode_index": episode_id,
                "tasks": [task_rows[task_index]["task"]],
                "dataset_from_index": start,
                "dataset_to_index": len(frames),
            }
        )
    pq.write_table(pa.Table.from_pylist(frames), data_dir / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(episodes), episode_dir / "file-000.parquet")
    return root


@pytest.fixture
def audit_dataset_root(tmp_path: Path) -> Path:
    return write_episode_dataset(tmp_path, [12, 13, 4, 11], [0, 0, 0, 1])


def test_all_task_descriptions_are_loaded_and_logged(tmp_path: Path):
    root = write_episode_dataset(tmp_path, [12] * 228, [0] * 50 + [1] * 178)
    dataset = IcedTeaDataset(IcedTeaDatasetConfig(root))
    audit = build_selection_audit(dataset)
    assert audit["counts"]["all_episodes"] == 228
    assert audit["counts"]["index_selected_episodes"] == 228
    assert audit["counts"]["index_selected_frames"] == 2736
    assert audit["counts"]["excluded_by_index_episodes"] == 0
    assert audit["counts"]["windows"] == 684
    assert audit["episode_indices"]["selected"] == list(range(228))
    assert audit["normalization"]["episodes"] == 228
    assert audit["normalization"]["frames"] == 2736
    assert audit["label_usage"] == "metadata_only"
    assert "task_filter" not in audit
    assert [task["episodes"] for task in audit["tasks"]] == [50, 178]
    text = "\n".join(format_selection_audit(audit))
    for value in ("grasp the red iced tea.", "fetch iced tea.", "228", "50", "178"):
        assert value in text
    assert "标签仅作元数据统计，不参与筛选" in text
    assert "任务匹配" not in text
    assert "不同描述" not in text


def test_only_explicit_episode_indices_are_filtered(audit_dataset_root: Path):
    dataset = IcedTeaDataset(
        IcedTeaDatasetConfig(audit_dataset_root, episode_indices=(0, 3))
    )
    audit = build_selection_audit(dataset)
    counts = audit["counts"]
    assert counts["all_episodes"] == 4
    assert counts["all_frames"] == 40
    assert counts["index_selected_episodes"] == 2
    assert counts["index_selected_frames"] == 23
    assert counts["excluded_by_index_episodes"] == 2
    assert audit["episode_indices"]["selected"] == [0, 3]
    assert audit["episode_indices"]["excluded_by_index"] == [1, 2]
    assert audit["normalization"]["frames"] == 23
    assert audit["normalization"]["episode_indices"] == [0, 3]
    assert "task_filter" not in audit


def test_short_episode_still_participates_in_statistics(audit_dataset_root: Path):
    dataset = IcedTeaDataset(
        IcedTeaDatasetConfig(audit_dataset_root, episode_indices=(0, 2))
    )
    audit = build_selection_audit(dataset)
    assert audit["counts"]["window_episodes"] == 1
    assert audit["counts"]["windows"] == len(dataset) == 3
    assert audit["counts"]["short_episodes"] == 1
    assert audit["counts"]["boundary_start_positions_excluded"] == 13
    assert audit["episode_indices"]["selected"] == [0, 2]
    assert audit["episode_indices"]["window_used"] == [0]
    assert audit["episode_indices"]["short"] == [2]
    assert audit["normalization"]["scope"] == "selected_episode_frames"
    assert audit["normalization"]["episodes"] == 2
    assert audit["normalization"]["frames"] == 16
    assert audit["normalization"]["episode_indices"] == [0, 2]
    assert audit["windows"]["needed_frames"] == 10
    assert audit["per_episode"][1]["windows"] == 0
    np.testing.assert_array_equal(dataset.action_range, np.full(8, 203.0))


def test_audit_does_not_change_dataset_values(audit_dataset_root: Path):
    dataset = IcedTeaDataset(IcedTeaDatasetConfig(audit_dataset_root))
    names = (
        "actions",
        "states",
        "action_min",
        "state_min",
        "action_range",
        "state_range",
    )
    arrays = {name: getattr(dataset, name).copy() for name in names}
    samples = list(dataset.samples)
    episodes = dict(dataset.episodes)
    audit = build_selection_audit(dataset)
    assert json.loads(json.dumps(audit)) == audit
    assert dataset.samples == samples
    assert dataset.episodes == episodes
    for name in names:
        np.testing.assert_array_equal(getattr(dataset, name), arrays[name])
    assert not dataset.containers
    assert not dataset.frame_cache


def test_audit_reports_longer_future_window(tmp_path: Path):
    root = write_episode_dataset(tmp_path, [30, 12, 4, 11], [0, 0, 0, 1])
    dataset = IcedTeaDataset(
        IcedTeaDatasetConfig(root, episode_indices=(0, 1), future_steps=20)
    )
    audit = build_selection_audit(dataset)
    assert audit["windows"]["needed_frames"] == 22
    assert audit["counts"]["windows"] == 9
    assert audit["counts"]["boundary_start_positions_excluded"] == 33
    assert audit["normalization"]["frames"] == 42


def test_audit_reports_raw_value_mode(audit_dataset_root: Path):
    dataset = IcedTeaDataset(IcedTeaDatasetConfig(audit_dataset_root, normalize=False))
    audit = build_selection_audit(dataset)
    assert audit["normalization"]["enabled"] is False
    assert "已禁用" in "\n".join(format_selection_audit(audit))


@pytest.fixture
def summary_inputs(audit_dataset_root: Path, monkeypatch):
    dataset = IcedTeaDataset(
        IcedTeaDatasetConfig(audit_dataset_root, episode_indices=(0, 2))
    )
    args = SimpleNamespace(
        dataset=audit_dataset_root,
        output=audit_dataset_root / "run",
        steps=100000,
        batch_size=128,
        workers=16,
        save_every=5000,
        resume=None,
        episode_indices=audit_dataset_root / "train_indices.json",
        cache_root=audit_dataset_root / "cache",
        cache_build_workers=8,
        device="cuda:0",
        log_every=20,
        split_seed=0,
        eval_batch_size=64,
        eval_chunk_size=4,
        eval_seed=12345,
        mlflow=True,
        mlflow_tracking_uri="https://mlflow.seer-robotics.cn",
        mlflow_experiment="pretrain-fm",
        mlflow_run_name=None,
    )
    optimization = create_optimization_config(args.steps, args.batch_size, args.workers)
    config = create_config(args.dataset, args.output, optimization)
    monkeypatch.setattr(
        training_startup,
        "collect_runtime_info",
        lambda device: {"device": device, "gpu": "CPU-only test double"},
    )
    monkeypatch.setattr(
        training_startup,
        "collect_code_info",
        lambda root: {
            "root": str(root),
            "commit": "test-commit",
            "branch": "test-branch",
            "working_tree_dirty": True,
        },
    )
    return config, dataset, args


def test_summary_records_actual_settings_and_selected_statistics(summary_inputs):
    config, dataset, args = summary_inputs
    summary = training_startup.build_training_summary(config, dataset, args)
    assert summary["arguments"]["steps"] == 100000
    assert summary["arguments"]["batch_size"] == 128
    assert summary["arguments"]["workers"] == 16
    assert summary["arguments"]["save_every"] == 5000
    assert summary["arguments"]["log_every"] == 20
    assert summary["config"]["optimization"]["gradient_steps"] == 100000
    assert summary["config"]["optimization"]["batch_size"] == 128
    assert summary["config"]["optimization"]["dataloader_num_workers"] == 16
    assert summary["schedule"]["scheduler_total_steps"] == 100000
    assert summary["selection"]["normalization"]["episode_indices"] == [0, 2]
    assert summary["selection"]["normalization"]["frames"] == 16
    assert summary["normalization"]["source"] == "selected_training_episode_frames_only"
    assert summary["normalization"]["clip_state"] is False
    assert summary["normalization"]["file"] == str(
        args.output / "normalization_stats.json"
    )
    for feature, minimum, scale in [
        ("action", dataset.action_min, dataset.action_range),
        ("observation.state", dataset.state_min, dataset.state_range),
    ]:
        recorded = summary["normalization"]["features"][feature]
        np.testing.assert_array_equal(recorded["min"], minimum)
        np.testing.assert_array_equal(recorded["range"], scale)


def test_summary_files_are_json_and_existing_run_is_preserved(summary_inputs):
    config, dataset, args = summary_inputs
    summary = training_startup.build_training_summary(config, dataset, args)
    training_startup.write_training_summary(summary, args.output)
    filenames = [
        "training_summary.json",
        "training_config.json",
        "dataset_selection.json",
    ]
    saved = {name: (args.output / name).read_bytes() for name in filenames}
    persisted = json.loads(saved["training_summary.json"])
    assert persisted["arguments"]["batch_size"] == 128
    assert persisted["selection"]["episode_indices"]["selected"] == [0, 2]
    assert json.loads(saved["training_config.json"]) == persisted["config"]
    assert json.loads(saved["dataset_selection.json"]) == persisted["selection"]
    with pytest.raises(FileExistsError):
        training_startup.write_training_summary(summary, args.output)
    for name, content in saved.items():
        assert (args.output / name).read_bytes() == content


def test_summary_does_not_record_unrequested_environment_or_arguments(
    summary_inputs, monkeypatch
):
    config, dataset, args = summary_inputs
    marker = "synthetic-private-marker-no-real-credential"
    monkeypatch.setenv("JEPA_TEST_PRIVATE_ENV", marker)
    args.private_environment = marker
    summary = training_startup.build_training_summary(config, dataset, args)
    serialized = json.dumps(summary)
    assert marker not in serialized
    assert "JEPA_TEST_PRIVATE_ENV" not in serialized
    assert "private_environment" not in summary["arguments"]


def test_runtime_info_does_not_collect_environment(monkeypatch):
    marker = "synthetic-runtime-marker-no-real-credential"
    monkeypatch.setenv("JEPA_TEST_RUNTIME_ENV", marker)
    monkeypatch.setattr(
        training_startup.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(name="CPU test", total_memory=0),
    )
    runtime = training_startup.collect_runtime_info("cuda:0")
    assert runtime["gpu"] == "CPU test"
    assert marker not in json.dumps(runtime)
    assert "JEPA_TEST_RUNTIME_ENV" not in runtime


def run_logging_probe(body: str, directory: Path):
    repo = Path(__file__).resolve().parents[1]
    script = directory / "logging_probe.py"
    prefix = [
        "import sys",
        "from pathlib import Path",
        f"sys.path.insert(0, {str(repo)!r})",
        "from loguru import logger",
        "from jepa_policy.training_logging import "
        "setup_training_logging, log_section, run_logged_training, log_training_progress",
        f"setup_training_logging(Path({str(directory)!r}))",
    ]
    script.write_text("\n".join(prefix) + "\n" + dedent(body))
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_model_info_is_file_only_but_sections_and_warnings_reach_console(
    tmp_path: Path,
):
    result = run_logging_probe(
        """
        logger.info("model-initialization-info-marker")
        log_section("数据筛选", ["selection-section-marker"])
        logger.warning("warning-visible-marker")
        logger.error("error-visible-marker")
        logger.complete()
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    saved = (tmp_path / "training.log").read_text()
    assert "model-initialization-info-marker" in saved
    assert "model-initialization-info-marker" not in result.stdout
    for marker in (
        "数据筛选",
        "selection-section-marker",
        "warning-visible-marker",
        "error-visible-marker",
    ):
        assert marker in result.stdout
        assert marker in saved
    assert not re.search(r"\| [\w.]+:\w+:\d+ -", result.stdout)
    assert not re.search(r"\| [\w.]+:\w+:\d+ -", saved)


def test_training_failure_logs_stack_and_exits_without_diagnostic_values(
    tmp_path: Path,
):
    result = run_logging_probe(
        """
        def fail():
            diagnostic_value = "synthetic-diagnostic-value-must-stay-hidden"
            raise (
                ValueError("intentional-startup-test-failure")
                if diagnostic_value
                else RuntimeError("unreachable")
            )

        run_logged_training(fail)
        """,
        tmp_path,
    )
    assert result.returncode == 1
    saved = (tmp_path / "training.log").read_text()
    for content in (result.stdout, saved):
        assert "训练失败" in content
        assert "Traceback (most recent call last)" in content
        assert "ValueError: intentional-startup-test-failure" in content
        assert "synthetic-diagnostic-value-must-stay-hidden" not in content
        assert "SystemExit" not in content
        assert not re.search(r"\| [\w.]+:\w+:\d+ -", content)


def test_training_progress_is_one_line_and_keeps_detailed_record(tmp_path: Path):
    result = run_logging_probe(
        """
        record = {
            "step": 190,
            "loss": 1.009515,
            "data_wait_seconds": 0.0003,
            "update_seconds": 0.0665,
            "gpu_peak_memory_gib": 1.73,
            "loss_action": 0.917741,
            "loss_future_weighted": 0.091774,
            "loss_future_raw": 4.817450,
        }
        before = record.copy()
        log_training_progress(record, 0.067)
        assert record == before
        logger.complete()
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1
    assert "step=190" in lines[0]
    assert "loss=1.009515" in lines[0]
    assert "elapsed=0.07s" in lines[0]
    assert "gpu_peak=1.7GiB" in lines[0]
    assert "梯度范数" not in lines[0]
    assert "未来原始" not in lines[0]
    assert (tmp_path / "training.log").read_text().strip() == lines[0]


def test_saved_statistics_include_selected_short_episodes(audit_dataset_root: Path):
    dataset = IcedTeaDataset(
        IcedTeaDatasetConfig(audit_dataset_root, episode_indices=(0, 2))
    )
    output = audit_dataset_root / "saved_run"
    prepare_run_directory(output, dataset, None)
    metadata = json.loads((output / "dataset_metadata.json").read_text())
    statistics = json.loads((output / "normalization_stats.json").read_text())
    assert metadata["episode_indices"] == [0, 2]
    assert metadata["window_episode_indices"] == [0]
    assert statistics["episode_indices"] == [0, 2]
    np.testing.assert_array_equal(
        statistics["features"]["action"]["range"], np.full(8, 203.0)
    )


def test_failure_before_log_setup_never_renders_local_arguments(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "early_failure.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from types import SimpleNamespace\n"
        "from jepa_policy.training_logging import run_logged_training\n"
        "def fail():\n"
        "    args = SimpleNamespace(output='existing-output', "
        "private_value='synthetic-private-marker')\n"
        "    raise FileExistsError(args.output)\n"
        "run_logged_training(fail)\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "FileExistsError: existing-output" in output
    assert "synthetic-private-marker" not in output
