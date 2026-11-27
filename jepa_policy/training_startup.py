"""Build and persist an auditable startup description of one training run."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import torch

from jepa_policy.datasets.icedtea_dataset import ACTION_NAMES, CAMERAS
from jepa_policy.datasets.icedtea_selection_audit import (
    build_selection_audit,
    format_selection_audit,
)
from jepa_policy.training_logging import describe_batch_tensors, log_section


def collect_runtime_info(device: str) -> dict:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "opencv": cv2.__version__,
        "device": device,
        "gpu": properties.name,
        "gpu_memory_gib": properties.total_memory / 1024**3,
        "cpu_available": len(os.sched_getaffinity(0)),
        "torch_cpu_threads": torch.get_num_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def collect_code_info(root: Path) -> dict:
    def read_git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    source_files = (
        "examples/train_x1_icedtea.py",
        "jepa_policy/config.py",
        "jepa_policy/agent.py",
        "jepa_policy/samplers.py",
        "jepa_policy/datasets/icedtea_dataset.py",
        "jepa_policy/datasets/icedtea_cache.py",
        "jepa_policy/datasets/icedtea_selection_audit.py",
        "jepa_policy/training_startup.py",
        "jepa_policy/training_logging.py",
    )
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in source_files
    }
    return {
        "source_files_sha256": hashes,
        "root": str(root),
        "commit": read_git("rev-parse", "HEAD"),
        "branch": read_git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_dirty": bool(read_git("status", "--porcelain")),
    }


def collect_normalization_info(dataset, output: Path) -> dict:
    features = {}
    for name, minimum, scale in [
        (
            "action",
            dataset.action_min.astype(float),
            dataset.action_range.astype(float),
        ),
        (
            "observation.state",
            dataset.state_min.astype(float),
            dataset.state_range.astype(float),
        ),
    ]:
        features[name] = {
            "names": ACTION_NAMES,
            "min": minimum.tolist(),
            "max": (minimum + scale).tolist(),
            "range": scale.tolist(),
        }
    return {
        "source": "selected_training_episode_frames_only",
        "file": str(output / "normalization_stats.json"),
        "formula": "2 * (raw - min) / range - 1",
        "range_minimum": 1e-6,
        "clip_state": False,
        "features": features,
    }


def build_training_summary(config, dataset, args) -> dict:
    output = Path(args.output).resolve()
    batch_size = config.optimization.batch_size
    argument_names = (
        "dataset",
        "output",
        "steps",
        "batch_size",
        "workers",
        "save_every",
        "resume",
        "episode_indices",
        "cache_root",
        "cache_build_workers",
        "device",
        "log_every",
    )
    arguments = {
        name: str(value) if isinstance(value, Path) else value
        for name in argument_names
        if (value := vars(args)[name]) is not None
    }
    return {
        "schema_version": 1,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "output": str(output),
        "arguments": arguments,
        "code": collect_code_info(Path(__file__).resolve().parents[1]),
        "runtime": collect_runtime_info(config.optimization.device),
        "config": asdict(config),
        "selection": build_selection_audit(dataset),
        "normalization": collect_normalization_info(dataset, output),
        "dataloader": {
            "batch_size": batch_size,
            "workers": args.workers,
            "persistent_workers": args.workers > 0,
            "batches_per_shuffle_cycle": len(dataset) // batch_size,
            "unused_windows_per_shuffle_cycle": len(dataset) % batch_size,
            "drop_last_per_cycle": True,
            "shuffle_each_cycle": True,
            "continuous_iterator": True,
            "prefetch_factor": 1 if args.workers > 0 else None,
            "pin_memory": True,
            "multiprocessing_context": "spawn" if args.workers > 0 else None,
        },
        "schedule": {
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "scheduler_total_steps": config.optimization.gradient_steps,
            "scheduler_eta_min": 0.0,
            "scheduler_warmup_steps": 0,
            "resume": str(args.resume) if args.resume is not None else None,
        },
    }


def format_normalization(summary: dict) -> list[str]:
    lines = [
        "来源=实际选中训练回合的全部帧，不按文本筛选；评估复用本文件",
        f"文件={summary['normalization']['file']}",
        "公式=2*(原始值-min)/range-1；最小range=1e-6；状态不截断",
    ]
    for feature, statistics in summary["normalization"]["features"].items():
        lines.append(
            f"{feature}: 名称                  min           max         range"
        )
        for name, minimum, maximum, scale in zip(
            statistics["names"],
            statistics["min"],
            statistics["max"],
            statistics["range"],
            strict=True,
        ):
            lines.append(
                f"  {name:<20} {minimum:>12.6g} {maximum:>12.6g} {scale:>12.6g}"
            )
    return lines


def format_hyperparameters(summary: dict) -> dict[str, list[str]]:
    config = summary["config"]
    optimization, network, task = (
        config["optimization"],
        config["network"],
        config["task"],
    )
    return {
        "超参数 · 训练与优化": [
            f"总更新={optimization['gradient_steps']:,} 步 | "
            f"batch={optimization['batch_size']} | seed={optimization['seed']}",
            f"AdamW: lr={optimization['lr']} | weight_decay={optimization['weight_decay']} | "
            "betas=(0.9,0.999) | eps=1e-8",
            f"CosineAnnealingLR: T_max={optimization['gradient_steps']} | "
            "eta_min=0 | 学习率预热=0 步",
            f"梯度裁剪范数={optimization['grad_clip_norm']} | "
            f"EMA={optimization['ema_rate']} | compile={optimization['use_compile']}",
            f"权重保存间隔={summary['arguments']['save_every']} 步 | "
            f"控制台间隔={summary['arguments']['log_every']} 步 | metrics逐步保存",
            "AMP=关闭；参数实际dtype见模型就绪日志",
            f"续训={summary['schedule']['resume']}",
        ],
        "超参数 · 损失与采样": [
            f"loss={optimization['loss_type']} | norm={optimization['norm_type']} | "
            f"loss_scale={optimization['loss_scale']}",
            f"联合未来预测={optimization['future_joint_mode']} | "
            f"未来目标={optimization['future_target_type']} | "
            f"模式={optimization['future_embed_loss_mode']}",
            f"未来权重模式={optimization['future_state_loss_mode']} | "
            f"目标损失比={optimization['future_state_loss_ratio']} | "
            f"权重上下限=[{optimization['future_state_loss_weight_min']}, "
            f"{optimization['future_state_loss_weight_max']}]",
            f"插值={optimization['interp_type']} | 初始动作={optimization['sample_mode']} | "
            f"动作两步分界={optimization['t_two_step']} | "
            f"未来两步分界={optimization['future_t_two_step']}",
            "联合MIP采样执行两次前向；num_steps=1不代表仅一次前向",
        ],
        "超参数 · 模型与窗口": [
            f"网络={network['network_type']} | hidden={network['emb_dim']} | "
            f"layers={network['num_layers']} | heads={network['n_heads']}",
            f"dropout={network['dropout']} | attention_dropout={network['attn_dropout']} | "
            f"future_tokens={network['n_future_tokens']}",
            f"视觉骨干={network['rgb_model_name']} | "
            f"预训练权重={network['rgb_model_weights']} | group_norm={task['use_group_norm']}",
            f"动作维度={task['act_dim']} | 绝对动作={task['abs_action']} | "
            f"观测帧={task['obs_steps']} | horizon={task['horizon']} | "
            f"执行帧={task['act_steps']} | 未来偏移={task['future_state_steps']} 帧",
        ],
    }


def format_observation_info(summary: dict) -> list[str]:
    task = summary["config"]["task"]
    return [
        *[f"{name} <- {source}" for name, source in CAMERAS.items()],
        f"图像: RGB uint8 -> {task['resize_shape']} 线性缩放 -> "
        f"设备转换/float32 [-1,1] -> 编码器内 {task['crop_shape']} 裁剪",
        f"训练裁剪={task['crop_mode']} | 评估裁剪={task['eval_crop_mode']}",
        f"ImageNet归一化={summary['config']['network']['imagenet_norm']}；"
        "不根据测试集计算图像均值方差",
        f"状态和动作顺序={', '.join(ACTION_NAMES)}",
        f"窗口示例起点s: 观测[s,s+{task['obs_steps'] - 1}]；"
        f"动作[s,s+{task['horizon'] - 1}]；"
        f"未来观测s+{task['obs_steps'] - 1 + task['future_state_steps']}",
        "视频解码=PyAV；取帧规则=首个不早于请求时间的解码帧",
        "完整配置见 training_config.json，完整回合筛选见 dataset_selection.json",
    ]


def write_training_summary(summary: dict, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    documents = {
        "training_summary.json": summary,
        "training_config.json": summary["config"],
        "dataset_selection.json": summary["selection"],
    }
    for filename in documents:
        if (output_path / filename).exists():
            raise FileExistsError(output_path / filename)
    for filename, document in documents.items():
        with (output_path / filename).open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    run_lines = [
        f"启动时间={summary['started_at']} | 输出目录={summary['output']}",
        f"分支={summary['code']['branch']} | commit={summary['code']['commit']} | "
        f"工作区有修改={summary['code']['working_tree_dirty']}",
        f"终端与模型日志={output_path / 'training.log'}",
    ]
    log_section("运行信息", run_lines)
    log_section(
        "设备与环境", [f"{name}={value}" for name, value in summary["runtime"].items()]
    )
    log_section("数据筛选", format_selection_audit(summary["selection"])[1:])
    log_section("归一化统计", format_normalization(summary))
    for title, lines in format_hyperparameters(summary).items():
        log_section(title, lines)
    log_section("输入预处理", format_observation_info(summary))
    log_section(
        "数据加载器",
        [f"{name}={value}" for name, value in summary["dataloader"].items()],
    )


def record_run_details(output_path: Path, section: str, details: dict) -> None:
    path = output_path / "training_summary.json"
    summary = json.loads(path.read_text())
    summary[section] = details
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def record_cache_info(reader, output: Path, elapsed: float) -> None:
    manifest_path = reader.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    details = {
        "path": str(reader.cache_dir),
        "num_windows": reader.num_frames,
        "shards": len(manifest["shard_ranges"]),
        "mapping_digest": manifest["sample_mapping_digest"],
        "preprocess_version": manifest["preprocess_version"],
        "estimated_gib": manifest["estimated_bytes"] / 1024**3,
        "prepare_seconds": elapsed,
    }
    record_run_details(output, "cache", details)
    log_section("缓存就绪", [f"{key}={value}" for key, value in details.items()])


def record_model_info(agent, output: Path, start_step: int) -> None:
    details = {}
    for name in ("encoder", "flow_map"):
        model = getattr(agent, name)
        details[name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "dtype": str(next(model.parameters()).dtype),
        }
    details["optimizer"] = {
        key: agent.optimizer.param_groups[0][key]
        for key in ("lr", "weight_decay", "betas", "eps", "amsgrad")
    }
    details["start_step"] = start_step
    statistics_file = output / "normalization_stats.json"
    details["normalization_sha256"] = hashlib.sha256(
        statistics_file.read_bytes()
    ).hexdigest()
    record_run_details(output, "loaded_model", details)
    log_section("模型就绪", [f"{name}={value}" for name, value in details.items()])


def record_first_batch(batch: dict, output: Path) -> None:
    entries = describe_batch_tensors(batch)
    record_run_details(output, "first_batch", {"tensors": entries})
    log_section(
        "首批实际张量 · 设备转换前",
        [
            f"{entry['name']}: shape={entry['shape']} dtype={entry['dtype']} "
            f"device={entry['device']} range=[{entry['min']:.6g}, {entry['max']:.6g}]"
            for entry in entries
        ],
    )
