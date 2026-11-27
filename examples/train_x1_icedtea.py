"""Train JEPA Policy on the X1 iced-tea task with one GPU."""

from __future__ import annotations

import argparse
import faulthandler
import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rolling_cache"))

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler

from jepa_policy.agent import TrainingAgent
from jepa_policy.config import (
    Config,
    EvalConfig,
    LogConfig,
    NetworkConfig,
    OptimizationConfig,
    TaskConfig,
)
from jepa_policy.datasets.icedtea_cache import CachedIcedTeaDataset, build_icedtea_cache
from jepa_policy.datasets.icedtea_dataset import (
    ACTION_NAMES,
    CAMERAS,
    IcedTeaDataset,
    IcedTeaDatasetConfig,
    initialize_icedtea_worker,
)
from jepa_policy.torch_utils import set_seed
from jepa_policy.training_logging import (
    CacheLogStream,
    log_section,
    log_training_progress,
    run_logged_training,
    save_training_checkpoint,
    setup_training_logging,
)
from jepa_policy.training_startup import (
    build_training_summary,
    collect_normalization_info,
    record_cache_info,
    record_first_batch,
    record_model_info,
    write_training_summary,
)

DATASET_PATH = (
    "/mnt/data/EIC01/vla/x1_pro/train/waterbar_2task_228eps_20261126_smoothed"
)
OUTPUT_PATH = Path("./outputs")


@dataclass(frozen=True)
class TrainingRunOptions:
    output_path: Path
    save_every: int
    resume_checkpoint: Path | None
    log_every: int = 10


class ContinuousShuffleBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset_size: int, batch_size: int, total_batches: int):
        if dataset_size < batch_size or batch_size <= 0 or total_batches <= 0:
            raise ValueError(
                "dataset_size must fit one batch; batch_size and total_batches must be positive"
            )
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.total_batches = total_batches
        self.batches_per_epoch = dataset_size // batch_size

    def __iter__(self):
        yielded = 0
        while yielded < self.total_batches:
            indices = torch.randperm(self.dataset_size).tolist()
            for batch_index in range(self.batches_per_epoch):
                start = batch_index * self.batch_size
                yield indices[start : start + self.batch_size]
                yielded += 1
                if yielded == self.total_batches:
                    return

    def __len__(self) -> int:
        return self.total_batches


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH
        / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--episode-indices", type=Path)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("/tmp/jepa_icedtea_cache")
    )
    parser.add_argument("--cache-build-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def create_optimization_config(
    steps: int, batch_size: int, workers: int
) -> OptimizationConfig:
    return OptimizationConfig(
        loss_type="mip",
        loss_scale=1.0,
        norm_type="l1",
        lr=1e-4,
        weight_decay=1e-5,
        num_steps=1,
        sample_mode="zero",
        t_two_step=0.9,
        future_t_two_step=0.9,
        ema_rate=0.995,
        batch_size=batch_size,
        dataloader_num_workers=workers,
        dataloader_persistent_workers=workers > 0,
        gradient_steps=steps,
        warmup_ratio=0.0,
        rampup_ratio=0.5,
        device="cuda:0",
        use_compile=False,
        auto_resume=False,
        use_future_embed_loss=True,
        future_embed_loss_mode="mip_two_step",
        future_joint_mode=True,
        future_state_loss_mode="ratio",
        future_state_loss_ratio=0.1,
        future_state_loss_weight_min=1e-6,
        future_state_loss_weight_max=0.1,
        future_target_type="embedding",
    )


def create_network_config() -> NetworkConfig:
    return NetworkConfig(
        network_type="chitransformer",
        emb_dim=384,
        num_layers=8,
        dropout=0.0,
        n_heads=6,
        attn_dropout=0.1,
        n_future_tokens=1,
        rgb_model_name="resnet18",
        imagenet_norm=False,
        use_seq=True,
        keep_horizon_dims=True,
    )


def create_task_config(dataset_path: Path) -> TaskConfig:
    observation_shapes = {
        key: {"shape": [3, 128, 128], "type": "rgb"} for key in CAMERAS
    }
    observation_shapes["agent_pos"] = {"shape": [8], "type": "low_dim"}
    return TaskConfig(
        env_name="x1_pro_icedtea_right_arm",
        obs_type="image",
        env_type="real",
        abs_action=True,
        dataset_path=str(dataset_path),
        obs_dim=384,
        act_dim=len(ACTION_NAMES),
        obs_steps=2,
        act_steps=8,
        horizon=10,
        shape_meta={
            "action": {"shape": [len(ACTION_NAMES)]},
            "obs": observation_shapes,
        },
        resize_shape=[128, 128],
        crop_shape=[112, 112],
        random_crop=True,
        crop_mode="temporal_consistent",
        temporal_consistent_crop=True,
        use_group_norm=True,
        use_seq=True,
        future_state_enabled=True,
        future_state_steps=4,
        future_state_steps_list=[4],
        future_target_type="embedding",
    )


def create_log_config(output_path: Path) -> LogConfig:
    return LogConfig(
        log_dir=str(output_path),
        model_dir=None,
        wandb_mode="disabled",
        project="jepa-policy",
        group="x1-icedtea-right-arm",
        exp_name="x1-icedtea-right-arm",
        entity=None,
        eval_freq=0,
        log_freq=10,
        save_freq=0,
        eval_episodes=0,
        save_video=False,
        gradient_diagnostic_freq=0,
        validation_freq=0,
    )


def create_config(
    dataset_path: Path, output_path: Path, optimization: OptimizationConfig
) -> Config:
    return Config(
        optimization=optimization,
        network=create_network_config(),
        task=create_task_config(dataset_path),
        log=create_log_config(output_path),
        eval=EvalConfig(),
        mode="train",
    )


def move_to_device(value, device: torch.device):
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    value = value.to(device=device, non_blocking=True)
    if value.dtype == torch.uint8:
        return value.float().div_(255).mul_(2).sub_(1)
    return value.float()


def create_dataloader(dataset: IcedTeaDataset, config: Config) -> DataLoader:
    workers = config.optimization.dataloader_num_workers
    batch_sampler = ContinuousShuffleBatchSampler(
        len(dataset),
        config.optimization.batch_size,
        config.optimization.gradient_steps,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=initialize_icedtea_worker if workers > 0 else None,
        multiprocessing_context="spawn" if workers > 0 else None,
        prefetch_factor=1 if workers > 0 else None,
    )
    if len(loader) == 0:
        raise ValueError("Dataset has fewer samples than the batch size")
    return loader


def restore_training_step(agent, config: Config, checkpoint: Path | None) -> int:
    if checkpoint is None:
        return 0
    state = agent.load(str(checkpoint), load_optimizer=True)
    if not state or "n_gradient_step" not in state:
        raise ValueError(f"Checkpoint has no training step: {checkpoint}")
    start_step = int(state["n_gradient_step"]) + 1
    if start_step >= config.optimization.gradient_steps:
        raise ValueError(
            f"Checkpoint step {start_step} already meets requested total steps "
            f"{config.optimization.gradient_steps}"
        )
    return start_step


def prepare_run_directory(
    output_path: Path, dataset: IcedTeaDataset, resume_checkpoint: Path | None
) -> tuple[Path, Path]:
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


def next_batch(iterator, loader: DataLoader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def update_agent(agent, config: Config, batch: dict) -> dict:
    device = torch.device(config.optimization.device)
    observations = move_to_device(batch["obs"], device)
    future = move_to_device(batch["future_obs"], device)
    actions = batch["action"].to(device, non_blocking=True)
    delta = torch.full((actions.shape[0],), 1.0, device=device)
    return agent.update(
        actions, observations, delta, future_obs=future, sync_metrics=True
    )


def log_training_step(
    metrics_path: Path, metrics: dict, step: int, learning_rate: float
) -> dict:
    record = {
        key: value.item() if torch.is_tensor(value) else value
        for key, value in metrics.items()
    }
    record.update(step=step + 1, learning_rate=learning_rate)
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=True) + "\n")
    return record


def train(config: Config, dataset: IcedTeaDataset, options: TrainingRunOptions) -> None:
    loader = create_dataloader(dataset, config)
    agent = TrainingAgent(config)
    start_step = restore_training_step(agent, config, options.resume_checkpoint)
    scheduler = CosineAnnealingLR(
        agent.optimizer,
        T_max=config.optimization.gradient_steps,
        last_epoch=start_step - 1,
    )
    model_dir, metrics_path = prepare_run_directory(
        options.output_path, dataset, options.resume_checkpoint
    )
    record_model_info(agent, options.output_path, start_step)
    log_section("数据加载器启动", ["正在初始化工作进程并预取首批数据"])
    iterator = iter(loader)
    agent.train()
    for step in range(start_step, config.optimization.gradient_steps):
        step_started = time.perf_counter()
        data_started = step_started
        batch, iterator = next_batch(iterator, loader)
        data_wait = time.perf_counter() - data_started
        if step == start_step:
            record_first_batch(batch, options.output_path)
        update_started = time.perf_counter()
        metrics = update_agent(agent, config, batch)
        update_time = time.perf_counter() - update_started
        metrics["data_wait_seconds"] = data_wait
        metrics["update_seconds"] = update_time
        metrics["gpu_peak_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        scheduler.step()
        record = log_training_step(
            metrics_path, metrics, step, scheduler.get_last_lr()[0]
        )
        if (step + 1) % options.log_every == 0 or step == start_step:
            log_training_progress(record, time.perf_counter() - step_started)
        if options.save_every > 0 and (step + 1) % options.save_every == 0:
            save_training_checkpoint(agent, model_dir, step)
    final_step = config.optimization.gradient_steps - 1
    checkpoint_path = save_training_checkpoint(agent, model_dir, final_step)
    log_section("训练完成", [f"最终权重={checkpoint_path}"])


def load_training_dataset(args, config: Config) -> IcedTeaDataset:
    return IcedTeaDataset(
        IcedTeaDatasetConfig(
            args.dataset,
            obs_steps=config.task.obs_steps,
            horizon=config.task.horizon,
            future_steps=config.task.future_state_steps,
            episode_indices=(
                tuple(json.loads(args.episode_indices.read_text()))
                if args.episode_indices is not None
                else None
            ),
        )
    )


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = parse_arguments()
    setup_training_logging(args.output)
    if (
        args.steps <= 0
        or args.batch_size <= 0
        or args.workers < 0
        or args.save_every < 0
        or args.cache_build_workers <= 0
        or args.log_every <= 0
    ):
        raise ValueError(
            "steps/batch-size/log-every must be positive; workers/save-every nonnegative"
        )
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("Training requires one available CUDA device at cuda:0")
    # 固定随机种子，建立统一日志，再核对训练数据与配置。
    log_section(
        "读取训练数据", [f"数据集={args.dataset}", f"索引文件={args.episode_indices}"]
    )
    set_seed(0)
    optimization = create_optimization_config(args.steps, args.batch_size, args.workers)
    config = create_config(args.dataset, args.output, optimization)
    config.optimization.device = args.device
    dataset = load_training_dataset(args, config)
    write_training_summary(build_training_summary(config, dataset, args), args.output)
    # 读取或构建与回合索引一致的缓存，并记录实际缓存身份。
    log_section(
        "缓存准备",
        [
            f"根目录={args.cache_root} | 构建进程={args.cache_build_workers}",
            "先校验缓存样本映射，再决定复用或构建",
        ],
    )
    cache_started = time.perf_counter()
    with redirect_stdout(CacheLogStream()):
        cache_reader = build_icedtea_cache(
            dataset, args.cache_root, cache_build_workers=args.cache_build_workers
        )
    record_cache_info(cache_reader, args.output, time.perf_counter() - cache_started)
    cached_dataset = CachedIcedTeaDataset(dataset, cache_reader)
    # 使用固定训练计划更新模型，按配置间隔记录进度与权重。
    options = TrainingRunOptions(
        args.output, args.save_every, args.resume, args.log_every
    )
    train(config, cached_dataset, options)


if __name__ == "__main__":
    run_logged_training(main)
