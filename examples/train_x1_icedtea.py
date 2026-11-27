"""Train JEPA Policy on the X1 iced-tea task with one GPU."""

from __future__ import annotations

import faulthandler
import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

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
from jepa_policy.datasets.icedtea_split import (
    EpisodeSplitOptions,
    prepare_episode_split,
)
from jepa_policy.icedtea_training_arguments import parse_arguments, validate_arguments
from jepa_policy.mlflow_tracking import MlflowOptions
from jepa_policy.torch_utils import set_seed
from jepa_policy.training_artifacts import prepare_run_directory, validate_run_output
from jepa_policy.training_logging import (
    CacheLogStream,
    log_section,
    log_training_progress,
    run_logged_training,
    setup_training_logging,
)
from jepa_policy.training_monitor import TrainingMonitor, TrainingMonitoringOptions
from jepa_policy.training_startup import (
    build_training_summary,
    record_cache_info,
    record_first_batch,
    record_model_info,
    write_training_summary,
)


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


def train(
    config: Config, dataset, options: TrainingRunOptions, monitor: TrainingMonitor
) -> None:
    loader = create_dataloader(dataset, config)
    agent = TrainingAgent(config)
    start_step = restore_training_step(agent, config, options.resume_checkpoint)
    scheduler = CosineAnnealingLR(
        agent.optimizer,
        T_max=config.optimization.gradient_steps,
        last_epoch=start_step - 1,
    )
    _, metrics_path = prepare_run_directory(
        options.output_path, dataset, options.resume_checkpoint
    )
    record_model_info(agent, options.output_path, start_step)
    log_section("数据加载器启动", ["正在初始化工作进程并预取首批数据"])
    iterator = iter(loader)
    agent.train()
    for step in range(start_step, config.optimization.gradient_steps):
        step_started = time.perf_counter()
        data_started = step_started
        batch = next(iterator)
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
        save_checkpoint = (
            options.save_every > 0 and (step + 1) % options.save_every == 0
        ) or step + 1 == config.optimization.gradient_steps
        if (step + 1) % options.log_every == 0 or step == start_step or save_checkpoint:
            monitor.log_training(record)
        if save_checkpoint:
            checkpoint_path = monitor.checkpoint(agent, step)
    log_section("训练完成", [f"最终权重={checkpoint_path}"])


def load_training_datasets(
    args, config: Config
) -> tuple[IcedTeaDataset, IcedTeaDataset]:
    split = prepare_episode_split(
        args.dataset,
        EpisodeSplitOptions(
            args.output, args.episode_indices, args.resume, args.split_seed
        ),
    )
    training = IcedTeaDataset(
        IcedTeaDatasetConfig(
            args.dataset,
            obs_steps=config.task.obs_steps,
            horizon=config.task.horizon,
            future_steps=config.task.future_state_steps,
            episode_indices=split.train,
        )
    )
    evaluation = IcedTeaDataset(
        IcedTeaDatasetConfig(
            args.dataset,
            obs_steps=config.task.obs_steps,
            horizon=config.task.horizon,
            future_steps=config.task.future_state_steps,
            normalize=False,
            episode_indices=split.evaluation,
        )
    )
    log_section(
        "完整回合划分",
        [
            f"候选={len(split.candidates)} | 训练={len(split.train)} | 评估={len(split.evaluation)}",
            f"训练窗口={len(training)} | 评估窗口={len(evaluation)} | 划分种子={split.seed}",
            "标签不参与筛选；训练与评估回合互斥；归一化只使用训练回合统计量",
            f"索引与划分信息={args.output / 'episode_split.json'}",
        ],
    )
    return training, evaluation


def monitoring_options(args) -> TrainingMonitoringOptions:
    return TrainingMonitoringOptions(
        output=args.output,
        mlflow=MlflowOptions(
            args.mlflow,
            args.mlflow_tracking_uri,
            args.mlflow_experiment,
            args.mlflow_run_name,
        ),
        evaluation_batch_size=args.eval_batch_size,
        evaluation_chunk_size=args.eval_chunk_size,
        evaluation_seed=args.eval_seed,
        resume_checkpoint=args.resume,
        save_every=args.save_every,
        split_seed=args.split_seed,
    )


def load_cached_dataset(dataset, args) -> CachedIcedTeaDataset:
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
    return CachedIcedTeaDataset(dataset, cache_reader)


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = parse_arguments()
    validate_run_output(args.output)
    setup_training_logging(args.output)
    validate_arguments(args)
    # 固定随机种子，建立日志并按完整回合划分训练和评估数据。
    log_section(
        "读取训练数据",
        [f"数据集={args.dataset}", f"候选索引文件={args.episode_indices}"],
    )
    set_seed(0)
    optimization = create_optimization_config(args.steps, args.batch_size, args.workers)
    config = create_config(args.dataset, args.output, optimization)
    config.optimization.device = args.device
    dataset, evaluation_dataset = load_training_datasets(args, config)
    write_training_summary(build_training_summary(config, dataset, args), args.output)
    # 先校验 MLflow，再构建训练缓存，避免认证失败后浪费解码时间。
    options = TrainingRunOptions(
        args.output, args.save_every, args.resume, args.log_every
    )
    with TrainingMonitor(
        config, (dataset, evaluation_dataset), monitoring_options(args)
    ) as monitor:
        cached_dataset = load_cached_dataset(dataset, args)
        # 更新模型；每个权重保存后完成影子评估，再继续后续训练。
        train(config, cached_dataset, options, monitor)


if __name__ == "__main__":
    run_logged_training(main)
