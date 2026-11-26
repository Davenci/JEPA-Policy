"""Train JEPA Policy on the X1 iced-tea task with one GPU."""

from __future__ import annotations

import argparse
import faulthandler
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from jepa_policy.agent import TrainingAgent
from jepa_policy.config import (
    Config,
    EvalConfig,
    LogConfig,
    NetworkConfig,
    OptimizationConfig,
    TaskConfig,
)
from jepa_policy.datasets.icedtea_dataset import (
    ACTION_NAMES,
    CAMERAS,
    IcedTeaDataset,
    IcedTeaDatasetConfig,
    initialize_icedtea_worker,
)
from jepa_policy.torch_utils import set_seed

DATASET_PATH = (
    "/home/x1/work/X1_Pro_data/waterbar_datasets/train/merged/"
    "CeMAT_12task_2160eps_20261024_smoothed"
)
OUTPUT_PATH = "./outputs/202611261556"


@dataclass(frozen=True)
class TrainingRunOptions:
    output_path: Path
    save_every: int
    resume_checkpoint: Path | None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--resume", type=Path)
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
    loader = DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=initialize_icedtea_worker if workers > 0 else None,
        multiprocessing_context="spawn" if workers > 0 else None,
        prefetch_factor=1 if workers > 0 else None,
        drop_last=True,
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
        "task": dataset.task_name,
        "action_names": ACTION_NAMES,
        "action_min": dataset.action_min.tolist(),
        "action_range": dataset.action_range.tolist(),
        "state_min": dataset.state_min.tolist(),
        "state_range": dataset.state_range.tolist(),
        "dataset": str(dataset.root),
    }
    (output_path / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
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
    iterator = iter(loader)
    agent.train()
    for step in range(start_step, config.optimization.gradient_steps):
        started = time.perf_counter()
        data_started = time.perf_counter()
        batch, iterator = next_batch(iterator, loader)
        data_wait = time.perf_counter() - data_started
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
        if (step + 1) % 10 == 0 or step == start_step:
            elapsed = time.perf_counter() - started
            print(
                f"step={step + 1} loss={record['loss']:.6f} "
                f"data={data_wait:.2f}s update={update_time:.2f}s "
                f"elapsed={elapsed:.2f}s gpu_peak={record['gpu_peak_memory_gib']:.1f}GiB",
                flush=True,
            )
        if options.save_every > 0 and (step + 1) % options.save_every == 0:
            agent.save(
                model_dir / "model_latest.pt", training_state={"n_gradient_step": step}
            )
    final_step = config.optimization.gradient_steps - 1
    agent.save(
        model_dir / "model_latest.pt", training_state={"n_gradient_step": final_step}
    )
    print(f"Training complete; checkpoint: {model_dir / 'model_latest.pt'}", flush=True)


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = parse_arguments()
    if (
        args.steps <= 0
        or args.batch_size <= 0
        or args.workers < 0
        or args.save_every < 0
    ):
        raise ValueError(
            "steps/batch-size must be positive; workers/save-every nonnegative"
        )
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("Training requires one available CUDA device at cuda:0")
    set_seed(0)
    optimization = create_optimization_config(args.steps, args.batch_size, args.workers)
    config = create_config(args.dataset, args.output, optimization)
    config.optimization.device = args.device
    dataset = IcedTeaDataset(
        IcedTeaDatasetConfig(
            args.dataset,
            obs_steps=config.task.obs_steps,
            horizon=config.task.horizon,
            future_steps=config.task.future_state_steps,
        )
    )
    print(
        f"Task={dataset.root} / iced tea; samples={len(dataset)}; actions={ACTION_NAMES}",
        flush=True,
    )
    options = TrainingRunOptions(args.output, args.save_every, args.resume)
    train(config, dataset, options)


if __name__ == "__main__":
    main()
