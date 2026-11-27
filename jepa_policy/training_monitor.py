"""Evaluate every saved checkpoint and keep training and holdout histories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jepa_policy.checkpoint_evaluation import (
    CheckpointEvaluationOptions,
    CheckpointEvaluator,
    max_plot_episodes,
)
from jepa_policy.mlflow_tracking import MlflowOptions, TrainingTracker
from jepa_policy.training_logging import log_section, save_training_checkpoint


@dataclass(frozen=True)
class TrainingMonitoringOptions:
    output: Path
    mlflow: MlflowOptions
    evaluation_batch_size: int = 64
    evaluation_chunk_size: int = 4
    evaluation_seed: int = 12345
    resume_checkpoint: Path | None = None
    save_every: int = 5000
    split_seed: int = 0


def tracking_parameters(
    config, datasets: tuple, options: TrainingMonitoringOptions
) -> dict:
    train_dataset, evaluation_dataset = datasets
    optimization = config.optimization
    parameters = {
        "gradient_steps": optimization.gradient_steps,
        "batch_size": optimization.batch_size,
        "learning_rate": optimization.lr,
        "weight_decay": optimization.weight_decay,
        "loss_type": optimization.loss_type,
        "ema_rate": optimization.ema_rate,
        "train_seed": optimization.seed,
        "train_episodes": len(train_dataset.episodes),
        "eval_episodes": len(evaluation_dataset.episodes),
        "train_windows": len(train_dataset),
        "eval_windows": len(evaluation_dataset),
        "eval_fraction": 0.05,
        "eval_weights": "checkpoint_ema",
        "eval_batch_size": options.evaluation_batch_size,
        "eval_chunk_size": options.evaluation_chunk_size,
        "eval_seed": options.evaluation_seed,
        "horizon": config.task.horizon,
        "observation_steps": config.task.obs_steps,
        "action_dim": config.task.act_dim,
        "future_steps": config.task.future_state_steps,
        "future_loss_ratio": optimization.future_state_loss_ratio,
        "save_every": options.save_every,
        "split_seed": options.split_seed,
        "gradient_clip_norm": optimization.grad_clip_norm,
        "image_resize": "128x128",
        "image_crop": "112x112",
        "workers": optimization.dataloader_num_workers,
        "embedding_dim": config.network.emb_dim,
        "network_layers": config.network.num_layers,
        "attention_heads": config.network.n_heads,
        "image_encoder": config.network.rgb_model_name,
        "two_step_time": optimization.t_two_step,
        "future_two_step_time": optimization.future_t_two_step,
        "future_loss_mode": optimization.future_embed_loss_mode,
        "future_joint_mode": optimization.future_joint_mode,
    }
    if options.resume_checkpoint is not None:
        parameters["resume_checkpoint"] = str(options.resume_checkpoint.resolve())
    return parameters


def render_loss_curve(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    training = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    evaluation = [
        json.loads(line)
        for line in (output / "evaluation_metrics.jsonl").read_text().splitlines()
    ]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(
        [point["step"] for point in training],
        [point["loss"] for point in training],
        alpha=0.5,
        linewidth=0.8,
        label="Train MIP (online weights, random crop)",
    )
    axes[0].plot(
        [point["step"] for point in evaluation],
        [point["loss"] for point in evaluation],
        "o-",
        label="Holdout MIP (checkpoint EMA, center crop)",
    )
    axes[0].set_ylabel("Training objective")
    axes[1].plot(
        [point["step"] for point in evaluation],
        [point["shadow_action_mse"] for point in evaluation],
        "o-",
        label="Shadow action MSE (training normalization)",
    )
    axes[1].set_ylabel("Action MSE")
    axes[1].set_xlabel("Optimizer step")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "loss_curve.png", dpi=150)
    plt.close(figure)


class TrainingMonitor:
    def __init__(self, config, datasets: tuple, options: TrainingMonitoringOptions):
        self.options = options
        self.tracker = TrainingTracker(
            options.mlflow,
            options.output,
            tracking_parameters(config, datasets, options),
        )
        self.evaluator = CheckpointEvaluator(
            config,
            datasets,
            CheckpointEvaluationOptions(
                output_path=options.output,
                batch_size=options.evaluation_batch_size,
                chunk_size=options.evaluation_chunk_size,
                seed=options.evaluation_seed,
            ),
        )

    def __enter__(self):
        self.tracker.__enter__()
        log_section(
            "训练与评估计划",
            [
                "每次保存后暂停更新，读取该权重的 EMA，评估全部留出回合后继续训练",
                "评估 loss 复用训练 MIP；另记真实影子推理动作 MSE；不按任务文本筛选",
                f"MLflow={self.options.mlflow.enabled} | 实验={self.options.mlflow.experiment_name}",
                f"评估 batch={self.options.evaluation_batch_size} | 执行步数={self.options.evaluation_chunk_size}",
                "权重旁保存全部逐帧预测，仅绘制前3回合；根目录保存 evaluation_metrics.jsonl 和 loss_curve.png",
            ],
        )
        return self

    def __exit__(self, exception_type, exception, traceback):
        return self.tracker.__exit__(exception_type, exception, traceback)

    def log_training(self, record: dict) -> None:
        self.tracker.log_train(record, record["step"])

    def checkpoint(self, agent, step: int) -> Path:
        path = save_training_checkpoint(agent, self.options.output / "models", step)
        metrics = self.evaluator.evaluate(agent, path, step + 1)
        record = {"step": step + 1, "checkpoint": str(path), **metrics}
        with (self.options.output / "evaluation_metrics.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        render_loss_curve(self.options.output)
        self.tracker.log_evaluation(metrics, step + 1)
        log_section(
            "权重影子评估完成",
            [
                f"评估回合={metrics['episodes']}（全部参与统计） | 绘图回合={min(metrics['episodes'], max_plot_episodes)}",
                f"step={step + 1} | MIP loss={metrics['loss']:.6f} | 动作 MSE={metrics['shadow_action_mse']:.6f}",
                f"七关节 MSE 之和={metrics['joint_mse_sum']:.8f} | 夹爪 MSE={metrics['gripper_mse']:.8f}",
                f"评估输出={path.parent / (path.stem + '_eval')}",
            ],
        )
        return path
