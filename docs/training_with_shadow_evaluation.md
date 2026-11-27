# 训练中自动进行影子评估

运行主机：`jepa`。项目目录：`/mnt/data/usr/yuanyh/new_test`。

## 数据与统计量

训练入口仍是 `python3 examples/train_x1_icedtea.py`。提供一个已经整理好的单任务数据目录，任务文本不参与筛选。

启动时先对完整回合做固定随机划分：训练约 95%，评估约 5%。评估回合数向上取整。228 条数据得到 216 条训练、12 条评估；200 条得到 190 条训练、10 条评估。窗口不会跨回合，因此两侧不共享帧。

`--split-seed` 默认 0，划分结果写入运行目录：

- `episode_split.json`：数据集路径、元数据指纹、种子及两侧索引。
- `train_episode_indices.json`：训练回合索引。
- `eval_episode_indices.json`：评估回合索引。

若指定 `--episode-indices`，该文件表示参与本次 95%/5% 划分的候选回合池，并非最终训练回合列表。

启动日志只保留训练与评估回合数等简短摘要，不输出详细“数据筛选”段落；完整审计仍写入 `dataset_selection.json`。各模块日志用空行和横线分隔，训练进度保持单行。

动作和状态的最小值、范围仅使用训练回合计算，保存到 `normalization_stats.json`。评估源以 `normalize=False` 读取，之后应用训练统计量；不计算或使用评估集自己的统计量。训练缓存只保存训练侧数据。评估不经过会重新构造统计量的缓存工厂。

图像预处理仍是三路 RGB、缩放到 128×128、映射到 [-1, 1]。训练随机裁剪到 112×112，评估中心裁剪到 112×112。没有增加按数据集计算的图像均值或标准差。

## 保存和评估顺序

每次达到 `--save-every`：

1. 完成当前优化器更新，保存权重与优化器状态。
2. 从这个刚保存的文件读取 EMA 编码器和策略权重，建立独立评估副本。
3. 对全部留出窗口计算与训练同公式的 MIP 损失。
4. 对全部留出回合做影子推理，保存原始逐帧预测与轨迹图片。
5. 保存标量指标，更新训练/评估曲线，在同一个 MLflow Run 记录当前优化器步数。
6. 释放评估副本，继续训练。

评估与训练串行使用同一张 GPU，评估及绘图期间暂停参数更新；没有机器人控制或仿真。评估隔离随机数状态，不改变在线模型、优化器、梯度和模块训练模式。

最后一步也会保存和评估；如果恰好达到保存间隔，只执行一次。`--save-every 0` 表示仅在最后执行。总计 10000 步、间隔 1000 步，会得到 10 个评估点。

## 指标含义

| MLflow 指标 | 来源与口径 | 频率 |
| --- | --- | --- |
| `train/loss` | 原训练循环的完整 MIP 损失，包含动作和未来辅助项；在线参数、随机裁剪 | 原日志周期、首步和保存步 |
| `train/loss_action`、`train/loss_future_raw` 等 | 原有训练指标，不额外反向传播 | 同训练日志 |
| `train/learning_rate` | 原调度器更新后的学习率，即下一步使用值 | 同训练日志 |
| `train/grad_norm` | 原训练更新已计算的裁剪前梯度范数 | 同训练日志 |
| `eval/loss` | `TrainingAgent.validation_metrics()` 复用完整训练公式；保存的 EMA 参数、中心裁剪、固定噪声种子；按窗口数加权 | 每个权重一次 |
| `eval/loss_action`、`eval/loss_future_raw` 等 | 同一验证调用的分项，按窗口数加权 | 每个权重一次 |
| `eval/shadow_action_mse` | 实际两次前向推理得到动作后，与真值比较；使用训练归一化空间，按帧与 8 个维度取均值 | 每个权重一次 |
| `eval/joint_mse_sum` | 原始动作单位下，7 个关节各自逐帧 MSE 的总和 | 每个权重一次 |
| `eval/gripper_mse` | 原始夹爪角度单位下的逐帧 MSE | 每个权重一次 |
| `eval/windows`、`eval/frames`、`eval/episodes`、`eval/seconds` | 实际评估规模与耗时 | 每个权重一次 |

完整 MIP 损失包含带真值加噪输入的第二阶段及未来辅助目标，不能把最终影子预测的动作 MSE 冒充这个训练目标。因此同时保留两种指标。训练在线参数和评估 EMA、随机裁剪和中心裁剪的差别，也会影响两条 loss 曲线的数值。

影子推理默认执行 4 步后重新查询，预测总长度仍为 10；历史位置不执行，最后一段只保留回合范围内的帧。`--eval-chunk-size` 可设为 1～9。本次在线评估不启用动作时间聚合；独立评估入口原有的 ACT 可选功能保持可用。

## 文件布局

```text
运行目录/
  episode_split.json
  train_episode_indices.json
  eval_episode_indices.json
  normalization_stats.json
  training_config.json
  training_summary.json
  training.log
  metrics.jsonl
  evaluation_metrics.jsonl
  loss_curve.png
  mlflow_run.json
  models/
    model_step_001000.pt
    model_step_001000_eval/
      metrics.json
      artifacts/episode_000004.npz
      figures/episode_000004_abs.png
      figures/episode_000004_delta.png
```

轨迹图片及原始点保存在对应权重旁。根目录 `loss_curve.png` 上半部叠加训练与验证 MIP，下半部连接影子动作 MSE。评估记录包含训练统计量文件 SHA256、相关代码 SHA256 和回合索引。

## MLflow 安装与连接

这台 DSW 沿用已有全局 `/usr/bin/python3`，Python 3.12.3、NVIDIA PyTorch 2.7.0a0、CUDA 12.8。新增客户端固定为 `mlflow-skinny==3.14.0`，安装清单位于 `requirements-mlflow.txt`。实际安装命令：

```bash
python3 -m pip install --break-system-packages -r /mnt/data/usr/yuanyh/new_test/requirements-mlflow.txt
```

没有重建 Torch/CUDA 环境，也未修改中央 MLflow 服务。主项目原有 `pyproject.toml` 与 `uv.lock` 保持不变；不为客户端安装解析机器人仿真的 Git 依赖。

默认启用 MLflow，Tracking 地址为 `https://mlflow.seer-robotics.cn`，实验为已经存在的 `pretrain-fm`。先只读确认实验存在且激活，再创建一个训练 Run；找不到或没有权限直接报错，不自动新建实验。

认证仅由实际训练进程的环境提供。既有服务的账号与访问令牌分别对应 `MLFLOW_TRACKING_USERNAME` 和 `MLFLOW_TRACKING_PASSWORD`；不要同时配置 Bearer 变量。账号或令牌不写入命令参数、配置、输出或 Git。用户在自己的 SSH 终端隐藏输入后，检查和训练必须从同一个终端启动。

只读检查命令为 `python3 tools/check_icedtea_mlflow.py`。提供 `--run-id` 时还会回读该 Run 的完整训练与评估 loss 历史，核对至少两个不同步数。

`--mlflow-run-name` 可设置名称，默认使用输出目录名；`--mlflow-tracking-uri`、`--mlflow-experiment` 可显式指定。`--no-mlflow` 关闭远程指标写入，保留本地评估与曲线。连接或写入失败直接报错，不自动改用其他后端。不会上传权重、图片、数据集或其他制品。

恢复训练使用 `--resume` 和一个新的输出目录；必须有原运行的 `episode_split.json`、`normalization_stats.json`，并核对划分及统计量一致。恢复创建新的 MLflow Run，记录来源权重，优化器步数接续；不会隐式续写旧 Run。

原有检查点没有保存完整随机数、采样器和调度器状态，因此续训不保证与未中断运行逐步完全复现。

## 训练示例

目标主机 `jepa`，工作目录 `/mnt/data/usr/yuanyh/new_test`。认证已在同一终端配置完成后：

```bash
cd /mnt/data/usr/yuanyh/new_test && python3 examples/train_x1_icedtea.py \
  --dataset /mnt/data/EIC01/vla/x1_pro/train/waterbar_2task_228eps_20261126_smoothed \
  --steps 10000 \
  --batch-size 256 \
  --workers 16 \
  --save-every 1000 \
  --log-every 10 \
  --cache-root /tmp/jepa_icedtea_cache \
  --cache-build-workers 8 \
  --split-seed 0 \
  --eval-batch-size 64 \
  --eval-chunk-size 4 \
  --eval-seed 12345 \
  --mlflow-run-name jepa-icedtea-holdout
```

未指定 `--output` 时自动使用新的 `./outputs/年月日_时分秒/`。非空输出目录会报错，避免覆盖既有训练。

## 2026-11-27 验证记录

真实单卡短验证使用准备好的数据目录中前 20 个回合作为候选池，划分为 19/1；原模型训练两步、batch 16、每步保存。使用显式指定的隔离本地 MLflow FileStore，不连接中央服务。

- 结果目录：`/mnt/data/usr/yuanyh/runs/jepa_train_eval_smoke_20261127_16a9c60a3c`。
- Run ID：`6c5bda9be5f64492a873d5de533f34d5`；状态 `FINISHED`。
- 两份权重、两次完整评估、四张轨迹图片、逐帧原始点及总 loss 曲线均生成。
- `train/loss`、学习率、梯度范数、`eval/loss`、影子动作 MSE 均回读到第 1、2 步；MLflow 制品列表为空。
- 完整记录：结果目录内 `smoke_verification.json`。两步实验只验证管线，不代表训练后的模型效果。
- 中央服务只读探测返回 401，agent 的 SSH 进程没有认证；用户训练终端的认证及中央服务读写仍待验证。
- `tests` 与 `tools/model_eval/tests` 共 237 项测试通过。仓库全目录收集包含未安装的机器人运行时与对照模型组件，不能在当前 DSW 当作全部通过。

本次实测 Python 物理行数（包含空行与注释）：

| 文件 | 行数 |
| --- | ---: |
| `examples/train_x1_icedtea.py` | 434 |
| `jepa_policy/checkpoint_evaluation.py` | 314 |
| `jepa_policy/datasets/icedtea_split.py` | 177 |
| `jepa_policy/icedtea_training_arguments.py` | 78 |
| `jepa_policy/mlflow_tracking.py` | 222 |
| `jepa_policy/training_artifacts.py` | 85 |
| `jepa_policy/training_logging.py` | 133 |
| `jepa_policy/training_monitor.py` | 177 |
| `jepa_policy/training_startup.py` | 360 |
| `tests/test_check_icedtea_mlflow.py` | 134 |
| `tests/test_checkpoint_evaluation.py` | 278 |
| `tests/test_icedtea_startup_log.py` | 452 |
| `tests/test_icedtea_training_split.py` | 207 |
| `tests/test_mlflow_tracking.py` | 217 |
| `tests/test_training_artifacts.py` | 102 |
| `tests/test_training_monitor.py` | 240 |
| `tools/check_icedtea_mlflow.py` | 129 |

生产函数均未超过 60 行或 4 个参数；测试替身 `joint_forward` 保留既有六参数接口，函数本身三行。Ruff 代码和格式审查通过。

未提交或推送本次修改。回滚仅撤回本次入口接线与新增模块；保留训练结果、数据及验证 Run，不执行自动删除。
