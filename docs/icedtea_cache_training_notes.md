# 冰红茶训练：解码缓存与连续供数记录

记录日期：2026-11-26（上海时间）

## 环境与目标

- 主机：阿里云 DSW，SSH 别名为 `jepa`。
- JEPA-Policy 部署目录：`/mnt/data/usr/yuanyh/new_test`。
- Rolling Cache 仓库：`/mnt/data/usr/yuanyh/new_test/rolling_cache`。
- 目标是减少视频反复解码和数据迭代边界造成的 GPU 等待。训练仍是单卡、右臂 8 维动作，batch size 为 256，总目标步数为 100,000。

版本记录：DSW 上的 /mnt/data/usr/yuanyh/new_test 当前不是 Git checkout；本次训练入口和缓存适配器修改保存在该部署目录，尚未提交到源仓库。后续若重新部署代码，需要把这两处改动同步到实际开发分支。

## 数据与窗口

训练集：`/mnt/data/EIC01/vla/x1_pro/train/waterbar_2task_228eps_20261126_smoothed`。

数据集共有 228 个回合。训练入口筛选任务名 `grasp the red iced tea.`，读到该任务 3,681 帧，构造出 3,231 个训练窗口。每条样本包含三路相机、2 帧观测、10 步动作序列及未来图像/状态目标。右臂动作维度为 8。

窗口起点必须保证 10 步动作范围留在同一回合内。长度为 L 的回合可产生 max(0, L - 9) 个窗口；不能跨回合取样。每轮 3,231 个窗口按 batch 256 组成 12 个完整批次，剩余 159 个样本按 drop_last 规则舍弃。

## 缓存构建

接入位置：`jepa_policy/datasets/icedtea_cache.py`；训练入口：`examples/train_x1_icedtea.py`。

适配器实现 Rolling Cache 的 SampleSource，并通过 StepSampleCache 构建解码后的磁盘缓存。缓存字段包含三路相机的观测图像、未来图像、右臂状态、动作和样本索引。训练期间读取本地 mmap 缓存，不再每轮从 CPFS 随机 seek 和解码原始视频。本次采用磁盘解码缓存，没有启用 Rolling Cache 的 /dev/shm 双槽滚动层；实测当前数据集的缓存读取已经很快。

- 缓存目录：`/tmp/jepa_icedtea_cache/0ffc3d656dfb4ac4/962e502e1e0a907a`。
- 容量约 1.33 GiB，4 个缓存构建进程约 18 秒完成。
- `READY` 标志存在，后续启动会复用缓存。
- /tmp 是临时盘；DSW 重启或清理临时盘后会重新构建。

参数含义：

- `--cache-build-workers`：只控制首次解码建缓存时的并发数。缓存命中后基本不起作用。
- `--workers`：控制训练 DataLoader 的读取进程数。本次用 16 个；缓存读取的等待数据能力已经充足，不建议盲目增加到 24。

## 为什么增加训练 worker 没有解决周期性掉速

旧训练每轮恰好 12 个 batch。日志中数据等待尖峰每隔 12 步出现一次，通常约 0.13–0.40 秒；其他 batch 的等待接近零。原因是 DataLoader 取完有限数据后重新开始下一轮，GPU 在轮次交界处等待读取进程恢复。增加 worker 数量不能消除这个重置边界，反而会增加启动和调度成本。

在训练入口加入 `ContinuousShuffleBatchSampler`：每次生成一轮随机排列，持续产出完整 batch，直到达到配置的总训练步数。它不让 DataLoader 在每轮末尾耗尽并重新创建迭代器；仍按轮随机打乱，仍丢弃每轮最后不足 256 的部分。该改动不改变模型、loss 或 batch size。

## 验证结果

- 3 个位置的缓存样本与原始解码逐字段相等。
- 16 个 DataLoader worker、batch 256 的缓存读取通过验证。
- 连续采样器使用 2 个 worker 连续读取 25 个 batch 的小型验证通过，轮次边界没有出现等待尖峰。
- 新训练从第 5,000 步检查点续跑后，步骤 5,001–5,035 的 data wait 约 0.0002 秒。
- 截至 2026-11-26 15:40（上海时间），续训运行最新日志为第 6,992 步；最近 30 步 data wait 均值约 0.00024 秒、最大约 0.00043 秒，update 用时约 0.235 秒。
- 修复后 15 秒 GPU 采样曾在 75%–99% 之间波动，没有再观察到掉到 0%。这说明数据供给断档已消除，但不能保证 GPU 利用率每个采样间隔都超过 90%；剩余波动需从模型计算与 GPU/CPU 同步开销继续分析。
- 已运行 Ruff 检查和 Python 语法检查；通过。

## 当前训练与检查点

旧运行目录：`/mnt/data/usr/yuanyh/runs/icedtea_cache_100k`。

检查点：`/mnt/data/usr/yuanyh/runs/icedtea_cache_100k/models/model_latest.pt`。其训练状态为 `n_gradient_step=4999`，含优化器状态，可从第 5,000 步续训。旧 metrics 曾记录到第 5,471 步，但检查点之后的 471 步没有保存进权重。

连续供数版本的新运行目录：`/mnt/data/usr/yuanyh/runs/icedtea_cache_100k_continuous`；日志：`/mnt/data/usr/yuanyh/runs/icedtea_cache_100k_continuous/train.log`。本次启动使用旧运行目录中的 5,000 步检查点，旧运行和检查点均保留。模型每 5,000 步保存一次；新运行的下一保存点为总步数 10,000。

## 本次续训启动参数

以下命令用于记录本次配置。正式续训时应换一个新的输出目录，并把 `--resume` 指向最新有效检查点；不要在同一训练任务运行期间重复启动：

```bash
export PYTHONPATH=/mnt/data/usr/yuanyh/new_test/rolling_cache:/mnt/data/usr/yuanyh/new_test:$PYTHONPATH
cd /mnt/data/usr/yuanyh/new_test

python examples/train_x1_icedtea.py \
  --dataset /mnt/data/EIC01/vla/x1_pro/train/waterbar_2task_228eps_20261126_smoothed \
  --output /mnt/data/usr/yuanyh/runs/icedtea_cache_100k_continuous \
  --steps 100000 \
  --batch-size 256 \
  --workers 16 \
  --save-every 5000 \
  --cache-root /tmp/jepa_icedtea_cache \
  --cache-build-workers 4 \
  --resume /mnt/data/usr/yuanyh/runs/icedtea_cache_100k/models/model_latest.pt
```

排查时重点查看 `metrics.jsonl` 中的 `data_wait_seconds` 和 `update_seconds`：若 data wait 很低但 GPU 利用率仍明显低于目标，应转查每步 GPU 同步、模型算子和 batch 计算效率，而不是继续增加 DataLoader worker。

