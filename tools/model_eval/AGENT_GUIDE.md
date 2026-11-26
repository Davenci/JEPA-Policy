# 给其他 Agent 的接入指南

本文说明如何用 `model_eval` 完成「**跑模型评估 → 生成中间产物 → 绘图**」的完整流程。

## 核心概念

```
┌─────────┐   evaluate()    ┌──────────────┐  render_episode()  ┌───────┐
│ 你的模型 │ ──────────────> │ 中间产物 .npz │ ─────────────────> │  PNG  │
└─────────┘                 └──────────────┘                    └───────┘
      ↑                            ↑
   你的数据集                 自动生成，不用手写
```

**中间产物是评估自动产生的**，你不需要手工构造 npz。

## 你要写的东西：一个脚本导出两个函数

本工具不知道你的模型和数据怎么加载，所以通过约定接入：

| 你必须导出 | 返回 | 作用 |
|---|---|---|
| `build_dataset()` | `model_eval.Dataset` | 提供回合数据 |
| `build_model()` | 任意模型对象 | 提供推理能力 |

模型对象只需满足两个条件：

| 属性/方法 | 要求 |
|---|---|
| `chunk_size` | 正整数，训练时的预测步长 H |
| `predict(obs)` | 返回 `[H, D]` 的 numpy 数组 |

`obs` 是你 `observe()` 返回的任意 dict，工具原样传递，不检查内部结构。

## 三步走

### 第一步：构造 Episode

```python
from model_eval import Episode, Dataset

def observe(frame):
    """返回第 frame 帧的模型输入，结构你自己定。"""
    return {
        "images": load_images(frame),      # 你的图像读取
        "state":  states[frame],           # 当前状态
        "prompt": task_text,
    }

ep = Episode(
    episode_id="ep000000",        # 会成为输出文件名
    length=600,                   # 整回合帧数 L
    ground_truth=states,          # [L, D] 逐帧真值
    observe=observe,              # 按帧取观测
    prompt="pick up the bottle",  # 会印在图上
)
```

**关键约束**：`ground_truth` 必须与模型输出**处在同一空间**
（同为绝对 state，或同为绝对 action），工具不做任何坐标系/单位换算。

### 第二步：包成模型

```python
class MyModel:
    chunk_size = 50               # 必须与训练配置一致

    def predict(self, obs):
        # 换成你的真实推理：torch / onnx / 远程 API 都行
        images, state = obs["images"], obs["state"]
        with torch.no_grad():
            action_chunk = self.policy(images, state)   # [H, D]
        return action_chunk.cpu().numpy()

def build_dataset():
    return Dataset([ep1, ep2, ...], name="my_dataset_v1")

def build_model():
    return MyModel()
```

### 第三步：一条命令跑完

```bash
python3 -m model_eval eval \
    --script    my_eval_script.py \
    --out-dir   runs/step50000/artifacts \
    --plot-dir  runs/step50000/figs \
    --checkpoint step50000
```

`--plot-dir` 给出时评估完直接出图；不给则只产出中间产物。

## 中间产物长什么样

`out-dir/` 下每个回合一个 npz，文件名就是 `episode_id`：

```
runs/step50000/artifacts/
├── ep000000.npz
├── ep000001.npz
└── ep000002.npz
```

评估会保存全回合逐帧真值，并把每次模型输出的完整 chunk 拼接成逐帧预测；
重叠位置采用最新一次推理结果。`metadata.inference_points` 记录模型实际推理帧，图中红点会标出这些位置。`stride` 必须不大于 `chunk_size`，否则时间轴会有未预测帧，工具会报错。产物包含 7 个基础键及可选 metadata。

验证一下：

```python
from model_eval import load_episode
e = load_episode("runs/step50000/artifacts/ep000000.npz")
print(e.dims, len(e.decision_points), e.chunk_size, e.episode_length)
print("inference points:", e.metadata["inference_points"])
```

## 导出指标（写报告用）

加 `--metrics-out`（eval）或 `--metrics`（plot）即可同时得到结构化指标 JSON，
报告可直接引用，不用手抄图上的数字：

```bash
python3 -m model_eval plot --arrays runs/a/artifacts --out-dir runs/a/figs --metrics
# -> runs/a/figs/metrics_abs.json / metrics_delta.json
```

JSON 里有 `episodes`（逐回合）与 `summary`（mean/median/min/max、分组、逐维）。
报告模板见 [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md)；只关心出图与标注含义见 [PLOT_PROMPT.md](PLOT_PROMPT.md)。

> 注意：`abs` 与 `delta` 的 MAE 数值恒等（数学必然），两者只提供不同曲线视角。

## 关键参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--stride` | 重规划间隔（帧）：每多少帧重新推理一次 | = `chunk_size` |
| `--on-error` | `fail` 遇错即停；`skip` 跳过坏回合继续 | `fail` |
| `--dim-mode` | `arm_joint8`（单臂 8 维）/ `arm_joint16`（双臂 16 维） | 按维数自动 |

`stride` 是最常调的参数：

- `stride = chunk_size`：chunk 连续拼接
- `stride < chunk_size`：重叠时采用最新 chunk 的输出
- `stride > chunk_size`：会产生帧缺口，工具会报错

## 分两步跑

评估和绘图也可以分开，便于反复调图而不重跑模型：

```bash
# 1) 只评估
python3 -m model_eval eval --script my_eval_script.py --out-dir runs/a/artifacts

# 2) 出图（可反复执行，不碰模型）
python3 -m model_eval plot --arrays runs/a/artifacts --out-dir runs/a/figs --kind both

# 3) 双权重对比：两份中间产物并排
python3 -m model_eval plot \
    --arrays  runs/new/artifacts \
    --compare runs/old/artifacts \
    --old-label new-ckpt --new-label old-ckpt \
    --out-dir runs/compare/figs --kind abs
```

## 在 Python 里直接调用

不想走命令行也可以：

```python
from model_eval import evaluate, render_episode

# 评估
results = evaluate(model, dataset, "runs/a/artifacts", stride=25)
# results: [{"episode_id", "path", "decisions", "chunk_size", "stride", "seconds"}, ...]

# 绘图
for r in results:
    render_episode(r["path"], f"runs/a/figs/{r["episode_id"]}.png", kind="abs")
```

## 输出组织建议

每次评估一个独立目录，不要覆盖历史：

```
runs/
└── YYYYMMDD_HHMMSS_<checkpoint>/
    ├── artifacts/       # 中间产物（可复现，留着）
    ├── figs/            # PNG
    └── REPORT.md        # 结论

## 排错

| 报错 | 原因与处理 |
|---|---|
| `model.chunk_size 必须是正整数` | 模型的 `chunk_size` 没设或类型不对 |
| `predict 返回形状 (a, b)，期望 (H, D)` | 检查模型输出 vs `ground_truth.shape[1]` |
| `predict 返回了 NaN/Inf` | 模型推理出 NaN，先查模型本身 |
| `ground_truth 行数 N 与 length L 不一致` | `Episode.length` 要和真值行数一致 |
| `评估失败：...` | 加了 `--on-error fail`（默认）；调试期可用 `skip` |

## 完整可运行示例

```bash
# 仓库自带，可直接跑
python3 -m model_eval eval \
    --script examples/torch_model_example.py \
    --out-dir /tmp/art --plot-dir /tmp/figs
```

照着 `examples/torch_model_example.py` 改两处（`load_episodes` 和 `predict`）即可接入真实模型。