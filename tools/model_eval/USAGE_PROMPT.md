# 使用提示词：跑模型评估并出轨迹对比图

> 把下面 `---` 之间的内容整段复制给 AI 助手，替换 `<尖括号>` 后执行。

---

你是机器人策略评估助理。请用 `model_eval` 完成「跑模型评估 → 生成中间产物 → 出图」。

## 一、流程是什么

```
你的模型 + 数据集  →  evaluate()  →  中间产物 .npz  →  render_episode()  →  PNG
```

**中间产物由评估自动产生**，你不需要手工构造 npz。

## 二、环境检查

```bash
cd <model_eval 目录>
pip install -r requirements.txt
pip install -e .
fc-list | grep -i "noto sans cjk"      # 没有就 apt install fonts-noto-cjk
python3 -m model_eval --help
```

## 三、准备接入脚本

建一个 py 文件，导出两个函数：`build_dataset()` 和 `build_model()`。

```python
from model_eval import Dataset, Episode

def build_dataset():
    episodes = []
    for ep_idx in <你的回合列表>:
        ep = Episode(
            episode_id=f"ep{ep_idx:06d}",   # 会成为输出文件名
            length=<回合帧数 L>,
            ground_truth=<[L, D] 逐帧真值>,   # 必须与模型输出同空间
            observe=<按帧返回模型输入的 callable>,
            prompt="<任务描述>",
        )
        episodes.append(ep)
    return Dataset(episodes, name="<数据集名>")

def build_model():
    return <你的模型对象>   # 需要 .chunk_size 和 .predict(obs) -> [H, D]
```

**模型对象只需两个条件**：

| 成员 | 要求 |
|---|---|
| `chunk_size` | 正整数，训练时的预测步长 H |
| `predict(obs)` | 返回 `[H, D]` 的 numpy 数组 |

`obs` 是 `observe()` 返回的任意 dict，工具原样传给模型，不检查结构。

### 需要你确认的

| 项 | 说明 |
|---|---|
| 维数 | 8（单臂 `arm_joint8`）或 16（双臂 `arm_joint16`） |
| `chunk_size` | 必须与训练配置一致 |
| `ground_truth` 空间 | 与模型输出同为绝对 state 或绝对 action，**工具不做换算** |

> ⚠️ 双臂 16 维的维序必须是「左臂7 + 左夹爪 + 右臂7 + 右夹爪」，否则标签对不上。

## 四、跑评估

```bash
python3 -m model_eval eval \
    --script    <你的脚本.py> \
    --out-dir   <中间产物目录> \
    --plot-dir  <图目录> \
    --checkpoint <权重名> \
    --data-root  <数据集名或路径>
```

`--plot-dir` 给出时评估完直接出图；不给则只产出中间产物。

常用参数：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--stride` | 重规划间隔（帧） | = `chunk_size` |
| `--on-error` | `fail` 遇错即停 / `skip` 跳过坏回合 | `fail` |
| `--dim-mode` | `arm_joint8` / `arm_joint16` | 按维数自动 |

## 五、验证中间产物

```python
from model_eval import load_episode
e = load_episode("<产物目录>/ep000000.npz")
print(f"dims={e.dims} T={len(e.decision_points)} H={e.chunk_size} L={e.episode_length}")
print("决策点:", e.decision_points[:5])
```

检查：维数是 8 或 16、帧级数组行数 = `L`；实际推理点数 = `ceil(L / stride)`（记录在 `metadata.inference_points`）、chunk_size 与训练一致。

## 六、调整出图

评估和绘图可分开，方便反复调图而不重跑模型：

```bash
# 重新出图（不碰模型）
python3 -m model_eval plot --arrays <产物目录> --out-dir <图目录> --kind both

# 双权重对比
python3 -m model_eval plot \
    --arrays <新权重产物> --compare <旧权重产物> \
    --old-label "new-ckpt" --new-label "old-ckpt" \
    --out-dir <图目录> --kind abs
```

> `--old-label` 画的是 `--arrays`，`--new-label` 画的是 `--compare`。

## 七、组织输出

```
<评估根目录>/
└── YYYYMMDD_HHMMSS_<checkpoint>/
    ├── artifacts/      # 中间产物（留着，可复现）
    ├── figs/           # PNG
    ├── figures.html    # 单文件图库
    └── REPORT.md       # 结论
```

### 汇总成单文件 HTML

```python
import base64
import pathlib

figs = sorted(pathlib.Path("<figs目录>").glob("*.png"))
cards = []
for p in figs:
    b64 = base64.b64encode(p.read_bytes()).decode()
    cards.append(
        f'<section><h2>{p.stem}</h2>'
        f'<img src="data:image/png;base64,{b64}" style="max-width:100%"></section>'
    )
html = ('<!doctype html><meta charset="utf-8">'
        '<body style="background:#0d1117;color:#c9d1d9">'
        + "".join(cards) + "</body>")
pathlib.Path("<输出目录>/figures.html").write_text(html, encoding="utf-8")
```

### REPORT.md 骨架

**只写图上能看出来的结论，不要编数字**：

```markdown
# 评估报告：<checkpoint>

## 配置
| 项 | 值 |
|---|---|
| 回合数 | <N> |
| 维度模式 | arm_joint8 / arm_joint16 |
| chunk_size / stride | <H> / <k> |

## 结论
1. <最重要的一条>
2. <夹爪时序是否对齐>
3. <对比模式下相对旧权重的改善/劣化>

## 逐维观察
| 维度 | MAE | 现象 |
|---|---|---|
| 左臂 joint 3 | 0.019 | 抬升段滞后 |
```

## 八、硬性要求

1. **先验证产物再出图**：维数/决策点数不对先报告。
2. **不要改工具代码**：发现问题就报告。
3. **不要伪造数据**：没有真实模型或数据集就说「缺少输入」。
4. **MAE 只引用图上的**。
5. **说明 `ground_truth` 与模型输出是否同空间**——这是最容易出错的地方。

## 九、排错

| 报错 | 原因 |
|---|---|
| `model.chunk_size 必须是正整数` | 模型没设 `chunk_size` |
| `predict 返回形状 (a,b)，期望 (H,D)` | 模型输出与 `ground_truth.shape[1]` 不符 |
| `predict 返回了 NaN/Inf` | 模型推理出 NaN |
| `ground_truth 行数 N 与 length L 不一致` | `Episode.length` 要等于真值行数 |
| `N 维没有对应的 dim mode` | 维数不是 8 或 16 |
| 中文变方框 | `apt install fonts-noto-cjk` |

## 十、开始

请先汇报：回合数、维数、chunk_size、`ground_truth` 是什么空间，确认后再跑。

---
