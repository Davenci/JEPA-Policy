# 整回合轨迹对比图

**跑模型评估 → 自动生成中间产物 → 出图**，一条流水线。

```
你的模型 + 数据集  →  evaluate()  →  中间产物 .npz  →  render_episode()  →  PNG
```

**中间产物由评估自动产生，不需要手写。**

> 🤖 其他 agent 接入请看 [AGENT_GUIDE.md](AGENT_GUIDE.md)。

## 支持的模式

| 模式 | 维数 | 维序 | 夹爪维 |
|---|---|---|---|
| `arm_joint8`（默认） | 8 | 单臂 7 关节 + 夹爪 | `[7]` |
| `arm_joint16` | 16 | 左臂 7 关节 + 左夹爪 + 右臂 7 关节 + 右夹爪 | `[7, 15]` |

## 安装

```bash
pip install -r requirements.txt          # 只需 numpy + matplotlib
pip install -e .                         # 可选，安装后可在任意目录运行
```

中文字体（系统级）：`sudo apt install fonts-noto-cjk`

## 快速开始

```bash
# 跑自带示例（评估 + 出图一条命令）
python3 -m model_eval eval \
    --script   examples/torch_model_example.py \
    --out-dir  /tmp/art \
    --plot-dir /tmp/figs
```

## 接入自己的模型

写一个脚本导出两个函数：

```python
from model_eval import Dataset, Episode

def build_dataset():
    ep = Episode(
        episode_id="ep000000",
        length=600,                    # 回合帧数 L
        ground_truth=states,           # [L, D] 逐帧真值（与模型输出同空间）
        observe=lambda f: {...},       # 按帧返回模型输入，结构随意
        prompt="pick up the bottle",
    )
    return Dataset([ep], name="my_dataset")

def build_model():
    return MyModel()                   # 只要有 chunk_size 和 predict(obs)->[H,D]
```

然后：

```bash
python3 -m model_eval eval \
    --script    my_eval.py \
    --out-dir   runs/step50000/artifacts \
    --plot-dir  runs/step50000/figs \
    --checkpoint step50000
```

## 两步分开跑

```bash
# 1) 评估 → 中间产物
python3 -m model_eval eval --script my_eval.py --out-dir runs/a/artifacts

# 2) 出图（可反复跑，不碰模型）
python3 -m model_eval plot --arrays runs/a/artifacts --out-dir runs/a/figs

# 3) 双权重对比
python3 -m model_eval plot \
    --arrays runs/new/artifacts --compare runs/old/artifacts \
    --old-label new-ckpt --new-label old-ckpt \
    --out-dir runs/cmp/figs --kind abs
```

## 参数

### `eval`

| 参数 | 说明 | 默认 |
|---|---|---|
| `--script` | **必填**，导出 `build_dataset()` 与 `build_model()` 的 py 文件 | — |
| `--out-dir` | **必填**，中间产物目录 | — |
| `--plot-dir` | 评估完直接出图到该目录 | 不出图 |
| `--stride` | 重规划间隔（帧）；必须不大于 `chunk_size` 才能覆盖整段轨迹 | = `chunk_size` |
| `--on-error` | `fail` / `skip` | `fail` |
| `--dim-mode` | `arm_joint8` / `arm_joint16` | 按维数自动 |
| `--checkpoint` / `--data-root` | 印在图上的名字 | 空 |

### `plot`

| 参数 | 说明 | 默认 |
|---|---|---|
| `--arrays` | **必填**，中间产物目录或单个 npz | — |
| `--out-dir` | **必填** | — |
| `--compare` | 对比权重的产物目录 | 无 |
| `--kind` | `abs` / `delta` / `both` | `both` |
| `--dim-mode` | 同 eval | 按维数自动 |
| `--episodes` | 只画指定回合 | 全部 |
| `--chunk-stride` | 红点间隔（帧），0 = 按 chunk_size | `0` |
| `--old-label` / `--new-label` | 图例文案 | `pred` / `compare` |

## 中间产物格式

评估自动写出，每回合一个 npz，键名由本项目定义（见 `model_eval/io.py`）：

| 键 | 形状 | 含义 |
|---|---|---|
| `prediction` | `[L, D]` | 全部 chunk 拼接后的逐帧预测；重叠区采用最近一次推理结果 |
| `ground_truth` | `[L, D]` | 完整逐帧真值 |
| `state_base` | `[L, D]` | delta 视图锚点（上一帧真值） |
| `decision_points` | `[L]` | 每行对应的帧号 `0..L-1` |
| `metadata.inference_points` | `[N]` | 实际调用模型推理的帧号，图中以红点标注 |
| `episode_length` / `chunk_size` / `prompt` | 标量 | |

也可用 `build_from_arrays()` 手工构造，或 `load_episode()` 校验。

## 输出

```
runs/step50000/
├── artifacts/          evaluate 产出
│   ├── ep000000.npz
│   └── ep000001.npz
└── figs/               plot 产出
    ├── ep000000_abs.png
    ├── ep000000_delta.png
    └── compare_ep000000_abs.png
```

## 测试

```bash
python -m unittest discover -s tests -v      # 34 个用例
```

## 目录结构

```
model_eval/
├── model_eval/
│   ├── evaluate.py      评估：驱动模型推理，产出中间产物
│   ├── io.py            中间产物格式定义与读写
│   ├── builders.py      从数组构建产物
│   ├── plotting.py      绘图（render_episode）
│   ├── cli.py           eval / plot 子命令
│   └── __main__.py
├── examples/torch_model_example.py   可运行的接入模板
├── tests/
├── scripts/
└── AGENT_GUIDE.md       给其他 agent 的接入指南
```