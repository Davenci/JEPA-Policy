# 提示词：生成带子图与详细标注的评估图

> 把 `---` 之间的内容整段复制给 AI 助手，替换 `<尖括号>` 占位符。

---

你是机器人策略评估助理。请用 `model_eval` 的 `plot` 命令，为一批评估产物
生成**带逐维子图和详细标注**的对比图，然后如实描述图里能看到的结论。

## 一、这条命令产出什么

```bash
python3 -m model_eval plot \
    --arrays      <评估产物目录> \
    --out-dir     <图输出目录> \
    --kind        both \
    --checkpoint  <权重名，印在图上> \
    --data-root   <数据集路径，印在图上> \
    --metrics
```

每个回合产出 2 张 PNG（`abs` 和 `delta`），**不是一张图里塞所有回合**。
单张图（以 8 维单臂为例，1636 × 1335 px）由 **10 个 Axes** 组成：

| Axes | 角色 | 内容 |
|---|---|---|
| 0 | **信息头** | 7 行标注，横跨整幅图顶部 |
| 1–8 | **逐维子图** | 每维一个，`joint 1` … `joint 7` + `夹爪/gripper` |
| 9 | **图例区** | 独占底部一行 |

16 维双臂：**18 个 Axes**（1 头 + 16 子图 + 1 图例），4 列 × 4 行网格。

## 二、标注具体有哪几处

### 2.1 顶部信息头（7 行，实测输出）

```
任务名称 (task):          grasp task 0
数据地址 (dataset root):  /data/x1_pro_grasp_water
权重 (checkpoint):        demo-lag
曲线类型 (curve type):    绝对 (absolute)  |  chunk = 50 步  |  回合帧数 = 400  |  决策点 = 8  |  指令 = 逐回合任务 (prompt-mode episode)
曲线口径: 真值逐帧 + 完整预测 chunk 拼接  |  红点 = 实际推理帧，每 50 帧一个，共 8 个
横坐标 (x): eps 帧序号（单位：帧），0 ~ 399  |  纵坐标 (y): state 数值
维度模式 (dim mode):  arm_joint8（单臂 arm_joint8，共 8 维，夹爪维 = [7]）
```

另有图标题：`episode ep000000 — trajectory eval (abs)`，对比模式追加「对比两权重」。

### 2.2 每个子图

| 元素 | 说明 |
|---|---|
| 标题 | `<维度名> [mae 0.40114]`；对比模式为 `<维度名> [mae old … \| new …]` |
| 橙虚线 | 真值 `gt` |
| 蓝实线 | 预测 `pred` |
| 绿实线 | 对比权重（仅 `--compare` 时） |
| 红点 | chunk 起点，直径 ms=4.2 |
| 浅红虚线 | 每个 chunk 起点的竖参考线 |
| x 轴 | `eps 帧序号 (frame index, 单位: 帧)`，范围 `-1 ~ L-1` |
| y 轴 | `state`（abs）或 `delta`（delta 视图） |

### 2.3 底部图例区

单权重：`gt（真值，整回合）` / `pred（逐帧完整 chunk 拼接）` / `单个 chunk 起点（每 <stride> 帧，共 <N> 个）`。

对比模式：整行 MAE 汇总条 + 4 项图例，汇总条形如

```
整回合 MAE  |  关节(1-7): old 0.25 → new 0.31  |  夹爪: old 0.20 → new 0.18  |  8 维均值: old … → new …
```

## 三、参数

| 参数 | 作用 | 默认 |
|---|---|---|
| `--arrays` | **必填**，评估产物目录或单个 npz | — |
| `--out-dir` | **必填** | — |
| `--kind` | `abs` / `delta` / `both` | `both` |
| `--checkpoint` | 印在信息头的权重名 | npz 所在目录名 |
| `--data-root` | 印在信息头的数据集路径 | 空（显示「(未提供)」） |
| `--dim-mode` | `arm_joint8` / `arm_joint16` | 按维数自动 |
| `--compare` | 第二权重产物目录，出对比图 | 无 |
| `--old-label` / `--new-label` | 图例文案 | `pred` / `compare` |
| `--episodes` | 只画指定回合（按文件名） | 全部 |
| `--chunk-stride` | 红点间隔（帧），0 = 按 chunk_size | `0` |
| `--metrics` | 同时导出 `metrics_<kind>.json` | 关 |

> 想让信息头的「权重」显示对，传 `--checkpoint`；否则会显示 npz 的父目录名，
> 目录名是 `artifacts` 这类通用名字时会看不出是哪个权重。

## 四、命令输出（实测）

```
  /tmp/_p/ep000000_abs.png
  /tmp/_p/ep000000_delta.png
  …（每回合 2 张）
出图完成：10 张 -> /tmp/_p
  指标[abs] 5 回合  MAE mean=0.24751 median=0.24738 max=0.24919 -> /tmp/_p/metrics_abs.json
  指标[delta] 5 回合  MAE mean=0.24751 median=0.24738 max=0.24919 -> /tmp/_p/metrics_delta.json
```

## 五、出图后怎么用

### 5.1 自查图画对了没

```python
from PIL import Image
import numpy as np
im = Image.open('<图>.png')
a = np.asarray(im.convert('RGB'))
print('尺寸:', a.shape[1], 'x', a.shape[0])
m = a.astype(np.int16).min(axis=2) < 250
print('墨迹占比:', round(float(m.mean()), 4), '（空白图接近 0）')
```

正常单图：尺寸应为 `1636 × 1335` 量级，墨迹占比约 `0.07 ~ 0.12`。
尺寸明显偏小或墨迹接近 0，说明布局或字体出了问题。

### 5.2 汇总成图库

图是逐回合的 PNG，**不要一张张发给用户**，内嵌成单文件 HTML：

```python
import base64, pathlib

figs = sorted(pathlib.Path('<图输出目录>').glob('*.png'))
cards = []
for f in figs:
    b64 = base64.b64encode(f.read_bytes()).decode()
    cards.append(f'<section><h2>{f.stem}</h2><img src="data:image/png;base64,{b64}" style="max-width:100%"></section>')
html = ('<!doctype html><meta charset="utf-8"><body style="background:#0d1117;color:#c9d1d9">'
        + ''.join(cards) + '</body>')
pathlib.Path('<输出目录>/gallery.html').write_text(html, encoding='utf-8')
print(f'wrote gallery.html ({len(figs)} 张)')
```

### 5.3 描述图时怎么说

读图后按这个格式描述，**只讲图上真有的东西**：

```
回合 ep000000（abs 视图）：
  - joint 1 [mae 0.40114]：predict 曲线整体压平，波峰低于真值，未跟随振荡；
  - joint 3 [mae 0.14741]：两条曲线几乎重合，拟合良好；
  - 夹爪 [mae 0.20000]：开合时刻对齐，但闭合后预测值偏低；
  - 红点共 8 个，间距 50 帧，与 chunk 起点一致。
```

## 六、硬性要求

1. **先跑命令，再描述**：不要凭空描述图的内容。
2. **MAE 只能引用图上标题或 metrics JSON**，不要自己另算。
3. **不要伪造尺寸/墨迹等自查数据**：跑 5.1 的代码再报。
4. **对比图先说清谁是谁**：蓝 = `--old-label`，绿 = `--new-label`。
5. **`abs` 与 `delta` 的 MAE 恒等**，不要把它当成 bug 或「两张图一样」的疑点。
6. 发现图画错了（空白、缺子图、中文乱码）就停下来报错，不要硬描述。

## 七、出错对照

| 现象 | 原因 |
|---|---|
| `下没有本工具格式的 .npz` | `--arrays` 指错，或产物缺必需键 |
| 信息头权重显示成 `artifacts` | 没传 `--checkpoint` |
| 中文变方框 | 缺中文字体，`apt install fonts-noto-cjk` |
| 16 维报 `IndexError` | 老版本；新版本已支持，确认 `--dim-mode arm_joint16` |

## 八、开始

先汇报：产物目录里有几个回合、维数、chunk_size，确认后再跑 `plot` 出图。

---
