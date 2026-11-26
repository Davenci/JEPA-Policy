# 模型评估报告模板

两份：**A 通用版**（任何机器人策略模型都能用）、**B 结合 model_eval 的完整版**（内置真实命令、指标字段与填好的示例）。

---

# A. 通用模型评估报告模板

> 不绑定任何具体工具，适用于 VLA / 扩散策略 / 行为克隆等。按 `<尖括号>` 填空即可。

```markdown
# 模型评估报告：<模型名> on <数据集名>

> 一句话结论：<这个模型什么水平，主要短板是什么>

## 1. 摘要

| 项 | 值 |
|---|---|
| 模型 / 权重 | <checkpoint 路径或 commit> |
| 数据集 | <路径 + 版本 + 回合数> |
| 评估方式 | 离线回放 / 闭环仿真 / 真机 |
| 核心指标 | <如 整回合 MAE 0.198 / 成功率 62%> |
| 基线对比 | <相对上一版或基线模型的变化> |
| 结论 | 可用 / 需改进 / 不可用 |

## 2. 评估设计

### 2.1 训练配置（决定可比性）

| 项 | 值 |
|---|---|
| 预测步长 H (chunk_size) | <H> |
| 重规划间隔 k | <k>，与 H 的关系 |
| 动作空间 | 绝对关节 / 相对增量 / 末端位姿 |
| 维度配置 | <如 单臂 8 维 = 7 关节 + 夹爪> |
| 归一化 | <是否用 stats，来源> |

### 2.2 评估口径（不写清楚数字就不可比）

- **决策点**：每 <step> 帧一个，共 <N> 个；
- **曲线取值**：每个决策点取预测 chunk 的第 <k> 步；
- **真值对齐**：取自 `action[t]` 还是 `state[t+1]`；
- **尾部处理**：最后一个不完整 chunk 截断 / 丢弃；
- **是否 teacher-forcing**：用数据集观测还是模型自身反馈。

### 2.3 可复现信息

| 项 | 值 |
|---|---|
| 代码版本 | <git commit> |
| 环境 | <Python / 框架 / 设备> |
| 随机种子 | <seed> |
| 评估日期 | <YYYY-MM-DD> |

## 3. 总体指标

| 指标 | mean | median | min | max | std |
|---|---|---|---|---|---|
| 整体 MAE | | | | | |
| 整体 RMSE | | | | | |
| 平均相关系数 | | | | | |

> **必须同时报 mean / median / max**。长尾回合（个别抓取失败）会被 mean 掩盖；
> 若各回合数字异常接近，说明是系统性偏差，要单独指出。

## 4. 分组指标

| 分组 | MAE | RMSE | 相关系数 | 说明 |
|---|---|---|---|---|
| 手臂关节 | | | | 影响任务完成度 |
| 夹爪 | | | | 影响抓取成败，常是短板 |
| （双臂）左臂 / 右臂 | | | | 看是否失衡 |

## 5. 逐维分析

| 维度 | MAE | 现象 | 严重度 |
|---|---|---|---|
| | | 滞后 / 波峰偏低 / 震荡 / 漂移 | 高 / 中 / 低 |

**关键维度详述**（取 MAE 最高的 2–3 维展开）：

- `<维度名>`：在 <阶段> 误差最大，图上表现为 <具体形状>，推测原因 <…>。

## 6. 分阶段分析（任务有明确阶段时用）

| 阶段 | 帧范围 | MAE | 观察 |
|---|---|---|---|
| 接近 / 抓取 / 移动 / 放置 | | | |

## 7. 与基线对比

| 指标 | 基线 | 当前 | 变化 | 是否显著 |
|---|---|---|---|---|
| 整体 MAE | | | | |
| 夹爪 MAE | | | | |

## 8. 失败案例

| 回合 | 现象 | 可能原因 | 优先级 |
|---|---|---|---|
| | | | |

## 9. 局限与风险

- 本评估是 <离线 / 闭环>，与真机有差距；
- 数据集 <N> 回合，统计置信度有限；
- <分布偏移、标注质量等>

## 10. 结论与建议

**结论**：<3 句以内>

**下一步**：
1. <最高优先级>
2. <…>

## 附录

- 图表清单 / 指标 JSON 路径 / 复现命令
```

### 写作原则

| 原则 | 说明 |
|---|---|
| 口径先行 | 不写清决策点 / 真值对齐方式，数字无法横向比较 |
| 分布优于均值 | mean + median + max 一起报，避免被长尾或极端值误导 |
| 数字可追溯 | 每个数字标明来自哪个文件 / 哪张图，不手抄、不另算 |
| 结论要能行动 | 「joint 1 偏差大」不如「joint 1 幅度压平，去查归一化」 |
| 主动写局限 | 回合数、teacher-forcing、分布偏移都写出来 |

---

# B. 结合 model_eval 的完整版模板

> 下面所有命令、字段、产物路径都来自本工具，已实测可跑。

## B0. 配套命令（先跑，再填模板）

```bash
# ① 评估：模型推理 → 中间产物
python3 -m model_eval eval \
    --script      my_task.py      # 需导出 build_dataset() / build_model()
    --out-dir     runs/artifacts \
    --metrics-out runs/metrics.json \
    --data-root   /data/x1_pro_grasp_water \
    --checkpoint  step50000 \
    --plot-dir    runs/figs        # 可选：评估完直接出图

# ② 出图 + 双权重对比 + 指标导出
python3 -m model_eval plot \
    --arrays      runs/artifacts \
    --compare     baseline/artifacts \
    --kind        abs \
    --old-label   step50000 --new-label step10000 \
    --checkpoint  step50000 \
    --data-root   /data/x1_pro_grasp_water \
    --out-dir     runs/figs \
    --metrics
```

产物：

```
runs/
├── artifacts/          # 每回合一个 npz（绘图输入）
├── figs/
│   ├── ep000000_abs.png
│   ├── ep000000_delta.png
│   ├── compare_ep000000_abs.png   # 带 --compare
│   ├── metrics_abs.json           # ← 报告引用这个
│   └── metrics_delta.json
└── metrics.json
```

## B1. 指标 JSON 字段

```jsonc
{
  "kind": "abs",                  // abs / delta
  "dim_mode": "arm_joint8",       // 或 arm_joint16
  "episodes": [
    {
      "episode_id": "ep000000",
      "num_decision_points": 8, "episode_length": 400, "chunk_size": 50,
      "prompt": "grasp task 0",
      "overall":  {"mae":…, "rmse":…, "mean_corr":…},
      "arms":     {"mae":…, "rmse":…, "mean_corr":…},   // 手臂关节组
      "grippers": {"mae":…, …},                          // 夹爪组
      "arm_groups": {"左臂(1-7)": {…}, "右臂(1-7)": {…}}, // 仅双臂
      "per_dim": [{"index":0,"label":"joint 1","is_gripper":false,
                   "mae":…,"rmse":…,"max_abs_err":…,"corr":…}]
    }
  ],
  "summary": {
    "num_episodes": 5,
    "overall":  {"mae":{"mean","median","min","max"}, "rmse":{…}, "mean_corr":{"mean"}},
    "arms": {…}, "grippers": {…},
    "per_dim": [{"label":"joint 1","is_gripper":false,"mae_mean":…,"mae_max":…}]
  }
}
```

> ⚠️ **`abs` 与 `delta` 的 MAE / RMSE 数值恒等**，这不是 bug。
> 因为 `(pred - base) - (gt - base) ≡ pred - gt`。两个视图的差别在**曲线形状**，
> 不在误差数值；报告里如实说明即可。

## B2. 填好的示例报告（真实工具产出）

> 数据来自 `python3 -m model_eval eval/plot` 实跑结果，可作填写参照。

```markdown
# 模型评估报告：demo-lag on x1_pro_grasp_water

> 结论：输出被系统性压平（幅度约 40%），**不可部署**；joint 1/6/7 最差。

## 1. 摘要

| 项 | 值 |
|---|---|
| 模型 / 权重 | `demo-lag` |
| 数据集 | `/data/x1_pro_grasp_water`，5 个回合 |
| 评估方式 | 离线回放（teacher-forcing） |
| 核心指标 | 整回合 MAE **0.2475**（median 0.2474，max 0.2492） |
| 基线对比 | —（尚无基线） |
| 结论 | **不可用**：整体幅度系统性偏低 |

## 2. 评估设计

| 项 | 值 |
|---|---|
| 维度模式 | `arm_joint8`（单臂 7 关节 + 夹爪） |
| 预测步长 H | 50 |
| 重规划间隔 k | 50（= H） |
| 曲线取值 | 决策点处 chunk 第 0 步 |
| 决策点数 | 8 / 10 / 11 / 12 / 13 |
| 尾部处理 | 最后一个不完整 chunk 截断 |
| 评估日期 | 2026-11-25 |

## 3. 总体指标

| 指标 | mean | median | min | max |
|---|---|---|---|---|
| 整体 MAE | 0.2475 | 0.2474 | 0.2449 | 0.2492 |
| 整体 RMSE | 0.2771 | — | — | — |
| 关节组 MAE | 0.2542 | — | — | — |
| 夹爪组 MAE | 0.2010 | — | — | — |

> 5 个回合 MAE 全落在 0.245–0.249，**异常集中**：这是系统性幅度偏差，
> 不是个别回合的偶发失误。

## 4. 逐维分析

| 维度 | MAE 均值 | MAE 最大 | 严重度 |
|---|---|---|---|
| joint 1 | **0.3958** | 0.4011 | 高 |
| joint 7 | 0.3223 | 0.3293 | 高 |
| joint 6 | 0.3036 | 0.3114 | 中 |
| joint 5 | 0.2229 | 0.2294 | 中 |
| joint 2 | 0.2198 | 0.2282 | 中 |
| 夹爪/gripper | 0.2010 | 0.2048 | 中 |
| joint 4 | 0.1731 | 0.1764 | 低 |
| joint 3 | 0.1416 | 0.1474 | 低 |

**joint 1 最差（0.396）**：从 `ep000000_abs.png` 看，预测曲线整体压平、
波峰明显低于真值且不跟随原始振荡，符合「输出被常数压缩」的特征。

## 5. 与基线对比

尚未建立基线。建议把 `--compare` 指向上一版权重的中间产物，
对比图底部会自动给出「左臂 / 右臂 / 夹爪 / 均值」的 old→new 变化。

## 6. 局限与风险

- 仅 5 个回合，统计置信度有限；
- 离线 teacher-forcing，未评估闭环累积误差；
- `abs` 与 `delta` 的 MAE 恒等，两者只提供不同曲线视角。

## 7. 结论与建议

**结论**：输出被系统性压平，所有回合误差高度一致，属欠拟合。

**下一步**：
1. 检查训练时的动作归一化 / stats 是否正确加载；
2. 优先修 joint 1、6、7 三维修；
3. 与上一版权重做 `--compare`，确认改动方向。

## 附录

- 图：`runs/figs/ep00000*_{abs,delta}.png`（10 张）
- 指标：`runs/figs/metrics_abs.json`
- 复现：见 B0 的两条命令
```

## B3. 从命令到报告的检查清单

| # | 检查项 | 对应命令 / 文件 |
|---|---|---|
| 1 | 中间产物已生成，回合数对得上 | `ls runs/artifacts/*.npz` |
| 2 | 图已产出，abs/delta 都有 | `ls runs/figs/*.png` |
| 3 | 指标 JSON 已导出 | `runs/figs/metrics_abs.json` |
| 4 | 报告数字全部来自 JSON，未手抄 | `summary.*` 字段 |
| 5 | 已报 mean / median / max 三个数 | `summary.overall.mae` |
| 6 | 已指出最差的 2–3 维 | `summary.per_dim` 排序 |
| 7 | 有对比时说明谁是谁 | `--old-label` / `--new-label` |
| 8 | 局限已写 | — |