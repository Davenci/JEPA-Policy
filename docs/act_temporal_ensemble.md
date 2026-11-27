# ACT 动作时间聚合

保留原版 ACT 的逐帧时间聚合，模型权重、结构及预测窗口保持原样。

## 聚合规则

- 每个控制时刻重新预测完整未来动作，只执行当前聚合后的一步。
- 保留所有仍覆盖当前绝对时刻的预测，按生成顺序从旧到新编号。
- 权重使用原作者 [ACT 源码](https://github.com/tonyzhaozh/act/blob/76cf30b4fed1d72dafbc3e1c270c0839d57e8bcf/imitate_episodes.py#L191-L259) 的指数公式：

$$
w_i=\frac{\exp(-k i)}{\sum_{j=0}^{n-1}\exp(-k j)},\qquad
\bar a_t=\sum_{i=0}^{n-1}w_i a_t^{(i)}
$$

默认 $k=0.01$，序号零对应最旧预测，因此较早预测的权重略高。
合法零动作参与聚合；过期预测自动移除；每个新回合清空历史。
此处不保留相邻四步动作块的融合模式。

## JEPA 时间对齐

观测为 $[t-1,t]$，网络输出十步，索引零是历史对齐点。
保留输出 `[1:10]` 的全部九个当前及未来动作，逐帧执行聚合后的一步。
启用时必须设置 `--chunk-size 1`；关闭聚合仍可选择原来的四步或八步执行。

## 离线评估

目标主机：DSW `jepa`；目录：`/mnt/data/usr/yuanyh/new_test`。

```bash
cd /mnt/data/usr/yuanyh/new_test && python3 examples/evaluate_x1_icedtea.py \
  --dataset /mnt/data/EIC01/vla/x1_pro/train/CeMAT_12task_2160eps_20261024_smoothed \
  --episode-indices /mnt/data/usr/yuanyh/runs/icedtea_holdout190_10_10k_20261127_091124/test_episode_indices.json \
  --checkpoint /mnt/data/usr/yuanyh/runs/icedtea_holdout190_10_10k_20261127_manual/models/model_step_065000.pt \
  --max-episodes 3 \
  --chunk-size 1 \
  --temporal-ensemble \
  --temporal-ensemble-k 0.01
```

结果写入仓库 `output/eval/YYYYMMDD_HHMM/`。
对照评估移除两个聚合参数，保留 `--chunk-size 1`，以保持推理频率相同。

## 可复用接口

```python
from jepa_policy.temporal_ensemble import ActTemporalEnsembler

ensemble = ActTemporalEnsembler(prediction_horizon=9, action_dim=8, decay=0.01)
ensemble.reset()  # 每个新回合或重新接管前调用
action = ensemble.update(full9)  # full9: [9,8]；返回当前一步 [8]
```

每个控制时刻调用一次，预测起点必须对齐当前时刻。
保持八个动作维度的名称、单位和坐标顺序一致；聚合器仅处理动作数组。
当前实现完成软件和离线验证，没有连接机器人或发送控制命令。
轨迹平滑程度及离线误差不能直接代表真机任务成功率。
