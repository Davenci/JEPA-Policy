"""从用户自己的数组构建输入文件（本项目格式）。

``model_eval`` 只负责画图；输入有两种来源：

1. **本项目内置构建器**（本模块）：你手上已有 ``prediction`` / ``ground_truth``
   两个 [T, D] 数组时，一行调用即可写出合规 npz。
2. **自己按 ``io.py`` 的键名写 npz**：任何能写 numpy 的流程都行，不依赖本项目代码。

下面这个函数覆盖最常见的情况：你有一段真值 ``ground_truth`` [L, D]，
每隔 ``stride`` 帧取一个决策点，模型在每个决策点给出 H 步预测。
"""

from __future__ import annotations

import pathlib

import numpy as np

from .io import EpisodeData, save_episode

__all__ = ["build_from_arrays", "build_from_chunks"]


def build_from_arrays(
    prediction,
    ground_truth,
    out_path,
    *,
    decision_points=None,
    chunk_size=None,
    episode_length=None,
    state_base=None,
    prompt="",
    episode_id="",
    metadata=None,
):
    """用现成的 [T, D] 预测与真值直接生成输入文件。

    prediction / ground_truth 的每一行对应一个**决策点**的取值
    （通常取该决策点上 chunk 的第 0 步）。

    Parameters
    ----------
    prediction, ground_truth : [T, D] 绝对量数组
    out_path : 输出的 .npz 路径
    decision_points : [T] 帧号；缺省假设第 i 个决策点在第 i*stride 帧
    chunk_size : 训练时的预测步长 H；缺省取 len(prediction)
    episode_length : 整回合帧数 L；缺省取最后一个决策点 + chunk_size
    state_base : [T, D] delta 视图锚点；缺省用 ground_truth 上一帧
    prompt : 任务描述
    episode_id : 回合标识；缺省用文件名

    Returns
    -------
    pathlib.Path 写出的文件路径
    """
    pred = np.asarray(prediction, dtype=np.float32)
    gt = np.asarray(ground_truth, dtype=np.float32)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("prediction / ground_truth 必须是 [T, D] 二维数组")
    if pred.shape != gt.shape:
        raise ValueError(f"形状不一致：{pred.shape} vs {gt.shape}")
    n, d = pred.shape
    h = int(chunk_size) if chunk_size else n
    dp = np.arange(n, dtype=np.int64) * h if decision_points is None else np.asarray(decision_points).reshape(-1).astype(np.int64)
    length = int(episode_length) if episode_length else int(dp[-1]) + h
    if state_base is None:
        base = np.empty_like(gt)
        base[0] = gt[0]
        if n > 1:
            base[1:] = gt[:-1]
    else:
        base = np.asarray(state_base, dtype=np.float32)
    ep = EpisodeData(
        prediction=pred, ground_truth=gt, state_base=base,
        decision_points=dp, episode_length=length, chunk_size=h,
        prompt=prompt, episode_id=episode_id or pathlib.Path(out_path).stem,
        metadata=metadata or {},
    )
    return save_episode(ep, pathlib.Path(out_path))


def build_from_chunks(
    chunks,
    ground_truth,
    out_path,
    *,
    decision_points,
    take=0,
    prompt="",
    episode_id="",
):
    """从完整的 chunk 张量构建输入。

    Parameters
    ----------
    chunks : [T, H, D] 每个决策点的完整预测 chunk
    ground_truth : [T, D] 决策点上的真值
    decision_points : [T] 决策点帧号
    take : 取 chunk 的第几步做主曲线（默认 0，即 chunk 首步）
    """
    ch = np.asarray(chunks, dtype=np.float32)
    if ch.ndim != 3:
        raise ValueError("chunks 必须是 [T, H, D] 三维数组")
    if not 0 <= int(take) < ch.shape[1]:
        raise ValueError(f"take={take} 超出 chunk 长度 {ch.shape[1]}")
    return build_from_arrays(
        ch[:, int(take), :], ground_truth, out_path,
        decision_points=decision_points, chunk_size=ch.shape[1],
        prompt=prompt, episode_id=episode_id,
    )
