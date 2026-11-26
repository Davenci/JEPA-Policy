"""逐回合指标计算与导出。

图上只写了 MAE；本模块把**结构化指标**算出来并落盘，供报告直接引用，
避免报告里手抄数字或另算一套。

每维都有 MAE / RMSE / 相关系数，并按维度模式分成手臂组和夹爪组。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .io import EpisodeData, load_episode
from .plotting import _curves, resolve_dim_mode

__all__ = ["compute_episode_metrics", "summarize", "export_metrics"]


def _corr(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    v = float(np.corrcoef(a, b)[0, 1])
    return v if np.isfinite(v) else None


def _per_dim(pred, gt):
    err = pred - gt
    out = []
    for d in range(pred.shape[1]):
        mae = float(np.mean(np.abs(err[:, d])))
        rmse = float(np.sqrt(np.mean(err[:, d] ** 2)))
        out.append(
            {
                "mae": mae,
                "rmse": rmse,
                "max_abs_err": float(np.max(np.abs(err[:, d]))),
                "corr": _corr(pred[:, d], gt[:, d]),
            }
        )
    return out


def _group(per_dim, indices):
    idx = list(indices)
    if not idx:
        return None
    corrs = [per_dim[i]["corr"] for i in idx if per_dim[i]["corr"] is not None]
    return {
        "mae": float(np.mean([per_dim[i]["mae"] for i in idx])),
        "rmse": float(np.mean([per_dim[i]["rmse"] for i in idx])),
        "mean_corr": float(np.mean(corrs)) if corrs else None,
    }


def compute_episode_metrics(ep, *, kind="abs", dim_mode=None):
    """算一个回合的逐维与分组指标。

    .. note::
       **abs 与 delta 的 MAE / RMSE 恒等**。因为
       (pred - base) - (gt - base) == pred - gt，均值也相同；
       夹爪维虽保持绝对值，但两侧都保持，差值仍是 pred - gt。
       两个视图的价值在于**曲线形状**（delta 能看出增量尺度对不对），
       不在于误差数值。提供 kind 参数只为让报告能如实写明这一点，
       不要因为两个数一样就以为算错了。
    """
    if not isinstance(ep, EpisodeData):
        ep = load_episode(ep)
    mode = resolve_dim_mode(dim_mode, ep.dims)
    pred, gt = _curves(ep, kind)
    per_dim = _per_dim(pred, gt)
    arm_idx = [d for d in range(ep.dims) if d not in mode.gripper_indices]
    groups = {
        "arms": _group(per_dim, arm_idx),
        "grippers": _group(per_dim, mode.gripper_indices),
        "all": _group(per_dim, range(ep.dims)),
    }
    named = {}
    for gname, idx in mode.arm_groups:
        named[gname] = _group(per_dim, idx)
    return {
        "episode_id": ep.episode_id,
        "kind": kind,
        "dim_mode": mode.name,
        "dims": ep.dims,
        "num_decision_points": int(
            len(ep.metadata.get("inference_points", ep.decision_points))
        ),
        "num_curve_points": int(len(ep.decision_points)),
        "episode_length": ep.episode_length,
        "chunk_size": ep.chunk_size,
        "prompt": ep.prompt,
        "overall": groups["all"],
        "arms": groups["arms"],
        "grippers": groups["grippers"],
        "arm_groups": named,
        "per_dim": [
            {
                "index": d,
                "label": mode.labels[d] if d < mode.dims else f"dim {d + 1}",
                "is_gripper": d in mode.gripper_indices,
                **per_dim[d],
            }
            for d in range(ep.dims)
        ],
    }


def summarize(episodes_metrics):
    """把多个回合的指标汇总成 mean / median / min / max。"""
    if not episodes_metrics:
        return None
    keys = ["mae", "rmse"]
    out = {"num_episodes": len(episodes_metrics)}
    for scope in ("overall", "arms", "grippers"):
        block = {}
        for k in keys:
            vals = [m[scope][k] for m in episodes_metrics if m.get(scope)]
            if vals:
                block[k] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        corrs = [
            m[scope]["mean_corr"]
            for m in episodes_metrics
            if m.get(scope) and m[scope]["mean_corr"] is not None
        ]
        if corrs:
            block["mean_corr"] = {"mean": float(np.mean(corrs))}
        out[scope] = block
    per_dim_labels = [d["label"] for d in episodes_metrics[0]["per_dim"]]
    per_dim = []
    for i, label in enumerate(per_dim_labels):
        maes = [m["per_dim"][i]["mae"] for m in episodes_metrics]
        per_dim.append(
            {
                "label": label,
                "is_gripper": episodes_metrics[0]["per_dim"][i]["is_gripper"],
                "mae_mean": float(np.mean(maes)),
                "mae_max": float(np.max(maes)),
            }
        )
    out["per_dim"] = per_dim
    return out


def export_metrics(episodes, out_path, *, kind="abs", dim_mode=None):
    """算一批回合的指标并写成 JSON。episodes 是路径列表或 EpisodeData 列表。"""
    per_ep = [
        compute_episode_metrics(e, kind=kind, dim_mode=dim_mode) for e in episodes
    ]
    payload = {
        "kind": kind,
        "dim_mode": per_ep[0]["dim_mode"] if per_ep else None,
        "episodes": per_ep,
        "summary": summarize(per_ep),
    }
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, payload
