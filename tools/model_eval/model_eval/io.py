"""本项目自有的评估数据格式（不依赖任何外部仓库）。

定义 ``EpisodeData`` 与 ``.npz`` 读写。所有键名、结构都由本项目规定：

=============  ==================  ==========================================
键             形状                含义
=============  ==================  ==========================================
prediction     [T, D]              模型预测（绝对量；评估产物为逐帧拼接）
ground_truth   [T, D]              对齐真值（绝对量）
state_base     [T, D]              delta 视图锚点（通常为前一帧真值）
decision_points[T]                 每行对应的时间帧坐标，严格递增
episode_length (T = 1)             整回合总帧数
chunk_size     (T = 1)             训练时的预测步长 H
prompt         ()                  任务描述文本
episode_id     ()                  回合标识，缺省用文件名
=============  ==================  ==========================================

可选扩展键（本工具用不到时可省略）：metadata 为一段 JSON 字符串。
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

REQUIRED_KEYS = (
    "prediction",
    "ground_truth",
    "state_base",
    "decision_points",
    "episode_length",
    "chunk_size",
    "prompt",
)

# 判断哪些维是「夹爪维」时用不到，但便于用户排查
OPTIONAL_KEYS = ("episode_id", "metadata")

__all__ = [
    "EpisodeData",
    "load_episode",
    "save_episode",
    "REQUIRED_KEYS",
    "OPTIONAL_KEYS",
]


@dataclass(slots=True)
class EpisodeData:
    """一个回合的绘图所需数据。

    prediction / ground_truth 为**绝对量**。评估器产物保存全帧曲线；其他构建器
    也可保存稀疏帧点。delta 视图由 ``state_base`` 现算，因此同一份文件可以画两种视图。
    """

    prediction: np.ndarray
    ground_truth: np.ndarray
    state_base: np.ndarray
    decision_points: np.ndarray
    episode_length: int
    chunk_size: int
    prompt: str
    episode_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.prediction = np.asarray(self.prediction, dtype=np.float32)
        self.ground_truth = np.asarray(self.ground_truth, dtype=np.float32)
        self.state_base = np.asarray(self.state_base, dtype=np.float32)
        self.decision_points = (
            np.asarray(self.decision_points).reshape(-1).astype(np.int64)
        )
        self.episode_length = int(self.episode_length)
        self.chunk_size = int(self.chunk_size)
        self.prompt = str(self.prompt)
        self.episode_id = str(self.episode_id)
        _validate(self)

    @property
    def dims(self):
        return int(self.prediction.shape[1])


def _validate(ep):
    for name in ("prediction", "ground_truth", "state_base"):
        arr = getattr(ep, name)
        if arr.ndim != 2:
            raise ValueError(f"{name} 必须是二维 [T, D]，实际 {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} 含有 NaN/Inf")
    shapes = {ep.prediction.shape, ep.ground_truth.shape, ep.state_base.shape}
    if len(shapes) != 1:
        raise ValueError(
            "prediction / ground_truth / state_base 形状必须一致，实际 "
            f"{ep.prediction.shape} / {ep.ground_truth.shape} / {ep.state_base.shape}"
        )
    n = ep.prediction.shape[0]
    if n == 0 or ep.prediction.shape[1] == 0:
        raise ValueError("数据不能为空（至少 1 个帧点、1 个维度）")
    if ep.decision_points.size != n:
        raise ValueError(
            f"decision_points 长度 {ep.decision_points.size} 与数据行数 {n} 不一致"
        )
    if n > 1 and np.any(np.diff(ep.decision_points) <= 0):
        raise ValueError("decision_points 必须严格递增")
    if ep.decision_points[0] < 0:
        raise ValueError("decision_points 不能为负")
    if ep.chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if ep.episode_length <= 0:
        raise ValueError("episode_length 必须为正整数")
    if ep.decision_points[-1] >= ep.episode_length:
        raise ValueError(
            f"最大决策点 {int(ep.decision_points[-1])} 超出回合长度 {ep.episode_length}"
        )


def load_episode(path):
    """读取本项目格式的 ``.npz``。缺键会给出明确的报错。"""
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        missing = [k for k in REQUIRED_KEYS if k not in files]
        if missing:
            raise ValueError(
                f"{path.name} 缺少必需键 {missing}；"
                f"本工具要求格式见 model_eval/io.py 顶部说明。"
                f"现有键：{sorted(files)}"
            )
        raw = {k: z[k] for k in REQUIRED_KEYS if k in files}
        raw["episode_id"] = str(z["episode_id"]) if "episode_id" in files else ""
        meta = {}
        if "metadata" in files:
            try:
                meta = json.loads(str(z["metadata"]))
            except json.JSONDecodeError:
                meta = {}
    if not raw["episode_id"]:
        # 缺省用文件名（去扩展名）作回合标识
        raw["episode_id"] = path.stem
    return EpisodeData(
        prediction=raw["prediction"],
        ground_truth=raw["ground_truth"],
        state_base=raw["state_base"],
        decision_points=raw["decision_points"],
        episode_length=raw["episode_length"],
        chunk_size=raw["chunk_size"],
        prompt=raw["prompt"],
        episode_id=raw["episode_id"],
        metadata=meta,
    )


def save_episode(ep, path):
    """按本项目格式写出 ``.npz``（供用户/脚本生成输入文件）。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        prediction=np.asarray(ep.prediction, dtype=np.float32),
        ground_truth=np.asarray(ep.ground_truth, dtype=np.float32),
        state_base=np.asarray(ep.state_base, dtype=np.float32),
        decision_points=np.asarray(ep.decision_points, dtype=np.int64),
        episode_length=np.asarray(int(ep.episode_length)),
        chunk_size=np.asarray(int(ep.chunk_size)),
        prompt=np.asarray(str(ep.prompt)),
        episode_id=np.asarray(str(ep.episode_id or path.stem)),
        metadata=np.asarray(json.dumps(ep.metadata or {}, ensure_ascii=False)),
    )
    return path
