"""整回合轨迹对比图：评估 → 中间产物 → 绘图。

典型流程::

    from model_eval import Dataset, Episode, evaluate, render_episode

    结果 = evaluate(model, dataset, "runs/artifacts")
    render_episode("runs/artifacts/ep000000.npz", "figs/ep0.png")
"""

from .builders import build_from_arrays, build_from_chunks
from .evaluate import Dataset, Episode, evaluate
from .io import REQUIRED_KEYS, EpisodeData, load_episode, save_episode
from .plotting import (
    DEFAULT_DIM_MODE,
    DIM_LABELS_8,
    DIM_MODES,
    NON_DELTA_INDICES,
    DimMode,
    render_episode,
    resolve_dim_mode,
)

__all__ = [
    # 评估
    "evaluate", "Dataset", "Episode",
    # 数据
    "EpisodeData", "load_episode", "save_episode",
    "build_from_arrays", "build_from_chunks", "REQUIRED_KEYS",
    # 绘图
    "render_episode", "DimMode", "DIM_MODES", "DEFAULT_DIM_MODE",
    "resolve_dim_mode", "NON_DELTA_INDICES", "DIM_LABELS_8",
]
