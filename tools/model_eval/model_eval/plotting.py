"""整回合轨迹图：逐维子图 + 信息头。

原方法产出一张带信息头的四行排版图：

* 第 0 行：信息头 —— 任务名称、数据地址、权重、曲线类型、曲线口径、横纵坐标说明；
* 第 1-3 行：逐维子图（3 列网格），每维一个子图，pred/gt 全帧曲线 + 推理帧红点；
* 单权重：第 4 行右格放图例；双权重对比：整行放 MAE 对比条 + 图例。

新版评估曲线逐帧保存真值，并把每次推理返回的完整 chunk 拼接到时间轴；
红点标示实际推理帧。旧版稀疏产物仍可读取并按原样绘制。

数据由本项目自有格式提供（见 ``io.py``），本模块不依赖任何外部仓库。
"""

from __future__ import annotations

import pathlib

import numpy as np

from .io import EpisodeData, load_episode

# 与训练 / ActionRelative 一致：夹爪维保持绝对值，不做差分


class DimMode:
    """一种「有多少维、各维是什么」的定义。

    arm_joint8  : 单臂 7 关节 + 1 夹爪（x1_pro_lerobot_arms8，维序 4-10,19）
    arm_joint16 : 双臂 7 关节 + 交叉夹爪（x1_pro_lerobot_dual16）
                  维序为 [左臂7关节, 左夹爪, 右臂7关节, 右夹爪]
    """

    __slots__ = ("name", "labels", "gripper_indices", "columns", "arm_groups", "title")

    def __init__(self, name, labels, gripper_indices, columns, arm_groups, title):
        self.name = name
        self.labels = list(labels)
        self.gripper_indices = list(gripper_indices)
        self.columns = columns
        self.arm_groups = [(label, idx) for label, idx in arm_groups]
        self.title = title

    @property
    def dims(self):
        return len(self.labels)

    @property
    def rows(self):
        return (self.dims + self.columns - 1) // self.columns

    def __repr__(self):
        return f"DimMode({self.name!r}, dims={self.dims})"


_SINGLE = DimMode(
    name="arm_joint8",
    labels=[
        "joint 1",
        "joint 2",
        "joint 3",
        "joint 4",
        "joint 5",
        "joint 6",
        "joint 7",
        "夹爪/gripper",
    ],
    gripper_indices=[7],
    columns=3,
    # 对比模式的 MAE 汇总行用：(组名, 参与平均的维索引)
    arm_groups=[("关节(1-7)", list(range(7)))],
    title="单臂 arm_joint8",
)

_DUAL = DimMode(
    name="arm_joint16",
    labels=[
        "左臂 joint 1",
        "左臂 joint 2",
        "左臂 joint 3",
        "左臂 joint 4",
        "左臂 joint 5",
        "左臂 joint 6",
        "左臂 joint 7",
        "左夹爪/gripper",
        "右臂 joint 1",
        "右臂 joint 2",
        "右臂 joint 3",
        "右臂 joint 4",
        "右臂 joint 5",
        "右臂 joint 6",
        "右臂 joint 7",
        "右夹爪/gripper",
    ],
    gripper_indices=[7, 15],
    columns=4,
    arm_groups=[
        ("左臂(1-7)", list(range(7))),
        ("右臂(1-7)", list(range(8, 15))),
    ],
    title="双臂 arm_joint16",
)

DIM_MODES = {"arm_joint8": _SINGLE, "arm_joint16": _DUAL}
DEFAULT_DIM_MODE = "arm_joint8"

# 向后兼容的旧常量（仅对应 8 维模式）
NON_DELTA_INDICES = list(_SINGLE.gripper_indices)
DIM_LABELS_8 = list(_SINGLE.labels)

__all__ = [
    "render_episode",
    "DimMode",
    "DIM_MODES",
    "DEFAULT_DIM_MODE",
    "resolve_dim_mode",
    "NON_DELTA_INDICES",
    "DIM_LABELS_8",
]


def resolve_dim_mode(mode=None, dims=None):
    """按名字或维数确定 DimMode。

    mode 优先；只给 dims 时按维数反查（8 -> arm_joint8，16 -> arm_joint16）；
    都不给时用默认 8 维。维数与模式不符时报错，避免画出错位的子图标题。
    """
    if mode is not None:
        key = str(mode).strip()
        if key not in DIM_MODES:
            raise ValueError(f"未知的 dim mode: {mode!r}，可选 {sorted(DIM_MODES)}")
        resolved = DIM_MODES[key]
        if dims is not None and int(dims) != resolved.dims:
            raise ValueError(
                f"dim mode {key!r} 需要 {resolved.dims} 维，但数据有 {int(dims)} 维"
            )
        return resolved
    if dims is not None:
        for cand in DIM_MODES.values():
            if cand.dims == int(dims):
                return cand
    if dims is not None:
        raise ValueError(
            f"{int(dims)} 维没有对应的 dim mode，可选 "
            f"{sorted(m.name for m in DIM_MODES.values())}；请显式指定 --dim-mode"
        )
    return DIM_MODES[DEFAULT_DIM_MODE]


def _plot_setup():
    """Matplotlib 延迟初始化（Agg + 中文字体）。

    字体顺序：先真实存在的 CJK 字体；``Droid Sans Fallback`` 只含中日韩字形、
    不含拉丁字母，若排在前面 matplotlib 会为每个拉丁字符递归回退并刷警告，
    因此放在最后兜底。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    from .plot_fonts import register_cjk_fonts

    cjk_families = register_cjk_fonts(font_manager)
    matplotlib.rcParams["font.sans-serif"] = [
        *cjk_families,
        "DejaVu Sans",
        "Droid Sans Fallback",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def _local_relative_delta(a, b, nd=NON_DELTA_INDICES):
    d = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    for i in nd:
        d[..., i] = np.asarray(a, dtype=np.float32)[..., i]
    return d


def _marker_indices(dec, stride):
    """挑选「chunk 起点」红点所在的决策点下标。

    ``stride`` 以**帧**为单位；``dec`` 是决策点的**帧号**数组。
    原实现写成 ``np.arange(0, len(dec), stride)``，把「决策点个数」当帧数用，
    量纲不匹配：决策点数 > stride 时只会剩 1 个红点（chunk 越多越糊）。
    这里改成按帧号取模，chunk_stride=0 时退化为所有决策点。

    返回被选中的下标数组（至少保留首点）。
    """
    dec = np.asarray(dec).reshape(-1)
    if dec.size == 0:
        return np.zeros((0,), dtype=int)
    step = max(1, int(stride))
    sel = np.flatnonzero(dec.astype(np.int64) % step == 0)
    return sel if sel.size else np.zeros((1,), dtype=int)


def _curves(ep, kind):
    """按视图类型取 pred / gt 曲线（abs 原样；delta 用 state_base 现算）。"""
    if kind == "delta":
        return (
            _local_relative_delta(ep.prediction, ep.state_base),
            _local_relative_delta(ep.ground_truth, ep.state_base),
        )
    return ep.prediction, ep.ground_truth


def render_episode(
    source,
    out_png,
    *,
    kind="abs",
    data_root="",
    chunk_stride=0,
    compare=None,
    old_label="",
    new_label="",
    dim_mode=None,
    checkpoint="",
):
    """单权重整回合图 / 双权重并排对比图（四行排版）。

    新版评估产物保存每帧真值和完整预测 chunk 拼接曲线；红点标出实际推理帧。
    旧版稀疏产物仍按其已保存的决策点绘制。

    参数
    ----
    source : 本项目格式的 ``.npz`` 路径，或已加载的 :class:`EpisodeData`。
        格式见 ``model_eval/io.py`` 顶部说明。
    out_png : 输出 PNG 路径（父目录自动创建）。
    kind : ``"abs"``（绝对量）或 ``"delta"``（相对量，夹爪维保持绝对值）。
    data_root : 信息头里显示的数据集路径。
    chunk_stride : 红点间隔（帧）；0 表示按 chunk_size。
    compare : 第二权重（路径或 EpisodeData），给出时画双权重对比图。
    old_label / new_label : 对比时两条曲线的图例文案。
    dim_mode : ``"arm_joint8"``（单臂 8 维，默认）或 ``"arm_joint16"``
        （双臂 16 维）。不给时按数据维数自动选择。
    checkpoint : 信息头里显示的权重名；不给时用文件所在目录名。
    """
    out_png = pathlib.Path(out_png)
    ep = source if isinstance(source, EpisodeData) else load_episode(source)
    cmp_ep = None
    if compare is not None:
        cmp_ep = compare if isinstance(compare, EpisodeData) else load_episode(compare)
    plt = _plot_setup()
    from matplotlib.gridspec import GridSpec

    pred_line, gt_line = _curves(ep, kind)
    dec = ep.decision_points
    L, D = ep.episode_length, ep.dims
    H = ep.chunk_size
    prompt = ep.prompt
    ep_no = ep.episode_id
    stride = int(chunk_stride) if chunk_stride else H
    dense_chunks = ep.metadata.get("curve_mode") == "dense_chunk"
    if dense_chunks:
        inference_points = np.asarray(ep.metadata["inference_points"], dtype=np.int64)
        sel = np.searchsorted(dec, inference_points)
        if np.any(sel >= len(dec)) or not np.array_equal(dec[sel], inference_points):
            raise ValueError("metadata.inference_points 不在曲线帧索引中")
        dec_s = inference_points
        curve_description = (
            f"曲线口径: 逐帧真值 + 完整chunk预测拼接；推理点={len(dec_s)}；"
            f"stride={ep.metadata['stride']}帧；重叠区采用最近chunk"
        )
        point_description = (
            f"回合帧数 = {L}      | 逐帧点 = {len(dec)}      | 推理点 = {len(dec_s)}"
        )
    else:
        sel = _marker_indices(dec, stride)
        dec_s = dec[sel]
        curve_description = f"曲线口径: 旧版稀疏决策点；红点 = 每 {stride} 帧一个"
        point_description = f"回合帧数 = {L}      | 决策点 = {len(dec)}"
    marker_label = (
        f"实际推理帧（共 {len(dec_s)} 个）"
        if dense_chunks
        else f"chunk 起点（每 {stride} 帧，共 {len(dec_s)} 个）"
    )
    kind_cn = "绝对 (absolute)" if kind == "abs" else "增量 (relative delta)"

    cmp_line = None
    if cmp_ep is not None:
        if cmp_ep.dims != D:
            raise ValueError(
                f"对比数据维度 {cmp_ep.dims} 与主数据 {D} 不一致，无法并排画"
            )
        cmp_line, cmp_gt = _curves(cmp_ep, kind)
        if cmp_gt.shape == gt_line.shape and not np.allclose(gt_line, cmp_gt):
            print("⚠ 两份数据的真值不一致，对比图仍会画但真值以第一份为准", flush=True)

    mode = resolve_dim_mode(dim_mode, D)

    mae = np.abs(pred_line - gt_line).mean(axis=0)
    mae_cmp = np.abs(cmp_line - gt_line).mean(axis=0) if cmp_line is not None else None

    # 网格几何随维度模式变化：8 维 3 列 3 行；16 维 4 列 4 行。
    # 图例行单独占一行，不挤占子图格（原版硬编码 8 维时图例会覆盖子图）。
    ncol = mode.columns
    nrows_dims = mode.rows
    nrow = 1 + nrows_dims + 1
    fig = plt.figure(
        figsize=(
            19 if cmp_line is not None else 17.5,
            max(14.5 if cmp_line is None else 17, 2.4 + 2.6 * nrows_dims),
        )
    )
    ratios = [0.52] + [1] * nrows_dims + ([0.42] if cmp_line is not None else [0.30])
    gs = GridSpec(nrow, ncol, figure=fig, height_ratios=ratios, hspace=0.42, wspace=0.2)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    info = [
        f"任务名称 (task):          {prompt}",
        f"数据地址 (dataset root):  {data_root or '(未提供)'}",
        f"权重 (checkpoint):        {checkpoint or (source.parent.name if isinstance(source, (str, pathlib.Path)) else '(内存数据)')}",
        f"曲线类型 (curve type):    {kind_cn}      |  chunk = {H} 步      |  "
        f"{point_description}      |  指令 = 逐回合任务 (prompt-mode episode)",
        curve_description,
        f"横坐标 (x): eps 帧序号（单位：帧），0 ~ {L - 1}（本回合最大帧数为上限）      |  "
        f"纵坐标 (y): {'state 数值' if kind == 'abs' else 'delta 数值'}",
        f"维度模式 (dim mode):     {mode.name}（{mode.title}，共 {mode.dims} 维，"
        f"夹爪维 = {mode.gripper_indices}）",
    ]
    ax.text(
        0.01,
        0.95,
        "\n".join(info),
        va="top",
        ha="left",
        fontsize=11.5,
        transform=ax.transAxes,
    )
    ax.set_title(
        f"episode {ep_no} — trajectory eval ({kind})"
        + ("   对比两权重" if cmp_line is not None else ""),
        fontsize=14,
        pad=6,
    )

    for d in range(D):
        r = 1 + d // ncol
        c = d % ncol
        axx = fig.add_subplot(gs[r, c])
        axx.plot(
            dec,
            gt_line[:, d],
            color="tab:orange",
            lw=1.7,
            ls="--",
            label="gt",
            zorder=4,
        )
        axx.plot(
            dec,
            pred_line[:, d],
            color="tab:blue",
            lw=1.4,
            label=(old_label or "pred"),
            zorder=3,
        )
        if cmp_line is not None:
            axx.plot(
                dec,
                cmp_line[:, d],
                color="tab:green",
                lw=1.4,
                label=(new_label or "compare"),
                zorder=3,
            )
        marker_line = cmp_line if cmp_line is not None else pred_line
        axx.plot(
            dec_s,
            marker_line[sel, d],
            "o",
            color="red",
            ms=4.2,
            zorder=5,
            label="推理点"
            if dense_chunks and d == 0
            else "chunk start"
            if d == 0
            else None,
        )
        for t0 in dec_s[1:]:
            axx.axvline(int(t0), color="red", lw=0.7, ls=":", alpha=0.3, zorder=0)
        nm = mode.labels[d] if d < mode.dims else f"dim {d + 1}"
        ttl = f"{nm} [mae {mae[d]:.5f}]"
        if mae_cmp is not None:
            ttl = f"{nm}   [mae old {mae[d]:.5f} | new {mae_cmp[d]:.5f}]"
        axx.set_title(ttl, fontsize=10)
        axx.set_xlim(-1, L - 1)
        axx.set_xlabel("eps 帧序号 (frame index, 单位: 帧)", fontsize=8.5)
        axx.set_ylabel("state" if kind == "abs" else "delta", fontsize=8.5)
        axx.grid(alpha=0.25, lw=0.5)
        axx.tick_params(labelsize=8)
        axx.ticklabel_format(useOffset=False, style="plain")
        if d == 0:
            axx.legend(fontsize=7.5, loc="best")

    if cmp_line is not None:
        axl = fig.add_subplot(gs[nrow - 1, :])
        axl.axis("off")
        # 汇总行按维度模式生成：单臂一组「关节(1-7)」；双臂两组「左臂/右臂」
        parts = []
        for group_name, idx in mode.arm_groups:
            o = float(np.mean(mae[idx]))
            n_ = float(np.mean(mae_cmp[idx]))
            parts.append(f"{group_name}: old {o:.5f}  →  new {n_:.5f}")
        grip = []
        for gi in mode.gripper_indices:
            tag = "夹爪" if mode.dims == 8 else ("左夹爪" if gi == 7 else "右夹爪")
            grip.append(f"{tag}: old {mae[gi]:.4f}  →  new {mae_cmp[gi]:.4f}")
        parts.append("   |   ".join(grip))
        parts.append(
            f"{mode.dims} 维均值: old {mae.mean():.5f}  →  new {mae_cmp.mean():.5f}"
        )
        axl.text(
            0.01,
            0.82,
            "整回合 MAE  |  " + "      |      ".join(parts),
            va="top",
            ha="left",
            fontsize=11.5,
            transform=axl.transAxes,
        )
        axl.legend(
            handles=[
                plt.Line2D(
                    [], [], color="tab:orange", lw=1.7, ls="--", label="gt（真值）"
                ),
                plt.Line2D([], [], color="tab:blue", lw=1.4, label=old_label or "pred"),
                plt.Line2D(
                    [], [], color="tab:green", lw=1.4, label=new_label or "compare"
                ),
                plt.Line2D(
                    [],
                    [],
                    color="red",
                    marker="o",
                    ls="",
                    ms=4.5,
                    label=marker_label,
                ),
            ],
            fontsize=9.5,
            loc="lower center",
            ncol=4,
            frameon=True,
        )
    else:
        # 图例独占最后一行（不再抢占子图格，否则 5-14 维时图例会盖掉子图）
        ax_leg = fig.add_subplot(gs[nrow - 1, :])
        ax_leg.axis("off")
        ax_leg.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    color="tab:orange",
                    lw=1.7,
                    ls="--",
                    label="gt（真值，整回合）",
                ),
                plt.Line2D(
                    [],
                    [],
                    color="tab:blue",
                    lw=1.4,
                    label="pred（逐帧完整 chunk 拼接）",
                ),
                plt.Line2D(
                    [],
                    [],
                    color="red",
                    marker="o",
                    ls="",
                    ms=5,
                    label=f"实际推理点（共 {len(dec_s)} 个）",
                ),
            ],
            fontsize=9.5,
            loc="center",
            ncol=3,
            frameon=True,
            title="图例 (legend)",
            title_fontsize=10,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return out_png
