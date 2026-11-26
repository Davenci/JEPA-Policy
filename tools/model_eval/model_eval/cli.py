"""命令行入口：评估（eval）与绘图（plot）。

``eval`` 需要用户提供一个 Python 文件，里面导出 ``build_dataset() -> Dataset`` 与
``build_model() -> 模型对象``。这样本工具不需要知道用户的模型和数据集怎么加载。
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib

from .evaluate import Dataset, evaluate
from .io import REQUIRED_KEYS
from .metrics import export_metrics
from .plotting import DIM_MODES, render_episode


def _load_user_module(path, required):
    """导入用户提供的 Python 文件，并校验必需的可调用对象。"""
    path = pathlib.Path(path)
    if not path.is_file():
        raise SystemExit(f"找不到脚本：{path}")
    spec = importlib.util.spec_from_file_location("_model_eval_user", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [n for n in required if not callable(getattr(module, n, None))]
    if missing:
        raise SystemExit(f"{path.name} 必须导出可调用的 {missing}")
    return module


def _find_episodes(root):
    """在目录里找本工具格式的 npz，返回 {stem: 路径}。"""
    root = pathlib.Path(root)
    if root.is_file():
        return {root.stem: root}
    if not root.is_dir():
        raise SystemExit(f"输入路径不存在：{root}")
    found, skipped = {}, []
    import numpy as np

    for p in sorted(root.rglob("*.npz")):
        try:
            with np.load(p, allow_pickle=False) as z:
                ok = all(k in z.files for k in REQUIRED_KEYS)
        except Exception:
            ok = False
        (found.__setitem__(p.stem, p) if ok else skipped.append(p))
    if not found:
        raise SystemExit(
            f"{root} 下没有本工具格式的 .npz（需含 {list(REQUIRED_KEYS)}）"
        )
    if skipped:
        print(
            f"  跳过 {len(skipped)} 个格式不符的 npz（如 {skipped[0].name}）",
            flush=True,
        )
    return found


def build_parser():
    parser = argparse.ArgumentParser(
        prog="model_eval",
        description="整回合轨迹对比图：eval 产出中间产物，plot 出图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="跑模型评估，产出中间产物")
    ev.add_argument(
        "--script",
        required=True,
        help="用户脚本，需导出 build_dataset() 与 build_model()",
    )
    ev.add_argument("--out-dir", required=True, help="中间产物输出目录")
    ev.add_argument(
        "--stride",
        type=int,
        default=None,
        help="重规划间隔（帧），缺省 = 模型 chunk_size",
    )
    ev.add_argument("--on-error", choices=["fail", "skip"], default="fail")
    ev.add_argument("--plot-dir", default=None, help="评估完直接出图到该目录")
    ev.add_argument("--kind", choices=["abs", "delta", "both"], default="both")
    ev.add_argument("--dim-mode", choices=sorted(DIM_MODES), default=None)
    ev.add_argument("--data-root", default="", help="评估数据集路径，并显示在图上")
    ev.add_argument("--checkpoint", default="", help="评估权重路径，并显示在图上")
    ev.add_argument(
        "--train-output", default="", help="训练运行目录，包含 dataset_metadata.json"
    )
    ev.add_argument("--device", default="cuda:0", help="模型推理设备")
    ev.add_argument("--task-name", default="", help="精确匹配的数据集任务名")
    ev.add_argument(
        "--max-episodes", type=int, default=None, help="最多评估前 N 个回合"
    )
    ev.add_argument(
        "--metrics-out",
        default=None,
        help="指标 JSON 路径（缺省写 <out-dir>/metrics_<kind>.json）",
    )

    pl = sub.add_parser("plot", help="用中间产物出图")
    pl.add_argument("--arrays", required=True, help="中间产物目录或单个 npz")
    pl.add_argument("--compare", default=None, help="对比权重的产物目录")
    pl.add_argument("--out-dir", required=True)
    pl.add_argument("--episodes", nargs="+", default=None)
    pl.add_argument("--kind", choices=["abs", "delta", "both"], default="both")
    pl.add_argument("--dim-mode", choices=sorted(DIM_MODES), default=None)
    pl.add_argument("--data-root", default="")
    pl.add_argument("--checkpoint", default="")
    pl.add_argument("--chunk-stride", type=int, default=0)
    pl.add_argument("--old-label", default="pred")
    pl.add_argument("--new-label", default="compare")
    pl.add_argument(
        "--metrics",
        action="store_true",
        help="导出指标 JSON（<out-dir>/metrics_<kind>.json）",
    )
    pl.add_argument("--metrics-out", default=None, help="指标 JSON 路径")
    return parser


def cmd_eval(args):
    module = _load_user_module(args.script, ["build_dataset", "build_model"])
    print("构建数据集与模型...", flush=True)
    configure = getattr(module, "configure", None)
    if callable(configure):
        configure(args)
    dataset = module.build_dataset()
    if not isinstance(dataset, Dataset):
        raise SystemExit("build_dataset() 必须返回 model_eval.Dataset")
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes 必须是正整数")
        dataset = Dataset(
            dataset.episodes[: args.max_episodes],
            name=dataset.name,
        )
    model = module.build_model()
    print(
        f"数据集 {dataset.name or '(未命名)'}：{len(dataset.episodes)} 个回合",
        flush=True,
    )
    print(f"开始评估 -> {args.out_dir}", flush=True)
    results = evaluate(
        model, dataset, args.out_dir, stride=args.stride, on_error=args.on_error
    )
    print(f"\n评估完成：{len(results)} 个回合 -> {args.out_dir}", flush=True)
    kinds = ["abs", "delta"] if args.kind == "both" else [args.kind]
    if args.metrics_out:
        _export(
            args.out_dir,
            pathlib.Path(args.out_dir),
            kinds,
            dim_mode=args.dim_mode,
            metrics_out=args.metrics_out,
        )
    if args.plot_dir:
        _plot(
            args.out_dir,
            args.plot_dir,
            kind=args.kind,
            compare=None,
            dim_mode=args.dim_mode,
            data_root=args.data_root or dataset.name,
            checkpoint=args.checkpoint,
            episodes=None,
            chunk_stride=0,
            old_label="pred",
            new_label="compare",
        )
    else:
        print(
            f"下一步出图：python3 -m model_eval plot --arrays {args.out_dir} "
            f"--out-dir <图目录>",
            flush=True,
        )


def _plot(
    arrays,
    out_dir,
    *,
    kind,
    compare,
    dim_mode,
    data_root,
    checkpoint,
    episodes,
    chunk_stride,
    old_label,
    new_label,
):
    main_map = _find_episodes(arrays)
    cmp_map = _find_episodes(compare) if compare else {}
    eps = episodes if episodes else sorted(main_map)
    kinds = ["abs", "delta"] if kind == "both" else [kind]
    out_dir = pathlib.Path(out_dir)
    n = 0
    for ep_id in eps:
        if ep_id not in main_map:
            print(f"  跳过 {ep_id}：无对应 npz", flush=True)
            continue
        for k in kinds:
            suffix = f"compare_{ep_id}_{k}.png" if compare else f"{ep_id}_{k}.png"
            render_episode(
                main_map[ep_id],
                out_dir / suffix,
                kind=k,
                data_root=data_root,
                chunk_stride=chunk_stride,
                compare=cmp_map.get(ep_id) if compare else None,
                old_label=old_label,
                new_label=new_label,
                dim_mode=dim_mode,
                checkpoint=checkpoint,
            )
            n += 1
            print(f"  {out_dir / suffix}", flush=True)
    print(f"出图完成：{n} 张 -> {out_dir}", flush=True)


def _export(arrays, out_dir, kinds, *, dim_mode, metrics_out=None):
    """算指标并落盘，报告可直接引用。"""
    main_map = _find_episodes(arrays)
    eps = sorted(main_map)
    for k in kinds:
        dest = out_dir / f"metrics_{k}.json"
        if metrics_out:
            mp = pathlib.Path(metrics_out)
            dest = mp if len(kinds) == 1 else mp.with_name(f"{mp.stem}_{k}.json")
        path, payload = export_metrics(
            [main_map[e] for e in eps], dest, kind=k, dim_mode=dim_mode
        )
        s = payload["summary"]
        if s:
            o = s["overall"]["mae"]
            print(
                f"  指标[{k}] {s['num_episodes']} 回合  "
                f"MAE mean={o['mean']:.5f} median={o['median']:.5f} "
                f"max={o['max']:.5f} -> {path}",
                flush=True,
            )
    return dest


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "eval":
        cmd_eval(args)
    else:
        _plot(
            args.arrays,
            args.out_dir,
            kind=args.kind,
            compare=args.compare,
            dim_mode=args.dim_mode,
            data_root=args.data_root,
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            chunk_stride=args.chunk_stride,
            old_label=args.old_label,
            new_label=args.new_label,
        )
        if args.metrics or args.metrics_out:
            kinds = ["abs", "delta"] if args.kind == "both" else [args.kind]
            _export(
                args.arrays,
                pathlib.Path(args.out_dir),
                kinds,
                dim_mode=args.dim_mode,
                metrics_out=args.metrics_out,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
