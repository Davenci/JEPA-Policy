"""生成 port_test_multi/gallery.html（单文件图库）。"""
import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "port_test_multi"
F = ROOT / "figures"
OUT = ROOT / "gallery.html"

SECTIONS = [
    ("单臂 arm_joint8 · 4 个决策点", "3 列 × 3 行网格",
     [("single_abs/ep000000_abs.png", "abs"), ("single_delta/ep000000_delta.png", "delta")]),
    ("单臂 arm_joint8 · 12 个决策点", "红点按 chunk 起点铺满",
     [("single_abs/ep000001_abs.png", "abs")]),
    ("单臂 arm_joint8 · 30 个决策点（chunk=20）", "高密度重规划",
     [("single_abs/ep000003_abs.png", "abs")]),
    ("单臂 arm_joint8 · 40 个决策点（bad 模型）", "劣化模型",
     [("single_abs/ep000004_abs.png", "abs")]),
    ("单臂 · 双权重对比", "蓝=arrays，绿=compare，底部 MAE 对比行",
     [("compare_abs/compare_ep000001_abs.png", "12 chunk")]),
    ("双臂 arm_joint16 · abs", "4 列 × 4 行，16 个子图",
     [("dual_abs/ep000000_abs.png", "12 chunk"),
      ("dual_abs/ep000001_abs.png", "20 chunk")]),
    ("双臂 arm_joint16 · delta", "夹爪维 7 和 15 保持绝对值",
     [("dual_delta/ep000000_delta.png", "12 chunk")]),
    ("双臂 arm_joint16 · 双权重对比", "MAE 分左臂 / 右臂 / 左夹爪 / 右夹爪 / 16 维均值",
     [("dual_compare/compare_ep000000_abs.png", "12 chunk")]),
]


def b64(p):
    return base64.b64encode(p.read_bytes()).decode()


cards = []
for title, desc, items in SECTIONS:
    imgs = []
    for rel, cap in items:
        p = F / rel
        if not p.exists():
            imgs.append(f'<p style="color:#f85149">缺失: {rel}</p>')
            continue
        imgs.append(
            f'<figure><img src="data:image/png;base64,{b64(p)}" alt="{rel}">'
            f'<figcaption>{cap} · {rel}</figcaption></figure>'
        )
    cards.append(f'<section><h2>{title}</h2><p class="d">{desc}</p>{"".join(imgs)}</section>')

DOC = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>model_eval 示例图</title><style>
body{{background:#0d1117;color:#c9d1d9;font:14px/1.6 -apple-system,"Noto Sans CJK SC",sans-serif;
margin:0;padding:24px 32px}}
h1{{color:#e6edf3;font-size:22px;border-bottom:1px solid #30363d;padding-bottom:10px}}
h2{{color:#58a6ff;font-size:15px;margin:0 0 4px}}
section{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 18px;margin:18px 0}}
.d{{color:#8b949e;margin:0 0 12px;font-size:13px}}
img{{max-width:100%;border:1px solid #30363d;border-radius:4px;background:#fff;display:block}}
figcaption{{color:#6e7681;font-size:12px;margin:6px 0 14px;font-family:ui-monospace,monospace}}
</style></head><body><h1>model_eval 示例图（本项目自带格式）</h1>{"".join(cards)}</body></html>"""

OUT.write_text(DOC, encoding="utf-8")
print(f"wrote {OUT}  {OUT.stat().st_size // 1024} KB")