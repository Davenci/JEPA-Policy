"""生成模拟测试数据与出图（写入 port_test_multi/）。

用法:  python scripts/gen_test_figures.py
"""
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "port_test_multi" / "data"
FIGS = ROOT / "port_test_multi" / "figures"
for d in (DATA, FIGS):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from model_eval.builders import build_from_arrays  # noqa: E402

PROMPTS = [
    "pick up the water bottle and place it on the shelf",
    "grasp the red cup and move it to the left tray",
    "open the drawer and take out the screwdriver",
    "stack the blue block on top of the green block",
    "insert the plug into the socket on the wall",
]


def traj(points, chunk, dims=8, seed=0):
    """生成 [points, dims] 的真值轨迹（按决策点采样）。"""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, points)
    smooth = 0.5 - 0.5 * np.cos(np.pi * np.clip(t * 1.3, 0, 1))
    lift = np.clip((t - 0.45) / 0.2, 0, 1) * 0.35
    move = np.clip((t - 0.65) / 0.3, 0, 1) * -0.4
    gt = np.empty((points, dims), dtype=np.float64)
    grips = [7] if dims == 8 else [7, 15]
    arms = [d for d in range(dims) if d not in grips]
    half = dims // 2
    for k, d in enumerate(arms):
        shift = 0.0 if d < half else 1.7
        j = k % 7
        gt[:, d] = (
            -0.8 + 0.22 * j
            + (0.6 + 0.20 * j) * smooth * np.sin(0.9 + 0.25 * j + shift)
            + lift * np.cos(0.3 * j + shift)
            + move * np.sin(0.5 + 0.2 * j + shift)
            + 0.04 * np.sin(np.linspace(0, 9 + j, points))
            + rng.normal(0, 0.008, points)
        )
    for gi in grips:
        off = 0 if gi == 7 else 0.12
        gt[:, gi] = np.clip((t - 0.35 - off) / 0.05, 0, 1)
    return gt


def predict(gt, quality, seed, lag=1.5, damp=0.92, noise=1.0):
    """模拟模型输出：good / bad / flat。"""
    rng = np.random.RandomState(seed)
    if quality == "flat":
        return np.repeat(gt[:1], len(gt), axis=0)
    if quality == "bad":
        lag, damp, noise = 6.0, 0.55, 1.8
    out = np.empty_like(gt)
    n = len(gt)
    for d in range(gt.shape[1]):
        col = gt[:, d]
        idx = np.clip(np.arange(n) - lag, 0, n - 1)
        lagged = np.interp(np.arange(n), idx, col)
        mean = col.mean()
        out[:, d] = mean + damp * (lagged - mean) + rng.normal(
            0, col.std() * 0.05 + 1e-4, n) * noise
    return out


def write(dest, name, points, chunk, dims, quality, seed):
    gt = traj(points, chunk, dims, seed)
    pred = predict(gt, quality, seed + 100)
    dp = np.arange(points) * chunk
    return build_from_arrays(
        pred, gt, dest / f"{name}.npz", decision_points=dp,
        chunk_size=chunk, episode_length=int(dp[-1]) + chunk,
        prompt=PROMPTS[seed % len(PROMPTS)],
    )


created = []

# ---- 单臂 arm_joint8：8 维 ----
single_new = DATA / "ckpt_single_50000"
single_old = DATA / "ckpt_single_10000"
SINGLE = [
    (0,  4,  50, "good"),
    (1, 12,  50, "good"),
    (2, 20,  50, "good"),
    (3, 30,  20, "good"),
    (4, 40,  50, "bad"),
]
for ep, points, chunk, q in SINGLE:
    write(single_new, f"ep{ep:06d}", points, chunk, 8, q, ep * 7 + 1)
    write(single_old, f"ep{ep:06d}", points, chunk, 8,
          "flat" if q == "good" else "bad", ep * 7 + 1)
    created.append(("arm_joint8", ep, points, chunk))

# ---- 双臂 arm_joint16：16 维 ----
dual_new = DATA / "ckpt_dual_50000"
dual_old = DATA / "ckpt_dual_10000"
DUAL = [
    (0, 12, 50, "good"),
    (1, 20, 50, "good"),
    (2, 20, 40, "bad"),
]
for ep, points, chunk, q in DUAL:
    write(dual_new, f"ep{ep:06d}", points, chunk, 16, q, ep * 13 + 3)
    write(dual_old, f"ep{ep:06d}", points, chunk, 16,
          "flat" if q == "good" else "bad", ep * 13 + 3)
    created.append(("arm_joint16", ep, points, chunk))

print("=== 生成的测试数据 ===")
print(f"{'模式':<14}{'ep':>4}{'决策点':>8}{'chunk':>7}")
for mode, ep, points, chunk in created:
    print(f"{mode:<14}{ep:>4}{points:>8}{chunk:>7}")

CWD = str(ROOT)
PY = sys.executable
DATA_ROOT = "/data/x1_pro_grasp_water"
DATA_ROOT_DUAL = "/data/x1_pro_dual16"

jobs = [
    ("单臂 单权重 abs", [PY, "-m", "model_eval", "--arrays", str(single_new),
                    "--kind", "abs", "--data-root", DATA_ROOT,
                    "--out-dir", str(FIGS / "single_abs")]),
    ("单臂 单权重 delta", [PY, "-m", "model_eval", "--arrays", str(single_new),
                      "--kind", "delta", "--data-root", DATA_ROOT,
                      "--out-dir", str(FIGS / "single_delta")]),
    ("单臂 双权重对比", [PY, "-m", "model_eval", "--arrays", str(single_new),
                    "--compare", str(single_old), "--kind", "abs",
                    "--old-label", "step50000", "--new-label", "step10000",
                    "--data-root", DATA_ROOT, "--out-dir", str(FIGS / "compare_abs")]),
    ("双臂 单权重 abs", [PY, "-m", "model_eval", "--arrays", str(dual_new),
                    "--kind", "abs", "--dim-mode", "arm_joint16",
                    "--data-root", DATA_ROOT_DUAL, "--out-dir", str(FIGS / "dual_abs")]),
    ("双臂 单权重 delta", [PY, "-m", "model_eval", "--arrays", str(dual_new),
                      "--kind", "delta", "--dim-mode", "arm_joint16",
                      "--data-root", DATA_ROOT_DUAL, "--out-dir", str(FIGS / "dual_delta")]),
    ("双臂 双权重对比", [PY, "-m", "model_eval", "--arrays", str(dual_new),
                    "--compare", str(dual_old), "--kind", "abs",
                    "--dim-mode", "arm_joint16",
                    "--old-label", "step50000", "--new-label", "step10000",
                    "--data-root", DATA_ROOT_DUAL, "--out-dir", str(FIGS / "dual_compare")]),
]

timing = []
for name, cmd in jobs:
    print(f"\n=== {name} ===")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=CWD, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    timing.append((name, dt, r.returncode))
    lines = [l for l in r.stdout.strip().splitlines() if l]
    print(lines[-1] if lines else r.stderr.strip()[-500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:])

print("\n=== 耗时 ===")
for n, dt, rc in timing:
    print(f"  {n:<20} {dt:6.2f}s  rc={rc}")

print(f"\n共 {len(list(FIGS.rglob(chr(42)+chr(46)+chr(112)+chr(110)+chr(103))))} 张 PNG")
