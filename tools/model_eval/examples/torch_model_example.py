"""端到端示例：跑评估 → 产出中间产物 → 绘图。

这是「如何接入自己的模型和数据集」的完整模板。照着改两处即可：

1. ``load_episodes()`` —— 换成你读数据集的代码；
2. ``MyModel.predict()`` —— 换成你真正的模型推理。

用法::

    python -m model_eval eval \
        --script examples/torch_model_example.py \
        --out-dir  runs/step50000/artifacts \
        --plot-dir runs/step50000/figs \
        --checkpoint step50000
"""

from __future__ import annotations

import numpy as np

from model_eval import Dataset, Episode

# ---------------------------------------------------------------- 数据集

DIM = 8          # 单臂 arm_joint8；双臂用 16
LENGTH = 600     # 整回合帧数
CAMERAS = ["head", "right_wrist"]


def load_episodes():
    """换成你自己的数据读取。每回合返回一个 Episode。

    真实项目里通常是 LeRobot / HDF5 / 自定义格式，只要构造出
    逐帧真值 ``[L, D]`` 和一个按帧取观测的 ``observe(t)``。
    """
    episodes = []
    for ep_idx in range(3):
        t = np.linspace(0, 1, LENGTH)
        gt = np.stack(
            [np.sin(np.pi * t) * (d + 1) * 0.2 - 0.3 for d in range(DIM)], axis=1
        ).astype(np.float32)
        gt[:, 7] = np.clip((t - 0.4) / 0.05, 0, 1)

        def observe(frame, _gt=gt, _seed=ep_idx):
            """返回第 frame 帧的模型输入。结构随意，原样交给 predict。"""
            return {
                "state": _gt[frame].copy(),
                "images": {c: np.zeros((224, 224, 3), np.uint8) for c in CAMERAS},
                "prompt": f"task {_seed}",
            }

        episodes.append(
            Episode(
                episode_id=f"ep{ep_idx:06d}",
                length=LENGTH,
                ground_truth=gt,
                observe=observe,
                prompt=f"pick up the water bottle #{ep_idx}",
            )
        )
    return episodes


def build_dataset():
    return Dataset(load_episodes(), name="my_dataset_v1")


# ------------------------------------------------------------------- 模型

class MyModel:
    """模型接口：只要有 chunk_size 与 predict(obs) -> [H, D]。

    真实项目里把 predict 换成你的推理，可以内部用 torch/CUDA、
    调远程服务、或读 ONNX，本工具不关心。
    """

    chunk_size = 50

    def __init__(self, lag=2.0, noise=0.01):
        self.lag = lag
        self.noise = noise

    def predict(self, obs):
        # 这里只是示例：拿当前 state 外推，真实模型请替换
        state = np.asarray(obs["state"], dtype=np.float32)
        seed = int(abs(state.sum()) * 1e6) % (2 ** 31)
        rng = np.random.RandomState(seed)
        drift = np.linspace(0, 0.05, self.chunk_size)[:, None]
        out = state[None, :] + drift * (state > 0)
        out = out + rng.normal(0, self.noise, out.shape)
        return out.astype(np.float32)


def build_model():
    return MyModel()
