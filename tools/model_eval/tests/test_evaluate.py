"""Tests for full-frame evaluation and chunk stitching."""

import pathlib
import tempfile
import unittest

import numpy as np
from model_eval import Dataset, Episode, evaluate, load_episode
from model_eval.cli import main


class EvaluateTests(unittest.TestCase):
    """评估：驱动模型推理，产出可绘图的中间产物。"""

    def _dataset(self, n_ep=2, length=300, dims=8):
        eps = []
        for i in range(n_ep):
            t = np.linspace(0, 1, length)
            gt = np.stack(
                [np.sin(np.pi * t) * (k + 1) * 0.2 for k in range(dims)], axis=1
            ).astype(np.float32)
            eps.append(
                Episode(
                    f"ep{i:06d}",
                    length,
                    gt,
                    observe=(lambda f, _g=gt: {"state": _g[f]}),
                    prompt=f"task {i}",
                )
            )
        return Dataset(eps, name="unit")

    def test_produces_loadable_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            res = evaluate(_ConstModel(50, 8), self._dataset(), pathlib.Path(d))
            self.assertEqual(len(res), 2)
            for r in res:
                e = load_episode(r["path"])
                self.assertEqual((e.dims, e.episode_length, e.chunk_size), (8, 300, 50))

    def test_decision_points_follow_stride(self):
        with tempfile.TemporaryDirectory() as d:
            res = evaluate(
                _ConstModel(50, 8), self._dataset(n_ep=1), pathlib.Path(d), stride=20
            )
            e = load_episode(res[0]["path"])
            np.testing.assert_array_equal(e.decision_points, np.arange(300))
            np.testing.assert_array_equal(
                e.metadata["inference_points"], np.arange(0, 300, 20)
            )

    def test_default_stride_is_chunk_size(self):
        with tempfile.TemporaryDirectory() as d:
            res = evaluate(_ConstModel(50, 8), self._dataset(n_ep=1), pathlib.Path(d))
            e = load_episode(res[0]["path"])
            self.assertEqual(len(e.decision_points), 300)
            self.assertEqual(len(e.metadata["inference_points"]), 6)

    def test_preserves_all_values_from_each_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            res = evaluate(
                _RampModel(10, 4),
                self._dataset(n_ep=1, length=23, dims=4),
                pathlib.Path(d),
            )
            episode = load_episode(res[0]["path"])
            expected = np.tile(np.arange(10, dtype=np.float32), 3)[:23]
            np.testing.assert_allclose(episode.prediction[:, 0], expected)
            np.testing.assert_array_equal(
                episode.ground_truth,
                self._dataset(n_ep=1, length=23, dims=4).episodes[0].ground_truth,
            )

    def test_overlapping_chunks_keep_latest_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            res = evaluate(
                _RampModel(10, 4),
                self._dataset(n_ep=1, length=18, dims=4),
                pathlib.Path(d),
                stride=5,
            )
            episode = load_episode(res[0]["path"])
            expected = np.array(
                [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2], dtype=np.float32
            )
            np.testing.assert_allclose(episode.prediction[:, 0], expected)

    def test_stride_larger_than_chunk_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "stride <= chunk_size"):
                evaluate(
                    _RampModel(10, 4),
                    self._dataset(n_ep=1, length=30, dims=4),
                    pathlib.Path(d),
                    stride=11,
                )

    def test_bad_shape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "形状"):
                evaluate(_BadShapeModel(50, 8), self._dataset(n_ep=1), pathlib.Path(d))

    def test_nan_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "NaN"):
                evaluate(
                    _NanModel(10, 8), self._dataset(n_ep=1, length=50), pathlib.Path(d)
                )

    def test_on_error_skip(self):
        with tempfile.TemporaryDirectory() as d:
            eps = [
                Episode(
                    "good",
                    50,
                    np.zeros((50, 8), np.float32),
                    observe=lambda f: {"state": -np.ones(8, np.float32)},
                ),
                Episode(
                    "bad",
                    50,
                    np.zeros((50, 8), np.float32),
                    observe=lambda f: {"state": np.ones(8, np.float32)},
                ),
            ]
            res = evaluate(
                _FailOnFirst(10, 8),
                Dataset(eps),
                pathlib.Path(d),
                on_error="skip",
                log=lambda *a, **k: None,
            )
            self.assertEqual([r["episode_id"] for r in res], ["good"])
            self.assertTrue((pathlib.Path(d) / "good.npz").is_file())
            self.assertFalse((pathlib.Path(d) / "bad.npz").is_file())

    def test_invalid_model_object(self):
        class NoChunk:
            def predict(self, obs):
                return np.zeros((1, 8))

        with self.assertRaisesRegex(ValueError, "chunk_size"):
            evaluate(NoChunk(), self._dataset(n_ep=1), "/tmp/x")

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            Dataset([])

    def test_artifacts_can_be_plotted(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            evaluate(_ConstModel(50, 8), self._dataset(n_ep=1), root / "art")
            out = root / "figs"
            self.assertEqual(
                main(
                    [
                        "plot",
                        "--arrays",
                        str(root / "art"),
                        "--out-dir",
                        str(out),
                        "--kind",
                        "abs",
                    ]
                ),
                0,
            )
            self.assertTrue((out / "ep000000_abs.png").is_file())


class _ConstModel:
    def __init__(self, chunk, dims):
        self.chunk_size = chunk
        self.dims = dims

    def predict(self, obs):
        return np.repeat(np.asarray(obs["state"])[None, :], self.chunk_size, axis=0)


class _RampModel:
    def __init__(self, chunk, dims):
        self.chunk_size = chunk
        self.dims = dims

    def predict(self, obs):
        return np.repeat(
            np.arange(self.chunk_size, dtype=np.float32)[:, None], self.dims, axis=1
        )


class _BadShapeModel:
    def __init__(self, chunk, dims):
        self.chunk_size = chunk
        self.dims = dims

    def predict(self, obs):
        return np.zeros((self.chunk_size - 1, self.dims), np.float32)


class _NanModel:
    def __init__(self, chunk, dims):
        self.chunk_size = chunk
        self.dims = dims

    def predict(self, obs):
        return np.full((self.chunk_size, self.dims), np.nan, np.float32)


class _FailOnFirst:
    """只在「每回合的第一个决策点」失败的话无法区分回合；

    这里按观测内容区分：第一个回合的 state 全为正，第二个为负。
    """

    def __init__(self, chunk, dims):
        self.chunk_size = chunk
        self.dims = dims

    def predict(self, obs):
        if float(np.asarray(obs["state"]).sum()) > 0:
            raise RuntimeError("boom")
        return np.zeros((self.chunk_size, self.dims), np.float32)
