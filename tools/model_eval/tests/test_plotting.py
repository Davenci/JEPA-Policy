"""整回合轨迹对比图测试：数据格式、绘图、维度模式、命令行。"""

import pathlib
import tempfile
import unittest

import numpy as np
from model_eval import (
    DIM_LABELS_8,
    DIM_MODES,
    NON_DELTA_INDICES,
    EpisodeData,
    load_episode,
    render_episode,
    resolve_dim_mode,
    save_episode,
)
from model_eval.builders import build_from_arrays, build_from_chunks
from model_eval.cli import _find_episodes, main


def make_ep(
    dims=8, points=4, chunk=50, seed=0, prompt="pick up the bottle", quality="good"
):
    rng = np.random.RandomState(seed)
    dp = np.arange(points) * chunk
    t = np.linspace(0, 1, points)
    gt = np.stack(
        [np.sin(np.pi * t) * (d + 1) * 0.2 - 0.3 for d in range(dims)], axis=1
    ).astype(np.float32)
    noise = (
        np.zeros(gt.shape) if quality == "perfect" else rng.normal(0, 0.05, gt.shape)
    )
    pred = (gt + noise).astype(np.float32)
    base = np.vstack([gt[:1], gt[:-1]]).astype(np.float32)
    return EpisodeData(
        prediction=pred,
        ground_truth=gt,
        state_base=base,
        decision_points=dp,
        episode_length=int(dp[-1]) + chunk,
        chunk_size=chunk,
        prompt=prompt,
    )


class IOFormatTests(unittest.TestCase):
    """本项目自有格式：不依赖任何外部仓库。"""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = save_episode(make_ep(), pathlib.Path(d) / "ep0.npz")
            e = load_episode(p)
            np.testing.assert_allclose(e.prediction, make_ep().prediction)
            self.assertEqual(e.dims, 8)
            self.assertEqual(e.episode_id, "ep0")

    def test_required_keys_are_ours(self):
        with tempfile.TemporaryDirectory() as d:
            p = save_episode(make_ep(), pathlib.Path(d) / "a.npz")
            with np.load(p) as z:
                for k in (
                    "prediction",
                    "ground_truth",
                    "state_base",
                    "decision_points",
                    "episode_length",
                    "chunk_size",
                    "prompt",
                ):
                    self.assertIn(k, z.files)

    def test_validation(self):
        good = {
            "prediction": np.zeros((3, 8)),
            "ground_truth": np.zeros((3, 8)),
            "state_base": np.zeros((3, 8)),
            "decision_points": np.array([0, 5, 10]),
            "episode_length": 20,
            "chunk_size": 5,
            "prompt": "x",
        }
        EpisodeData(**good)  # 不抛错
        bad_cases = [
            {**good, "prediction": np.zeros((3, 7))},  # 形状不一致
            {**good, "decision_points": np.array([0, 5])},  # 长度不符
            {**good, "decision_points": np.array([0, 10, 5])},  # 非递增
            {**good, "decision_points": np.array([0, 5, 99])},  # 超出回合
            {**good, "chunk_size": 0},
            {**good, "episode_length": 0},
            {**good, "prediction": np.full((3, 8), np.nan)},
            {**good, "ground_truth": np.zeros((3, 8, 1))},
        ]
        for i, kw in enumerate(bad_cases):
            with self.subTest(i=i), self.assertRaises(ValueError):
                EpisodeData(**kw)

    def test_missing_key_message(self):
        with tempfile.TemporaryDirectory() as d:
            bad = pathlib.Path(d) / "bad.npz"
            np.savez(bad, prediction=np.zeros((3, 8)))
            with self.assertRaisesRegex(ValueError, "缺少必需键"):
                load_episode(bad)


class BuilderTests(unittest.TestCase):
    def test_from_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.RandomState(0)
            p = build_from_arrays(
                rng.rand(12, 8),
                rng.rand(12, 8),
                pathlib.Path(d) / "e.npz",
                chunk_size=50,
                prompt="t",
            )
            e = load_episode(p)
            self.assertEqual(
                (e.dims, len(e.decision_points), e.chunk_size), (8, 12, 50)
            )

    def test_from_chunks_takes_step_zero(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.RandomState(1)
            chunks = rng.rand(5, 10, 16)
            gt = rng.rand(5, 16)
            p = build_from_chunks(
                chunks, gt, pathlib.Path(d) / "c.npz", decision_points=np.arange(5) * 10
            )
            e = load_episode(p)
            self.assertEqual((e.dims, e.chunk_size), (16, 10))
            np.testing.assert_allclose(e.prediction, chunks[:, 0, :].astype(np.float32))

    def test_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                build_from_arrays(
                    np.zeros((3, 8)), np.zeros((3, 7)), pathlib.Path(d) / "x.npz"
                )


class DimModeTests(unittest.TestCase):
    def test_mode_table(self):
        single = DIM_MODES["arm_joint8"]
        dual = DIM_MODES["arm_joint16"]
        self.assertEqual((single.dims, single.columns, single.rows), (8, 3, 3))
        self.assertEqual((dual.dims, dual.columns, dual.rows), (16, 4, 4))
        self.assertEqual(single.gripper_indices, [7])
        self.assertEqual(dual.gripper_indices, [7, 15])
        self.assertEqual(dual.labels[0], "左臂 joint 1")
        self.assertEqual(dual.labels[7], "左夹爪/gripper")
        self.assertEqual(dual.labels[8], "右臂 joint 1")
        self.assertEqual(dual.labels[15], "右夹爪/gripper")

    def test_resolve(self):
        self.assertEqual(resolve_dim_mode("arm_joint8", 8).name, "arm_joint8")
        self.assertEqual(resolve_dim_mode(None, 16).name, "arm_joint16")
        self.assertEqual(resolve_dim_mode().name, "arm_joint8")

    def test_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "16 维"):
            resolve_dim_mode("arm_joint8", 16)
        with self.assertRaisesRegex(ValueError, "未知的 dim mode"):
            resolve_dim_mode("arm_joint24", 24)
        with self.assertRaisesRegex(ValueError, "没有对应的 dim mode"):
            resolve_dim_mode(None, 15)


class RenderTests(unittest.TestCase):
    def test_single_weight(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            ep = make_ep()
            for kind in ("abs", "delta"):
                p = render_episode(ep, root / f"{kind}.png", kind=kind)
                self.assertTrue(p.is_file())
                self.assertGreater(p.stat().st_size, 10_000)

    def test_compare(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            p = render_episode(
                make_ep(seed=1),
                root / "c.png",
                kind="abs",
                compare=make_ep(seed=2),
                old_label="A",
                new_label="B",
            )
            self.assertTrue(p.is_file())

    def test_compare_dim_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "维度"):
                render_episode(
                    make_ep(dims=8), pathlib.Path(d) / "x.png", compare=make_ep(dims=16)
                )

    def test_dual_arm(self):
        with tempfile.TemporaryDirectory() as d:
            p = render_episode(
                make_ep(dims=16), pathlib.Path(d) / "dual.png", kind="abs"
            )
            self.assertTrue(p.is_file())

    def test_accepts_path_and_object(self):
        """同一个 EpisodeData 走「路径」和「对象」两条入口，画出的内容要一致。

        注：走路径时数据经 npz 落盘（float32），两者像素级可能有极小差异，
        因此比较尺寸与整体墨迹，而不是比较文件字节。
        """
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            ep = make_ep(quality="perfect")
            p = save_episode(ep, root / "e.npz")
            a = render_episode(p, root / "a.png", kind="abs", checkpoint="ckpt")
            b = render_episode(
                load_episode(p), root / "b.png", kind="abs", checkpoint="ckpt"
            )
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_object_input_renders(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            out = render_episode(make_ep(), root / "obj.png", kind="delta")
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 10_000)

    def test_marker_indices(self):
        from model_eval.plotting import _marker_indices

        for n, stride in ((4, 50), (12, 50), (40, 50), (30, 20)):
            dec = np.arange(n) * stride
            self.assertEqual(len(_marker_indices(dec, stride)), n)
        dec = np.arange(0, 600, 20)
        self.assertEqual(len(_marker_indices(dec, 10)), 30)

    def test_delta_keeps_gripper_absolute(self):
        from model_eval.plotting import _local_relative_delta

        a = np.arange(16, dtype=np.float32).reshape(1, 16)
        out = _local_relative_delta(
            a,
            np.zeros((1, 16), dtype=np.float32),
            nd=DIM_MODES["arm_joint16"].gripper_indices,
        )
        self.assertEqual(float(out[0, 7]), 7.0)
        self.assertEqual(float(out[0, 15]), 15.0)

    def test_constants(self):
        self.assertEqual(len(DIM_LABELS_8), 8)
        self.assertEqual(NON_DELTA_INDICES, [7])


class CliTests(unittest.TestCase):
    def test_directory_input(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            src = root / "ckpt_a"
            save_episode(make_ep(), src / "ep000000.npz")
            out = root / "figs"
            self.assertEqual(
                main(["plot", "--arrays", str(src), "--out-dir", str(out)]), 0
            )
            self.assertTrue((out / "ep000000_abs.png").is_file())
            self.assertTrue((out / "ep000000_delta.png").is_file())

    def test_single_file_input(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            f = save_episode(make_ep(), root / "only.npz")
            out = root / "figs"
            self.assertEqual(
                main(
                    ["plot", "--arrays", str(f), "--out-dir", str(out), "--kind", "abs"]
                ),
                0,
            )
            self.assertTrue((out / "only_abs.png").is_file())

    def test_compare_mode(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            save_episode(make_ep(seed=1), root / "new" / "e3.npz")
            save_episode(make_ep(seed=2), root / "old" / "e3.npz")
            out = root / "figs"
            self.assertEqual(
                main(
                    [
                        "plot",
                        "--arrays",
                        str(root / "new"),
                        "--compare",
                        str(root / "old"),
                        "--kind",
                        "abs",
                        "--out-dir",
                        str(out),
                    ]
                ),
                0,
            )
            self.assertTrue((out / "compare_e3_abs.png").is_file())

    def test_skips_foreign_npz(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            np.savez(root / "unrelated.npz", foo=np.zeros(3))
            save_episode(make_ep(), root / "good.npz")
            found = _find_episodes(root)
            self.assertEqual(list(found), ["good"])

    def test_empty_dir_exits(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                main(["plot", "--arrays", d, "--out-dir", d])
