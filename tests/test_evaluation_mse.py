import numpy as np
import pytest

from examples.evaluate_x1_icedtea import calculate_evaluation_mse


def test_evaluation_mse_sums_joint_dimensions_and_pools_frames(tmp_path):
    first_prediction = np.zeros((1, 8), dtype=np.float32)
    first_ground_truth = np.ones((1, 8), dtype=np.float32)
    second_prediction = np.zeros((3, 8), dtype=np.float32)
    second_ground_truth = np.ones((3, 8), dtype=np.float32)
    second_ground_truth[:, :7] *= 2
    second_ground_truth[:, 7] *= 3
    paths = []
    for index, (prediction, ground_truth) in enumerate(
        (
            (first_prediction, first_ground_truth),
            (second_prediction, second_ground_truth),
        )
    ):
        path = tmp_path / f"episode_{index}.npz"
        np.savez(path, prediction=prediction, ground_truth=ground_truth)
        paths.append({"path": str(path)})

    metrics = calculate_evaluation_mse(paths)

    assert metrics["frames"] == 4
    assert metrics["per_dimension_mse"] == [3.25] * 7 + [7.0]
    assert metrics["joint_mse_sum"] == 22.75
    assert metrics["joint_mse_mean"] == 3.25
    assert metrics["gripper_mse"] == 7.0


def test_evaluation_mse_rejects_misaligned_arrays(tmp_path):
    path = tmp_path / "episode.npz"
    np.savez(
        path,
        prediction=np.zeros((2, 8), dtype=np.float32),
        ground_truth=np.zeros((3, 8), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="aligned"):
        calculate_evaluation_mse([{"path": str(path)}])
