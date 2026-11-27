"""Check the original ACT aggregation order and control-time alignment."""

import numpy as np
import pytest

from jepa_policy.temporal_ensemble import ActTemporalEnsembler


def test_matches_dense_act_reference_over_overlapping_predictions():
    horizon, action_dim, ticks = 9, 8, 30
    ensemble = ActTemporalEnsembler(horizon, action_dim, 0.01)
    predictions = np.zeros((ticks, ticks + horizon, action_dim), dtype=np.float64)
    populated = np.zeros((ticks, ticks + horizon), dtype=bool)
    generator = np.random.default_rng(7)
    for tick in range(ticks):
        chunk = generator.uniform(-2.0, 2.0, (horizon, action_dim)).astype(np.float32)
        predictions[tick, tick : tick + horizon] = chunk
        populated[tick, tick : tick + horizon] = True
        candidates = predictions[:, tick][populated[:, tick]]
        weights = np.exp(-0.01 * np.arange(len(candidates), dtype=np.float64))
        weights /= weights.sum()
        expected = np.sum(candidates * weights[:, None], axis=0)

        actual = ensemble.update(chunk)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        assert actual.dtype == np.float32
        assert actual.shape == (action_dim,)


def test_aligns_same_time_and_keeps_more_than_two_historical_predictions():
    ensemble = ActTemporalEnsembler(3, 1, 0.0)
    first = np.array([[1.0], [20.0], [300.0]], dtype=np.float32)
    second = np.array([[40.0], [500.0], [6000.0]], dtype=np.float32)
    third = np.array([[700.0], [8000.0], [90000.0]], dtype=np.float32)
    np.testing.assert_array_equal(ensemble.update(first), [1.0])
    np.testing.assert_array_equal(ensemble.update(second), [30.0])
    np.testing.assert_array_equal(ensemble.update(third), [500.0])


def test_positive_decay_weights_oldest_valid_prediction_more():
    ensemble = ActTemporalEnsembler(3, 1, np.log(2.0))
    ensemble.update(np.full((3, 1), 10.0))
    actual = ensemble.update(np.full((3, 1), 20.0))
    np.testing.assert_allclose(actual, [(10.0 + 0.5 * 20.0) / 1.5])
    assert actual[0] < 15.0


def test_zero_actions_are_valid_candidates():
    ensemble = ActTemporalEnsembler(3, 2, 0.0)
    first = np.ones((3, 2), dtype=np.float32)
    first[1] = 0.0
    ensemble.update(first)
    np.testing.assert_array_equal(ensemble.update(np.full((3, 2), 2.0)), np.ones(2))


def test_copies_predictions_and_never_modifies_input():
    ensemble = ActTemporalEnsembler(3, 1, 0.0)
    first = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    original = first.copy()
    output = ensemble.update(first)
    np.testing.assert_array_equal(first, original)
    first[:] = 1000
    output[:] = -1000
    second = np.zeros((3, 1), dtype=np.float32)
    np.testing.assert_array_equal(ensemble.update(second), [1.0])
    np.testing.assert_array_equal(second, np.zeros((3, 1)))


def test_history_is_bounded_and_expired_chunks_are_removed():
    ensemble = ActTemporalEnsembler(3, 2)
    for tick in range(20):
        ensemble.update(np.full((3, 2), tick, dtype=np.float32))
        assert len(ensemble.history) == min(tick + 1, 3)
        assert [start for start, _ in ensemble.history] == list(
            range(max(0, tick - 2), tick + 1)
        )


def test_reset_clears_history_and_control_tick():
    ensemble = ActTemporalEnsembler(3, 2)
    ensemble.update(np.full((3, 2), 100.0))
    ensemble.reset()
    assert len(ensemble.history) == 0
    assert ensemble.step == 0
    np.testing.assert_array_equal(ensemble.update(np.full((3, 2), 3.0)), [3.0, 3.0])


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
@pytest.mark.parametrize("parameter", ["prediction_horizon", "action_dim"])
def test_invalid_dimensions_are_rejected(parameter, value):
    arguments = {"prediction_horizon": 3, "action_dim": 2, "decay": 0.01}
    arguments[parameter] = value
    with pytest.raises(ValueError, match="positive integer"):
        ActTemporalEnsembler(**arguments)


@pytest.mark.parametrize("decay", [-0.01, np.nan, np.inf, -np.inf])
def test_invalid_decay_is_rejected(decay):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ActTemporalEnsembler(3, 2, decay)


@pytest.mark.parametrize("shape", [(2, 2), (4, 2), (3, 3), (3, 2, 1), (0, 2)])
def test_invalid_shape_leaves_history_unchanged(shape):
    ensemble = ActTemporalEnsembler(3, 2)
    ensemble.update(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        ensemble.update(np.zeros(shape))
    assert ensemble.step == 1
    assert len(ensemble.history) == 1
    np.testing.assert_array_equal(ensemble.history[0][1], np.ones((3, 2)))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "not-a-number", 1j])
def test_invalid_values_leave_history_empty(value):
    ensemble = ActTemporalEnsembler(3, 2)
    with pytest.raises(ValueError, match="finite real"):
        ensemble.update(np.full((3, 2), value))
    assert ensemble.step == 0
    assert len(ensemble.history) == 0
