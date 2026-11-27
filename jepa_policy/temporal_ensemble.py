"""ACT temporal ensembling over predictions for the same control tick."""

from __future__ import annotations

from collections import deque

import numpy as np


class ActTemporalEnsembler:
    """Fuse overlapping predictions in original ACT order.

    Call update once per control tick with [prediction_horizon, action_dim]
    absolute actions, starting at the current tick. Chunks are ordered from
    oldest to newest, with normalized weights exp(-decay * index). Thus a
    positive decay gives the oldest still-valid prediction the highest weight.
    """

    def __init__(self, prediction_horizon: int, action_dim: int, decay: float = 0.01):
        for name, value in (
            ("prediction_horizon", prediction_horizon),
            ("action_dim", action_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(decay) or decay < 0:
            raise ValueError("decay must be finite and nonnegative")
        self.prediction_horizon = prediction_horizon
        self.action_dim = action_dim
        self.decay = float(decay)
        self.history: deque[tuple[int, np.ndarray]] = deque()
        self.step = 0

    def reset(self) -> None:
        """Discard predictions from the preceding episode."""
        self.history.clear()
        self.step = 0

    def update(self, actions: np.ndarray) -> np.ndarray:
        """Add this tick's full prediction and return one fused float32 action."""
        chunk = np.asarray(actions)
        expected_shape = (self.prediction_horizon, self.action_dim)
        if chunk.shape != expected_shape:
            raise ValueError(
                f"Expected actions with shape {expected_shape}, got {chunk.shape}"
            )
        if (
            not np.issubdtype(chunk.dtype, np.number)
            or np.issubdtype(chunk.dtype, np.complexfloating)
            or not np.isfinite(chunk).all()
        ):
            raise ValueError("Actions must contain finite real numbers")
        while (
            self.history and self.history[0][0] + self.prediction_horizon <= self.step
        ):
            self.history.popleft()
        self.history.append((self.step, chunk.copy()))
        predictions = np.stack(
            [prediction[self.step - start] for start, prediction in self.history]
        ).astype(np.float64, copy=False)
        weights = np.exp(-self.decay * np.arange(len(predictions), dtype=np.float64))
        weights /= weights.sum()
        action = np.sum(predictions * weights[:, None], axis=0).astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError("Ensembled actions must be finite float32 values")
        self.step += 1
        return action
