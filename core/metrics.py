from __future__ import annotations

import numpy as np


def nash_pseudo_regret_curve(A: np.ndarray, game_value: float, x_hist: np.ndarray, y_hist: np.ndarray) -> np.ndarray:
    """
    R_N(t) = sum_{s<=t} [V* - x_s^T A y_s]
    """
    A = np.asarray(A, dtype=float)
    losses = []
    for x, y in zip(x_hist, y_hist):
        losses.append(game_value - float(x @ A @ y))
    return np.cumsum(np.asarray(losses, dtype=float))

