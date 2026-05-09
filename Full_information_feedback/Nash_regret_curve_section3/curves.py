"""
Cumulative Nash regret vs time for Section 3 baseline (diagonal full-information).

Logic mirrors experiments/section3.py (run_nash, run_hedge, run_official_diag_algo).
Keep in sync with that module if baseline behaviour changes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.section3 import (  # noqa: E402
    _update_official_diag,
    _value_of_diag_game,
    adversary,
    generate_bernoulli_diagonal_matrix,
    generate_diagonal_matrix,
    nash1_diag,
)


def nash_cumulative_curve(seed: int, horizon: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = generate_diagonal_matrix(n)
    V = _value_of_diag_game(B)
    B1 = np.zeros((n, n), dtype=float)
    curve = np.zeros(horizon, dtype=float)
    reg = 0.0
    for t in range(horizon):
        x = nash1_diag(B1)
        val, _ = adversary(B, x)
        reg += V - val
        curve[t] = reg
        Bsamp = generate_bernoulli_diagonal_matrix(B, rng)
        B1 = (t / (t + 1)) * B1 + (1.0 / (t + 1)) * Bsamp
    return np.maximum(curve, 1e-12)


def hedge_cumulative_curve(seed: int, horizon: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = generate_diagonal_matrix(n)
    V = _value_of_diag_game(B)
    weights = np.ones(n, dtype=float)
    eta = (math.log(n) / max(1, horizon)) ** 0.5
    curve = np.zeros(horizon, dtype=float)
    reg = 0.0
    for t in range(horizon):
        x = weights / np.sum(weights)
        val, idx = adversary(B, x)
        reg += V - val
        curve[t] = reg
        Bsamp = generate_bernoulli_diagonal_matrix(B, rng)
        reward_vector = Bsamp[:, idx]
        weights *= np.exp(eta * reward_vector)
    return np.maximum(curve, 1e-12)


def official_diag_cumulative_curve(seed: int, horizon: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = generate_diagonal_matrix(n)
    V = _value_of_diag_game(B)
    B1 = np.zeros((n, n), dtype=float)
    B2 = np.zeros((n, n), dtype=float)
    jt = 0
    x = np.ones(n, dtype=float) / n
    count0 = 1
    t0 = 1
    threshold = min(math.log(max(horizon, 2)) ** 2, horizon**0.5)
    curve = np.zeros(horizon, dtype=float)
    reg = 0.0
    for t in range(horizon):
        if count0 > 0 and t > threshold:
            x = _update_official_diag(B2, x, jt, t0)
            count0 -= 1
        else:
            t0 = t + 1
            x = nash1_diag(B1)
            count0 = t0 - 1
            B2 = B1.copy()
        val, jt = adversary(B, x)
        reg += V - val
        Bsamp = generate_bernoulli_diagonal_matrix(B, rng)
        B1 = (t / (t + 1)) * B1 + (1.0 / (t + 1)) * Bsamp
        reg += V - val
        curve[t] = reg
    return np.maximum(curve, 1e-12)


def mean_std_curves(
    fn,
    horizon: int,
    n: int,
    n_runs: int,
    base_seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for r in range(n_runs):
        seed = base_seed + 10007 * r
        curves.append(fn(seed, horizon, n))
    arr = np.stack(curves, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)
