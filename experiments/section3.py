from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from core.utils import ensure_dir


@dataclass
class RunConfig:
    horizons: list[int]
    n_runs: int
    seed: int = 7
    preset: str = "quick"
    n_actions: int = 20


def section3_horizons_for_preset(preset: str) -> tuple[list[int], int]:
    # Mirrors official code scale with runtime-aware presets.
    if preset == "quick":
        return [10, 100], 2
    if preset == "medium":
        return [10, 100, 1000], 4
    if preset == "paper-lite":
        return [10**i for i in range(1, 6)], 12
    if preset in {"final", "paper"}:
        return [10**i for i in range(1, 8)], 100
    raise ValueError(f"Unknown preset: {preset}")


def nash1_diag(A: np.ndarray) -> np.ndarray:
    A = np.maximum(A, 1e-6)
    den = np.sum(1.0 / np.diag(A))
    return 1.0 / (np.diag(A) * den)


def adversary(A: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    column_sums = A @ x
    idx = int(np.argmin(column_sums))
    return float(column_sums[idx]), idx


def generate_diagonal_matrix(n: int) -> np.ndarray:
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        B[i, i] = 0.4 + 0.2 * i / (n - 1)
    return B


def generate_bernoulli_diagonal_matrix(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = A.shape[0]
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        B[i, i] = rng.binomial(1, A[i, i])
    return B


def update(A: np.ndarray, x1: np.ndarray, j: int, t: int) -> np.ndarray:
    vec = A[:-1, j] - A[-1, j]
    x1[:-1] = np.clip(x1[:-1] + vec * (1.0 / t), 0.0, 1.0)
    s = np.sum(x1[:-1])
    if s > 1:
        x1[:-1] /= s
        x1[-1] = 0.0
    else:
        x1[-1] = 1.0 - s
    return x1


def _value_of_diag_game(A: np.ndarray) -> float:
    den = 0.0
    for i in range(A.shape[0]):
        den += 1.0 / A[i, i]
    return 1.0 / den


def run_nash(seed: int, horizon: int, n: int) -> float:
    rng = np.random.default_rng(seed)
    B = generate_diagonal_matrix(n)
    V = _value_of_diag_game(B)
    B1 = np.zeros((n, n), dtype=float)
    reg = 0.0
    for t in range(horizon):
        x = nash1_diag(B1)
        val, _ = adversary(B, x)
        reg += V - val
        Bsamp = generate_bernoulli_diagonal_matrix(B, rng)
        B1 = (t / (t + 1)) * B1 + (1.0 / (t + 1)) * Bsamp
    return float(max(reg, 1e-12))


def run_our_algo(seed: int, horizon: int, n: int) -> float:
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
    reg = 0.0
    for t in range(horizon):
        if count0 > 0 and t > threshold:
            x = update(B2, x, jt, t0)
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
        # Keep the official script behavior exactly (double accumulation).
        reg += V - val
    return float(max(reg, 1e-12))


def run_hedge(seed: int, horizon: int, n: int) -> float:
    rng = np.random.default_rng(seed)
    B = generate_diagonal_matrix(n)
    V = _value_of_diag_game(B)
    weights = np.ones(n, dtype=float)
    eta = (math.log(n) / max(1, horizon)) ** 0.5
    reg = 0.0
    for _ in range(horizon):
        x = weights / np.sum(weights)
        val, idx = adversary(B, x)
        reg += V - val
        Bsamp = generate_bernoulli_diagonal_matrix(B, rng)
        reward_vector = Bsamp[:, idx]
        weights *= np.exp(eta * reward_vector)
    return float(max(reg, 1e-12))


def run(config: RunConfig) -> None:
    ensure_dir("plots")
    x_axis = np.asarray(config.horizons, dtype=float)
    algo_specs = [
        ("Nash", run_nash, "#1f77b4"),
        ("Our-Algo", run_our_algo, "#ff7f0e"),
        ("Hedge", run_hedge, "#2ca02c"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, fn, color in algo_specs:
        means: list[float] = []
        stds: list[float] = []
        for T in config.horizons:
            vals = []
            for r in range(config.n_runs):
                seed = config.seed + 10007 * r + 37 * T
                vals.append(fn(seed, T, config.n_actions))
            arr = np.log10(np.maximum(np.asarray(vals, dtype=float), 1e-12))
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))

        y = np.asarray(means, dtype=float)
        ci = np.asarray(stds, dtype=float)
        ax.plot(x_axis, y, marker="o", label=label, color=color)
        ax.fill_between(x_axis, y - ci, y + ci, alpha=0.25, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("log10(Total Nash regret)")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper left")
    ax.set_title(
        f"Section 3 official-style (n={config.n_actions}, preset={config.preset}, runs={config.n_runs}, horizons={config.horizons})"
    )
    plt.tight_layout()
    plt.savefig(f"plots/section3_official_style_{config.preset}.png", dpi=170)
    plt.show()


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default="quick", choices=["quick", "medium", "paper-lite", "final", "paper"])
    p.add_argument("--horizons", type=int, nargs="*", default=None, help="Optional custom horizons list.")
    p.add_argument("--n_runs", type=int, default=None, help="Optional override of preset trial count.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n_actions", type=int, default=20, help="Diagonal game size n (official default: 20).")
    a = p.parse_args()
    preset_horizons, preset_runs = section3_horizons_for_preset(a.preset)
    horizons = a.horizons if a.horizons else preset_horizons
    n_runs = a.n_runs if a.n_runs is not None else preset_runs
    return RunConfig(horizons=sorted(horizons), n_runs=n_runs, seed=a.seed, preset=a.preset, n_actions=a.n_actions)


if __name__ == "__main__":
    run(parse_args())

