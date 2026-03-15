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
    seed: int = 19
    preset: str = "quick"


def section4_horizons_for_preset(preset: str) -> tuple[list[int], int]:
    # Mirrors official scripts Regret-log{1,2,3}.py
    if preset == "quick":
        return [10, 100], 2
    if preset == "medium":
        return [10, 100, 1000], 4
    if preset == "paper-lite":
        return [10**i for i in range(1, 6)], 12
    if preset in {"final", "paper"}:
        return [10**i for i in range(1, 9)], 128
    raise ValueError(f"Unknown preset: {preset}")


def nash1(A: np.ndarray) -> tuple[float, float]:
    if A[0][0] <= A[0][1] and A[0][0] >= A[1][0]:
        return 1.0, 0.0
    if A[1][0] <= A[1][1] and A[1][0] >= A[0][0]:
        return 0.0, 1.0
    if A[0][1] <= A[0][0] and A[0][1] >= A[1][1]:
        return 1.0, 0.0
    if A[1][1] <= A[1][0] and A[1][1] >= A[0][1]:
        return 0.0, 1.0
    D = A[0][0] - A[0][1] - A[1][0] + A[1][1]
    N1 = A[1][1] - A[1][0]
    N2 = A[0][0] - A[0][1]
    return float(N1 / D), float(N2 / D)


def adv22gd(A: np.ndarray, x1: float, x2: float) -> tuple[float, int]:
    a1 = A[0][0] * x1 + A[1][0] * x2
    a2 = A[0][1] * x1 + A[1][1] * x2
    if a1 <= a2:
        return float(a1), 0
    return float(a2), 1


def val22(A: np.ndarray, x0: float, y0: float) -> float:
    x1 = 1.0 - x0
    y1 = 1.0 - y0
    return float(A[0][0] * x0 * y0 + A[0][1] * x0 * y1 + A[1][0] * x1 * y0 + A[1][1] * x1 * y1)


def update(A: np.ndarray, x1: float, j: int, t: int, error: float) -> float:
    a = A[0][j] - A[1][j]
    D = abs(A[0][0] - A[0][1] - A[1][0] + A[1][1])
    z1, z2 = nash1(A)
    if z1 > 0 and z2 > 0:
        xmax = min(1.0, x1 + error / max(D, 1e-12))
        xmin = max(0.0, x1 - error / max(D, 1e-12))
        if a >= 0:
            return min(x1 + a * max(1.0, math.log(max(t, 2))) / (max(D, 1e-12) * t), xmax)
        return max(x1 + a * max(1.0, math.log(max(t, 2))) / (max(D, 1e-12) * t), xmin)
    if a > 0:
        return min(x1 + 1.0 * max(1.0, math.log(max(t, 2))) / (2 * t), 1.0)
    if a < 0:
        return max(x1 - 1.0 * max(1.0, math.log(max(t, 2))) / (2 * t), 0.0)
    return 0.5


def advnew(mode: str, x1: float, T: int) -> tuple[float, float]:
    if mode == "adv1":
        # Regret-log1
        if x1 < 1 / 3:
            return 1.0, 0.0
        if x1 > 1 / 3:
            return 0.0, 1.0
        return 1 / 3, 2 / 3
    if mode == "adv2":
        # Regret-log2
        eps = (1.0 / T) ** 0.5
        if x1 < 1 / 3 - eps:
            return 1.0, 0.0
        if x1 > 1 / 3 + eps:
            return 0.0, 1.0
        return 1 / 3, 2 / 3
    if mode == "adv3":
        # Regret-log3 first half equilibrium behavior
        return 1 / 3, 2 / 3
    raise ValueError(f"Unknown mode: {mode}")


def _base_matrix_and_value() -> tuple[np.ndarray, float]:
    B = np.array([[2 / 3, 0.0], [0.0, 1 / 3]], dtype=float)
    V = (B[0][0] * B[1][1] - B[0][1] * B[1][0]) / (B[0][0] - B[1][0] - B[0][1] + B[1][1])
    return B, float(V)


def run_ucb(seed: int, horizon: int, adv_mode: str) -> float:
    rng = np.random.default_rng(seed)
    T1 = int(horizon / 2)
    B, V = _base_matrix_and_value()
    B1 = np.zeros((2, 2), dtype=float)
    U1 = np.zeros((2, 2), dtype=float)
    N1 = np.zeros((2, 2), dtype=float)
    reg = 0.0

    for _ in range(T1):
        x1, x2 = nash1(U1)
        it = rng.binomial(1, x2)
        y1, y2 = advnew(adv_mode, x1, horizon)
        val = val22(B, x1, y1)
        jt = rng.binomial(1, y2)
        a = rng.binomial(1, B[it][jt])
        B1[it][jt] = (B1[it][jt] * N1[it][jt] + a) / (N1[it][jt] + 1)
        reg += V - val
        for ui in range(2):
            for uj in range(2):
                U1[ui][uj] = B1[ui][uj] + ((2 * np.log(8 * horizon**2)) / (N1[ui][uj] + 1)) ** 0.5
        N1[it][jt] += 1

    for _ in range(T1):
        x1, x2 = nash1(U1)
        it = rng.binomial(1, x2)
        val, jt = adv22gd(B, x1, x2)
        a = rng.binomial(1, B[it][jt])
        B1[it][jt] = (B1[it][jt] * N1[it][jt] + a) / (N1[it][jt] + 1)
        reg += V - val
        for ui in range(2):
            for uj in range(2):
                U1[ui][uj] = B1[ui][uj] + ((2 * np.log(8 * horizon**2)) / (N1[ui][uj] + 1)) ** 0.5
        N1[it][jt] += 1

    return float(max(reg, 1e-12))


def run_our_algo(seed: int, horizon: int, adv_mode: str) -> float:
    rng = np.random.default_rng(seed)
    T1 = int(horizon / 2)
    B, V = _base_matrix_and_value()
    B2 = np.zeros((2, 2), dtype=float)
    U2 = np.zeros((2, 2), dtype=float)
    F2 = np.zeros((2, 2), dtype=float)
    N2 = np.zeros((2, 2), dtype=float)
    jt = 0
    x1 = 0.5
    count0 = 1
    t0 = 1
    error = 1.0
    reg = 0.0

    for t in range(T1):
        if count0 > 0 and t > np.log(max(horizon, 2)) ** 2:
            x1 = update(F2, x1, jt, t0, error)
            count0 -= 1
        else:
            t0 = t + 1
            F2[:, :] = U2[:, :]
            z1, z2 = nash1(F2)
            if z1 > 0 and z2 > 0:
                x1 = z1
            x1 = update(F2, x1, jt, t0, error)
            count0 = t0 - 1
        x2 = 1.0 - x1
        it = rng.binomial(1, x2)
        y1, y2 = advnew(adv_mode, x1, horizon)
        val = val22(B, x1, y1)
        jt = rng.binomial(1, y2)
        a = rng.binomial(1, B[it][jt])
        B2[it][jt] = (B2[it][jt] * N2[it][jt] + a) / (N2[it][jt] + 1)
        reg += V - val
        maxdev = 0.0
        for ui in range(2):
            for uj in range(2):
                dev = ((2 * np.log(8 * horizon**2)) / (N2[ui][uj] + 1)) ** 0.5
                if maxdev < dev:
                    maxdev = dev
                U2[ui][uj] = B2[ui][uj] + dev
        if maxdev < error:
            error = maxdev
        N2[it][jt] += 1

    x1, x2 = nash1(B2)
    for _ in range(T1):
        x1 = update(B2, x1, jt, max(T1, 1), error)
        x2 = 1.0 - x1
        val, jt = adv22gd(B, x1, x2)
        reg += V - val

    return float(max(reg, 1e-12))


def run_exp3(seed: int, horizon: int, adv_mode: str) -> float:
    rng = np.random.default_rng(seed)
    T1 = int(horizon / 2)
    eta = (np.log(2) / max(1, horizon)) ** 0.5
    W = [0.0, 0.0]
    B, V = _base_matrix_and_value()
    N3 = np.zeros((2, 2), dtype=float)
    x = [0.0, 0.0]
    reg = 0.0

    for _ in range(T1):
        x[0] = np.exp(-eta * W[0]) / (np.exp(-eta * W[0]) + np.exp(-eta * W[1]))
        x[1] = 1.0 - x[0]
        it = rng.binomial(1, x[1])
        y1, y2 = advnew(adv_mode, x[0], horizon)
        val = val22(B, x[0], y1)
        jt = rng.binomial(1, y2)
        a = rng.binomial(1, B[it][jt])
        l = (1 - a) / max(x[it], 1e-12)
        reg += V - val
        N3[it][jt] += 1
        W[it] = W[it] + l
        mn = min(W[0], W[1])
        W[0] -= mn
        W[1] -= mn

    for _ in range(T1):
        x[0] = np.exp(-eta * W[0]) / (np.exp(-eta * W[0]) + np.exp(-eta * W[1]))
        x[1] = 1.0 - x[0]
        it = rng.binomial(1, x[1])
        val, jt = adv22gd(B, x[0], x[1])
        a = rng.binomial(1, B[it][jt])
        l = (1 - a) / max(x[it], 1e-12)
        reg += V - val
        N3[it][jt] += 1
        W[it] = W[it] + l
        mn = min(W[0], W[1])
        W[0] -= mn
        W[1] -= mn

    return float(max(reg, 1e-12))


def run(config: RunConfig) -> None:
    ensure_dir("plots")
    x_axis = np.asarray(config.horizons, dtype=float)
    adv_specs = [
        ("adv1", "Adversary 1 (Regret-log1)"),
        ("adv2", "Adversary 2 (Regret-log2)"),
        ("adv3", "Adversary 3 (Regret-log3)"),
    ]
    algo_specs = [
        ("UCB", run_ucb, "#1f77b4"),
        ("Our-Algo", run_our_algo, "#ff7f0e"),
        ("EXP3", run_exp3, "#2ca02c"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, (adv_key, adv_title) in zip(axes, adv_specs):
        for label, fn, color in algo_specs:
            means: list[float] = []
            stds: list[float] = []
            for T in config.horizons:
                vals = []
                for r in range(config.n_runs):
                    seed = config.seed + 100003 * r + 97 * T + (1 if adv_key == "adv1" else 2 if adv_key == "adv2" else 3)
                    vals.append(fn(seed, T, adv_key))
                arr = np.log10(np.maximum(np.asarray(vals, dtype=float), 1e-12))
                means.append(float(np.mean(arr)))
                stds.append(float(np.std(arr)))
            y = np.asarray(means, dtype=float)
            ci = np.asarray(stds, dtype=float)
            ax.plot(x_axis, y, marker="o", label=label, color=color)
            ax.fill_between(x_axis, y - ci, y + ci, alpha=0.25, color=color)

        ax.set_xscale("log")
        ax.grid(True, which="both", ls=":")
        ax.set_title(adv_title)
        ax.set_xlabel("Horizon T")

    axes[0].set_ylabel("log10(Total Nash regret)")
    axes[0].legend(loc="best")
    fig.suptitle(f"Section 4 official-style (preset={config.preset}, runs={config.n_runs}, horizons={config.horizons})")
    plt.tight_layout()
    plt.savefig(f"plots/section4_official_style_{config.preset}.png", dpi=170)
    plt.show()


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default="quick", choices=["quick", "medium", "paper-lite", "final", "paper"])
    p.add_argument("--horizons", type=int, nargs="*", default=None, help="Optional custom horizons list.")
    p.add_argument("--n_runs", type=int, default=None, help="Optional override of preset trial count.")
    p.add_argument("--seed", type=int, default=19)
    a = p.parse_args()
    preset_horizons, preset_runs = section4_horizons_for_preset(a.preset)
    horizons = a.horizons if a.horizons else preset_horizons
    n_runs = a.n_runs if a.n_runs is not None else preset_runs
    return RunConfig(horizons=sorted(horizons), n_runs=n_runs, seed=a.seed, preset=a.preset)


if __name__ == "__main__":
    run(parse_args())

