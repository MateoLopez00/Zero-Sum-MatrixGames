from __future__ import annotations
import argparse
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.optimize import linprog

from core.utils import ensure_dir

@dataclass
class RunConfig:
    horizons: list[int]
    n_runs: int
    seed: int = 7
    preset: str = "quick"
    n_actions: int = 20
    variant: str = "official"


def section3_horizons_for_preset(preset: str) -> tuple[list[int], int]:
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

def generate_bernoulli_diagonal_matrix(
    A: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n = A.shape[0]
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        B[i, i] = rng.binomial(1, A[i, i])
    return B

def _value_of_diag_game(A: np.ndarray) -> float:
    den = 0.0
    for i in range(A.shape[0]):
        den += 1.0 / A[i, i]
    return 1.0 / den

# ─────────────────────────────────────────────────────────────────────────────
# Original Algo Just Row
# ─────────────────────────────────────────────────────────────────────────────

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


def run_hedge(seed: int, horizon: int, n: int) -> float:
    rng = np.random.default_rng(seed)
    B = generate_diagonal_matrix(n)
    V = _value_of_diag_game(B)
    weights = np.ones(n, dtype=float)
    eta = math.sqrt(math.log(max(n, 2)) / max(1, horizon))
    reg = 0.0
    for _ in range(horizon):
        x = weights / np.sum(weights)
        val, idx = adversary(B, x)
        reg += V - val
        Bsamp = generate_bernoulli_diagonal_matrix(B, rng)
        reward_vector = Bsamp[:, idx]
        weights *= np.exp(eta * reward_vector)
    return float(max(reg, 1e-12))


def _update_official_diag(
    A: np.ndarray, x1: np.ndarray, j: int, t: int
) -> np.ndarray:
    vec = A[:-1, j] - A[-1, j]
    x1[:-1] = np.clip(x1[:-1] + vec * (1.0 / max(1, t)), 0.0, 1.0)
    s = float(np.sum(x1[:-1]))
    if s > 1.0:
        x1[:-1] /= s
        x1[-1] = 0.0
    else:
        x1[-1] = 1.0 - s
    return x1


def run_official_diag_algo(seed: int, horizon: int, n: int) -> float:
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
        #reg += V - val
    return float(max(reg, 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# setting bilateral
# ─────────────────────────────────────────────────────────────────────────────

class Player(ABC):

    @abstractmethod
    def get_strategy(self) -> np.ndarray:
        ...

    @abstractmethod
    def update(self, A_sample: np.ndarray, opponent_strategy: np.ndarray) -> None:
        ...

    def reset(self) -> None:
        pass


class HedgeRowPlayer(Player):

    def __init__(self, n: int, horizon: int) -> None:
        self.n = n
        self.horizon = horizon
        self.eta = math.sqrt(math.log(max(n, 2)) / max(1, horizon))
        self.weights = np.ones(n, dtype=float)

    def reset(self) -> None:
        self.weights = np.ones(self.n, dtype=float)

    def get_strategy(self) -> np.ndarray:
        w = self.weights
        return w / w.sum()

    def update(self, A_sample: np.ndarray, opponent_strategy: np.ndarray) -> None:
        reward = A_sample @ opponent_strategy
        self.weights *= np.exp(self.eta * reward)


class HedgeColumnPlayer(Player):

    def __init__(self, m: int, horizon: int) -> None:
        self.m = m
        self.horizon = horizon
        self.eta = math.sqrt(math.log(max(m, 2)) / max(1, horizon))
        self.weights = np.ones(m, dtype=float)

    def reset(self) -> None:
        self.weights = np.ones(self.m, dtype=float)

    def get_strategy(self) -> np.ndarray:
        w = self.weights
        return w / w.sum()

    def update(self, A_sample: np.ndarray, opponent_strategy: np.ndarray) -> None:
        # negativo: columna minimiza
        reward = -(A_sample.T @ opponent_strategy)
        self.weights *= np.exp(self.eta * reward)


class NashRowPlayer(Player):

    def __init__(self, n: int) -> None:
        self.n = n
        self.Abar_accum = np.zeros((n, n), dtype=float)
        self.t = 0

    def reset(self) -> None:
        self.Abar_accum = np.zeros((self.n, self.n), dtype=float)
        self.t = 0

    def get_strategy(self) -> np.ndarray:
        if self.t == 0:
            return np.ones(self.n, dtype=float) / self.n
        return nash1_diag(self.Abar_accum / self.t)

    def update(self, A_sample: np.ndarray, opponent_strategy: np.ndarray) -> None:
        self.Abar_accum += A_sample
        self.t += 1


class NashColumnPlayer(Player):

    def __init__(self, m: int) -> None:
        self.m = m
        self.Abar_accum = np.zeros((m, m), dtype=float)
        self.t = 0

    def reset(self) -> None:
        self.Abar_accum = np.zeros((self.m, self.m), dtype=float)
        self.t = 0

    def get_strategy(self) -> np.ndarray:
        if self.t == 0:
            return np.ones(self.m, dtype=float) / self.m
        return nash1_diag(self.Abar_accum / self.t)

    def update(self, A_sample: np.ndarray, opponent_strategy: np.ndarray) -> None:
        self.Abar_accum += A_sample
        self.t += 1


class OurRowPlayer(Player):

    def __init__(self, n: int, horizon: int) -> None:
        self.n = n
        self.horizon = horizon
        self._init_state()

    def _init_state(self) -> None:
        n, horizon = self.n, self.horizon
        self.Abar_accum = np.zeros((n, n), dtype=float)
        self.t = 0
        self.log_T = math.log(max(horizon, 2))
        self.t_star = min(max(n, int(math.ceil(self.log_T ** 2))), horizon // 2)
        self._phase = "burnin"
        self._delta = None
        self._x_prime = None
        self._A_hat = None
        self._eta_sub = None
        self._clip = None
        self._t0 = self.t_star

    def reset(self) -> None:
        self._init_state()

    def get_strategy(self) -> np.ndarray:
        if self._phase == "burnin" or self._delta is None:
            if self.t == 0:
                return np.ones(self.n, dtype=float) / self.n
            return nash1_diag(self.Abar_accum / self.t)
        n = self.n
        vec = np.empty(n, dtype=float)
        vec[:-1] = self._delta
        vec[-1] = -float(np.sum(self._delta))
        x_t = self._x_prime + vec
        x_t = np.clip(x_t, 0.0, 1.0)
        s = x_t.sum()
        return x_t / s if s > 0 else np.ones(n) / n

    def update(self, A_sample: np.ndarray, opponent_strategy: np.ndarray) -> None:
        self.Abar_accum += A_sample
        self.t += 1
        if self._phase == "burnin":
            if self.t >= self.t_star:
                self._init_subroutine()
            return
        # Use full opponent strategy (mixed) instead of just argmax
        g = (self._A_hat[:-1, :] - self._A_hat[-1, :]) @ opponent_strategy
        self._delta = np.clip(
            self._delta + self._eta_sub * g, -self._clip, self._clip
        )

    def _init_subroutine(self) -> None:
        n = self.n
        Abar = self.Abar_accum / max(1, self.t)
        diag_vals = np.maximum(np.diag(Abar), 1e-6)
        D1 = max(float(np.min(diag_vals)) * n, 1.0)
        T1 = max(self._t0 / self.log_T, 1.0)
        self._x_prime = nash1_diag(Abar)
        self._A_hat = Abar.copy()
        self._eta_sub = 1.0 / (D1 * T1)
        self._clip = 1.0 / (D1 * math.sqrt(T1))
        self._delta = np.full(n - 1, -self._clip, dtype=float)
        self._phase = "subroutine"


class OurColumnPlayer(Player):

    def __init__(self, m: int, horizon: int) -> None:
        self.m = m
        self.horizon = horizon
        self._init_state()

    def _init_state(self) -> None:
        m, horizon = self.m, self.horizon
        self.Abar_accum = np.zeros((m, m), dtype=float)
        self.t = 0
        self.log_T = math.log(max(horizon, 2))
        self.t_star = min(max(m, int(math.ceil(self.log_T ** 2))), horizon // 2)
        self._phase = "burnin"
        self._delta = None
        self._y_prime = None
        self._A_hat = None
        self._eta_sub = None
        self._clip = None
        self._t0 = self.t_star

    def reset(self) -> None:
        self._init_state()

    def get_strategy(self) -> np.ndarray:
        if self._phase == "burnin" or self._delta is None:
            if self.t == 0:
                return np.ones(self.m, dtype=float) / self.m
            return nash1_diag(self.Abar_accum / self.t)
        m = self.m
        vec = np.empty(m, dtype=float)
        vec[:-1] = self._delta
        vec[-1] = -float(np.sum(self._delta))
        y_t = self._y_prime + vec
        y_t = np.clip(y_t, 0.0, 1.0)
        s = y_t.sum()
        return y_t / s if s > 0 else np.ones(m) / m

    def update(self, A_sample: np.ndarray, opponent_strategy: np.ndarray) -> None:
        self.Abar_accum += A_sample
        self.t += 1
        if self._phase == "burnin":
            if self.t >= self.t_star:
                self._init_subroutine()
            return
        # Minimize: use full opponent strategy (mixed) instead of just argmax
        g = -(self._A_hat[:, :-1].T @ opponent_strategy - self._A_hat[:, -1] @ opponent_strategy)
        self._delta = np.clip(
            self._delta + self._eta_sub * g, -self._clip, self._clip
        )

    def _init_subroutine(self) -> None:
        m = self.m
        Abar = self.Abar_accum / max(1, self.t)
        diag_vals = np.maximum(np.diag(Abar), 1e-6)
        D1 = max(float(np.min(diag_vals)) * m, 1.0)
        T1 = max(self._t0 / self.log_T, 1.0)
        self._y_prime = nash1_diag(Abar)
        self._A_hat = Abar.copy()
        self._eta_sub = 1.0 / (D1 * T1)
        self._clip = 1.0 / (D1 * math.sqrt(T1))
        self._delta = np.full(m - 1, -self._clip, dtype=float)
        self._phase = "subroutine"


# ─────────────────────────────────────────────────────────────────────────────
# Loop bilateral & combo
# ─────────────────────────────────────────────────────────────────────────────

def run_match(
    row_player: Player,
    col_player: Player,
    seed: int,
    horizon: int,
    n: int,
) -> float:
    rng = np.random.default_rng(seed)
    A = generate_diagonal_matrix(n)
    V = _value_of_diag_game(A)
    row_player.reset()
    col_player.reset()
    reg = 0.0
    for _ in range(horizon):
        x = row_player.get_strategy()
        y = col_player.get_strategy()
        val = float(x @ A @ y)
        reg += abs(V - val)
        A_sample = generate_bernoulli_diagonal_matrix(A, rng)
        row_player.update(A_sample, y)
        col_player.update(A_sample, x)
    return float(reg)


def run_match_curve(
    row_player: Player,
    col_player: Player,
    seed: int,
    horizon: int,
    n: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = generate_diagonal_matrix(n)
    V = _value_of_diag_game(A)
    row_player.reset()
    col_player.reset()
    regrets = np.zeros(horizon, dtype=float)
    reg = 0.0
    for t in range(horizon):
        x = row_player.get_strategy()
        y = col_player.get_strategy()
        val = float(x @ A @ y)
        reg += abs(V - val)
        regrets[t] = reg
        A_sample = generate_bernoulli_diagonal_matrix(A, rng)
        row_player.update(A_sample, y)
        col_player.update(A_sample, x)
    return regrets


def run_our_vs_our(seed: int, horizon: int, n: int) -> float:
    return run_match(
        OurRowPlayer(n, horizon), OurColumnPlayer(n, horizon), seed, horizon, n
    )

def run_hedge_vs_hedge(seed: int, horizon: int, n: int) -> float:
    return run_match(
        HedgeRowPlayer(n, horizon), HedgeColumnPlayer(n, horizon), seed, horizon, n
    )

def run_our_vs_hedge(seed: int, horizon: int, n: int) -> float:
    return run_match(
        OurRowPlayer(n, horizon), HedgeColumnPlayer(n, horizon), seed, horizon, n
    )

def run_nash_vs_nash(seed: int, horizon: int, n: int) -> float:
    return run_match(
        NashRowPlayer(n), NashColumnPlayer(n), seed, horizon, n
    )

def run_our_vs_nash(seed: int, horizon: int, n: int) -> float:
    return run_match(
        OurRowPlayer(n, horizon), NashColumnPlayer(n), seed, horizon, n
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run(config: RunConfig) -> None:
    ensure_dir("plots_bilateral")
    horizons_axis = np.asarray(config.horizons, dtype=float)

    # Algoritmos originales (adversario best-response)
    original_specs = [
        ("Hedge (adv)",    run_hedge,              "#2ca02c"),
        ("Our-Algo (adv)", run_official_diag_algo, "#ff7f0e"),
        ("Nash (adv)",     run_nash,               "#1f77b4"),
    ]

    # Algoritmos bilaterales (ambos jugadores aprenden)
    bilateral_player_specs = [
        ("Our vs Our",     lambda n, T: (OurRowPlayer(n, T), OurColumnPlayer(n, T)), "#9467bd"),
        ("Hedge vs Hedge", lambda n, T: (HedgeRowPlayer(n, T), HedgeColumnPlayer(n, T)), "#8c564b"),
        ("Our vs Hedge",   lambda n, T: (OurRowPlayer(n, T), HedgeColumnPlayer(n, T)), "#e377c2"),
        ("Nash vs Nash",   lambda n, T: (NashRowPlayer(n), NashColumnPlayer(n)), "#7f7f7f"),
        ("Our vs Nash",    lambda n, T: (OurRowPlayer(n, T), NashColumnPlayer(n)), "#bcbd22"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, fn, color in original_specs:
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

        y  = np.asarray(means, dtype=float)
        ci = np.asarray(stds,  dtype=float)
        ax.plot(horizons_axis, y, marker="o", label=label, color=color, linestyle="--")
        ax.fill_between(horizons_axis, y - ci, y + ci, alpha=0.15, color=color)

    T_max = max(config.horizons)
    curve_axis = np.arange(1, T_max + 1)

    for label, make_players, color in bilateral_player_specs:
        mean_curve = np.zeros(T_max, dtype=float)
        for r in range(config.n_runs):
            seed = config.seed + 10007 * r
            row_player, col_player = make_players(config.n_actions, T_max)
            mean_curve += run_match_curve(
                row_player, col_player, seed, T_max, config.n_actions
            )
        mean_curve /= max(1, config.n_runs)
        log_curve = np.log10(np.maximum(mean_curve, 1e-12))
        ax.plot(curve_axis, log_curve, label=label, color=color, linestyle="-")

    ax.set_xscale("linear")
    ax.set_xlabel("Time Horizon T")
    ax.set_ylabel("Log of Nash Regret (row player)")
    ax.set_xlim(1, T_max)
    ax.set_ylim(None, None)  # Automatic scaling
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title(
         f"{config.n_actions}×{config.n_actions} diagonal matrix — "
         f"original (--) vs bilateral (—)"
    )
    #ax.set_title(
    #    f"{config.n_actions}×{config.n_actions} diagonal matrix — Our vs Our"
    #)
    plt.tight_layout()
    # plt.savefig(
    #     f"plots/section3_bilateral_{config.preset}_n{config.n_actions}.png",
    #     dpi=170,
    # )
    plt.savefig(
        f"plots_bilateral/section3_bilateral_{config.preset}_n{config.n_actions}.png",
        dpi=170,
    )
    plt.show()


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preset", default="quick",
        choices=["quick", "medium", "paper-lite", "final", "paper"]
    )
    p.add_argument("--horizons", type=int, nargs="*", default=None)
    p.add_argument("--n_runs",   type=int, default=None)
    p.add_argument("--seed",     type=int, default=7)
    p.add_argument("--n_actions", type=int, default=20)
    p.add_argument(
        "--variant", default="subroutine",
        choices=["official", "subroutine", "theory-lp"]
    )
    a = p.parse_args()
    preset_horizons, preset_runs = section3_horizons_for_preset(a.preset)
    horizons = a.horizons if a.horizons else preset_horizons
    n_runs   = a.n_runs   if a.n_runs is not None else preset_runs
    return RunConfig(
        horizons=sorted(horizons),
        n_runs=n_runs,
        seed=a.seed,
        preset=a.preset,
        n_actions=a.n_actions,
        variant=a.variant,
    )


if __name__ == "__main__":
    run(parse_args())
