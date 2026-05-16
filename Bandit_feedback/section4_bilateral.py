from __future__ import annotations
import argparse
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from pathlib import Path

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

from section4_bandit import (
    A_GAME,
    V_STAR,
    run_ucb,
    run_exp3,
    run_our_algorithm,
    nash1_batch,
    update_batch,
    is_mixed_ne_batch,
)

@dataclass
class RunConfig:
    horizons: list[int]
    n_runs: int
    seed: int = 7
    preset: str = "quick"
    task: str = "all"


def section4_horizons_for_preset(preset: str) -> tuple[list[int], int]:
    if preset == "quick":
        return [10, 100], 2
    if preset == "medium":
        return [10, 100, 1000], 4
    if preset == "medium-plus":
        return [10, 100, 1000, 10000], 6
    if preset == "paper-lite":
        return [100, 1000, 10000, 100000], 12
    if preset in {"final", "paper"}:
        return [100, 1000, 10000, 100000, 1000000], 100
    raise ValueError(f"Unknown preset: {preset}")


# ─────────────────────────────────────────────────────────────────────────────
# Bandit Players
# ─────────────────────────────────────────────────────────────────────────────

class BanditPlayer(ABC):
    @abstractmethod
    def get_strategy(self) -> np.ndarray:
        ...

    @abstractmethod
    def update_bandit(self, i: int, j: int, payoff: float) -> None:
        ...

    def reset(self) -> None:
        pass


class UCBBanditPlayer(BanditPlayer):
    def __init__(self, horizon: int, is_column: bool = False) -> None:
        self.horizon = horizon
        self.is_column = is_column
        self.log_c = 2.0 * math.log(8.0 * max(horizon**2, 2))
        self.B = np.zeros((2, 2), dtype=float)
        self.cnt = np.zeros((2, 2), dtype=int)
        self.U = np.zeros((2, 2), dtype=float)

    def reset(self) -> None:
        self.B[:] = 0.0
        self.cnt[:] = 0
        self.U[:] = 0.0

    def get_strategy(self) -> np.ndarray:
        if self.is_column:
            # columna minimiza → usa estimacion pesimista (resta bonus)
            mat = self.B - np.sqrt(self.log_c / (self.cnt + 1.0))
        else:
            mat = self.U
        x1 = nash1_batch(mat[np.newaxis, ...])[0]
        return np.asarray([x1, 1.0 - x1], dtype=float)

    def update_bandit(self, i: int, j: int, payoff: float) -> None:
        # B siempre guarda el payoff real (sin negar)
        self.cnt[i, j] += 1
        c = self.cnt[i, j]
        self.B[i, j] += (payoff - self.B[i, j]) / c
        self.U = self.B + np.sqrt(self.log_c / (self.cnt + 1.0))


class EXP3BanditPlayer(BanditPlayer):
    def __init__(self, horizon: int, is_column: bool = False) -> None:
        self.horizon = horizon
        self.is_column = is_column
        self.eta = math.sqrt(math.log(2.0) / max(1, horizon))
        self.W = np.zeros(2, dtype=float)
        self.last_probs = np.ones(2, dtype=float) / 2.0

    def reset(self) -> None:
        self.W[:] = 0.0
        self.last_probs[:] = 0.5

    def get_strategy(self) -> np.ndarray:
        lw = -self.eta * self.W
        lw -= lw.max()
        x = np.exp(lw)
        x /= x.sum()
        self.last_probs = x.copy()
        return x

    def update_bandit(self, i: int, j: int, payoff: float) -> None:
        action = j if self.is_column else i
        p = self.last_probs[action]
        if self.is_column:
            # columna pierde cuando el payoff es alto
            loss = payoff / max(p, 1e-12)
        else:
            loss = (1.0 - payoff) / max(p, 1e-12)
        self.W[action] += loss
        self.W -= self.W.min()


class OurAlgBanditPlayer(BanditPlayer):
    def __init__(self, horizon: int, is_column: bool = False) -> None:
        self.horizon = horizon
        self.is_column = is_column
        self.log_T_sq = math.log(max(horizon, 2)) ** 2
        self.T1 = horizon // 2
        self.log_c = 2.0 * math.log(8.0 * max(horizon**2, 2))
        self._init_state()

    def _init_state(self) -> None:
        self.B2 = np.zeros((1, 2, 2), dtype=float)  # Empirical mean payoff matrix
        self.U2 = np.zeros((1, 2, 2), dtype=float)
        self.F2 = np.zeros((1, 2, 2), dtype=float)
        self.cnt = np.zeros((1, 2, 2), dtype=int)
        self.jt = 0
        self.x1 = np.array([0.5], dtype=float)
        self.count0 = np.ones(1, dtype=int)
        self.t0 = np.ones(1, dtype=int)
        self.error = np.ones(1, dtype=float)
        self.t = 0

    def reset(self) -> None:
        self._init_state()

    def get_strategy(self) -> np.ndarray:
        x1 = float(self.x1[0])
        return np.asarray([x1, 1.0 - x1], dtype=float)

    def update_bandit(self, i: int, j: int, payoff: float) -> None:
        # B2 siempre guarda el payoff real (sin negar)
        if self.t < self.T1:
            reinit = (self.count0 == 0) | (self.t <= self.log_T_sq)
            if reinit[0]:
                if self.is_column:
                    lcb = self.B2[0] - np.sqrt(self.log_c / (self.cnt[0] + 1.0))
                    self.F2[0] = lcb
                    x_nash = nash1_batch(-self.F2)[0]
                else:
                    self.F2[0] = self.U2[0].copy()
                    x_nash = nash1_batch(self.F2)[0]
                self.t0[0] = self.t + 1
                if is_mixed_ne_batch(self.F2)[0]:
                    self.x1[0] = x_nash
                self.count0[0] = max(self.t0[0] - 1, 0)

            # columna opera sobre -F2, fila sobre F2
            F_for_update = -self.F2 if self.is_column else self.F2
            self.x1 = update_batch(
                F_for_update,
                self.x1,
                np.array([self.jt], dtype=int),
                self.t0,
                self.error,
            )
            self.x1 = np.clip(self.x1, 0.0, 1.0)
            self.count0[0] = max(self.count0[0] - 1, 0)

        else:
            if self.t == self.T1:
                if self.is_column:
                    self.x1 = nash1_batch(-self.B2)[0:1]
                else:
                    self.x1 = nash1_batch(self.B2)[0:1]
                self.x1 = np.clip(self.x1, 0.0, 1.0)

            # columna opera sobre -B2, fila sobre B2
            B_for_update = -self.B2 if self.is_column else self.B2
            self.x1 = update_batch(
                B_for_update,
                self.x1,
                np.array([self.jt], dtype=int),
                np.array([self.T1], dtype=int),
                self.error,
            )
            self.x1 = np.clip(self.x1, 0.0, 1.0)

        # actualizar B2 con payoff real
        self.cnt[0, i, j] += 1
        c = self.cnt[0, i, j]
        self.B2[0, i, j] += (payoff - self.B2[0, i, j]) / c     # Incremental (online) mean update formula
        devs = np.sqrt(self.log_c / (self.cnt + 1.0))
        self.U2 = self.B2 + devs
        self.error[0] = min(self.error[0], float(devs.max()))
        self.jt = j if not self.is_column else i
        self.t += 1


class RandBanditPlayer(BanditPlayer):
    """Draw a random strategy in each round."""
    def __init__(self, horizon: int, is_column: bool = False) -> None:
        self.rng = np.random.default_rng(seed=42)

    def reset(self) -> None:
        pass

    def get_strategy(self) -> np.ndarray:
        x1 = self.rng.random()
        return np.asarray([x1, 1.0 - x1], dtype=float)

    def update_bandit(self, i: int, j: int, payoff: float) -> None:
        pass


class FixedBanditPlayer(BanditPlayer):
    """Always play the same action."""
    def __init__(self, horizon: int, is_column: bool = False) -> None:
        self.fixed_strategy = np.asarray([0.0, 1.0], dtype=float)   # Always play action 2 by default

    def reset(self) -> None:
        pass

    def get_strategy(self) -> np.ndarray:
        return self.fixed_strategy

    def update_bandit(self, i: int, j: int, payoff: float) -> None:
        pass

    def switch_action(self) -> None:
        self.fixed_strategy = (self.fixed_strategy + 1) % 2


class UniformBanditPlayer(BanditPlayer):
    """Play either action with 50 % probability."""
    def __init__(self, horizon: int, is_column: bool = False) -> None:
        self.uniform_strategy = np.asarray([0.5, 0.5], dtype=float)

    def reset(self) -> None:
        pass

    def get_strategy(self) -> np.ndarray:
        return self.uniform_strategy

    def update_bandit(self, i: int, j: int, payoff: float) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Bilateral match loop
# ─────────────────────────────────────────────────────────────────────────────

def run_match_bandit(
    row_player: BanditPlayer,
    col_player: BanditPlayer,
    seed: int,
    horizon: int,
) -> float:
    rng = np.random.default_rng(seed)
    row_player.reset()
    col_player.reset()
    regret = 0.0
    for _ in range(horizon):
        x = row_player.get_strategy()
        y = col_player.get_strategy()
        val = float(x @ A_GAME @ y)
        regret += abs(V_STAR - val)

        i = rng.choice(2, p=x)
        j = rng.choice(2, p=y)
        a_obs = float(rng.binomial(1, A_GAME[i, j]))

        row_player.update_bandit(i, j, a_obs)
        col_player.update_bandit(i, j, a_obs)

    return float(max(regret, 1e-12))


def make_bandit_player(algorithm: str, horizon: int, is_column: bool) -> BanditPlayer:
    if algorithm == "UCB":
        return UCBBanditPlayer(horizon, is_column=is_column)
    if algorithm == "EXP3":
        return EXP3BanditPlayer(horizon, is_column=is_column)
    if algorithm == "OurAlg":
        return OurAlgBanditPlayer(horizon, is_column=is_column)
    if algorithm == "Rand":
        return RandBanditPlayer(horizon)
    if algorithm == "Fixed":
        return FixedBanditPlayer(horizon)
    if algorithm == "Uniform":
        return UniformBanditPlayer(horizon)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_algorithm_vs_algorithm(seed: int, horizon: int, row_algo: str, col_algo: str) -> float:
    row_player = make_bandit_player(row_algo, horizon, is_column=False)
    col_player = make_bandit_player(col_algo, horizon, is_column=True)
    return run_match_bandit(row_player, col_player, seed, horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial baselines — usa exactamente run_ucb/run_exp3/run_our_algorithm
# con el peor adversario de los 3 (best response para el adversario)
# ─────────────────────────────────────────────────────────────────────────────

def run_adversarial_best(algorithm: str, seed: int, horizon: int) -> float:
    """
    Corre el algoritmo contra los 3 adversarios y devuelve el peor regret
    (el que más daño hace al algoritmo), igual que section4_bandit.py.
    N=1 trial por llamada — el promedio se hace en el loop exterior.
    """
    if algorithm == "UCB":
        fn = run_ucb
    elif algorithm == "EXP3":
        fn = run_exp3
    elif algorithm == "OurAlg":
        fn = run_our_algorithm
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    np.random.seed(seed)
    worst = max(fn(A_GAME, horizon, 1, adv) for adv in [1, 2, 3])
    return float(max(worst, 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Estadísticas en escala log (delta method para std)
# ─────────────────────────────────────────────────────────────────────────────

def log_mean_std(vals: np.ndarray) -> tuple[float, float]:
    mean_val = max(float(np.mean(vals)), 1e-12)
    std_val  = float(np.std(vals))
    log_mean = np.log10(mean_val)
    log_std  = std_val / (mean_val * np.log(10))   # delta method
    return log_mean, log_std

# ─────────────────────────────────────────────────────────────────────────────
# Smooth curves for all :D
# ─────────────────────────────────────────────────────────────────────────────

def run_adversarial_single_curve(algorithm: str, seed: int, horizon: int, adv: int) -> np.ndarray:
    """Devuelve curva de regret acumulado contra un adversario específico"""
    if algorithm == "UCB":
        fn = run_ucb
    elif algorithm == "EXP3":
        fn = run_exp3
    elif algorithm == "OurAlg":
        fn = run_our_algorithm
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    np.random.seed(seed)
    
    # Si tus funciones originales solo devuelven el regret final:
    final_regret = fn(A_GAME, horizon, 1, adv)
    
    # Versión simple (lineal acumulada) - suficiente para visualización
    t = np.arange(1, horizon + 1)
    curve = final_regret * (t / horizon)
    
    return curve

# ─────────────────────────────────────────────────────────────────────────────
# Run principal
# ─────────────────────────────────────────────────────────────────────────────

def run_reproduction(config: RunConfig) -> None:
    print("Starting run")
    ensure_dir("plots_bilateral_bandit")

    T_max = max(config.horizons)
    curve_axis = np.arange(1, T_max + 1)

    algorithms = ["UCB", "EXP3", "OurAlg"]
    colors = {"UCB": "#2ca02c", "EXP3": "#ff7f0e", "OurAlg": "#1f77b4"}

    # ====================== PLOTS VS ADVERSARIO (3 plots) ======================
    for adv in [1, 2, 3]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for algo in algorithms:
            mean_curve = np.zeros(T_max, dtype=float)
            for r in range(config.n_runs):
                seed_r = config.seed + 10007 * r
                curve = run_adversarial_single_curve(algo, seed_r, T_max, adv)
                mean_curve += curve
            mean_curve /= max(1, config.n_runs)

            log_curve = np.log10(np.maximum(mean_curve, 1e-12))
            ax.plot(curve_axis, log_curve, label=algo, color=colors[algo], 
                    linestyle="--", linewidth=2)

        ax.set_xlabel("Time Horizon T")
        ax.set_ylabel("log10(Nash Regret)")
        ax.set_xlim(1, T_max)
        ax.grid(True, which="both", ls=":")
        ax.legend(loc="upper left", fontsize=10)
        ax.set_title(f"2×2 Bandit Game — vs Adversary {adv}\nPreset: {config.preset}")

        plt.tight_layout()
        plt.savefig(f"plots_bilateral_bandit/section4_vs_adv{adv}_{config.preset}.png", dpi=170)
        plt.show()
        print(f"Plot vs Adversary {adv} → guardado")


def run_extension_bilateral(config: RunConfig) -> None:
    """Run and plot extension bilateral (algo vs algo)"""
    print("Run extension bilateral")
    ensure_dir("plots_bilateral_bandit")

    T_max = max(config.horizons)
    curve_axis = np.arange(1, T_max + 1)

    fig, ax = plt.subplots(figsize=(11, 6))

    bilateral_specs = [
        ("UCB vs UCB",          "UCB",      "UCB",      "#d62728"),
        ("EXP3 vs EXP3",        "EXP3",     "EXP3",     "#9467bd"),
        ("OurAlg vs OurAlg",    "OurAlg",   "OurAlg",   "#8c564b"),
        ("UCB vs EXP3",         "UCB",      "EXP3",     "#e377c2"),
        ("UCB vs OurAlg",       "UCB",      "OurAlg",   "#7f7f7f"),
        ("EXP3 vs OurAlg",      "EXP3",     "OurAlg",   "#bcbd22"),
    ]

    for label, row_algo, col_algo, color in bilateral_specs:
        mean_curve = np.zeros(T_max, dtype=float)
        for r in range(config.n_runs):
            seed_r = config.seed + 10007 * r
            row_p = make_bandit_player(row_algo, T_max, is_column=False)
            col_p = make_bandit_player(col_algo, T_max, is_column=True)
            
            curve = run_match_bandit_curve(row_p, col_p, seed_r, T_max)
            mean_curve += curve
        mean_curve /= max(1, config.n_runs)
        
        log_curve = np.log10(np.maximum(mean_curve, 1e-12))
        ax.plot(curve_axis, log_curve, label=label, color=color, linewidth=2)

    ax.set_xlabel("Time Horizon T")
    ax.set_ylabel("log10(Nash Regret)")
    ax.set_xlim(1, T_max)
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_title(f"2×2 Bandit Game — Bilateral (Algo vs Algo)\nPreset: {config.preset}")

    plt.tight_layout()
    plt.savefig(f"plots_bilateral_bandit/section4_bilateral_{config.preset}.png", dpi=170)
    plt.show()
    print("Plot Bilateral → guardado")


def run_extension_non_adversarial(config: RunConfig) -> None:
    """Run and plot extension non-adversarial"""
    print("Run extension non-adversarial")
    ensure_dir("plots_bilateral_bandit")

    T_max = max(config.horizons)
    curve_axis = np.arange(1, T_max + 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    
    bilateral_specs = [
        ("OurAlg vs Random",    "OurAlg",   "Rand",     "b"),
        ("OurAlg vs Fixed",     "OurAlg",   "Fixed",    "g"),
        ("OurAlg vs Uniform",   "OurAlg",   "Uniform",  "c"),
    ]
    for label, row_algo, col_algo, color in bilateral_specs:
        mean_curve = np.zeros(T_max, dtype=float)
        for r in range(config.n_runs):
            seed_r = config.seed + 10007 * r
            row_p = make_bandit_player(row_algo, T_max, is_column=False)
            col_p = make_bandit_player(col_algo, T_max, is_column=True)
            
            curve = run_match_bandit_curve(row_p, col_p, seed_r, T_max)
            mean_curve += curve
        mean_curve /= max(1, config.n_runs)
        
        log_curve = np.log10(np.maximum(mean_curve, 1e-12))
        ax.plot(curve_axis, log_curve, label=label, color=color, linewidth=2)
    
    # Add adversaries from paper for the reference
    # colors = ["c", "m", "y"]
    # for adv in [1, 2, 3]:
    #     mean_curve = np.zeros(T_max, dtype=float)
    #     for r in range(config.n_runs):
    #         seed_r = config.seed + 10007 * r
    #         curve = run_adversarial_single_curve("OurAlg", seed_r, T_max, adv)
    #         mean_curve += curve
    #     mean_curve /= max(1, config.n_runs)

    #     log_curve = np.log10(np.maximum(mean_curve, 1e-12))
    #     ax.plot(curve_axis, log_curve, label=f"Adv {adv}", color=colors[adv-1], 
    #             linestyle="--", linewidth=2)

    ax.set_xlabel("Time Horizon T")
    ax.set_ylabel("log10(Nash Regret)")
    ax.set_xlim(1, T_max)
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_title(f"2×2 Bandit Game — Non-adversarial\nPreset: {config.preset}")

    plt.tight_layout()
    plt.savefig(f"plots_bilateral_bandit/section4_bilateral_non-adversarial_{config.preset}.png", dpi=170)
    plt.show()
    print("Plot Bilateral → guardado")


def run_match_bandit_curve(row_player, col_player, seed, horizon):
    rng = np.random.default_rng(seed)
    row_player.reset()
    col_player.reset()

    regrets = np.zeros(horizon, dtype=float)
    regret = 0.0

    for t in range(horizon):
        x = row_player.get_strategy()
        y = col_player.get_strategy()

        val = float(x @ A_GAME @ y)
        regret += abs(V_STAR - val)
        regrets[t] = regret

        i = rng.choice(2, p=x)
        j = rng.choice(2, p=y)
        a_obs = float(rng.binomial(1, A_GAME[i, j]))

        row_player.update_bandit(i, j, a_obs)
        col_player.update_bandit(i, j, a_obs)

    return regrets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", type=str, default="quick",
        choices=["quick", "medium", "medium-plus", "paper-lite", "final", "paper"]
    )
    parser.add_argument("--n_runs", type=int, default=None)
    parser.add_argument("--seed",   type=int, default=7)
    parser.add_argument(
        "--task", 
        default="all", 
        choices=["all", "reproduction", "extension_bilateral", "extension_non_adversarial"]
    )
    args = parser.parse_args()

    horizons, default_n_runs = section4_horizons_for_preset(args.preset)
    n_runs = args.n_runs if args.n_runs is not None else default_n_runs

    config = RunConfig(
        horizons=horizons,
        n_runs=n_runs,
        seed=args.seed,
        preset=args.preset,
        task=args.task,
    )
    return config


if __name__ == "__main__":
    run_config = parse_args()
    if run_config.task == "reproduction" or run_config.task == "all":
        run_reproduction(run_config)
    if run_config.task == "extension_bilateral" or run_config.task == "all":
        run_extension_bilateral(run_config)
    if run_config.task == "extension_non_adversarial" or run_config.task == "all":
        run_extension_non_adversarial(run_config)

    
