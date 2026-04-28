"""
Section 4 extension: non-adversarial and structured column opponents.

This module keeps the original Section 4 reproduction untouched. It reuses the
game constants and update helpers from ../section4_bandit.py, then runs the same
row-player algorithms against uniform, Nash, and Hedge column opponents.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np


PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from section4_bandit import (  # noqa: E402
    A_GAME,
    V_STAR,
    X1_NASH,
    is_mixed_ne_batch,
    nash1_batch,
    update_batch,
    val22_batch,
)


OPPONENTS = ("uniform", "nash", "hedge")
ALGORITHM_ORDER = ("UCB", "EXP3", "OurAlg")


def preset_config(preset: str) -> tuple[list[int], int]:
    """Return (T_list, n_runs) for the extension presets."""
    if preset == "quick":
        return [10, 100, 1000], 16
    if preset == "medium":
        return [10, 100, 1000, 10000], 32
    if preset == "paper-lite":
        return [10, 100, 1000, 10000, 100000], 64
    raise ValueError(f"Unknown preset: {preset}")


def init_hedge_state(N: int, T: int) -> dict[str, Any]:
    """Column-player Hedge state. Loss is row payoff, since the column minimizes it."""
    return {
        "losses": np.zeros((N, 2), dtype=float),
        "eta": float(np.sqrt(np.log(2.0) / max(T, 1))),
    }


def _hedge_y1(hedge_state: dict[str, Any]) -> np.ndarray:
    eta = hedge_state["eta"]
    losses = hedge_state["losses"]
    logits = -eta * losses
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights[:, 0]


def _column_expected_payoffs(A: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x2 = 1.0 - x1
    col0 = A[0, 0] * x1 + A[1, 0] * x2
    col1 = A[0, 1] * x1 + A[1, 1] * x2
    return col0, col1


def non_adversarial_opponent(
    A: np.ndarray,
    x1: np.ndarray,
    opponent_type: str,
    hedge_state: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None, np.ndarray]:
    """
    Return expected row payoff, sampled columns, updated opponent state, and y1.

    y1 is the probability that the column player chooses column 0. The sampled
    column action jt follows the convention from section4_bandit.py: jt=1 means
    column 1 was played.
    """
    N = len(x1)

    if opponent_type == "uniform":
        y1 = np.full(N, 0.5)
    elif opponent_type == "nash":
        y1 = np.full(N, 1.0 / 3.0)
    elif opponent_type == "hedge":
        if hedge_state is None:
            raise ValueError("hedge_state is required for the Hedge opponent")
        y1 = _hedge_y1(hedge_state)
    else:
        raise ValueError(f"Unknown opponent_type: {opponent_type}")

    y2 = 1.0 - y1
    val = val22_batch(A, x1, y1)
    jt = (np.random.rand(N) < y2).astype(int)

    if opponent_type == "hedge":
        col0, col1 = _column_expected_payoffs(A, x1)
        hedge_state["losses"][:, 0] += col0
        hedge_state["losses"][:, 1] += col1
        hedge_state["losses"] -= hedge_state["losses"].min(axis=1, keepdims=True)

    return val, jt, hedge_state, y1


def _final_metrics(regret: np.ndarray, payoff: np.ndarray, x1: np.ndarray, T: int) -> dict[str, float]:
    signed_regret = float(regret.mean())
    return {
        "regret": max(signed_regret, 0.0),
        "signed_nash_gap": signed_regret,
        "avg_payoff": float((payoff / max(T, 1)).mean()),
        "dist_to_nash": float(np.abs(x1 - X1_NASH).mean()),
    }


def run_ucb_non_adversarial(
    A: np.ndarray,
    T: int,
    N: int,
    opponent_type: str,
) -> dict[str, float]:
    """UCB row player against one non-adversarial opponent for the full horizon."""
    log_c = 2.0 * np.log(8.0 * max(T**2, 2))
    B1 = np.zeros((N, 2, 2))
    U1 = np.zeros((N, 2, 2))
    cnt = np.zeros((N, 2, 2))
    regret = np.zeros(N)
    payoff = np.zeros(N)
    idx = np.arange(N)
    hedge_state = init_hedge_state(N, T) if opponent_type == "hedge" else None

    x1 = np.full(N, 0.5)
    for _ in range(T):
        x1 = nash1_batch(U1)
        x2 = 1.0 - x1
        it = (np.random.rand(N) < x2).astype(int)

        val, jt, hedge_state, _ = non_adversarial_opponent(A, x1, opponent_type, hedge_state)
        a_obs = (np.random.rand(N) < A[it, jt]).astype(float)

        cnt[idx, it, jt] += 1
        c = cnt[idx, it, jt]
        B1[idx, it, jt] += (a_obs - B1[idx, it, jt]) / c
        np.add(B1, np.sqrt(log_c / (cnt + 1.0)), out=U1)

        regret += V_STAR - val
        payoff += val

    return _final_metrics(regret, payoff, x1, T)


def run_exp3_non_adversarial(
    A: np.ndarray,
    T: int,
    N: int,
    opponent_type: str,
) -> dict[str, float]:
    """EXP3 row player against one non-adversarial opponent for the full horizon."""
    eta = float(np.sqrt(np.log(2.0) / max(T, 1)))
    W = np.zeros((N, 2))
    regret = np.zeros(N)
    payoff = np.zeros(N)
    idx_N = np.arange(N)
    hedge_state = init_hedge_state(N, T) if opponent_type == "hedge" else None

    x1 = np.full(N, 0.5)
    for _ in range(T):
        logits = -eta * W
        logits -= logits.max(axis=1, keepdims=True)
        x = np.exp(logits)
        x /= x.sum(axis=1, keepdims=True)
        x1 = x[:, 0]

        it = (np.random.rand(N) < x[:, 1]).astype(int)
        val, jt, hedge_state, _ = non_adversarial_opponent(A, x1, opponent_type, hedge_state)
        a_obs = (np.random.rand(N) < A[it, jt]).astype(float)

        p_it = x[idx_N, it]
        loss = (1.0 - a_obs) / np.maximum(p_it, 1e-12)
        W[idx_N, it] += loss
        W -= W.min(axis=1, keepdims=True)

        regret += V_STAR - val
        payoff += val

    return _final_metrics(regret, payoff, x1, T)


def run_our_algorithm_non_adversarial(
    A: np.ndarray,
    T: int,
    N: int,
    opponent_type: str,
) -> dict[str, float]:
    """Paper's row update logic against one non-adversarial opponent for the full horizon."""
    log_T_sq = np.log(max(T, 2)) ** 2
    B2 = np.zeros((N, 2, 2))
    U2 = np.zeros((N, 2, 2))
    F2 = np.zeros((N, 2, 2))
    cnt = np.zeros((N, 2, 2))
    jt = np.zeros(N, dtype=int)
    x1 = np.full(N, 0.5)
    count0 = np.ones(N, dtype=int)
    t0 = np.ones(N, dtype=int)
    error = np.ones(N)
    regret = np.zeros(N)
    payoff = np.zeros(N)
    idx_N = np.arange(N)
    log_c = 2.0 * np.log(8.0 * max(T**2, 2))
    hedge_state = init_hedge_state(N, T) if opponent_type == "hedge" else None

    for t in range(T):
        reinit = (count0 == 0) | (t <= log_T_sq)
        if reinit.any():
            F2[reinit] = U2[reinit]
            t0[reinit] = t + 1
            x_nash = nash1_batch(F2)
            mixed = is_mixed_ne_batch(F2)
            x1 = np.where(reinit & mixed, x_nash, x1)
            count0[reinit] = np.maximum(t0[reinit] - 1, 0)

        x1 = update_batch(F2, x1, jt, t0, error)
        x1 = np.clip(x1, 0.0, 1.0)
        count0 = np.maximum(count0 - 1, 0)

        val, jt, hedge_state, _ = non_adversarial_opponent(A, x1, opponent_type, hedge_state)
        it = (np.random.rand(N) < (1.0 - x1)).astype(int)
        a_obs = (np.random.rand(N) < A[it, jt]).astype(float)

        cnt[idx_N, it, jt] += 1
        c = cnt[idx_N, it, jt]
        B2[idx_N, it, jt] += (a_obs - B2[idx_N, it, jt]) / c

        devs = np.sqrt(log_c / (cnt + 1.0))
        np.add(B2, devs, out=U2)
        error = np.minimum(error, devs.max(axis=(1, 2)))

        regret += V_STAR - val
        payoff += val

    return _final_metrics(regret, payoff, x1, T)


def run_section4_non_adversarial(
    A: np.ndarray,
    T_list: list[int],
    N: int,
    verbose: bool = True,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Run all row algorithms against all structured opponents."""
    algorithms = {
        "UCB": run_ucb_non_adversarial,
        "EXP3": run_exp3_non_adversarial,
        "OurAlg": run_our_algorithm_non_adversarial,
    }
    notes = {
        "uniform": "Uniform opponent -> all algorithms should be similar",
        "nash": "Nash opponent -> stable payoff around the game value is expected",
        "hedge": "Hedge opponent -> intermediate difficulty",
    }
    results: dict[str, dict[str, dict[str, list[float]]]] = {}

    for opponent in OPPONENTS:
        results[opponent] = {
            name: {"regret": [], "signed_nash_gap": [], "avg_payoff": [], "dist_to_nash": []}
            for name in algorithms
        }
        if verbose:
            print(f"\n-- {opponent.capitalize()} opponent {'-' * 42}")
            print(notes[opponent])

        for T in T_list:
            if verbose:
                print(f"  T={T:>8d}  ", end="", flush=True)
            for name, fn in algorithms.items():
                metrics = fn(A, T, N, opponent)
                for key, value in metrics.items():
                    results[opponent][name][key].append(value)
                if verbose:
                    print(
                        f"{name}: R={metrics['regret']:8.2f}, "
                        f"P={metrics['avg_payoff']:.3f}  ",
                        end="",
                        flush=True,
                    )
            if verbose:
                print()

    return results


def _plot_common(ax: plt.Axes, title: str) -> None:
    ax.set_xlabel("Log of Time Step", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))


def plot_extension_non_adversarial(
    T_list: list[int],
    results: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """Plot log10 Nash regret for each non-adversarial opponent."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    palette = {"UCB": "#2196F3", "EXP3": "#4CAF50", "OurAlg": "#FF9800"}
    markers = {"UCB": "o", "EXP3": "s", "OurAlg": "^"}
    log_T = np.log10(np.asarray(T_list, dtype=float))
    labels = {
        "uniform": "(a) Uniform stochastic",
        "nash": "(b) Nash stochastic",
        "hedge": "(c) Hedge opponent",
    }

    for ax, opponent in zip(axes, OPPONENTS):
        for name in ALGORITHM_ORDER:
            vals = np.asarray(results[opponent][name]["regret"], dtype=float)
            log_r = np.log10(np.maximum(vals, 1.0))
            ax.plot(
                log_T,
                log_r,
                color=palette[name],
                marker=markers[name],
                label=name,
                linewidth=1.8,
                markersize=6,
            )

        r0 = max(float(results[opponent]["UCB"]["regret"][0]), 1.0)
        ax.plot(
            log_T,
            np.log10(r0) + 0.5 * (log_T - log_T[0]),
            "k--",
            linewidth=0.9,
            alpha=0.45,
            label="slope 1/2",
        )
        ax.set_ylabel("Log of Nash Regret", fontsize=10)
        _plot_common(ax, labels[opponent])

    plt.suptitle(
        "Section 4 Extension: non-adversarial opponents\n"
        "A = [[2/3,0],[0,1/3]]   Bernoulli observations",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved -> {save_path}")
    plt.close(fig)


def plot_extension_average_payoff(
    T_list: list[int],
    results: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """Plot final average payoff for each non-adversarial opponent."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    palette = {"UCB": "#2196F3", "EXP3": "#4CAF50", "OurAlg": "#FF9800"}
    markers = {"UCB": "o", "EXP3": "s", "OurAlg": "^"}
    log_T = np.log10(np.asarray(T_list, dtype=float))
    labels = {
        "uniform": "(a) Uniform stochastic",
        "nash": "(b) Nash stochastic",
        "hedge": "(c) Hedge opponent",
    }

    for ax, opponent in zip(axes, OPPONENTS):
        for name in ALGORITHM_ORDER:
            vals = np.asarray(results[opponent][name]["avg_payoff"], dtype=float)
            ax.plot(
                log_T,
                vals,
                color=palette[name],
                marker=markers[name],
                label=name,
                linewidth=1.8,
                markersize=6,
            )
        ax.axhline(V_STAR, color="black", linestyle="--", linewidth=0.9, alpha=0.5, label="V*")
        ax.set_ylabel("Average Row Payoff", fontsize=10)
        _plot_common(ax, labels[opponent])

    plt.suptitle(
        "Section 4 Extension: average payoff under structured opponents\n"
        "Reference line is the zero-sum game value V* = 2/9",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved -> {save_path}")
    plt.close(fig)


def print_summary(
    T_list: list[int],
    results: dict[str, dict[str, dict[str, list[float]]]],
) -> None:
    """Print final horizon metrics in a compact table."""
    print("\nFinal horizon summary")
    print("-" * 60)
    final_T = T_list[-1]
    for opponent in OPPONENTS:
        print(f"\n{opponent.capitalize()} opponent (T={final_T})")
        print(f"{'Algorithm':>10}  {'NashRegret':>12}  {'AvgPayoff':>10}  {'|x-x*|':>10}")
        for name in ALGORITHM_ORDER:
            regret = results[opponent][name]["regret"][-1]
            payoff = results[opponent][name]["avg_payoff"][-1]
            dist = results[opponent][name]["dist_to_nash"][-1]
            print(f"{name:>10}  {regret:>12.2f}  {payoff:>10.4f}  {dist:>10.4f}")

