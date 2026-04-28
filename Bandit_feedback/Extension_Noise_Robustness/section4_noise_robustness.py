"""Noise robustness extension for Section 4 bandit feedback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np


BANDIT_DIR = Path(__file__).resolve().parents[1]
if str(BANDIT_DIR) not in sys.path:
    sys.path.insert(0, str(BANDIT_DIR))

from section4_bandit import (  # noqa: E402
    A_GAME,
    V_STAR,
    adv22gd_batch,
    advnew_batch,
    is_mixed_ne_batch,
    nash1_batch,
    update_batch,
    val22_batch,
)


ADVERSARIES = [1, 2, 3]
ALGORITHM_ORDER = ["UCB", "EXP3", "OurAlg"]


def preset_config(preset: str) -> tuple[list[float], int, int]:
    if preset == "quick":
        return [0.0, 0.1, 0.3], 8, 10_000
    if preset == "medium":
        return [0.0, 0.05, 0.1, 0.2, 0.3], 16, 100_000
    if preset == "paper-lite":
        return [0.0, 0.05, 0.1, 0.2, 0.3], 32, 100_000
    raise ValueError(f"Unknown preset: {preset}")


def sample_bandit_gaussian_reward(
    A: np.ndarray,
    it: np.ndarray,
    jt: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian bandit feedback at the played cell."""
    mean = A[it, jt]
    reward = mean + sigma * rng.normal(size=len(it))
    return np.clip(reward, 0.0, 1.0)


def _metrics(regret: np.ndarray, payoff: np.ndarray, T: int) -> dict[str, float]:
    return {
        "regret": float(max(float(regret.mean()), 1e-12)),
        "avg_payoff": float((payoff / max(T, 1)).mean()),
    }


def run_ucb_gaussian(
    A: np.ndarray,
    T: int,
    N: int,
    adv_type: int,
    sigma: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    T1 = T // 2
    log_c = 2.0 * np.log(8.0 * max(T**2, 2))
    B1 = np.zeros((N, 2, 2))
    U1 = np.zeros((N, 2, 2))
    cnt = np.zeros((N, 2, 2))
    regret = np.zeros(N)
    payoff = np.zeros(N)
    idx = np.arange(N)

    def step(use_adv_type: bool) -> None:
        x1 = nash1_batch(U1)
        x2 = 1.0 - x1
        it = (rng.random(N) < x2).astype(int)

        if use_adv_type:
            y1, y2 = advnew_batch(x1, T, adv_type)
            val = val22_batch(A, x1, y1)
            jt = (rng.random(N) < y2).astype(int)
        else:
            val, jt = adv22gd_batch(A, x1)

        obs = sample_bandit_gaussian_reward(A, it, jt, sigma, rng)
        cnt[idx, it, jt] += 1
        c = cnt[idx, it, jt]
        B1[idx, it, jt] += (obs - B1[idx, it, jt]) / c
        np.add(B1, np.sqrt(log_c / (cnt + 1.0)), out=U1)

        regret[:] += V_STAR - val
        payoff[:] += val

    for _ in range(T1):
        step(use_adv_type=True)
    for _ in range(T1):
        step(use_adv_type=False)

    return _metrics(regret, payoff, T)


def run_exp3_gaussian(
    A: np.ndarray,
    T: int,
    N: int,
    adv_type: int,
    sigma: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    T1 = T // 2
    eta = float(np.sqrt(np.log(2.0) / max(T, 1)))
    W = np.zeros((N, 2))
    regret = np.zeros(N)
    payoff = np.zeros(N)
    idx_N = np.arange(N)

    def step(use_adv_type: bool) -> None:
        logits = -eta * W
        logits -= logits.max(axis=1, keepdims=True)
        x = np.exp(logits)
        x /= x.sum(axis=1, keepdims=True)
        x1 = x[:, 0]

        it = (rng.random(N) < x[:, 1]).astype(int)
        if use_adv_type:
            y1, y2 = advnew_batch(x1, T, adv_type)
            val = val22_batch(A, x1, y1)
            jt = (rng.random(N) < y2).astype(int)
        else:
            val, jt = adv22gd_batch(A, x1)

        obs = sample_bandit_gaussian_reward(A, it, jt, sigma, rng)
        p_it = x[idx_N, it]
        loss = (1.0 - obs) / np.maximum(p_it, 1e-12)
        W[idx_N, it] += loss
        W[:] -= W.min(axis=1, keepdims=True)

        regret[:] += V_STAR - val
        payoff[:] += val

    for _ in range(T1):
        step(use_adv_type=True)
    for _ in range(T1):
        step(use_adv_type=False)

    return _metrics(regret, payoff, T)


def run_our_algorithm_gaussian(
    A: np.ndarray,
    T: int,
    N: int,
    adv_type: int,
    sigma: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    T1 = T // 2
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

    for t in range(T1):
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

        y1, y2 = advnew_batch(x1, T, adv_type)
        val = val22_batch(A, x1, y1)
        jt = (rng.random(N) < y2).astype(int)
        it = (rng.random(N) < (1.0 - x1)).astype(int)
        obs = sample_bandit_gaussian_reward(A, it, jt, sigma, rng)

        cnt[idx_N, it, jt] += 1
        c = cnt[idx_N, it, jt]
        B2[idx_N, it, jt] += (obs - B2[idx_N, it, jt]) / c
        devs = np.sqrt(log_c / (cnt + 1.0))
        np.add(B2, devs, out=U2)
        error = np.minimum(error, devs.max(axis=(1, 2)))

        regret += V_STAR - val
        payoff += val

    x1 = nash1_batch(B2)
    x1 = np.clip(x1, 0.0, 1.0)
    t0_fixed = np.full(N, T1, dtype=int)

    for _ in range(T1):
        x1 = update_batch(B2, x1, jt, t0_fixed, error)
        x1 = np.clip(x1, 0.0, 1.0)
        val, jt = adv22gd_batch(A, x1)
        regret += V_STAR - val
        payoff += val

    return _metrics(regret, payoff, T)


def run_section4_noise_robustness(
    preset: str,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[dict[int, dict[str, dict[str, list[float]]]], list[float], int, int]:
    sigma_values, n_runs, horizon = preset_config(preset)
    algorithms = {
        "UCB": run_ucb_gaussian,
        "EXP3": run_exp3_gaussian,
        "OurAlg": run_our_algorithm_gaussian,
    }
    results: dict[int, dict[str, dict[str, list[float]]]] = {}

    for adv_type in ADVERSARIES:
        results[adv_type] = {
            name: {"regret": [], "avg_payoff": []}
            for name in ALGORITHM_ORDER
        }
        if verbose:
            print(f"\n-- Adversary {adv_type} {'-' * 48}")
        for sigma in sigma_values:
            if verbose:
                print(f"  sigma={sigma:>4.2f}  ", end="", flush=True)
            for name, fn in algorithms.items():
                run_seed = seed + 1009 * adv_type + int(10_000 * sigma)
                rng = np.random.default_rng(run_seed)
                metrics = fn(A_GAME, horizon, n_runs, adv_type, sigma, rng)
                results[adv_type][name]["regret"].append(metrics["regret"])
                results[adv_type][name]["avg_payoff"].append(metrics["avg_payoff"])
                if verbose:
                    print(f"{name}: R={metrics['regret']:8.2f}, P={metrics['avg_payoff']:.4f}  ", end="", flush=True)
            if verbose:
                print()

    return results, sigma_values, n_runs, horizon


def plot_section4_noise_regret(
    results: dict[int, dict[str, dict[str, list[float]]]],
    sigma_values: list[float],
    save_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    palette = {"UCB": "#2196F3", "EXP3": "#4CAF50", "OurAlg": "#FF9800"}
    markers = {"UCB": "o", "EXP3": "s", "OurAlg": "^"}

    for ax, adv_type in zip(axes, ADVERSARIES):
        for name in ALGORITHM_ORDER:
            vals = np.asarray(results[adv_type][name]["regret"], dtype=float)
            ax.plot(
                sigma_values,
                np.log10(np.maximum(vals, 1e-12)),
                marker=markers[name],
                color=palette[name],
                linewidth=1.8,
                label=name,
            )
        ax.set_title(f"Adversary {adv_type}")
        ax.set_xlabel("Noise level sigma")
        ax.set_ylabel("log10(total Nash regret)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))

    plt.suptitle("Section 4 noise robustness: Gaussian bandit feedback", y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"Figure saved -> {save_path}")
    plt.close(fig)


def plot_section4_noise_payoff(
    results: dict[int, dict[str, dict[str, list[float]]]],
    sigma_values: list[float],
    save_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    palette = {"UCB": "#2196F3", "EXP3": "#4CAF50", "OurAlg": "#FF9800"}
    markers = {"UCB": "o", "EXP3": "s", "OurAlg": "^"}

    for ax, adv_type in zip(axes, ADVERSARIES):
        for name in ALGORITHM_ORDER:
            vals = np.asarray(results[adv_type][name]["avg_payoff"], dtype=float)
            ax.plot(
                sigma_values,
                vals,
                marker=markers[name],
                color=palette[name],
                linewidth=1.8,
                label=name,
            )
        ax.axhline(V_STAR, color="black", linestyle="--", linewidth=0.9, alpha=0.5, label="V*")
        ax.set_title(f"Adversary {adv_type}")
        ax.set_xlabel("Noise level sigma")
        ax.set_ylabel("average row payoff")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))

    plt.suptitle("Section 4 noise robustness: average payoff", y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"Figure saved -> {save_path}")
    plt.close(fig)


def print_summary(
    results: dict[int, dict[str, dict[str, list[float]]]],
    sigma_values: list[float],
) -> None:
    print("\nFinal sigma summary")
    print("-" * 72)
    final_idx = len(sigma_values) - 1
    final_sigma = sigma_values[final_idx]
    for adv_type in ADVERSARIES:
        print(f"\nAdversary {adv_type}, sigma={final_sigma}")
        print(f"{'Algorithm':>10}  {'NashRegret':>12}  {'AvgPayoff':>10}")
        for name in ALGORITHM_ORDER:
            regret = results[adv_type][name]["regret"][final_idx]
            payoff = results[adv_type][name]["avg_payoff"][final_idx]
            print(f"{name:>10}  {regret:>12.2f}  {payoff:>10.4f}")


def run_and_plot(preset: str, seed: int = 42) -> tuple[dict[str, Any], Path, Path]:
    results, sigma_values, n_runs, horizon = run_section4_noise_robustness(preset, seed=seed, verbose=True)
    here = Path(__file__).resolve().parent
    plots_dir = here / "plots"
    regret_path = plots_dir / f"section4_noise_regret_{preset}.png"
    payoff_path = plots_dir / f"section4_noise_payoff_{preset}.png"
    plot_section4_noise_regret(results, sigma_values, regret_path)
    plot_section4_noise_payoff(results, sigma_values, payoff_path)
    print_summary(results, sigma_values)
    metadata = {"preset": preset, "sigma_values": sigma_values, "n_runs": n_runs, "horizon": horizon}
    return {"results": results, "metadata": metadata}, regret_path, payoff_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 4 noise robustness extension.")
    parser.add_argument("--preset", default="quick", choices=["quick", "medium", "paper-lite"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("Section 4 Extension: Noise Robustness")
    print("=" * 72)
    sigma_values, n_runs, horizon = preset_config(args.preset)
    print(f"Preset: {args.preset}")
    print(f"Sigma values: {sigma_values}")
    print(f"n_runs: {n_runs}")
    print(f"T: {horizon}")
    run_and_plot(args.preset, seed=args.seed)


if __name__ == "__main__":
    main()

