"""Noise robustness extension for Section 3 full-information feedback."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_INFO_DIR = REPO_ROOT / "Full_information_feedback"
if str(FULL_INFO_DIR) not in sys.path:
    sys.path.insert(0, str(FULL_INFO_DIR))

from experiments.section3 import (  # noqa: E402
    _update_official_diag,
    _value_of_diag_game,
    adversary,
    generate_diagonal_matrix,
    nash1_diag,
)


N_VALUES = [10, 20, 50, 100]
ALGORITHM_ORDER = ["Nash", "Hedge", "Our-Algo"]


def preset_config(preset: str) -> tuple[list[float], int, int]:
    if preset == "quick":
        return [0.0, 0.1, 0.3], 8, 10_000
    if preset == "medium":
        return [0.0, 0.1, 0.2, 0.3], 8, 30_000
    if preset == "paper-lite":
        return [0.0, 0.1, 0.2, 0.3], 12, 50_000
    raise ValueError(f"Unknown preset: {preset}")


def sample_full_information_gaussian(
    A: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Full-information Gaussian feedback: observe the whole noisy matrix."""
    noisy = A + sigma * rng.normal(size=A.shape)
    return np.clip(noisy, 0.0, 1.0)


def _metrics(regret: float, payoff_sum: float, horizon: int) -> dict[str, float]:
    return {
        "regret": float(max(regret, 1e-12)),
        "avg_payoff": float(payoff_sum / max(horizon, 1)),
    }


def run_nash_gaussian(seed: int, horizon: int, n: int, sigma: float) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    A = generate_diagonal_matrix(n)
    V = _value_of_diag_game(A)
    Abar = np.zeros((n, n), dtype=float)
    regret = 0.0
    payoff = 0.0

    for t in range(horizon):
        x = nash1_diag(Abar)
        val, _ = adversary(A, x)
        regret += V - val
        payoff += val
        sample = sample_full_information_gaussian(A, sigma, rng)
        Abar = (t / (t + 1)) * Abar + (1.0 / (t + 1)) * sample

    return _metrics(regret, payoff, horizon)


def run_hedge_gaussian(seed: int, horizon: int, n: int, sigma: float) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    A = generate_diagonal_matrix(n)
    V = _value_of_diag_game(A)
    weights = np.ones(n, dtype=float)
    eta = math.sqrt(math.log(n) / max(1, horizon))
    regret = 0.0
    payoff = 0.0

    for _ in range(horizon):
        x = weights / np.sum(weights)
        val, idx = adversary(A, x)
        regret += V - val
        payoff += val
        sample = sample_full_information_gaussian(A, sigma, rng)
        reward_vector = sample[:, idx]
        weights *= np.exp(eta * reward_vector)

    return _metrics(regret, payoff, horizon)


def run_our_algo_official_gaussian(
    seed: int,
    horizon: int,
    n: int,
    sigma: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    A = generate_diagonal_matrix(n)
    V = _value_of_diag_game(A)

    Abar = np.zeros((n, n), dtype=float)
    frozen = np.zeros((n, n), dtype=float)
    jt = 0
    x = np.ones(n, dtype=float) / n
    count0 = 1
    t0 = 1
    threshold = min(math.log(max(horizon, 2)) ** 2, horizon**0.5)
    regret = 0.0
    payoff = 0.0

    for t in range(horizon):
        if count0 > 0 and t > threshold:
            x = _update_official_diag(frozen, x, jt, t0)
            count0 -= 1
        else:
            t0 = t + 1
            x = nash1_diag(Abar)
            count0 = t0 - 1
            frozen = Abar.copy()

        val, jt = adversary(A, x)
        # Match the official Section 3 runner's regret accounting.
        regret += 2.0 * (V - val)
        payoff += val
        sample = sample_full_information_gaussian(A, sigma, rng)
        Abar = (t / (t + 1)) * Abar + (1.0 / (t + 1)) * sample

    return _metrics(regret, payoff, horizon)


def run_section3_noise_robustness(
    preset: str,
    seed: int = 7,
    verbose: bool = True,
) -> tuple[dict[int, dict[str, dict[str, list[float]]]], list[float], int, int]:
    sigma_values, n_runs, horizon = preset_config(preset)
    algorithms = {
        "Nash": run_nash_gaussian,
        "Hedge": run_hedge_gaussian,
        "Our-Algo": run_our_algo_official_gaussian,
    }
    results: dict[int, dict[str, dict[str, list[float]]]] = {}

    for n in N_VALUES:
        results[n] = {
            name: {"regret": [], "avg_payoff": []}
            for name in ALGORITHM_ORDER
        }
        if verbose:
            print(f"\n-- n={n} {'-' * 55}")
        for sigma in sigma_values:
            if verbose:
                print(f"  sigma={sigma:>4.2f}  ", end="", flush=True)
            for name, fn in algorithms.items():
                regrets = []
                payoffs = []
                for run_idx in range(n_runs):
                    run_seed = seed + 10007 * run_idx + 7919 * n + int(1000 * sigma)
                    metrics = fn(run_seed, horizon, n, sigma)
                    regrets.append(metrics["regret"])
                    payoffs.append(metrics["avg_payoff"])
                mean_regret = float(np.mean(regrets))
                mean_payoff = float(np.mean(payoffs))
                results[n][name]["regret"].append(mean_regret)
                results[n][name]["avg_payoff"].append(mean_payoff)
                if verbose:
                    print(f"{name}: R={mean_regret:8.2f}, P={mean_payoff:.4f}  ", end="", flush=True)
            if verbose:
                print()

    return results, sigma_values, n_runs, horizon


def plot_section3_noise_regret(
    results: dict[int, dict[str, dict[str, list[float]]]],
    sigma_values: list[float],
    save_path: str | Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=False)
    palette = {"Nash": "#1f77b4", "Hedge": "#2ca02c", "Our-Algo": "#ff7f0e"}
    markers = {"Nash": "o", "Hedge": "s", "Our-Algo": "^"}

    for ax, n in zip(axes.ravel(), N_VALUES):
        for name in ALGORITHM_ORDER:
            vals = np.asarray(results[n][name]["regret"], dtype=float)
            ax.plot(
                sigma_values,
                np.log10(np.maximum(vals, 1e-12)),
                marker=markers[name],
                color=palette[name],
                linewidth=1.8,
                label=name,
            )
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Noise level sigma")
        ax.set_ylabel("log10(total Nash regret)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    plt.suptitle("Section 3 noise robustness: full-information Gaussian feedback", y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"Figure saved -> {save_path}")
    plt.close(fig)


def plot_section3_noise_payoff(
    results: dict[int, dict[str, dict[str, list[float]]]],
    sigma_values: list[float],
    save_path: str | Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=False)
    palette = {"Nash": "#1f77b4", "Hedge": "#2ca02c", "Our-Algo": "#ff7f0e"}
    markers = {"Nash": "o", "Hedge": "s", "Our-Algo": "^"}

    for ax, n in zip(axes.ravel(), N_VALUES):
        for name in ALGORITHM_ORDER:
            vals = np.asarray(results[n][name]["avg_payoff"], dtype=float)
            ax.plot(
                sigma_values,
                vals,
                marker=markers[name],
                color=palette[name],
                linewidth=1.8,
                label=name,
            )
        ax.axhline(_value_of_diag_game(generate_diagonal_matrix(n)), color="black", linestyle="--", alpha=0.5, linewidth=0.9, label="V*")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Noise level sigma")
        ax.set_ylabel("average row payoff")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    plt.suptitle("Section 3 noise robustness: average payoff", y=1.01)
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
    for n in N_VALUES:
        print(f"\nn={n}, sigma={final_sigma}")
        print(f"{'Algorithm':>10}  {'NashRegret':>12}  {'AvgPayoff':>10}")
        for name in ALGORITHM_ORDER:
            regret = results[n][name]["regret"][final_idx]
            payoff = results[n][name]["avg_payoff"][final_idx]
            print(f"{name:>10}  {regret:>12.2f}  {payoff:>10.4f}")


def run_and_plot(preset: str, seed: int = 7) -> tuple[dict[str, Any], Path, Path]:
    results, sigma_values, n_runs, horizon = run_section3_noise_robustness(preset, seed=seed, verbose=True)
    here = Path(__file__).resolve().parent
    plots_dir = here / "plots"
    regret_path = plots_dir / f"section3_noise_regret_{preset}.png"
    payoff_path = plots_dir / f"section3_noise_payoff_{preset}.png"
    plot_section3_noise_regret(results, sigma_values, regret_path)
    plot_section3_noise_payoff(results, sigma_values, payoff_path)
    print_summary(results, sigma_values)
    metadata = {"preset": preset, "sigma_values": sigma_values, "n_runs": n_runs, "horizon": horizon}
    return {"results": results, "metadata": metadata}, regret_path, payoff_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 3 noise robustness extension.")
    parser.add_argument("--preset", default="quick", choices=["quick", "medium", "paper-lite"])
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("Section 3 Extension: Noise Robustness")
    print("=" * 72)
    sigma_values, n_runs, horizon = preset_config(args.preset)
    print(f"Preset: {args.preset}")
    print(f"Sigma values: {sigma_values}")
    print(f"n_runs: {n_runs}")
    print(f"T: {horizon}")
    run_and_plot(args.preset, seed=args.seed)


if __name__ == "__main__":
    main()

