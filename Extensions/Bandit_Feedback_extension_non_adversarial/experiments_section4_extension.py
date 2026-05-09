"""Command-line runner for the Section 4 non-adversarial opponent extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from section4_non_adversarial import (
    A_GAME,
    plot_extension_average_payoff,
    plot_extension_non_adversarial,
    preset_config,
    print_summary,
    run_section4_non_adversarial,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 4 non-adversarial extension.")
    parser.add_argument(
        "--preset",
        default="quick",
        choices=["quick", "medium", "paper-lite"],
        help="Runtime preset for horizons and number of runs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    T_list, n_runs = preset_config(args.preset)
    here = Path(__file__).resolve().parent
    plots_dir = here / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Section 4 Extension: Non-Adversarial / Structured Opponents")
    print("=" * 72)
    print(f"Preset: {args.preset}")
    print(f"T values: {T_list}")
    print(f"N runs: {n_runs}")
    print()

    results = run_section4_non_adversarial(A_GAME, T_list, n_runs, verbose=True)

    regret_path = plots_dir / f"section4_non_adversarial_{args.preset}.png"
    payoff_path = plots_dir / f"section4_non_adversarial_payoff_{args.preset}.png"

    plot_extension_non_adversarial(T_list, results, save_path=regret_path)
    plot_extension_average_payoff(T_list, results, save_path=payoff_path)
    print_summary(T_list, results)


if __name__ == "__main__":
    main()

