from __future__ import annotations

import argparse

from experiments.section3 import RunConfig as S3Config
from experiments.section3 import run as run_section3
from experiments.section3 import section3_horizons_for_preset
from experiments.section4 import RunConfig as S4Config
from experiments.section4 import run as run_section4
from experiments.section4 import section4_horizons_for_preset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default="quick", choices=["quick", "medium", "paper-lite", "final", "paper"])
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    h3, n3 = section3_horizons_for_preset(args.preset)
    h4, n4 = section4_horizons_for_preset(args.preset)

    run_section3(S3Config(horizons=h3, n_runs=n3, seed=args.seed, preset=args.preset))
    run_section4(S4Config(horizons=h4, n_runs=n4, seed=args.seed + 1000, preset=args.preset))


if __name__ == "__main__":
    main()

