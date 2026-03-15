from __future__ import annotations

import numpy as np


class BestResponseColumnAdversary:
    """
    Adversarial column player:
    picks a column minimizing row expected payoff x^T A e_j.
    """

    def __init__(self, A: np.ndarray, tie_break: str = "random", seed: int | None = None):
        self.A = np.asarray(A, dtype=float)
        self.tie_break = tie_break
        self.rng = np.random.default_rng(seed)

    def select_column(self, x: np.ndarray) -> int:
        vals = self.A.T @ x
        min_val = float(np.min(vals))
        idx = np.where(np.isclose(vals, min_val))[0]
        if len(idx) == 1 or self.tie_break != "random":
            return int(idx[0])
        return int(self.rng.choice(idx))

