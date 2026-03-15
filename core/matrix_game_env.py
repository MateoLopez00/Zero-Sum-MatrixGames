from __future__ import annotations

import numpy as np


class MatrixGameEnv:
    """
    Zero-sum matrix game with noisy feedback.

    A[i, j] is the row payoff (in [0, 1] for this reproduction).
    Column payoff is -A[i, j].
    """

    def __init__(self, payoff_matrix: np.ndarray, seed: int | None = None):
        self.A = np.asarray(payoff_matrix, dtype=float)
        if self.A.ndim != 2:
            raise ValueError("payoff_matrix must be 2D.")
        if np.any(self.A < 0.0) or np.any(self.A > 1.0):
            raise ValueError("payoff_matrix entries must be in [0,1].")

        self.n_rows, self.n_cols = self.A.shape
        self.rng = np.random.default_rng(seed)

    def sample_noisy_matrix(self) -> np.ndarray:
        # Bernoulli matrix noise as in paper setup (E[A_t] = A).
        return self.rng.binomial(1, self.A, size=self.A.shape).astype(float)

    def sample_entry(self, i: int, j: int) -> float:
        return float(self.rng.binomial(1, self.A[i, j]))

