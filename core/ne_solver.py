from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    linprog = None


def _sanitize_prob(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return x / s


def solve_row_ne(A: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Solve row player's minimax strategy and game value:
        max_x min_j (A^T x)_j
    """
    A = np.asarray(A, dtype=float)
    n, m = A.shape

    if linprog is None:
        # Fallback: multiplicative weights approximation
        x = np.ones(n) / n
        eta = 0.15
        for _ in range(3000):
            j = int(np.argmin(A.T @ x))
            g = A[:, j]
            x = x * np.exp(eta * g)
            x = _sanitize_prob(x)
        return x, float(np.min(A.T @ x))

    # LP variables are [x_1..x_n, v]
    c = np.zeros(n + 1)
    c[-1] = -1.0  # maximize v == minimize -v

    A_ub = np.zeros((m, n + 1))
    A_ub[:, :n] = -A.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(m)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        x = np.ones(n) / n
        return x, float(np.min(A.T @ x))

    x = _sanitize_prob(res.x[:n])
    v = float(np.min(A.T @ x))
    return x, v


def solve_col_ne(A: np.ndarray) -> tuple[np.ndarray, float]:
    y_row, v_neg = solve_row_ne(-A.T)
    return y_row, -v_neg


def solve_ne(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x, v = solve_row_ne(A)
    y, _ = solve_col_ne(A)
    return x, y, v

