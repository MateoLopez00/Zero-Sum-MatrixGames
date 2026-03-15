from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ne_solver import solve_ne


def project_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if np.all(v >= 0.0) and np.isclose(v.sum(), 1.0):
        return v
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones(n) / n
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def _diag_ne(diag_vals: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    d = np.maximum(np.asarray(diag_vals, dtype=float), eps)
    inv = 1.0 / d
    z = float(inv.sum())
    x = inv / z
    v = 1.0 / z
    return x, v


def _diag_matrix_from_diag(diag_vals: np.ndarray) -> np.ndarray:
    n = len(diag_vals)
    A = np.zeros((n, n), dtype=float)
    A[np.arange(n), np.arange(n)] = diag_vals
    return A


def _m_matrix(B: np.ndarray) -> np.ndarray:
    n = B.shape[0]
    M = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        M[i, :] = B[0, :] - B[i + 1, :]
    M[n - 1, :] = 1.0
    return M


def _replace_column(M: np.ndarray, col: int) -> np.ndarray:
    n = M.shape[0]
    out = M.copy()
    out[:, col] = 0.0
    out[n - 1, col] = 1.0
    return out


def full_support_det_stats(B: np.ndarray) -> tuple[float, float]:
    n = B.shape[0]
    MB = _m_matrix(B)
    MBt = _m_matrix(B.T)
    vals = [abs(np.linalg.det(MB)), abs(np.linalg.det(MBt))]
    for i in range(n):
        vals.append(abs(np.linalg.det(_replace_column(MBt, i))))
        vals.append(abs(np.linalg.det(_replace_column(MB, i))))
    det_min_tilde = float(min(vals))
    D_tilde = float(abs(np.linalg.det(MBt)))
    return det_min_tilde, D_tilde


class FullInfoRowAlgo:
    def select_x(self, t: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def update(self, t: int, j_t: int, A_t: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class HedgeRowPlayer(FullInfoRowAlgo):
    n_rows: int
    horizon: int
    eta_scale: float = 1.0

    def __post_init__(self) -> None:
        self.w = np.ones(self.n_rows, dtype=float)
        self.x = np.ones(self.n_rows, dtype=float) / self.n_rows

    def select_x(self, t: int) -> np.ndarray:
        self.x = self.w / np.sum(self.w)
        return self.x

    def update(self, t: int, j_t: int, A_t: np.ndarray) -> None:
        eta = self.eta_scale * math.sqrt(max(1e-12, math.log(self.n_rows + 1) / max(1, t)))
        gains = A_t[:, j_t]
        self.w *= np.exp(eta * gains)
        self.w = np.clip(self.w, 1e-300, 1e300)


@dataclass
class EmpiricalNashRowPlayer(FullInfoRowAlgo):
    n_rows: int
    n_cols: int

    def __post_init__(self) -> None:
        self.t = 0
        self.A_bar = np.zeros((self.n_rows, self.n_cols), dtype=float)
        self.x = np.ones(self.n_rows, dtype=float) / self.n_rows

    def select_x(self, t: int) -> np.ndarray:
        return self.x

    def update(self, t: int, j_t: int, A_t: np.ndarray) -> None:
        self.t += 1
        alpha = 1.0 / self.t
        self.A_bar = (1.0 - alpha) * self.A_bar + alpha * A_t
        self.x, _, _ = solve_ne(self.A_bar)


@dataclass
class LogRegretFullInfoRowPlayer(FullInfoRowAlgo):
    n_rows: int
    n_cols: int
    horizon: int

    def __post_init__(self) -> None:
        self.t = 0
        self.A_bar = np.zeros((self.n_rows, self.n_cols), dtype=float)
        self.x = np.ones(self.n_rows, dtype=float) / self.n_rows

        # Algorithm 5 state
        self.t_star = self.horizon + 1
        self.fallback_mode = False

        # Algorithm 2 call state
        self.in_alg2 = False
        self.t1 = None
        self.alg2_steps_left = 0
        self.alg2_x_prime = self.x.copy()
        self.alg2_bA = self.A_bar.copy()
        self.alg2_delta = np.zeros(max(1, self.n_rows - 1), dtype=float)
        self.alg2_eta = 0.0
        self.alg2_bound = 0.0

    def _delta_t(self, t: int) -> float:
        return math.sqrt(2.0 * math.log(self.n_rows * self.n_cols * (self.horizon**2)) / max(1, t))

    def _has_unique_full_support_ne(self, A: np.ndarray) -> bool:
        x, y, _ = solve_ne(A)
        tol = 1e-6
        return bool(np.all(x > tol) and np.all(y > tol))

    def _start_alg2_call(self, t_start: int) -> None:
        x_prime, _, _ = solve_ne(self.A_bar)
        self.alg2_x_prime = x_prime
        self.alg2_bA = self.A_bar.copy()

        Delta = self._delta_t(t_start)
        _, D_tilde = full_support_det_stats(self.A_bar[:, : self.n_rows])
        D1 = max(D_tilde / (5.0 * self.n_rows * math.factorial(self.n_rows)), 1e-6)
        T1 = max((1.0 / max(Delta, 1e-6)) ** 2, 1.0)
        T2 = int(max(1, min(t_start, self.horizon - t_start)))

        self.alg2_eta = 1.0 / (D1 * T1)
        self.alg2_bound = 1.0 / (D1 * math.sqrt(T1))
        self.alg2_delta = -self.alg2_bound * np.ones(max(1, self.n_rows - 1), dtype=float)
        self.alg2_steps_left = T2
        self.in_alg2 = True
        self.x = self._alg2_current_x()

    def _alg2_current_x(self) -> np.ndarray:
        if self.n_rows == 1:
            return np.array([1.0], dtype=float)
        x = self.alg2_x_prime.copy()
        x[:-1] += self.alg2_delta
        x[-1] -= float(np.sum(self.alg2_delta))
        return project_simplex(x)

    def select_x(self, t: int) -> np.ndarray:
        return self.x

    def update(self, t: int, j_t: int, A_t: np.ndarray) -> None:
        self.t += 1
        alpha = 1.0 / self.t
        self.A_bar = (1.0 - alpha) * self.A_bar + alpha * A_t

        if self.in_alg2:
            for i in range(self.n_rows - 1):
                self.alg2_delta[i] += self.alg2_eta * (self.alg2_bA[i, j_t] - self.alg2_bA[self.n_rows - 1, j_t])
                self.alg2_delta[i] = float(np.clip(self.alg2_delta[i], -self.alg2_bound, self.alg2_bound))
            self.alg2_steps_left -= 1
            self.x = self._alg2_current_x()
            if self.alg2_steps_left <= 0:
                self.in_alg2 = False
                self.t1 = int(2 * max(1, self.t1))
                if self.t1 < self.horizon:
                    self._start_alg2_call(self.t1)
            return

        if self.fallback_mode:
            x_hat, _, _ = solve_ne(self.A_bar)
            self.x = x_hat
            return

        x_hat, _, _ = solve_ne(self.A_bar)
        self.x = x_hat

        # Algorithm 5 line 8-12 trigger condition
        Delta = self._delta_t(self.t)
        det_min_tilde, _ = full_support_det_stats(self.A_bar[:, : self.n_rows])
        c = 2.0 * (self.n_rows**2) * math.factorial(self.n_rows) * Delta
        denom = det_min_tilde - c
        if self.t_star > self.horizon and denom > 1e-12:
            ratio = (det_min_tilde + c) / denom
            if 1.0 <= ratio <= 1.5:
                self.t_star = int(math.ceil(6.25 * self.t))

        # Algorithm 5 line 13 onward
        if self.t >= self.t_star:
            if self._has_unique_full_support_ne(self.A_bar[:, : self.n_rows]):
                self.t1 = self.t
                self._start_alg2_call(self.t1)
            else:
                self.fallback_mode = True

