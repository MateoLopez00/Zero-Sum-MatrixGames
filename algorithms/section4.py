from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.ne_solver import solve_ne


def _safe_prob_2(x1: float) -> np.ndarray:
    x1 = float(np.clip(x1, 0.0, 1.0))
    return np.array([x1, 1.0 - x1], dtype=float)


@dataclass
class MatrixUCBRowPlayer:
    n_rows: int
    n_cols: int
    horizon: int

    def __post_init__(self) -> None:
        self.counts = np.zeros((self.n_rows, self.n_cols), dtype=float)
        self.means = np.zeros((self.n_rows, self.n_cols), dtype=float)
        self.logterm = math.log(2.0 * (self.horizon**2) * self.n_rows * self.n_cols)
        self.x = np.ones(self.n_rows, dtype=float) / self.n_rows

    def select_x(self, t: int) -> np.ndarray:
        bonus = np.sqrt(2.0 * self.logterm / np.maximum(1.0, self.counts))
        A_tilde = np.clip(self.means + bonus, 0.0, 1.0)
        x, _, _ = solve_ne(A_tilde)
        self.x = x
        return self.x

    def update(self, i_t: int, j_t: int, reward_t: float) -> None:
        n = self.counts[i_t, j_t]
        self.counts[i_t, j_t] = n + 1.0
        self.means[i_t, j_t] += (reward_t - self.means[i_t, j_t]) / (n + 1.0)


@dataclass
class Exp3RowPlayer:
    n_actions: int
    horizon: int
    gamma: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.gamma is None:
            self.gamma = min(1.0, math.sqrt((self.n_actions * math.log(self.n_actions)) / ((math.e - 1.0) * max(2, self.horizon))))
        self.rng = np.random.default_rng(self.seed)
        self.w = np.ones(self.n_actions, dtype=float)
        self.last_p = np.ones(self.n_actions, dtype=float) / self.n_actions

    def select_x(self, t: int) -> np.ndarray:
        base = self.w / np.sum(self.w)
        p = (1.0 - self.gamma) * base + self.gamma / self.n_actions
        p = np.clip(p, 1e-12, 1.0)
        p = p / np.sum(p)
        self.last_p = p
        return p

    def update(self, i_t: int, j_t: int, reward_t: float) -> None:
        est = np.zeros(self.n_actions, dtype=float)
        est[i_t] = reward_t / max(self.last_p[i_t], 1e-12)
        eta = self.gamma / self.n_actions
        self.w *= np.exp(np.clip(eta * est, -100.0, 100.0))
        self.w = np.clip(self.w, 1e-300, 1e300)


@dataclass
class LogRegret2x2BanditRowPlayer:
    """Algorithm-6/3 style 2x2 bandit player."""

    horizon: int

    def __post_init__(self) -> None:
        self.counts = np.zeros((2, 2), dtype=float)
        self.means = np.zeros((2, 2), dtype=float)
        self.x = np.array([0.5, 0.5], dtype=float)
        self.t = 0

        self._phase = "explore"
        self._pure_row: int | None = None

        # Algorithm 6 loop variables.
        self._t1 = None
        self._t2 = None
        self._alg3_active = False

        # Algorithm 3 state for current call.
        self._x_prime = np.array([0.5, 0.5], dtype=float)
        self._bA = np.zeros((2, 2), dtype=float)
        self._D1 = 1.0
        self._T1 = 1.0
        self._T2 = 1
        self._eta = 0.0
        self._bound = 0.0
        self._delta = 0.0
        self._local_counts = np.zeros((2, 2), dtype=float)

    def _cell_conf(self, i: int, j: int) -> float:
        n = max(1.0, self.counts[i, j])
        return math.sqrt(2.0 * math.log(self.horizon**2) / n)

    def _delta_max(self) -> float:
        return max(self._cell_conf(0, 0), self._cell_conf(0, 1), self._cell_conf(1, 0), self._cell_conf(1, 1))

    def _tilde_delta_min(self, Ahat: np.ndarray) -> float:
        gaps = [
            abs(Ahat[0, 0] - Ahat[0, 1]),
            abs(Ahat[1, 0] - Ahat[1, 1]),
            abs(Ahat[0, 0] - Ahat[1, 0]),
            abs(Ahat[0, 1] - Ahat[1, 1]),
        ]
        return float(min(gaps))

    def _exploration_done(self) -> bool:
        if np.min(self.counts) < 2:
            return False
        Ahat = np.clip(self.means, 0.0, 1.0)
        Delta = self._delta_max()
        tdm = self._tilde_delta_min(Ahat)
        denom = tdm - 2.0 * Delta
        if denom <= 1e-12:
            return False
        ratio = (tdm + 2.0 * Delta) / denom
        return 1.0 <= ratio <= 1.5

    def _strongly_dominant_row(self) -> int | None:
        Ahat = np.clip(self.means, 0.0, 1.0)
        Delta = self._delta_max()
        if (Ahat[0, 0] - Ahat[1, 0] > 2 * Delta) and (Ahat[0, 1] - Ahat[1, 1] > 2 * Delta):
            return 0
        if (Ahat[1, 0] - Ahat[0, 0] > 2 * Delta) and (Ahat[1, 1] - Ahat[0, 1] > 2 * Delta):
            return 1
        return None

    def _find_psne_row(self) -> int | None:
        A = np.clip(self.means, 0.0, 1.0)
        for i in [0, 1]:
            for j in [0, 1]:
                row_best = A[i, j] >= A[1 - i, j]
                col_best = A[i, j] <= A[i, 1 - j]
                if row_best and col_best:
                    return i
        return None

    def _start_alg3_call(self) -> None:
        Ahat = np.clip(self.means, 0.0, 1.0)
        x_prime, _, _ = solve_ne(Ahat)
        self._x_prime = x_prime
        self._bA = Ahat

        D_tilde = abs(Ahat[0, 0] - Ahat[0, 1] - Ahat[1, 0] + Ahat[1, 1])
        Delta = self._delta_max()
        self._D1 = max(D_tilde / 2.0, 1e-6)
        self._T1 = max((1.0 / max(Delta, 1e-6)) ** 2, 1.0)
        self._T2 = max(1, int(self._t2))
        self._eta = 1.0 / (self._D1 * self._T1)
        self._bound = 1.0 / (self._D1 * math.sqrt(self._T1))
        self._delta = -self._bound
        self._local_counts = np.zeros((2, 2), dtype=float)
        self._alg3_active = True

    def select_x(self, t: int) -> np.ndarray:
        if self._pure_row is not None:
            self.x = np.array([1.0, 0.0], dtype=float) if self._pure_row == 0 else np.array([0.0, 1.0], dtype=float)
            return self.x

        if self._phase == "explore":
            self.x = np.array([0.5, 0.5], dtype=float)
            return self.x

        # Algorithm 3 line 4
        self.x = _safe_prob_2(self._x_prime[0] + self._delta)
        return self.x

    def update(self, i_t: int, j_t: int, reward_t: float) -> None:
        self.t += 1
        n = self.counts[i_t, j_t]
        self.counts[i_t, j_t] = n + 1.0
        self.means[i_t, j_t] += (reward_t - self.means[i_t, j_t]) / (n + 1.0)

        if self._pure_row is not None:
            return

        if self._phase == "explore":
            if not self._exploration_done():
                return

            # Algorithm 6 lines 3-4
            dom = self._strongly_dominant_row()
            if dom is not None:
                self._pure_row = dom
                return

            psne_row = self._find_psne_row()
            if psne_row is not None:
                self._pure_row = psne_row
                return

            # Algorithm 6 lines 5-10
            self._t1 = self.t
            self._t2 = int(np.min(self.counts))
            self._phase = "exploit"
            self._start_alg3_call()
            return

        if self._phase == "exploit" and self._alg3_active:
            # Algorithm 3 line 5/6 updates
            self._delta += self._eta * (self._bA[0, j_t] - self._bA[1, j_t])
            self._delta = float(np.clip(self._delta, -self._bound, self._bound))
            self._local_counts[i_t, j_t] += 1.0

            # Algorithm 3 line 9 termination
            if np.min(self._local_counts) >= self._T2 or self.t >= self.horizon:
                self._alg3_active = False
                self._t1 = self.t
                self._t2 = int(np.min(self.counts))
                if self._t1 < self.horizon:
                    self._start_alg3_call()

