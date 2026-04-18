from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.optimize import linprog

from core.utils import ensure_dir


@dataclass
class RunConfig:
    horizons: list[int]
    n_runs: int
    seed: int = 7
    preset: str = "quick"
    n_actions: int = 20
    variant: str = "official"  # "official" matches authors' new_experiments.py; "subroutine" matches subroutine mechanics; "theory-lp" matches Appendix Alg. (support ID + alg-nxn-full) using LP NE.


def section3_horizons_for_preset(preset: str) -> tuple[list[int], int]:
    # Mirrors official code scale with runtime-aware presets.
    if preset == "quick":
        return [10, 100], 2
    if preset == "medium":
        return [10, 100, 1000], 4
    if preset == "paper-lite":
        return [10**i for i in range(1, 6)], 12
    if preset in {"final", "paper"}:
        return [10**i for i in range(1, 8)], 100
    raise ValueError(f"Unknown preset: {preset}")


def nash1_diag(A: np.ndarray) -> np.ndarray:
    A = np.maximum(A, 1e-6)
    den = np.sum(1.0 / np.diag(A))
    return 1.0 / (np.diag(A) * den)


def adversary(A: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    column_sums = A @ x
    idx = int(np.argmin(column_sums))
    return float(column_sums[idx]), idx


def generate_diagonal_matrix(n: int) -> np.ndarray:
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        B[i, i] = 0.4 + 0.2 * i / (n - 1)
    return B


def generate_bernoulli_diagonal_matrix(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = A.shape[0]
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        B[i, i] = rng.binomial(1, A[i, i])
    return B


def _value_of_diag_game(A: np.ndarray) -> float:
    den = 0.0
    for i in range(A.shape[0]):
        den += 1.0 / A[i, i]
    return 1.0 / den


# Baseline: play Nash equilibrium of empirical estimate each round.
def run_nash(seed: int, horizon: int, n: int) -> float:
    rng = np.random.default_rng(seed)
    B = generate_diagonal_matrix(n)
    V = _value_of_diag_game(B)
    B1 = np.zeros((n, n), dtype=float)
    reg = 0.0
    for t in range(horizon):
        x = nash1_diag(B1)
        val, _ = adversary(B, x)
        reg += V - val
        Bsamp = generate_bernoulli_diagonal_matrix(B, rng)
        B1 = (t / (t + 1)) * B1 + (1.0 / (t + 1)) * Bsamp
    return float(max(reg, 1e-12))


def _adj_diag_and_D_from_diag(diag: np.ndarray) -> tuple[np.ndarray, float]:
    """
    For a diagonal matrix B=diag(diag), adj(B) is also diagonal with entries:
      adj(B)[i,i] = prod_{k != i} diag[k].
    The paper uses D = 1^T adj(B) 1 (scalar) and x' proportional to adj(B^T) 1.
    For diagonal B, B^T=B and x' is proportional to 1/diag (as expected).
    """
    d = np.maximum(np.asarray(diag, dtype=float), 1e-12)
    prod_all = float(np.prod(d))
    adj_diag = prod_all / d
    D = float(np.sum(adj_diag))
    return adj_diag, D


def _project_interval(v: np.ndarray, bound: float) -> np.ndarray:
    return np.clip(v, -bound, bound)


def _subroutine_paper_nxm(
    *,
    rng: np.random.Generator,
    true_A: np.ndarray,
    V_true: float,
    A_hat_fixed: np.ndarray,
    x_prime: np.ndarray,
    D1: float,
    T1: int,
    T2: int,
    Abar_accum: np.ndarray,
    t_global_start: int,
) -> tuple[float, np.ndarray]:
    """
    Implements Algorithm (Subroutine for n×m games) from the paper, specialized to our setting:
    - We still use the paper's update rule for delta with eta=1/(D1*T1) and projection box.
    - A_hat_fixed stays fixed during these T2 rounds.
    - We keep collecting matrix feedback samples and return updated Abar_accum (sum of samples).
    Returns: (regret accumulated over these T2 rounds, updated Abar_accum).
    """
    n = true_A.shape[0]
    if n < 2:
        raise ValueError("Subroutine requires n>=2.")
    T1 = max(int(T1), 1)
    D1 = float(max(D1, 1e-12))

    eta = 1.0 / (D1 * T1)
    bound = 1.0 / (D1 * math.sqrt(T1))
    delta = np.full(n - 1, -bound, dtype=float)

    reg = 0.0
    for tau in range(1, T2 + 1):
        vec_delta = np.empty(n, dtype=float)
        vec_delta[:-1] = delta
        vec_delta[-1] = -float(np.sum(delta))
        x_t = x_prime + vec_delta
        # Numerical safety: clip and renormalize to simplex if needed.
        x_t = np.clip(x_t, 0.0, 1.0)
        s = float(np.sum(x_t))
        if s <= 0:
            x_t = np.ones(n, dtype=float) / n
        else:
            x_t /= s

        val, j_t = adversary(true_A, x_t)
        reg += V_true - val

        # Observe full-information noisy matrix sample and accumulate for future Abar updates.
        A_samp = generate_bernoulli_diagonal_matrix(true_A, rng)
        Abar_accum += A_samp

        # Paper update: delta_{t+1/2}(i) = delta_t(i) + eta*(Ahat_{i,j}-Ahat_{n,j})
        col = int(j_t)
        g = A_hat_fixed[:-1, col] - A_hat_fixed[-1, col]
        delta_half = delta + eta * g
        delta = _project_interval(delta_half, bound)

    return float(reg), Abar_accum


def _update_official_diag(A: np.ndarray, x1: np.ndarray, j: int, t: int) -> np.ndarray:
    """
    Official diagonal-experiment update used in authors' `new_experiments.py`:
    x_{t+1} = clip(x_t + (A_{:,j}-A_{n,j})/t) with simplex fix by renormalizing first n-1 coords.
    """
    vec = A[:-1, j] - A[-1, j]
    x1[:-1] = np.clip(x1[:-1] + vec * (1.0 / max(1, t)), 0.0, 1.0)
    s = float(np.sum(x1[:-1]))
    if s > 1.0:
        x1[:-1] /= s
        x1[-1] = 0.0
    else:
        x1[-1] = 1.0 - s
    return x1


def run_official_diag_algo(seed: int, horizon: int, n: int) -> float:
    """
    Faithful to the authors' released `official_paper_code/official_paper_code/new_experiments.py`
    for the n×n diagonal full-noisy-feedback experiment (the one used for the paper figure).

    Key behaviors that match the official script:
    - threshold = min(log(T)^2, sqrt(T))
    - alternate between (i) Nash equilibrium of running empirical matrix B1 and (ii) repeated small updates
      using a frozen copy B2, the last adversary column j_t, and step size 1/t0.
    - "double accumulation" of regret per step (sum += V - val twice) exactly as in the official script.
    """
    rng = np.random.default_rng(seed)
    B = generate_diagonal_matrix(n)
    V = _value_of_diag_game(B)

    B1 = np.zeros((n, n), dtype=float)
    B2 = np.zeros((n, n), dtype=float)
    jt = 0
    x = np.ones(n, dtype=float) / n
    count0 = 1
    t0 = 1
    threshold = min(math.log(max(horizon, 2)) ** 2, horizon**0.5)
    reg = 0.0

    for t in range(horizon):
        if count0 > 0 and t > threshold:
            x = _update_official_diag(B2, x, jt, t0)
            count0 -= 1
        else:
            t0 = t + 1
            x = nash1_diag(B1)
            count0 = t0 - 1
            B2 = B1.copy()

        val, jt = adversary(B, x)
        reg += V - val
        Bsamp = generate_bernoulli_diagonal_matrix(B, rng)
        B1 = (t / (t + 1)) * B1 + (1.0 / (t + 1)) * Bsamp
        # Official script behavior: add regret twice per time step.
        reg += V - val

    return float(max(reg, 1e-12))


def run_full_information_algo(seed: int, horizon: int, n: int) -> float:
    """
    Closest faithful implementation of the paper's Section 3 algorithm structure for our diagonal full-support setting:
    - Phase 1 (burn-in): play Nash equilibrium of running empirical matrix for t_star rounds.
      (Paper uses stopping conditions based on support identification + concentration; we implement a conservative,
       horizon-dependent burn-in length for this diagonal experiment.)
    - Phase 2: invoke the paper subroutine with doubling trick; A_hat is held fixed within each invocation.
    - Subroutine parameters match the paper's description: (D1, T1, T2, x', A_hat).
    """
    rng = np.random.default_rng(seed)
    true_A = generate_diagonal_matrix(n)
    V_true = _value_of_diag_game(true_A)

    # Running sum of observed matrices; Abar = Abar_accum / t
    Abar_accum = np.zeros((n, n), dtype=float)
    reg = 0.0

    # --- Phase 1: burn-in (support ID is trivial here: full support for this diagonal family) ---
    # Paper burn-in is instance-dependent; we choose a small polylog schedule to get into the "concentrated" regime.
    t_star = int(max(1, math.ceil(math.log(max(horizon, 3)) ** 2)))
    t = 0
    while t < min(t_star, horizon):
        Abar = Abar_accum / max(1, t)
        x_t = nash1_diag(Abar) if t > 0 else np.ones(n, dtype=float) / n
        val, _ = adversary(true_A, x_t)
        reg += V_true - val
        A_samp = generate_bernoulli_diagonal_matrix(true_A, rng)
        Abar_accum += A_samp
        t += 1

    if t >= horizon:
        return float(max(reg, 1e-12))

    # --- Phase 2: doubling trick invocations of the subroutine ---
    # We follow the paper's invocation schedule:
    # first call at t_star+1, then 2*t_star+1, then 4*t_star+1, ...
    base = max(t, 1)
    next_invoke_at = base + 1
    inv_len = base
    while t < horizon:
        if t + 1 < next_invoke_at:
            # In between invocations, continue playing Nash of running empirical mean.
            Abar = Abar_accum / max(1, t)
            x_t = nash1_diag(Abar)
            val, _ = adversary(true_A, x_t)
            reg += V_true - val
            A_samp = generate_bernoulli_diagonal_matrix(true_A, rng)
            Abar_accum += A_samp
            t += 1
            continue

        # Snapshot empirical matrix at time t0 (=inv_len) and keep fixed during this run.
        Abar = Abar_accum / max(1, t)
        A_hat_fixed = Abar.copy()

        # Paper parameters (specialized):
        # T1 ≈ t0 / log(nT), T2 = min{t0, T - t0} (we cap by remaining horizon)
        T_total = max(horizon, 2)
        log_factor = max(1.0, math.log(n * T_total))
        T1 = max(1, int(inv_len / log_factor))
        T2 = int(min(inv_len, horizon - t))

        # x' = adj(Bhat^T)1 / (1^T adj(Bhat^T)1); for diagonal this equals Nash of diag(A_hat_fixed).
        x_prime = nash1_diag(A_hat_fixed)

        # D1 ≈ |1^T adj(Bhat) 1| / (k alpha_k); for our diagonal full-support case k=n.
        # alpha_k is not specified numerically in the paper text; for this experiment we set alpha_k=1.
        adj_diag, D_hat = _adj_diag_and_D_from_diag(np.diag(A_hat_fixed))
        alpha_k = 1.0
        D1 = abs(D_hat) / (n * alpha_k)

        reg_seg, Abar_accum = _subroutine_paper_nxm(
            rng=rng,
            true_A=true_A,
            V_true=V_true,
            A_hat_fixed=A_hat_fixed,
            x_prime=x_prime,
            D1=D1,
            T1=T1,
            T2=T2,
            Abar_accum=Abar_accum,
            t_global_start=t,
        )
        reg += reg_seg
        t += T2

        # Update doubling schedule
        inv_len *= 2
        next_invoke_at = inv_len + 1

    return float(max(reg, 1e-12))


def _nash_equilibrium_lp(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Appendix algorithms require repeatedly computing the Nash equilibrium of an empirical matrix.
    We implement this via LP formulations of minimax using SciPy HiGHS.
    Returns (x, y, V) for row/column strategies and game value.
    Assumes payoffs are in [0,1] (as in the Appendix after rescaling).
    """
    A = np.asarray(A, dtype=float)
    n, m = A.shape

    # Row player's maximin: maximize v s.t. A^T x >= v 1, sum x=1, x>=0.
    c = np.zeros(n + 1, dtype=float)
    c[-1] = -1.0
    A_ub = np.hstack([-A.T, np.ones((m, 1), dtype=float)])
    b_ub = np.zeros(m, dtype=float)
    A_eq = np.zeros((1, n + 1), dtype=float)
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0], dtype=float)
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Row-player LP failed: {res.message}")
    x = np.maximum(res.x[:n], 0.0)
    x /= max(1e-12, float(np.sum(x)))
    v = float(res.x[-1])

    # Column player's minimax: minimize w s.t. A y <= w 1, sum y=1, y>=0.
    c2 = np.zeros(m + 1, dtype=float)
    c2[-1] = 1.0
    A_ub2 = np.hstack([A, -np.ones((n, 1), dtype=float)])
    b_ub2 = np.zeros(n, dtype=float)
    A_eq2 = np.zeros((1, m + 1), dtype=float)
    A_eq2[0, :m] = 1.0
    b_eq2 = np.array([1.0], dtype=float)
    bounds2 = [(0.0, 1.0)] * m + [(None, None)]
    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")
    if not res2.success:
        raise RuntimeError(f"Column-player LP failed: {res2.message}")
    y = np.maximum(res2.x[:m], 0.0)
    y /= max(1e-12, float(np.sum(y)))
    w = float(res2.x[-1])

    return x, y, 0.5 * (v + w)


def _support(v: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return np.where(v > tol)[0]


def _M_matrix(A_sq: np.ndarray) -> np.ndarray:
    A_sq = np.asarray(A_sq, dtype=float)
    k = A_sq.shape[0]
    M = np.zeros((k, k), dtype=float)
    for i in range(k - 1):
        M[i, :] = A_sq[0, :] - A_sq[i + 1, :]
    M[k - 1, :] = 1.0
    return M


def _M_replace_col(M: np.ndarray, col_idx: int) -> np.ndarray:
    k = M.shape[0]
    out = M.copy()
    b = np.zeros(k, dtype=float)
    b[-1] = 1.0
    out[:, col_idx] = b
    return out


def _det(M: np.ndarray) -> float:
    return float(np.linalg.det(np.asarray(M, dtype=float)))


def _tilde_delta_min(Bbar: np.ndarray) -> float:
    Bbar = np.asarray(Bbar, dtype=float)
    k = Bbar.shape[0]
    M_B = _M_matrix(Bbar)
    M_BT = _M_matrix(Bbar.T)
    cands = [abs(_det(M_B)), abs(_det(M_BT))]
    for i in range(k):
        cands.append(abs(_det(_M_replace_col(M_BT, i))))
    for j in range(k):
        cands.append(abs(_det(_M_replace_col(M_B, j))))
    return float(min(cands))


def _has_unique_full_support_ne_square(Bbar: np.ndarray, det_tol: float = 1e-12) -> bool:
    Bbar = np.asarray(Bbar, dtype=float)
    k = Bbar.shape[0]
    M_B = _M_matrix(Bbar)
    M_BT = _M_matrix(Bbar.T)
    det_MB = _det(M_B)
    det_MBT = _det(M_BT)
    if abs(det_MB) <= det_tol or abs(det_MBT) <= det_tol:
        return False
    s_row = np.sign(det_MBT)
    s_col = np.sign(det_MB)
    for i in range(k):
        if np.sign(_det(_M_replace_col(M_BT, i))) != s_row:
            return False
    for j in range(k):
        if np.sign(_det(_M_replace_col(M_B, j))) != s_col:
            return False
    return True


def _identify_optimal_submatrix_indices(
    *,
    rng: np.random.Generator,
    true_A: np.ndarray,
    horizon: int,
    support_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray]:
    """
    Implements Appendix Algorithm 'Algorithm to identify the optimal submatrix' (Alg. ref{alg-nxm-instance}).
    Returns (I_rows, J_cols, t_stop, regret_incurred, Abar_accum).
    """
    n, m = true_A.shape
    Abar_accum = np.zeros((n, m), dtype=float)
    x_t = np.ones(n, dtype=float) / n
    reg = 0.0
    _, _, V_true = _nash_equilibrium_lp(true_A)

    for t in range(1, horizon + 1):
        # Play x_t, adversary best response.
        col_sums = true_A.T @ x_t
        j_t = int(np.argmin(col_sums))
        reg += V_true - float(col_sums[j_t])

        # Observe full information noisy matrix.
        if n == m:
            A_samp = generate_bernoulli_diagonal_matrix(true_A, rng)
        else:
            A_samp = rng.binomial(1, np.clip(true_A, 0.0, 1.0))
        Abar_accum += A_samp
        Abar = Abar_accum / t

        x_prime, y_prime, V_bar = _nash_equilibrium_lp(Abar)
        x_t = x_prime

        I = _support(x_prime, tol=support_tol)
        J = _support(y_prime, tol=support_tol)
        if I.size != J.size or I.size == 0:
            continue
        k = int(I.size)
        I = I[:k]
        J = J[:k]
        Bbar = Abar[np.ix_(I, J)]

        Delta = math.sqrt(2.0 * math.log(max(2.0, n * m * (horizon**2))) / t)
        tildeDelta = k * math.factorial(k) * math.sqrt(2.0 * math.log(max(2.0, n * m * (horizon**2))) / t)
        tdel_min = _tilde_delta_min(Bbar)

        denom = tdel_min - 2.0 * (k**2) * math.factorial(k) * Delta
        if denom <= 0:
            continue
        ratio = (tdel_min + 2.0 * (k**2) * math.factorial(k) * Delta) / denom
        if not (1.0 <= ratio <= 1.5):
            continue

        # Empirical gap estimates
        if k != n:
            outside = np.setdiff1d(np.arange(n), I, assume_unique=False)
            tildeDelta_g1 = float("inf") if outside.size == 0 else float(V_bar - float(np.max(Abar[outside, :] @ y_prime)))
        else:
            tildeDelta_g1 = float("inf")
        if k != m:
            outsideJ = np.setdiff1d(np.arange(m), J, assume_unique=False)
            tildeDelta_g2 = float("inf") if outsideJ.size == 0 else float(float(np.min(x_prime @ Abar[:, outsideJ])) - V_bar)
        else:
            tildeDelta_g2 = float("inf")
        tildeDelta_g = float(min(tildeDelta_g1, tildeDelta_g2))

        tildeD = float(min(abs(_det(_M_matrix(Bbar.T))), abs(_det(_M_matrix(Bbar)))))
        if tildeD <= 0:
            continue

        if tildeDelta_g >= (5.0 * k * tildeDelta / tildeD + 2.0 * Delta):
            return I, J, t, float(reg), Abar_accum

    # Fallback: no identification within horizon
    return np.arange(n), np.arange(min(n, m)), horizon, float(reg), Abar_accum


def run_theory_lp_algo(seed: int, horizon: int, n: int) -> float:
    """
    Implements the full Appendix pipeline for Section 3 (best-effort faithful):
    - Burn-in/support identification: Alg. ref{alg-nxm-instance}
    - Then full-row-support algorithm: Alg. ref{alg-nxn-full} with subroutine ref{subroutine-nxn}
    Uses LP-based Nash equilibrium computations for empirical matrices.
    """
    rng = np.random.default_rng(seed)
    true_A = generate_diagonal_matrix(n)

    I, J, t_burn, reg, Abar_accum = _identify_optimal_submatrix_indices(rng=rng, true_A=true_A, horizon=horizon)
    if t_burn >= horizon:
        return float(max(reg, 1e-12))

    A_red = true_A[I, :]
    n_red, m_red = A_red.shape
    fact = math.factorial(n_red)
    J = np.asarray(J, dtype=int)
    if J.size < n_red:
        J = np.arange(n_red, dtype=int)

    _, _, V_true = _nash_equilibrium_lp(A_red)

    def Abar_at(t_now: int) -> np.ndarray:
        return Abar_accum[I, :] / max(1, t_now)

    x_t = np.ones(n_red, dtype=float) / n_red
    t_star = horizon + 1

    for t in range(t_burn + 1, horizon + 1):
        col_sums = A_red.T @ x_t
        j_t = int(np.argmin(col_sums))
        reg += V_true - float(col_sums[j_t])

        A_samp = generate_bernoulli_diagonal_matrix(true_A, rng)
        Abar_accum += A_samp

        Abar = Abar_at(t)
        x_prime, _, _ = _nash_equilibrium_lp(Abar)
        x_t = x_prime

        Delta = math.sqrt(2.0 * math.log(max(2.0, n_red * m_red * (horizon**2))) / t)
        Bbar = Abar[:, J[:n_red]]
        tdel_min = _tilde_delta_min(Bbar)
        denom = tdel_min - 2.0 * (n_red**2) * fact * Delta
        if denom > 0 and t_star > horizon:
            ratio = (tdel_min + 2.0 * (n_red**2) * fact * Delta) / denom
            if 1.0 <= ratio <= 1.5:
                t_star = int(math.ceil(6.25 * t))

        if t >= t_star:
            if _has_unique_full_support_ne_square(Bbar):
                t1 = int(t)
                while t1 < horizon:
                    Abar = Abar_at(t1)
                    x_prime, _, _ = _nash_equilibrium_lp(Abar)
                    A_hat = Abar.copy()
                    tildeD = abs(_det(_M_matrix(Abar[:, J[:n_red]].T)))
                    Delta1 = math.sqrt(2.0 * math.log(max(2.0, n_red * m_red * (horizon**2))) / t1)
                    D1 = tildeD / (5.0 * n_red * fact)
                    T1 = max(1, int((1.0 / max(Delta1, 1e-12)) ** 2))
                    T2 = int(min(t1, horizon - t1))
                    reg_seg, Abar_accum = _subroutine_paper_nxm(
                        rng=rng,
                        true_A=A_red,
                        V_true=V_true,
                        A_hat_fixed=A_hat,
                        x_prime=x_prime,
                        D1=D1,
                        T1=T1,
                        T2=T2,
                        Abar_accum=Abar_accum,
                        t_global_start=t1,
                    )
                    reg += reg_seg
                    t1 *= 2
                return float(max(reg, 1e-12))

            # con2 branch: continue with empirical NE until end
            for tt in range(t + 1, horizon + 1):
                Abar = Abar_at(tt - 1)
                x_prime, _, _ = _nash_equilibrium_lp(Abar)
                col_sums = A_red.T @ x_prime
                j_t = int(np.argmin(col_sums))
                reg += V_true - float(col_sums[j_t])
                A_samp = generate_bernoulli_diagonal_matrix(true_A, rng)
                Abar_accum += A_samp
            return float(max(reg, 1e-12))

    return float(max(reg, 1e-12))

# treats each row independently like a bandit, ignores game structure (Nash equilibrium) → wastes information → slower learning
def run_hedge(seed: int, horizon: int, n: int) -> float:
    rng = np.random.default_rng(seed)
    B = generate_diagonal_matrix(n)
    V = _value_of_diag_game(B)
    weights = np.ones(n, dtype=float)
    eta = (math.log(n) / max(1, horizon)) ** 0.5
    reg = 0.0
    for _ in range(horizon):
        x = weights / np.sum(weights)
        val, idx = adversary(B, x)
        reg += V - val
        Bsamp = generate_bernoulli_diagonal_matrix(B, rng)
        reward_vector = Bsamp[:, idx]
        weights *= np.exp(eta * reward_vector)
    return float(max(reg, 1e-12))


def run(config: RunConfig) -> None:
    ensure_dir("plots")
    # Paper figures use x = log10(T) with integer ticks 1..7 for T in {10^1,...,10^7}.
    x_axis = np.log10(np.asarray(config.horizons, dtype=float))
    if config.variant == "official":
        our_fn = run_official_diag_algo
        our_label = "Our-Algo"
    elif config.variant == "subroutine":
        our_fn = run_full_information_algo
        our_label = "Our-Algo (subroutine)"
    elif config.variant == "theory-lp":
        our_fn = run_theory_lp_algo
        our_label = "Our-Algo (theory LP)"
    else:
        raise ValueError(f"Unknown variant: {config.variant}")

    # Plot order matters when curves overlap. We draw Nash last so it's always visible.
    # Additionally, for the slow/expensive theory-LP variant we disable the orange band
    # because it often covers the Nash curve when they are close.
    show_our_band = config.variant != "theory-lp"
    algo_specs = [
        ("Hedge", run_hedge, "#2ca02c", False, 1),
        (our_label, our_fn, "#ff7f0e", show_our_band, 2),
        ("Nash", run_nash, "#1f77b4", True, 3),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, fn, color, show_band, z in algo_specs:
        means: list[float] = []
        stds: list[float] = []
        for T in config.horizons:
            vals = []
            for r in range(config.n_runs):
                seed = config.seed + 10007 * r + 37 * T
                vals.append(fn(seed, T, config.n_actions))
            arr = np.log10(np.maximum(np.asarray(vals, dtype=float), 1e-12))
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))

        y = np.asarray(means, dtype=float)
        ci = np.asarray(stds, dtype=float)
        ax.plot(x_axis, y, marker="o", label=label, color=color, zorder=z)
        if show_band:
            ax.fill_between(x_axis, y - ci, y + ci, alpha=0.25, color=color, zorder=z - 0.1)

    ax.set_xscale("linear")
    ax.set_xlabel("Log of Time Step")
    ax.set_ylabel("Log of Regret")
    xmin, xmax = float(np.min(x_axis)), float(np.max(x_axis))
    ax.set_xlim(max(0.5, xmin - 0.25), min(7.5, xmax + 0.25))
    ax.set_xticks(np.arange(int(np.floor(xmin)), int(np.ceil(xmax)) + 1))
    ax.set_ylim(-1.2, 2.8)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper left")
    ax.set_title(f"(a) {config.n_actions} × {config.n_actions} matrix")
    plt.tight_layout()
    plt.savefig(
        f"plots/section3_official_style_{config.preset}_n{config.n_actions}.png",
        dpi=170,
    )
    plt.show()


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default="quick", choices=["quick", "medium", "paper-lite", "final", "paper"])
    p.add_argument("--horizons", type=int, nargs="*", default=None, help="Optional custom horizons list.")
    p.add_argument("--n_runs", type=int, default=None, help="Optional override of preset trial count.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n_actions", type=int, default=20, help="Diagonal game size n (official default: 20).")
    p.add_argument("--variant", type=str, default="official", choices=["official", "subroutine", "theory-lp"])
    a = p.parse_args()
    preset_horizons, preset_runs = section3_horizons_for_preset(a.preset)
    horizons = a.horizons if a.horizons else preset_horizons
    n_runs = a.n_runs if a.n_runs is not None else preset_runs
    return RunConfig(
        horizons=sorted(horizons),
        n_runs=n_runs,
        seed=a.seed,
        preset=a.preset,
        n_actions=a.n_actions,
        variant=a.variant,
    )


if __name__ == "__main__":
    run(parse_args())
