"""
Cumulative Nash regret vs time for Section 4 baseline (2×2 bandit).

Mirrors Bandit_feedback/section4_bandit.py (run_ucb, run_exp3, run_our_algorithm).
Uses batch helpers with N=1 and numpy.random.Generator for reproducibility.
Keep in sync with section4_bandit.py if baseline behaviour changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

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

from notebook_plot_backend import ensure_inline_notebook_backend  # noqa: E402

ensure_inline_notebook_backend()

A = A_GAME


def _ucb_one_trajectory(rng: np.random.Generator, T: int, adv_type: int) -> np.ndarray:
    N = 1
    idx = np.arange(N)
    T1 = T // 2
    log_c = 2.0 * np.log(8.0 * max(T**2, 2))
    B1 = np.zeros((N, 2, 2))
    U1 = np.zeros((N, 2, 2))
    cnt = np.zeros((N, 2, 2))
    regret_inst = np.zeros(N)
    curve = np.zeros(T)
    t_global = 0

    def step(use_adv_type: bool) -> None:
        nonlocal t_global
        x1 = nash1_batch(U1)
        it = (rng.random(N) < (1.0 - x1)).astype(int)
        if use_adv_type:
            y1, y2 = advnew_batch(x1, T, adv_type)
            val = val22_batch(A, x1, y1)
            jt = (rng.random(N) < y2).astype(int)
        else:
            val, jt = adv22gd_batch(A, x1)
        a_obs = (rng.random(N) < A[it, jt]).astype(float)
        cnt[idx, it, jt] += 1
        c = cnt[idx, it, jt]
        B1[idx, it, jt] += (a_obs - B1[idx, it, jt]) / c
        np.add(B1, np.sqrt(log_c / (cnt + 1.0)), out=U1)
        regret_inst[:] = V_STAR - val
        curve[t_global] = float(regret_inst.sum())
        t_global += 1

    for _ in range(T1):
        step(True)
    for _ in range(T1):
        step(False)
    return np.cumsum(np.maximum(curve, 0.0))


def _exp3_one_trajectory(rng: np.random.Generator, T: int, adv_type: int) -> np.ndarray:
    N = 1
    idx_N = np.arange(N)
    T1 = T // 2
    eta = (np.log(2.0) / max(T, 1)) ** 0.5
    W = np.zeros((N, 2))
    curve = np.zeros(T)
    t_global = 0

    def one_step(use_adv_type: bool) -> None:
        nonlocal t_global
        lw = -eta * W
        lw -= lw.max(axis=1, keepdims=True)
        x = np.exp(lw)
        x /= x.sum(axis=1, keepdims=True)
        x1 = x[:, 0]
        it = (rng.random(N) < x[:, 1]).astype(int)
        if use_adv_type:
            y1, y2 = advnew_batch(x1, T, adv_type)
            val = val22_batch(A, x1, y1)
            jt = (rng.random(N) < y2).astype(int)
        else:
            val, jt = adv22gd_batch(A, x1)
        a_obs = (rng.random(N) < A[it, jt]).astype(float)
        p_it = x[idx_N, it]
        loss = (1.0 - a_obs) / np.maximum(p_it, 1e-12)
        W[idx_N, it] += loss
        W[:] -= W.min(axis=1, keepdims=True)
        curve[t_global] = float((V_STAR - val).sum())
        t_global += 1

    for _ in range(T1):
        one_step(True)
    for _ in range(T1):
        one_step(False)
    return np.cumsum(np.maximum(curve, 0.0))


def _ouralg_one_trajectory(rng: np.random.Generator, T: int, adv_type: int) -> np.ndarray:
    N = 1
    idx_N = np.arange(N)
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
    log_c = 2.0 * np.log(8.0 * max(T**2, 2))
    curve = np.zeros(T)
    t_global = 0

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
        a_obs = (rng.random(N) < A[it, jt]).astype(float)
        cnt[idx_N, it, jt] += 1
        c = cnt[idx_N, it, jt]
        B2[idx_N, it, jt] += (a_obs - B2[idx_N, it, jt]) / c
        devs = np.sqrt(log_c / (cnt + 1.0))
        np.add(B2, devs, out=U2)
        maxdev = devs.max(axis=(1, 2))
        error = np.minimum(error, maxdev)
        curve[t_global] = float((V_STAR - val).sum())
        t_global += 1

    x1 = nash1_batch(B2)
    x1 = np.clip(x1, 0.0, 1.0)
    t0_fixed = np.full(N, T1, dtype=int)
    for _ in range(T1):
        x1 = update_batch(B2, x1, jt, t0_fixed, error)
        x1 = np.clip(x1, 0.0, 1.0)
        val, jt = adv22gd_batch(A, x1)
        curve[t_global] = float((V_STAR - val).sum())
        t_global += 1

    return np.cumsum(np.maximum(curve, 0.0))


def ucb_cumulative_curve(seed: int, T: int, adv_type: int) -> np.ndarray:
    return _ucb_one_trajectory(np.random.default_rng(seed), T, adv_type)


def exp3_cumulative_curve(seed: int, T: int, adv_type: int) -> np.ndarray:
    return _exp3_one_trajectory(np.random.default_rng(seed), T, adv_type)


def ouralg_cumulative_curve(seed: int, T: int, adv_type: int) -> np.ndarray:
    return _ouralg_one_trajectory(np.random.default_rng(seed), T, adv_type)


def mean_std_curves(
    fn,
    T: int,
    adv_type: int,
    n_runs: int,
    base_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for r in range(n_runs):
        seed = base_seed + 10007 * r
        curves.append(fn(seed, T, adv_type))
    arr = np.stack(curves, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)
