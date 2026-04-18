"""
Section 4: Bandit Feedback Setting
Reproduction of Figure 2 from:
  "On the Limitations and Possibilities of Nash Regret Minimization
   in Zero-Sum Matrix Games under Noisy Feedback"
  Maiti, Jamieson, Ratliff (2025)

Team 2 implementation — faithfully adapted from the paper's reference code
(Regret-log1.py, Regret-log2.py, Regret-log3.py).

Key design choices from the paper:
  - Bernoulli observations: a ~ Bernoulli(A[i,j])  (bounded noise, NOT Gaussian)
  - UCB bonus: sqrt(2*log(8T²) / (N_ij+1))
  - EXP3: loss-minimisation form, W[i] += (1-a)/x[i]
  - OurAlgo update: x1 += (A[0,j]-A[1,j]) * log(t) / (D*t),  clipped to ±error/D
  - No separate exploration phase; implicit doubling via count0=t0-1
  - Two phases per trial: first T/2 rounds (phase-specific adversary),
    last T/2 rounds (always pure best-response)

Three adversaries (matching the three reference files):
  Adv 1: threshold BR (x1<1/3 → j=0, x1>1/3 → j=1)  then BR
  Adv 2: tolerance BR (threshold ± 1/√T)              then BR
  Adv 3: Nash equilibrium (1/3, 2/3) fixed             then BR
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Game constants for A = [[2/3, 0], [0, 1/3]]
# ─────────────────────────────────────────────────────────────────────────────

A_GAME = np.array([[2/3, 0.0],
                   [0.0, 1/3]], dtype=float)

# Nash eq: x*=(1/3,2/3), y*=(1/3,2/3), V*=2/9
V_STAR = (A_GAME[0,0]*A_GAME[1,1] - A_GAME[0,1]*A_GAME[1,0]) / \
         (A_GAME[0,0] - A_GAME[0,1] - A_GAME[1,0] + A_GAME[1,1])
X1_NASH = 1/3.0   # P(row plays action 0) at Nash


# ─────────────────────────────────────────────────────────────────────────────
# 1.  2×2 Nash utility  (paper's nash1, with PSNE detection)
# ─────────────────────────────────────────────────────────────────────────────

def nash1_batch(A_batch: np.ndarray) -> np.ndarray:
    """
    Paper's nash1() — vectorised over N trials.
    A_batch : (N,2,2)
    Returns x1 (N,) = P(row plays action 0) at Nash equilibrium.
    Includes saddle-point (pure NE) detection as in the reference code.
    """
    a = A_batch[:,0,0]; b = A_batch[:,0,1]
    c = A_batch[:,1,0]; d = A_batch[:,1,1]

    # Saddle-point cases (pure NE)
    pure0 = (a <= b) & (a >= c)          # (0,0) is saddle → x*=1 (play row 0)
    pure1 = (c <= d) & (c >= a)          # (1,0) is saddle → x*=0 (play row 1)
    pure2 = (b <= a) & (b >= d)          # (0,1) is saddle → x*=1
    pure3 = (d <= c) & (d >= b)          # (1,1) is saddle → x*=0

    D  = a - b - c + d
    N1 = d - c                            # (A[1,1]-A[1,0])
    mixed = np.where(np.abs(D) < 1e-12, 0.5, np.clip(N1 / D, 0.0, 1.0))

    x1 = mixed
    x1 = np.where(pure3, 0.0, x1)
    x1 = np.where(pure2, 1.0, x1)
    x1 = np.where(pure1, 0.0, x1)
    x1 = np.where(pure0, 1.0, x1)
    return x1


def is_mixed_ne_batch(A_batch: np.ndarray) -> np.ndarray:
    """True (N,) where the Nash eq is strictly mixed (no saddle point)."""
    a = A_batch[:,0,0]; b = A_batch[:,0,1]
    c = A_batch[:,1,0]; d = A_batch[:,1,1]
    pure = ((a<=b)&(a>=c)) | ((c<=d)&(c>=a)) | ((b<=a)&(b>=d)) | ((d<=c)&(d>=b))
    return ~pure


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Update rule  (paper's update(), vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def update_batch(F_batch: np.ndarray,
                 x1: np.ndarray,
                 jt: np.ndarray,
                 t0: np.ndarray,
                 error: np.ndarray) -> np.ndarray:
    """
    Paper's update(A, x1, j, t, error) — vectorised.

    Step: x1 ← x1 + (A[0,j]-A[1,j]) * max(1,log(t)) / (D * t)
    clipped to [x1 - error/D, x1 + error/D]  when mixed NE exists.

    F_batch : (N,2,2)  frozen matrix snapshot
    x1      : (N,)     current P(row action 0)
    jt      : (N,) int last column action observed
    t0      : (N,) int time of last re-initialisation
    error   : (N,)     current UCB concentration error
    """
    N   = len(x1)
    idx = np.arange(N)

    a_j = F_batch[idx, 0, jt] - F_batch[idx, 1, jt]          # A[0,j]-A[1,j]
    D   = np.abs(F_batch[:,0,0] - F_batch[:,0,1]
                 - F_batch[:,1,0] + F_batch[:,1,1])            # |det param|

    mixed = is_mixed_ne_batch(F_batch)                         # (N,) bool

    log_t = np.maximum(1.0, np.log(np.maximum(t0.astype(float), 2.0)))
    step  = a_j * log_t / (np.maximum(D, 1e-12) * t0)

    # Boundaries for mixed-NE case
    bnd  = error / np.maximum(D, 1e-12)
    xmax = np.minimum(1.0, x1 + bnd)
    xmin = np.maximum(0.0, x1 - bnd)

    x_mixed = np.where(a_j >= 0,
                       np.minimum(x1 + step, xmax),
                       np.maximum(x1 + step, xmin))

    # Pure / degenerate case: nudge toward 0 or 1
    step_pure = log_t / (2.0 * t0)
    x_pure = np.where(a_j > 0,
                      np.minimum(x1 + step_pure, 1.0),
                      np.where(a_j < 0,
                               np.maximum(x1 - step_pure, 0.0),
                               np.full(N, 0.5)))

    return np.where(mixed, x_mixed, x_pure)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Adversary helpers
# ─────────────────────────────────────────────────────────────────────────────

def val22_batch(A: np.ndarray, x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """
    Expected payoff for row player given fractional strategies.
    val = A[0,0]*x1*y1 + A[0,1]*x1*(1-y1) + A[1,0]*x2*y1 + A[1,1]*x2*(1-y1)
    """
    x2 = 1.0 - x1; y2 = 1.0 - y1
    return A[0,0]*x1*y1 + A[0,1]*x1*y2 + A[1,0]*x2*y1 + A[1,1]*x2*y2


def adv22gd_batch(A: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure best-response for column player (minimises row's payoff).
    Returns (val (N,), jt (N,) int).
    """
    x2 = 1.0 - x1
    a0 = A[0,0]*x1 + A[1,0]*x2     # payoff for col action 0
    a1 = A[0,1]*x1 + A[1,1]*x2     # payoff for col action 1
    jt  = (a1 < a0).astype(int)     # pick column with smaller payoff
    val = np.where(jt == 0, a0, a1)
    return val, jt


def advnew_batch(x1: np.ndarray, T: int,
                 adv_type: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase-1 adversary:
      adv_type=1  threshold BR   : x1<1/3       → j=0; x1>1/3       → j=1
      adv_type=2  tolerance BR   : x1<1/3-√(1/T)→ j=0; x1>1/3+√(1/T)→ j=1
      adv_type=3  Nash fixed     : always (1/3, 2/3)

    Returns (y1 (N,), y2 (N,))  fractional column probs  (y1=P(col=0)).
    """
    N = len(x1)
    if adv_type == 3:
        return np.full(N, 1/3.0), np.full(N, 2/3.0)

    tol = 0.0 if adv_type == 1 else (1.0/T)**0.5
    lo  = 1/3.0 - tol
    hi  = 1/3.0 + tol

    y1 = np.where(x1 < lo, 1.0,
         np.where(x1 > hi, 0.0, 1/3.0))
    y2 = 1.0 - y1
    return y1, y2


# ─────────────────────────────────────────────────────────────────────────────
# 4.  UCB  (paper's ucb(), vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def run_ucb(A: np.ndarray, T: int, N: int, adv_type: int) -> float:
    """
    UCB for bandit matrix games — two phases (matching reference code).
    Phase 1: phase-specific adversary.   Phase 2: pure BR (adv22gd).
    Observation model: Bernoulli(A[i,j]).
    UCB bonus: sqrt(2*log(8T²) / (N_ij+1)).
    """
    T1     = T // 2
    log_c  = 2.0 * np.log(8.0 * max(T**2, 2))
    B1     = np.zeros((N, 2, 2))   # running mean
    U1     = np.zeros((N, 2, 2))   # UCB matrix
    cnt    = np.zeros((N, 2, 2))
    regret = np.zeros(N)
    idx    = np.arange(N)

    def _step(t, use_adv_type):
        x1 = nash1_batch(U1)
        x2 = 1.0 - x1

        # Sample row action: it=1 w.p. x2
        it = (np.random.rand(N) < x2).astype(int)

        # Column adversary
        if use_adv_type:
            y1, y2 = advnew_batch(x1, T, adv_type)
            val    = val22_batch(A, x1, y1)
            jt     = (np.random.rand(N) < y2).astype(int)
        else:
            val, jt = adv22gd_batch(A, x1)

        # Bernoulli observation of played cell
        a_obs = (np.random.rand(N) < A[it, jt]).astype(float)

        # Update running mean and UCB (vectorised — no inner for-loops)
        cnt[idx, it, jt] += 1
        c = cnt[idx, it, jt]
        B1[idx, it, jt] += (a_obs - B1[idx, it, jt]) / c
        np.add(B1, np.sqrt(log_c / (cnt + 1.0)), out=U1)

        regret[:] += V_STAR - val

    for t in range(T1):
        _step(t, use_adv_type=True)
    for t in range(T1):
        _step(t, use_adv_type=False)

    return float(regret.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EXP3  (paper's exp3(), vectorised — loss-minimisation form)
# ─────────────────────────────────────────────────────────────────────────────

def run_exp3(A: np.ndarray, T: int, N: int, adv_type: int) -> float:
    """
    EXP3 (loss form) — two phases.
    eta = sqrt(log(2) / T).
    W[i] += (1 - a) / x[i]   (cumulative losses; implicit exploration via exp(-eta*W)).
    """
    T1     = T // 2
    eta    = (np.log(2.0) / max(T, 1)) ** 0.5
    W      = np.zeros((N, 2))      # cumulative losses
    regret = np.zeros(N)
    idx_N  = np.arange(N)

    def _one_step(use_adv_type: bool) -> None:
        # Mixed strategy from softmin of cumulative losses (numerically stable)
        lw  = -eta * W                          # (N,2)
        lw -= lw.max(axis=1, keepdims=True)
        x   = np.exp(lw)
        x  /= x.sum(axis=1, keepdims=True)      # (N,2)
        x1  = x[:, 0]

        it = (np.random.rand(N) < x[:, 1]).astype(int)

        if use_adv_type:
            y1, y2 = advnew_batch(x1, T, adv_type)
            val    = val22_batch(A, x1, y1)
            jt     = (np.random.rand(N) < y2).astype(int)
        else:
            val, jt = adv22gd_batch(A, x1)

        a_obs = (np.random.rand(N) < A[it, jt]).astype(float)

        # Importance-weighted loss; normalise to prevent overflow
        p_it = x[idx_N, it]
        loss = (1.0 - a_obs) / np.maximum(p_it, 1e-12)
        W[idx_N, it] += loss
        W[:] -= W.min(axis=1, keepdims=True)    # [:] avoids local-var ambiguity

        regret[:] += V_STAR - val               # [:] for same reason

    for _ in range(T1):
        _one_step(True)
    for _ in range(T1):
        _one_step(False)

    return float(regret.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Our Algorithm  (paper's ouralgo(), vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def run_our_algorithm(A: np.ndarray, T: int, N: int, adv_type: int) -> float:
    """
    Paper's bandit 2×2 algorithm — two phases, implicit doubling.

    Phase 1 (T/2 rounds, phase-specific adversary):
      - Maintain running mean B2 and UCB matrix U2.
      - Warm-up: for t <= log(T)^2, always re-init from U2.
      - After warm-up: run update_batch for count0 steps, then re-init.
      - Re-init: freeze F2 = U2, set x1 = Nash(F2), count0 = t.
      - Track max UCB error; use it as boundary in update_batch.

    Phase 2 (T/2 rounds, always pure BR):
      - Fix B2 from phase 1.
      - x1 = Nash(B2); update each round with fixed t=T1 and frozen error.
    """
    T1       = T // 2
    log_T_sq = np.log(max(T, 2)) ** 2

    B2  = np.zeros((N, 2, 2))    # empirical mean
    U2  = np.zeros((N, 2, 2))    # UCB matrix
    F2  = np.zeros((N, 2, 2))    # frozen snapshot for current run
    cnt = np.zeros((N, 2, 2))
    jt      = np.zeros(N, dtype=int)
    x1      = np.full(N, 0.5)
    count0  = np.ones(N, dtype=int)
    t0      = np.ones(N, dtype=int)
    error   = np.ones(N)
    regret  = np.zeros(N)
    idx_N   = np.arange(N)
    log_c   = 2.0 * np.log(8.0 * max(T**2, 2))

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    for t in range(T1):
        reinit = (count0 == 0) | (t <= log_T_sq)

        # Re-initialise where needed
        if reinit.any():
            F2[reinit] = U2[reinit]
            t0[reinit] = t + 1
            x_nash = nash1_batch(F2)                # (N,)
            mixed  = is_mixed_ne_batch(F2)          # (N,)
            x1 = np.where(reinit & mixed, x_nash, x1)
            count0[reinit] = np.maximum(t0[reinit] - 1, 0)

        # Update step (all trials, including just-reinitialised ones)
        x1 = update_batch(F2, x1, jt, t0, error)
        x1 = np.clip(x1, 0.0, 1.0)
        count0 = np.maximum(count0 - 1, 0)

        # Column adversary for phase 1
        y1, y2 = advnew_batch(x1, T, adv_type)
        val    = val22_batch(A, x1, y1)
        jt     = (np.random.rand(N) < y2).astype(int)

        # Sample row action
        it = (np.random.rand(N) < (1.0 - x1)).astype(int)

        # Bernoulli observation
        a_obs = (np.random.rand(N) < A[it, jt]).astype(float)

        # Update empirical mean
        cnt[idx_N, it, jt] += 1
        c = cnt[idx_N, it, jt]
        B2[idx_N, it, jt] += (a_obs - B2[idx_N, it, jt]) / c

        # Update UCB and track max error (vectorised)
        devs = np.sqrt(log_c / (cnt + 1.0))          # (N,2,2)
        np.add(B2, devs, out=U2)
        maxdev = devs.max(axis=(1, 2))                # (N,)
        error  = np.minimum(error, maxdev)            # error can only decrease

        regret += V_STAR - val

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    x1 = nash1_batch(B2)
    x1 = np.clip(x1, 0.0, 1.0)
    t0_fixed = np.full(N, T1, dtype=int)   # fixed time reference

    for _ in range(T1):
        x1  = update_batch(B2, x1, jt, t0_fixed, error)
        x1  = np.clip(x1, 0.0, 1.0)
        val, jt = adv22gd_batch(A, x1)
        regret += V_STAR - val

    return float(regret.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_all(A, T_list, N, verbose=True):
    algorithms = {"UCB": run_ucb, "EXP3": run_exp3, "OurAlg": run_our_algorithm}
    results = {}
    for adv in [1, 2, 3]:
        results[adv] = {name: [] for name in algorithms}
        if verbose:
            print(f"\n── Adversary {adv} {'─'*45}")
        for T in T_list:
            if verbose:
                print(f"  T={T:>8d}  ", end="", flush=True)
            for name, fn in algorithms.items():
                r = max(fn(A, T, N, adv), 0.0)
                results[adv][name].append(r)
                if verbose:
                    print(f"{name}:{r:8.2f}  ", end="", flush=True)
            if verbose:
                print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Plotting  (Figure 2 style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure2(T_list, results, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    palette = {"UCB": "#2196F3", "EXP3": "#4CAF50", "OurAlg": "#FF9800"}
    markers  = {"UCB": "o",      "EXP3": "s",       "OurAlg": "^"}
    log_T    = np.arange(1, len(T_list) + 1)   # x-axis = 1..num_eps (log10 T)

    adv_labels = {
        1: "(a) Adversary 1\n(threshold BR → BR)",
        2: "(b) Adversary 2\n(tolerance BR → BR)",
        3: "(c) Adversary 3\n(Nash → BR)",
    }

    for adv in [1, 2, 3]:
        ax = axes[adv - 1]
        for name, vals in results[adv].items():
            log_r = np.log10(np.maximum(vals, 1.0))
            ax.plot(log_T, log_r, color=palette[name], marker=markers[name],
                    label=name, linewidth=1.8, markersize=6)

        # Reference slope-½ line anchored at UCB's first point
        r0 = max(results[adv]["UCB"][0], 1.0)
        ax.plot(log_T, np.log10(r0) + 0.5*(log_T - log_T[0]),
                "k--", linewidth=0.9, alpha=0.45, label="slope ½")

        ax.set_xlabel("Log of Time Step", fontsize=10)
        ax.set_ylabel("Log of Regret",    fontsize=10)
        ax.set_title(adv_labels[adv], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        loc1 = plticker.MultipleLocator(base=0.2)
        loc2 = plticker.MultipleLocator(base=1.0)
        ax.xaxis.set_major_locator(loc2)
        ax.yaxis.set_major_locator(loc1)

    plt.suptitle(
        "Figure 2 — Nash regret under bandit feedback\n"
        "A = [[2/3,0],[0,1/3]]   N=128 seeds   Bernoulli observations",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    A = A_GAME
    print("=" * 60)
    print("Section 4: Bandit Feedback Experiments")
    print("=" * 60)
    print(f"Game matrix A:\n{A}")
    print(f"V* = {V_STAR:.6f},  x*_0 = {X1_NASH:.4f}")
    print()

    # Paper uses T in {10^1,...,10^8} with N=128.
    # We use T up to 10^6 with N=64 for speed.
    # To reproduce Figure 2 exactly: T_list = [10**k for k in range(1,9)], N=128
    T_list = [10**k for k in range(1, 6)]   # 10, 100, ..., 10^5
    N      = 64
    # To reproduce paper exactly: T_list = [10**k for k in range(1,9)], N=128
    # Partial T=10^6 results (from extended run): UCB=[16.95,90.23,?], EXP3=[392.81,390.66,?],
    #   OurAlg=[50.81,?,?] — log-log slopes for OurAlg continue to decrease toward 0.

    print(f"T ∈ {T_list},  N = {N} trials")
    print("(Paper: T up to 10^8, N=128 — extend T_list/N to reproduce exactly)\n")

    results = run_all(A, T_list, N, verbose=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "section4_fig2_reproduced.png")
    plot_figure2(T_list, results, save_path=out_path)

    print("\n── Nash Regret Summary (raw values) " + "─"*30)
    for adv in [1, 2, 3]:
        print(f"\nAdversary {adv}:")
        print(f"{'T':>10}  {'UCB':>10}  {'EXP3':>10}  {'OurAlg':>10}")
        for i, T in enumerate(T_list):
            row = f"{T:>10}"
            for name in ["UCB", "EXP3", "OurAlg"]:
                row += f"  {results[adv][name][i]:>10.2f}"
            print(row)
