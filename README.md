# Zero-Sum Matrix Games — Paper Reproduction

Reproduction of the experimental setup from *On the Limitations and Possibilities of Nash Regret Minimization in Zero-Sum Matrix Games under Noisy Feedback* (arXiv:2306.13233v3).

- Section 3 (full-information feedback): `Full_information_feedback/`
- Section 4 (bandit feedback, 2x2): `Bandit_feedback/`

---

# Full-information feedback (Section 3) reproduction

## Install

```bash
cd Full_information_feedback
pip install -r requirements.txt
```

## Setting

`n x n` diagonal matrix game with `A[i,i] = 0.4 + 0.2*(i-1)/(n-1)`. Each round the row player sees the **full noisy payoff row** (full-information feedback) and the column player always plays best-response. Plots show `log(total Nash regret)` vs `log(T)` for Our-Algo, Nash-empirical baseline, and Hedge, across `n = 10, 20, 50, 100`. The claim: Our-Algo achieves `polylog(T)` Nash regret while Hedge grows as `sqrt(T)`.

## Reproduce the 4 plots (paper-lite, official)

```bash
cd Full_information_feedback
python experiments_section3.py --preset paper-lite --variant official --n_actions 10
```

At **n=10**, Our-Algo grows much slower than Nash and Hedge.

![Section 3 paper-lite n=10](Full_information_feedback/plots/section3_official_style_paper-lite_n10.png)

```bash
cd Full_information_feedback
python experiments_section3.py --preset paper-lite --variant official --n_actions 20
```

At **n=20**, the same qualitative behavior holds: the proposed method has a flatter regret curve than the baselines.

![Section 3 paper-lite n=20](Full_information_feedback/plots/section3_official_style_paper-lite_n20.png)

```bash
cd Full_information_feedback
python experiments_section3.py --preset paper-lite --variant official --n_actions 50
```

At **n=50**, the proposed method continues to outperform the baselines across horizons.

![Section 3 paper-lite n=50](Full_information_feedback/plots/section3_official_style_paper-lite_n50.png)

```bash
cd Full_information_feedback
python experiments_section3.py --preset paper-lite --variant official --n_actions 100
```

At **n=100**, the gap vs Nash/Hedge is still visible under the same experimental protocol.

![Section 3 paper-lite n=100](Full_information_feedback/plots/section3_official_style_paper-lite_n100.png)

---

# Bandit feedback (Section 4) reproduction

## Install

```bash
cd Bandit_feedback
pip install numpy matplotlib pandas jupyter
```

## Setting

2x2 diagonal matrix game `A = [[2/3, 0], [0, 1/3]]` with Nash equilibrium `x* = y* = (1/3, 2/3)` and value `V* = 2/9`. Each round the row player observes **only the Bernoulli-sampled entry `A[i_t, j_t]`** at the played cell (bandit feedback), not the full row. Every trial runs in two phases of length `T/2`: Phase 1 uses a phase-specific adversary, Phase 2 always uses pure best-response. Our-Algo (Algorithm 6) is compared against UCB and EXP3 against three column adversaries:

- **Adversary 1 (threshold BR):** plays pure best-response the moment `x1` deviates from `1/3`.
- **Adversary 2 (tolerance BR):** same as Adversary 1 but with a `±1/sqrt(T)` tolerance band around Nash before punishing.
- **Adversary 3 (Nash -> BR):** plays Nash `y* = (1/3, 2/3)` during Phase 1, then switches to pure best-response in Phase 2.

The plot shows `log(total Nash regret)` vs `log(T)` for each adversary. The claim: Our-Algo achieves `polylog(T)` Nash regret against **all three** adversaries.

## Reproduce Figure 2 (paper Section 4.1)

```bash
cd Bandit_feedback
jupyter notebook section4_reproduction.ipynb
```

Run all cells top-to-bottom. The standalone script `section4_bandit.py` runs the same pipeline from the command line:

```bash
cd Bandit_feedback
python section4_bandit.py
```

Across all three adversaries, Our-Algo stays essentially flat while UCB and EXP3 grow polynomially — most dramatically against Adversary 3, matching the paper's core claim.

![Section 4 Figure 2](Bandit_feedback/section4_fig2.png)
