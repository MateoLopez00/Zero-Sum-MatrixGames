# Zero-Sum Matrix Games — Paper Reproduction

Reproduction of the experimental setup from *On the Limitations and Possibilities of Nash Regret Minimization in Zero-Sum Matrix Games under Noisy Feedback* (arXiv:2306.13233v3).

- Section 3 (full-information feedback): `Full_information_feedback/`
- Section 4 (bandit feedback, 2x2): `Bandit_feedback/`

---

# Full-information feedback (Section 3) reproduction

Everything needed to reproduce the paper’s **Section 3 full-information feedback** setting lives in:

- `Full_information_feedback/`

## Install

```bash
cd Full_information_feedback
pip install -r requirements.txt
```

## Reproduce the 4 plots (paper-lite, official)

```bash
cd Full_information_feedback
python experiments_section3.py --preset paper-lite --variant official --n_actions 10
```

In this setting (diagonal matrix game + full-information noisy matrix feedback + best-response adversary), **Our-Algo grows much slower than Nash and Hedge**.

![Section 3 paper-lite n=10](Full_information_feedback/plots/section3_official_style_paper-lite_n10.png)

```bash
cd Full_information_feedback
python experiments_section3.py --preset paper-lite --variant official --n_actions 20
```

In this setting, the same qualitative behavior holds at **n=20**: the proposed method has a flatter regret curve than the baselines.

![Section 3 paper-lite n=20](Full_information_feedback/plots/section3_official_style_paper-lite_n20.png)

```bash
cd Full_information_feedback
python experiments_section3.py --preset paper-lite --variant official --n_actions 50
```

In this setting at **n=50**, the proposed method continues to outperform the baselines across horizons.

![Section 3 paper-lite n=50](Full_information_feedback/plots/section3_official_style_paper-lite_n50.png)

```bash
cd Full_information_feedback
python experiments_section3.py --preset paper-lite --variant official --n_actions 100
```

In this setting at **n=100**, the gap vs Nash/Hedge is still visible (same experimental protocol).

![Section 3 paper-lite n=100](Full_information_feedback/plots/section3_official_style_paper-lite_n100.png)

---

# Bandit feedback (Section 4) reproduction

Everything needed to reproduce the paper’s **Section 4 bandit feedback** setting (2x2 game) lives in:

- `Bandit_feedback/`

## Install

```bash
cd Bandit_feedback
pip install numpy matplotlib pandas jupyter
```

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

The game matrix is

```
A = [[2/3,  0 ],
     [ 0 , 1/3]]
```

with Nash equilibrium `x* = y* = (1/3, 2/3)` and value `V* = 2/9`. The row player **only observes the Bernoulli-sampled entry `A[i_t, j_t]`** at the played cell (not the full row), which is the central challenge of this setting. Each trial has two phases of length T/2: Phase 1 uses a phase-specific adversary, Phase 2 always uses pure best-response.

The algorithm is evaluated against three column-player adversaries:

- **Adversary 1 (threshold BR):** plays pure best-response the moment `x1` deviates from `1/3`; applies maximum adversarial pressure from round 1.
- **Adversary 2 (tolerance BR):** same as Adversary 1 but with a `±1/√T` tolerance band around Nash before punishing.
- **Adversary 3 (Nash fixed → BR):** plays the Nash mixed strategy `y* = (1/3, 2/3)` during Phase 1, then switches to pure best-response in Phase 2.

Achieving `polylog(T)` Nash regret against **all three** adversaries is the key empirical claim of Section 4. The reproduced Figure 2 below shows log(total Nash regret) vs log(T) for Our-Algorithm (Algorithm 6), UCB, and EXP3 against each adversary.

![Section 4 Figure 2](Bandit_feedback/section4_fig2.png)

