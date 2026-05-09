# Zero-Sum Matrix Games: Paper Reproduction

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

## Section 3 plots

The four plots below were generated with `Full_information_feedback/experiments_section3.py` using the `paper-lite` preset and the `official` variant for `n_actions = 10, 20, 50, 100`.

<table>
  <tr>
    <td width="50%"><img src="Full_information_feedback/plots/section3_official_style_paper-lite_n10.png" width="100%"></td>
    <td width="50%"><img src="Full_information_feedback/plots/section3_official_style_paper-lite_n20.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>n = 10</b></td>
    <td align="center"><b>n = 20</b></td>
  </tr>
  <tr>
    <td width="50%"><img src="Full_information_feedback/plots/section3_official_style_paper-lite_n50.png" width="100%"></td>
    <td width="50%"><img src="Full_information_feedback/plots/section3_official_style_paper-lite_n100.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>n = 50</b></td>
    <td align="center"><b>n = 100</b></td>
  </tr>
</table>

At `n=10`, Our-Algo grows much slower than Nash and Hedge. The same qualitative behavior holds at `n=20`, where the proposed method has a flatter regret curve than the baselines. At `n=50` and `n=100`, Our-Algo continues to outperform the baselines across horizons, although the gap shrinks as the matrix size grows.

## Empirical vs theoretical

On log-log axes, a `sqrt(T)` regret rate shows up as a straight line with slope `0.5`, while a `polylog(T)` rate appears as a curve that *flattens* toward slope `0` as `T` grows. The paper's theoretical rates for this setting are:

- **Our-Algo:** `polylog(T)` (with an extra dependence on `n`).
- **Hedge:** `O(sqrt(T log n))`, the standard online learning bound.
- **Nash (empirical):** `O(sqrt(T))`, a baseline that plays Nash of the empirical matrix.

The four plots above match this: Hedge and Nash trace approximately straight lines with slope near `0.5`, while Our-Algo's curve visibly flattens across horizons, consistent with the `polylog(T)` rate. The `n`-dependence predicted by the theory is also visible, since the gap between Our-Algo and the baselines shrinks as `n` grows from 10 to 100, though Our-Algo still stays clearly below `sqrt(T)` behavior.

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

## Figure 2 reproduction (paper Section 4.1)

The figure below was reproduced with `Bandit_feedback/section4_reproduction.ipynb`. The same experiment is also available in `Bandit_feedback/section4_bandit.py`.

Across all three adversaries, Our-Algo stays essentially flat while UCB and EXP3 grow polynomially, most dramatically against Adversary 3, matching the paper's core claim.

![Section 4 Figure 2](Bandit_feedback/section4_fig2.png)

## Empirical vs theoretical

The paper's theoretical rates for the `2x2` bandit setting are:

- **Our-Algo (Algorithm 6):** `polylog(T)` Nash regret against any column adversary.
- **UCB and EXP3:** both are `Omega(sqrt(T))` in this adversarial regime. UCB fails because it is built for stochastic, not adversarial, columns; EXP3 fails because of the general lower bound in the paper's Theorem 3.

On log-log axes this means Our-Algo should have a slope that flattens toward `0`, while UCB and EXP3 should sit on straight lines with slope near `0.5`. Figure 2 matches this prediction: Our-Algo's curve is essentially flat against all three adversaries (most visibly against Adversary 3), while UCB and EXP3 grow at roughly `sqrt(T)` rate. The empirical results therefore align with the theoretical regret bounds claimed in Section 4.

---

# Extension: Noise Robustness

We inject **Gaussian noise** into the feedback (entries clipped to `[0, 1]`): `clip(A + sigma * N(0,1), 0, 1)`. Section 3 keeps **full-matrix** observations each round; Section 4 keeps **bandit** observations (noisy reward only at the played cell).

Code and notebooks:

- `Extensions/Extension_Noise_Robustness_Full_info_feedback/section3_noise_robustness.py` and `section3_noise_robustness.ipynb`
- `Extensions/Extension_Noise_Robustness_Bandit_feedback/section4_noise_robustness.py` and `section4_noise_robustness.ipynb`

Figures below use the **`medium`** preset (`T = 30_000` for Section 3 convergence; `T = 100_000` for Section 4 convergence), multi-$\sigma$ convergence runs at **`sigma in {0.1, 0.2, 0.3}`**, and seed **`7`** (Section 3) / **`42`** (Section 4) unless you change them in the notebooks.

## Section 3 — noise-aware threshold (full information)

Baselines are unchanged. We add **Our-Algo-NoiseAware**, which waits longer before switching out of the exploration phase when feedback is noisier:

```python
threshold = min((1 + 2 * sigma) * log(T)**2, sqrt(T))
```

compared to the original `threshold = min(log(T)**2, sqrt(T))`.

### Summary at `sigma = 0.3` (noise sweep, `medium`)

| n | Nash regret | Hedge regret | Our-Algo regret | Our-Algo-NoiseAware regret | Reduction vs Our-Algo |
|---:|---:|---:|---:|---:|---:|
| 10 | 35.56 | 8.96 | 5.96 | 5.90 | 0.9% |
| 20 | 21.26 | 8.57 | 4.91 | 4.73 | 3.6% |
| 50 | 8.71 | 8.41 | 4.07 | 3.84 | 5.8% |
| 100 | 4.68 | 8.46 | 3.85 | 3.55 | 8.0% |

### Convergence (`n = 100`)

Cumulative Nash regret vs time for four algorithms; **Our-Algo-NoiseAware** tracks below **Our-Algo** at high noise, with the gap most visible at **`n = 100`** (last row above).

![Section 3 convergence sigma=0.1](Extensions/Extension_Noise_Robustness_Full_info_feedback/plots/section3_convergence_medium_n100_sigma0p1.png)

![Section 3 convergence sigma=0.2](Extensions/Extension_Noise_Robustness_Full_info_feedback/plots/section3_convergence_medium_n100_sigma0p2.png)

![Section 3 convergence sigma=0.3](Extensions/Extension_Noise_Robustness_Full_info_feedback/plots/section3_convergence_medium_n100_sigma0p3.png)

## Section 4 — bandit noise (UCB, EXP3, OurAlg)

We compare only the **original** bandit **OurAlg** from the paper with **UCB** and **EXP3**. Gaussian noise at the observed cell changes regret ordering relative to the noiseless Figure 2 experiment; **OurAlg** stays strongest **against Adversary 3**, where UCB and EXP3 accumulate large regret under noisy feedback.

### Final Nash regret at `sigma = 0.3` (`medium`, seed 42)

| Adversary | UCB regret | EXP3 regret | OurAlg regret |
|---:|---:|---:|---:|
| 1 | 8.27 | 142.42 | 73.58 |
| 2 | 58.50 | 139.81 | 89.23 |
| 3 | 773.81 | 1265.62 | 2.81 |

### Convergence

Three adversaries per figure (panels); one plot file per noise level.

![Section 4 convergence sigma=0.1](Extensions/Extension_Noise_Robustness_Bandit_feedback/plots/section4_convergence_medium_sigma0p1.png)

![Section 4 convergence sigma=0.2](Extensions/Extension_Noise_Robustness_Bandit_feedback/plots/section4_convergence_medium_sigma0p2.png)

![Section 4 convergence sigma=0.3](Extensions/Extension_Noise_Robustness_Bandit_feedback/plots/section4_convergence_medium_sigma0p3.png)

On Adversary 3, regret jumps near **`T / 2`** when the column player switches phase (same protocol as the base Section 4 experiment); OurAlg remains comparatively stable after the switch.

## Extension conclusion

- **Section 3:** A noise-aware **exploration delay** improves Our-Algo under noisy full-information feedback, especially at larger `n` (see ~8% regret reduction at `n = 100`, `sigma = 0.3`).
- **Section 4:** The paper’s bandit OurAlg already uses optimism/confidence in the design; we report **UCB**, **EXP3**, and **OurAlg** under Gaussian bandit noise across **`sigma`**, with convergence curves showing how regret accumulates over time for each adversary.
