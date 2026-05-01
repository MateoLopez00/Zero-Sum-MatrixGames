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

At **n=10**, Our-Algo grows much slower than Nash and Hedge.

![Section 3 paper-lite n=10](Full_information_feedback/plots/section3_official_style_paper-lite_n10.png)

At **n=20**, the same qualitative behavior holds: the proposed method has a flatter regret curve than the baselines.

![Section 3 paper-lite n=20](Full_information_feedback/plots/section3_official_style_paper-lite_n20.png)

At **n=50**, the proposed method continues to outperform the baselines across horizons.

![Section 3 paper-lite n=50](Full_information_feedback/plots/section3_official_style_paper-lite_n50.png)

At **n=100**, the gap vs Nash/Hedge is still visible under the same experimental protocol.

![Section 3 paper-lite n=100](Full_information_feedback/plots/section3_official_style_paper-lite_n100.png)

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

## Motivation

The original reproduction studies the paper's algorithms under the paper's feedback models. This extension asks a complementary question: if the feedback becomes noisier, can the proposed method be made explicitly noise-aware?

The experiment varies a Gaussian noise parameter `sigma` and measures both total Nash regret and average row payoff. The noisy feedback is generated as `clip(A + sigma * N(0,1), 0, 1)`. Section 3 uses full-information feedback, so the learner observes a full noisy matrix signal each round. Section 4 uses bandit feedback, so the learner observes only the noisy reward at the played cell.

The extension is implemented in:

- `Extensions/Extension_Noise_Robustness_Full_info_feedback/`
- `Extensions/Extension_Noise_Robustness_Bandit_feedback/`

The plots below were generated from:

- `Extensions/Extension_Noise_Robustness_Full_info_feedback/section3_noise_robustness.ipynb` with the `medium` preset.
- `Extensions/Extension_Noise_Robustness_Bandit_feedback/section4_noise_robustness.ipynb` with the `paper-lite` preset.

## Algorithmic change

The baselines are kept unchanged. The extension only modifies the paper's proposed algorithm, adding a second variant that is compared directly against the original method.

In Section 3, the original Our-Algo uses the update threshold:

```python
threshold = min(log(T)**2, sqrt(T))
```

The noise-aware version uses:

```python
threshold = min((1 + sigma**2) * log(T)**2, sqrt(T))
```

The intuition is that, under noisier full-information feedback, the algorithm should wait slightly longer before trusting the empirical matrix estimate.

In Section 4, the original bandit algorithm uses the confidence radius:

```python
devs = sqrt(log_c / (count + 1))
```

The noise-aware version uses:

```python
devs = (1 + sigma**2) * sqrt(log_c / (count + 1))
```

Here the idea is to widen the confidence radius mildly when feedback noise increases. The factor uses `sigma**2` rather than `sigma` because Gaussian noise variance scales with `sigma**2`; this keeps the modification conservative and avoids over-widening the confidence set.

## Section 3 results

<table>
  <tr>
    <td width="50%"><img src="Extensions/Extension_Noise_Robustness_Full_info_feedback/plots/section3_noise_regret_medium.png" width="100%"></td>
    <td width="50%"><img src="Extensions/Extension_Noise_Robustness_Full_info_feedback/plots/section3_noise_payoff_medium.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Section 3: Nash regret vs noise</b></td>
    <td align="center"><b>Section 3: Average payoff vs noise</b></td>
  </tr>
</table>

For Section 3, the noise-aware variant improves the original Our-Algo consistently at the highest tested noise level, `sigma = 0.3`:

- `n = 10`: regret improves from `5.96` to `5.84`.
- `n = 20`: regret improves from `4.91` to `4.82`.
- `n = 50`: regret improves from `4.07` to `4.02`.
- `n = 100`: regret improves from `3.85` to `3.82`.

The improvement is numerically small, but it appears consistently across all tested matrix sizes. This supports the interpretation that, in the full-information setting, slightly delaying updates under higher noise makes the empirical matrix estimates more reliable. The average-payoff plot is less sensitive because payoffs stay close to the game value, but it shows that the noise-aware change does not damage payoff performance.

## Section 4 results

<p align="center"><b>Section 4: Nash regret vs noise</b></p>

![Section 4 noise regret](Extensions/Extension_Noise_Robustness_Bandit_feedback/plots/section4_noise_regret_paper-lite.png)

<p align="center"><b>Section 4: Average payoff vs noise</b></p>

![Section 4 noise payoff](Extensions/Extension_Noise_Robustness_Bandit_feedback/plots/section4_noise_payoff_paper-lite.png)

For Section 4, the same noise-aware idea does not improve the original OurAlg overall. At `sigma = 0.3`, the comparison is:

- **Adversary 1:** original OurAlg has regret `77.90`, while OurAlg-NoiseAware has regret `84.64`.
- **Adversary 2:** original OurAlg has regret `86.07`, while OurAlg-NoiseAware has regret `92.29`.
- **Adversary 3:** both versions are essentially identical, with regret `2.86`.

This is still informative. In the bandit setting, the original algorithm already contains uncertainty handling through optimistic estimates and confidence radii. Increasing the radius further makes the algorithm slightly too conservative against Adversaries 1 and 2, where it needs to react accurately to best-response-style behavior. Against Adversary 3, both versions remain very stable and stay close to the game value `V* = 2/9`.

## Extension conclusion

The extension shows two different outcomes from the same noise-aware idea:

- In Section 3, noise-aware adaptation improves the proposed method consistently under high Gaussian feedback noise.
- In Section 4, the original proposed method is already robust to bandit uncertainty, and additional confidence widening does not improve it.

This suggests that noise-aware tuning is useful in the full-information setting, while the bandit algorithm from the paper is already carefully calibrated for uncertainty.
