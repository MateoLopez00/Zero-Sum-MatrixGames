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

