# Zero-Sum Matrix Games (Seminar Reproduction)

This repository reproduces the experimental setup requested for:

- **Paper**: *On the Limitations and Possibilities of Nash Regret Minimization in Zero-Sum Matrix Games under Noisy Feedback* (arXiv:2306.13233v3)
- **Target sections**: **Section 3** (full-information) and **Section 4** (bandit feedback, 2x2)

## What is implemented

- Paper-locked experiment protocols (matrix/adversary/horizon grids).
- Section 3: Hedge, Empirical-Nash baseline, and Section-3-style algorithm.
- Section 4: UCB baseline, EXP3 baseline, and Section-4-style 2x2 algorithm.
- Nash regret evaluation against game value.
- Log-log plots saved as PNG files.

## Locked Experiment Specs (Paper-Aligned)

### Section 3

- Input matrices: 4 diagonal matrices of size `10x10`, `20x20`, `50x50`, `100x100`.
- Diagonal entries: `A[i,i] = 0.4 + 0.2*(i-1)/(n-1)`.
- Adversary: best-response at every step.
- Plot type: `log(total Nash regret)` vs `log(T)` for multiple horizons.

### Section 4

- Input matrix: `[[2/3, 0], [0, 1/3]]`.
- Adversaries:
  - `adv1`: equilibrium first half, best-response second half.
  - `adv2`: best-response throughout.
  - `adv3`: adaptive (equilibrium vs best-response) first half, best-response second half.
- Plot type: `log(total Nash regret)` vs `log(T)`.

## Main Scripts

- `experiments_section3.py`: Section 3 log-log plots.
- `experiments_section4.py`: Section 4 log-log plots.
- `reproduction.py`: runs Section 3 and Section 4 in sequence.

## Project Structure

- `core/`: shared helper utilities used by the experiment scripts.
- `core/utils.py`: helper utilities (seed setup, directory creation, small utility functions).
- `experiments/`: main experiment runners and plotting logic.
- `experiments/section3.py`: official-style Section 3 pipeline and plotting.
- `experiments/section4.py`: official-style Section 4 pipeline and plotting.
- `experiments_section3.py`: lightweight CLI wrapper for Section 3.
- `experiments_section4.py`: lightweight CLI wrapper for Section 4.
- `reproduction.py`: one-command runner for both sections.
- `plots/`: generated PNG figures from experiment runs.
- `requirements.txt`: Python dependency list for reproducible setup.
- `pyproject.toml`: project metadata and Python tooling configuration.

## Runtime Presets (Staged Plan)

Each script supports `--preset {quick|medium|paper-lite|final|paper}`:

- `quick`: fast sanity check
- `medium`: stronger validation
- `paper-lite`: paper-style setup with reduced compute (around 30-60 min on CPU, machine-dependent)
- `final`: paper-like horizon/trial settings (slow)
- `paper`: exact official full settings for true paper reproduction
  - Section 3: `T=10^1..10^7`, `runs=100`
  - Section 4: `T=10^1..10^8`, `runs=128`

## Simple Step-by-Step

1. Open a terminal in this project folder.
2. Install libraries:
   - `pip install -r requirements.txt`
3. Run a very fast check:
   - `python reproduction.py --preset quick`
4. Run a better check:
   - `python reproduction.py --preset medium`
5. Look at generated plots in:
   - `plots/`

If Python env is already active, just run steps 3 and 4.

## Run (workload from low-to-high because computing times can be extensive, first try with quick!)

```bash
python experiments_section3.py --preset quick
python experiments_section4.py --preset quick
```

Medium:

```bash
python experiments_section3.py --preset medium
python experiments_section4.py --preset medium
```

Paper-lite:

```bash
python experiments_section3.py --preset paper-lite
python experiments_section4.py --preset paper-lite
```

Final:

```bash
python experiments_section3.py --preset final
python experiments_section4.py --preset final
```

Paper (exact official full settings):

```bash
python experiments_section3.py --preset paper
python experiments_section4.py --preset paper
```

Run both sections:

```bash
python reproduction.py --preset quick
```

One-command paper-lite reproduction:

```bash
python reproduction.py --preset paper-lite
```

True one-command paper reproduction:

```bash
python reproduction.py --preset paper
```

## Output

PNG files are generated under `plots/`:

- `plots/section3_official_style_<preset>.png`
- `plots/section4_official_style_<preset>.png`

Plot titles include the active runtime config (`preset`, `runs`, `horizons`).

