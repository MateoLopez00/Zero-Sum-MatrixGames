from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int | None) -> np.random.Generator:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clipped_sqrt_confidence(t: int, scale: float = 2.0, log_factor: float = 1.0) -> float:
    t = max(1, t)
    return scale * math.sqrt(log_factor / t)
