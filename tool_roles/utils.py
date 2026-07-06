from __future__ import annotations
import numpy as np


def bernoulli_kl(p: float, q: float, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1 - eps))
    q = float(np.clip(q, eps, 1 - eps))
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def chernoff_info_bernoulli(p: float, q: float, grid: int = 501, eps: float = 1e-12) -> float:
    """
    Chernoff information:
      D_ch = - min_{s in [0,1]} log( p^s q^(1-s) + (1-p)^s (1-q)^(1-s) )
    Approximate with a grid search for simplicity.
    """
    p = float(np.clip(p, eps, 1 - eps))
    q = float(np.clip(q, eps, 1 - eps))
    ss = np.linspace(0.0, 1.0, grid)
    vals = []
    for s in ss:
        mix = (p ** s) * (q ** (1 - s)) + ((1 - p) ** s) * ((1 - q) ** (1 - s))
        vals.append(-np.log(mix))
    return float(np.min(vals))
