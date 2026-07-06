from __future__ import annotations
import numpy as np
from .base import Agent
from ..data import OfflineDataset


class OfflineRLAgent(Agent):
    """
    Fits Q(u,v,a) as empirical mean reward for each (u,v,a).
    """
    def __init__(self, m: int):
        self.m = m
        self.sum_r = np.zeros((m, m, 2), dtype=float)
        self.cnt = np.zeros((m, m, 2), dtype=float)

    def fit(self, ds: OfflineDataset) -> "OfflineRLAgent":
        for uu, vv, aa, rr in zip(ds.u, ds.v, ds.a, ds.r):
            self.sum_r[uu, vv, aa] += rr
            self.cnt[uu, vv, aa] += 1.0
        return self

    def act(self, u: int, v: int) -> str:
        q = np.zeros(2, dtype=float)
        for a in (0, 1):
            if self.cnt[u, v, a] > 0:
                q[a] = self.sum_r[u, v, a] / self.cnt[u, v, a]
            else:
                q[a] = 0.0
        aa = int(np.argmax(q))
        return "execute" if aa == 1 else "reject"
