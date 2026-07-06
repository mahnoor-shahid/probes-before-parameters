from __future__ import annotations
import numpy as np
from .base import Agent
from ..data import OfflineDataset


class BehaviorCloningAgent(Agent):
    """
    Learns pi(a|u,v) from logged (u,v)->a using a simple frequency table.
    """
    def __init__(self, m: int):
        self.m = m
        self.counts = np.zeros((m, m, 2), dtype=float)  # action counts

    def fit(self, ds: OfflineDataset) -> "BehaviorCloningAgent":
        for uu, vv, aa in zip(ds.u, ds.v, ds.a):
            self.counts[uu, vv, aa] += 1.0
        return self

    def act(self, u: int, v: int) -> str:
        c = self.counts[u, v]
        # default to reject if unseen
        aa = int(np.argmax(c)) if c.sum() > 0 else 0
        return "execute" if aa == 1 else "reject"
