from __future__ import annotations
import numpy as np
from .base import Agent
from ..data import OfflineDataset


class ERMPredictorAgent(Agent):
    """
    Trains a predictor yhat(u,v) using executed samples only.
    In diagonal logging, executed mostly means u==v, so it effectively learns g on the diagonal.
    """
    def __init__(self, m: int):
        self.m = m
        self.mean_y = np.full((m, m), 0.5, dtype=float)
        self.counts = np.zeros((m, m), dtype=float)

    def fit(self, ds: OfflineDataset) -> "ERMPredictorAgent":
        mask = ds.executed_mask()
        for uu, vv, yy in zip(ds.u[mask], ds.v[mask], ds.y[mask]):
            self.mean_y[uu, vv] = (self.mean_y[uu, vv] * self.counts[uu, vv] + yy) / (self.counts[uu, vv] + 1.0)
            self.counts[uu, vv] += 1.0
        return self

    def act(self, u: int, v: int) -> str:
        return "execute" if self.mean_y[u, v] > 0.5 else "reject"
