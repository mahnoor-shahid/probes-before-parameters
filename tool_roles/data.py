from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .env import ToolRoleEnv, Episode
from .logging_policy import LoggingPolicy


@dataclass
class OfflineDataset:
    """
    Stores offline episodes:
      (u, v, a, y, r, executed_flag)
    """
    u: np.ndarray
    v: np.ndarray
    a: np.ndarray          # 0 reject, 1 execute
    y: np.ndarray
    r: np.ndarray
    executed: np.ndarray   # bool

    def __len__(self) -> int:
        return int(self.u.shape[0])

    def executed_mask(self) -> np.ndarray:
        return self.executed.astype(bool)

    def as_dict(self) -> Dict[str, Any]:
        return {"u": self.u, "v": self.v, "a": self.a, "y": self.y, "r": self.r, "executed": self.executed}


def generate_offline_dataset(env: ToolRoleEnv, pi0: LoggingPolicy, N: int) -> OfflineDataset:
    u = np.zeros(N, dtype=int)
    v = np.zeros(N, dtype=int)
    a = np.zeros(N, dtype=int)
    y = np.zeros(N, dtype=int)
    r = np.zeros(N, dtype=float)
    executed = np.zeros(N, dtype=bool)

    for i in range(N):
        ep = env.sample_episode()
        u[i], v[i] = ep.u, ep.v
        act = pi0.act(ep.u, ep.v)
        executed[i] = (act == "execute")
        a[i] = 1 if executed[i] else 0
        out = env.step(ep, act)
        y[i] = int(out["y"])
        r[i] = float(out["r"])

    return OfflineDataset(u=u, v=v, a=a, y=y, r=r, executed=executed)


def estimate_g_from_diagonal(env_m: int, ds: OfflineDataset, alpha: float = 1.0) -> np.ndarray:
    """
    Estimate g(u) using only executed diagonal samples (u==v).
    For alpha=1 (deterministic), y reveals g(u) exactly when executed.
    For alpha<1, we estimate Bernoulli mean and threshold at 0.5.
    """
    g_hat = np.zeros(env_m, dtype=float)
    counts = np.zeros(env_m, dtype=int)

    mask = ds.executed_mask()
    # Under diagonal logging, executed implies u==v; but we don't assume it.
    for uu, vv, yy in zip(ds.u[mask], ds.v[mask], ds.y[mask]):
        if uu != vv:
            continue
        g_hat[uu] += yy
        counts[uu] += 1

    # Default for unseen u: 0.5 (unknown)
    out = np.full(env_m, 0.5, dtype=float)
    seen = counts > 0
    out[seen] = g_hat[seen] / counts[seen]

    # Convert to {0,1} estimate if needed by agents
    if alpha == 1.0:
        return (out >= 0.5).astype(int)
    return (out >= 0.5).astype(int)
