from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from .env import ToolRoleEnv


def make_query_set(m: int, max_pairs: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Returns array of pairs (u,v) with u!=v. Optionally subsample.
    """
    pairs = [(u, v) for u in range(m) for v in range(m) if u != v]
    pairs = np.array(pairs, dtype=int)
    if max_pairs is None or max_pairs >= len(pairs):
        return pairs
    rng = rng or np.random.default_rng()
    idx = rng.choice(len(pairs), size=max_pairs, replace=False)
    return pairs[idx]


def separating_mask(pairs: np.ndarray, g: np.ndarray) -> np.ndarray:
    u = pairs[:, 0]
    v = pairs[:, 1]
    return (g[u] != g[v])


def accuracy_on_queries(env: ToolRoleEnv, agent, pairs: np.ndarray, model_id: int | None = None) -> float:
    correct = 0
    for (u, v) in pairs:
        act = agent.act(u, v)
        opt = env.oracle_opt_action(u, v, model_id=model_id)
        correct += int(act == opt)
    return correct / max(1, len(pairs))


def id_error_rate(true_model: int, hat_model: int) -> float:
    return float(hat_model != true_model)
