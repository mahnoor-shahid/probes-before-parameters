from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .env import ToolRoleEnv, Episode


def random_probe_pairs(m: int, n: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    pairs = []
    while len(pairs) < n:
        u = int(rng.integers(0, m))
        v = int(rng.integers(0, m))
        if u != v:
            pairs.append((u, v))
    return pairs


def run_probes(env: ToolRoleEnv, pairs: List[Tuple[int, int]]) -> List[int]:
    ys = []
    for (u, v) in pairs:
        ep = Episode(u=u, v=v)
        out = env.step(ep, "execute")  # do(execute)
        ys.append(int(out["y"]))
    return ys
