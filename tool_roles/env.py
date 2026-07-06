from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, Dict, Any


@dataclass(frozen=True)
class Episode:
    """One episode corresponds to one tool call with arguments (u, v)."""
    u: int
    v: int


class ToolRoleEnv:
    """
    H=2 episodic environment:
      t=1: observe (Apply, u, v), choose execute/reject
      if reject: Y=0, r=0
      if execute: environment reveals Y in {0,1}, r=Y

    Two models:
      M1: P(Y=1 | u,v, execute) = alpha*g(u) + (1-alpha)*0.5
      M2: P(Y=1 | u,v, execute) = alpha*g(v) + (1-alpha)*0.5
    """

    def __init__(self, m: int, p: float = 0.5, alpha: float = 1.0, rng: Optional[np.random.Generator] = None):
        if not (0 < m):
            raise ValueError("m must be positive.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0,1].")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0,1].")

        self.m = m
        self.p = p
        self.alpha = alpha
        self.rng = rng or np.random.default_rng()

        # latent property g: U -> {0,1}
        self.g = self.rng.binomial(1, p, size=m).astype(int)

        # model id: 1 or 2
        self.model_id = int(self.rng.integers(1, 3))

    def set_model(self, model_id: int) -> None:
        if model_id not in (1, 2):
            raise ValueError("model_id must be 1 or 2.")
        self.model_id = model_id

    def sample_episode(self) -> Episode:
        u = int(self.rng.integers(0, self.m))
        v = int(self.rng.integers(0, self.m))
        return Episode(u=u, v=v)

    def obs_t1(self, ep: Episode) -> Tuple[int, int]:
        # keep observation minimal; "Apply" token can be implicit
        return (ep.u, ep.v)

    def _p_y1(self, u: int, v: int, model_id: Optional[int] = None) -> float:
        mid = self.model_id if model_id is None else model_id
        decisive = u if mid == 1 else v
        return float(self.alpha * self.g[decisive] + (1.0 - self.alpha) * 0.5)

    def step(self, ep: Episode, action: str) -> Dict[str, Any]:
        """
        action: 'execute' or 'reject'
        Returns a dict containing:
          y, r, o2 (same as y), done=True
        """
        if action not in ("execute", "reject"):
            raise ValueError("action must be 'execute' or 'reject'.")

        if action == "reject":
            y = 0
            r = 0.0
            return {"y": y, "r": r, "o2": y, "done": True}

        p1 = self._p_y1(ep.u, ep.v)
        y = int(self.rng.random() < p1)
        r = float(y)
        return {"y": y, "r": r, "o2": y, "done": True}

    def oracle_opt_action(self, u: int, v: int, model_id: Optional[int] = None) -> str:
        """
        Optimal policy for the true model if g is known:
          execute iff decisive argument is valid (g=1).
        """
        mid = self.model_id if model_id is None else model_id
        decisive = u if mid == 1 else v
        return "execute" if self.g[decisive] == 1 else "reject"

    def separating_pairs(self) -> np.ndarray:
        """
        Return all separating pairs (u,v) with u!=v and g(u)!=g(v).
        Warning: O(m^2) memory; use only for small m.
        """
        pairs = []
        for u in range(self.m):
            for v in range(self.m):
                if u != v and self.g[u] != self.g[v]:
                    pairs.append((u, v))
        return np.array(pairs, dtype=int)
