from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Agent
from ..env import ToolRoleEnv
from ..data import OfflineDataset, estimate_g_from_diagonal
from ..probing import random_probe_pairs, run_probes
from ..utils import chernoff_info_bernoulli


class ProbeLRAgent(Agent):
    """
    Probe+LR:
      - estimate g_hat from diagonal executed samples
      - probe n times with do(execute) on chosen (u,v)
      - compute log-likelihood ratio for M1 vs M2 using g_hat and alpha
      - output hat_model and act optimally for hat_model
    """
    def __init__(
        self,
        env: ToolRoleEnv,
        ds: OfflineDataset,
        n_probes: int,
        rng: np.random.Generator,
        use_oed: bool = False,
        k_candidates: int = 256,
    ):
        self.env = env
        self.m = env.m
        self.alpha = env.alpha
        self.rng = rng
        self.n_probes = n_probes
        self.use_oed = use_oed
        self.k_candidates = k_candidates
        self.g_hat = estimate_g_from_diagonal(env.m, ds, alpha=env.alpha)
        self.selector = OEDProbeSelector(self.g_hat, env.alpha, rng) if use_oed else None
        self.hat_model: Optional[int] = None

    def _p_y1(self, u: int, v: int, model_id: int) -> float:
        decisive = u if model_id == 1 else v
        return float(self.alpha * self.g_hat[decisive] + (1.0 - self.alpha) * 0.5)

    def _select_probe_pairs(self) -> List[Tuple[int, int]]:
        if self.use_oed and self.selector is not None:
            return [self.selector.select(self.m, k_candidates=self.k_candidates) for _ in range(self.n_probes)]
        return random_probe_pairs(self.m, self.n_probes, self.rng)

    def infer_model(self, probe_pairs: Optional[List[Tuple[int, int]]] = None) -> int:
        if self.n_probes <= 0:
            self.hat_model = 1  # arbitrary under no probes
            return self.hat_model

        pairs = probe_pairs or self._select_probe_pairs()
        ys = run_probes(self.env, pairs)

        llr = 0.0
        eps = 1e-12
        for (u, v), y in zip(pairs, ys):
            p1 = np.clip(self._p_y1(u, v, 1), eps, 1 - eps)
            p2 = np.clip(self._p_y1(u, v, 2), eps, 1 - eps)
            llr += y * np.log(p1 / p2) + (1 - y) * np.log((1 - p1) / (1 - p2))

        self.hat_model = 1 if llr >= 0 else 2
        return self.hat_model

    def act(self, u: int, v: int) -> str:
        if self.hat_model is None:
            self.infer_model()
        decisive = u if self.hat_model == 1 else v
        return "execute" if self.g_hat[decisive] == 1 else "reject"


class OEDProbeSelector:
    """
    Simple OED-style greedy probe selector:
    choose (u,v) from candidate pool maximizing estimated Chernoff information
    between Bernoulli(p1) and Bernoulli(p2) given g_hat.
    """
    def __init__(self, g_hat: np.ndarray, alpha: float, rng: np.random.Generator):
        self.g_hat = g_hat
        self.alpha = alpha
        self.rng = rng

    def score_pair(self, u: int, v: int) -> float:
        p1 = float(self.alpha * self.g_hat[u] + (1.0 - self.alpha) * 0.5)
        p2 = float(self.alpha * self.g_hat[v] + (1.0 - self.alpha) * 0.5)
        return chernoff_info_bernoulli(p1, p2)

    def select(self, m: int, k_candidates: int = 256) -> Tuple[int, int]:
        best = None
        best_score = -1.0
        for _ in range(k_candidates):
            u = int(self.rng.integers(0, m))
            v = int(self.rng.integers(0, m))
            if u == v:
                continue
            sc = self.score_pair(u, v)
            if sc > best_score:
                best_score = sc
                best = (u, v)
        if best is None:
            # fallback
            u = int(self.rng.integers(0, m))
            v = int((u + 1) % m)
            best = (u, v)
        return best
