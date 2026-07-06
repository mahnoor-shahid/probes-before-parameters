from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
from .base import Agent
from ..env import ToolRoleEnv
from ..data import OfflineDataset, estimate_g_from_diagonal
from ..probing import run_probes
from ..utils import chernoff_info_bernoulli
from .probe_lr import OEDProbeSelector


class ProbeCommitAgent(Agent):
    """
    Implements the probe/commit tradeoff experiment:
      - choose n via theorem-inspired closed form
      - probe n times
      - infer model via LR (same as ProbeLR logic but embedded)
      - optionally "commit" (adds cost externally in eval)
    """
    def __init__(
        self,
        env: ToolRoleEnv,
        ds: OfflineDataset,
        C_probe: float,
        C_commit: float,
        L: float,
        rng: np.random.Generator,
        use_oed: bool = True,
        k_candidates: int = 256,
    ):
        self.env = env
        self.m = env.m
        self.alpha = env.alpha
        self.C_probe = float(C_probe)
        self.C_commit = float(C_commit)
        self.L = float(L)
        self.rng = rng
        self.use_oed = use_oed
        self.k_candidates = k_candidates

        self.g_hat = estimate_g_from_diagonal(env.m, ds, alpha=env.alpha)
        self.selector = OEDProbeSelector(self.g_hat, env.alpha, rng)
        self.hat_model: Optional[int] = None
        self.n_star: int = 0

        # pick n* (approximate D_ch by averaging over some candidate separating-like pairs)
        self.n_star = self._choose_n_star()

    def _approx_Dch(self, trials: int = 256) -> float:
        # approximate a typical discriminability between M1 and M2
        vals = []
        for _ in range(trials):
            u = int(self.rng.integers(0, self.m))
            v = int(self.rng.integers(0, self.m))
            if u == v:
                continue
            p1 = float(self.alpha * self.g_hat[u] + (1.0 - self.alpha) * 0.5)
            p2 = float(self.alpha * self.g_hat[v] + (1.0 - self.alpha) * 0.5)
            vals.append(chernoff_info_bernoulli(p1, p2))
        return float(np.mean(vals)) if vals else 1e-6

    def _choose_n_star(self) -> int:
        D = max(self._approx_Dch(), 1e-9)
        # theorem: n* = ceil( (1/D) * log( L D / C_probe ) )_+
        arg = (self.L * D) / max(self.C_probe, 1e-12)
        if arg <= 1.0:
            return 0
        n_cont = (1.0 / D) * np.log(arg)
        return int(max(0, np.ceil(n_cont)))

    def _pick_probe_pairs(self, n: int) -> List[Tuple[int, int]]:
        pairs = []
        for _ in range(n):
            if self.use_oed:
                pairs.append(self.selector.select(self.m, k_candidates=self.k_candidates))
            else:
                # random
                u = int(self.rng.integers(0, self.m))
                v = int(self.rng.integers(0, self.m))
                while v == u:
                    v = int(self.rng.integers(0, self.m))
                pairs.append((u, v))
        return pairs

    def infer_model(self) -> int:
        n = self.n_star
        if n <= 0:
            self.hat_model = 1
            return self.hat_model

        pairs = self._pick_probe_pairs(n)
        ys = run_probes(self.env, pairs)

        llr = 0.0
        eps = 1e-12
        for (u, v), y in zip(pairs, ys):
            p1 = np.clip(self.alpha * self.g_hat[u] + (1.0 - self.alpha) * 0.5, eps, 1 - eps)
            p2 = np.clip(self.alpha * self.g_hat[v] + (1.0 - self.alpha) * 0.5, eps, 1 - eps)
            llr += y * np.log(p1 / p2) + (1 - y) * np.log((1 - p1) / (1 - p2))

        self.hat_model = 1 if llr >= 0 else 2
        return self.hat_model

    def act(self, u: int, v: int) -> str:
        if self.hat_model is None:
            self.infer_model()
        decisive = u if self.hat_model == 1 else v
        return "execute" if self.g_hat[decisive] == 1 else "reject"
