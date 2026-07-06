"""Run the tool-role benchmark experiments and SAVE results to disk.

This reproduces the data behind the three paper figures:
  - Fig 1: passive-active separation (accuracy on recombination queries Q vs probe budget n)
  - Fig 2: exponential role identification (Pr(hat M != M) vs n, random vs OED probes)
  - Fig 3: probe-commit tradeoff (expected total cost vs n for several commit costs)

Results are written under --outdir (default: results/) as JSON (one file per figure)
plus a flat summary.csv. The plotting script `scripts/make_plots.py` reads these files
and regenerates the PDFs, so the whole pipeline is reproducible end-to-end:

    python scripts/run_experiments.py            # writes results/*.json, results/summary.csv
    python scripts/make_plots.py                 # writes the three *.pdf figures

Nothing here reads or overwrites the committed PDFs; those are kept as our reference results.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

# Allow `python scripts/run_experiments.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import trange

from tool_roles.env import ToolRoleEnv
from tool_roles.logging_policy import diagonal_logging_policy
from tool_roles.data import generate_offline_dataset
from tool_roles.metrics import make_query_set, separating_mask, accuracy_on_queries, id_error_rate
from tool_roles.agents import (
    BehaviorCloningAgent,
    OfflineRLAgent,
    ERMPredictorAgent,
    ProbeLRAgent,
)


@dataclass
class Config:
    # environment / generator (Appendix B defaults)
    m: int = 100
    p: float = 0.5
    alpha: float = 0.9
    N: int = 50_000
    # evaluation
    max_query_pairs: int = 2_000
    # sweeps (match the figures)
    budgets: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 8, 10, 12])
    commit_costs: List[float] = field(default_factory=lambda: [0.0, 10.0, 50.0, 200.0])
    seeds: int = 20
    # probe-commit cost model
    C_probe: float = 1.0
    L: float = 100.0
    k_candidates: int = 256

    def to_dict(self) -> Dict:
        return {
            "m": self.m, "p": self.p, "alpha": self.alpha, "N": self.N,
            "max_query_pairs": self.max_query_pairs, "budgets": self.budgets,
            "commit_costs": self.commit_costs, "seeds": self.seeds,
            "C_probe": self.C_probe, "L": self.L, "k_candidates": self.k_candidates,
        }


def run_seed(seed: int, cfg: Config) -> Dict:
    """Run every agent/sweep for one seed. Returns per-seed raw numbers."""
    rng = np.random.default_rng(seed)

    env = ToolRoleEnv(m=cfg.m, p=cfg.p, alpha=cfg.alpha, rng=rng)
    true_model = env.model_id

    pi0 = diagonal_logging_policy()
    ds = generate_offline_dataset(env, pi0, N=cfg.N)

    # Recombination queries Q = {(u,v): u != v}; Q_sep is the decision-relevant subset.
    Q = make_query_set(cfg.m, max_pairs=cfg.max_query_pairs, rng=rng)
    Q_sep = Q[separating_mask(Q, env.g)]

    # --- Passive baselines (constant in n) ---
    bc = BehaviorCloningAgent(cfg.m).fit(ds)
    offrl = OfflineRLAgent(cfg.m).fit(ds)
    erm = ERMPredictorAgent(cfg.m).fit(ds)

    passive = {
        "BC": accuracy_on_queries(env, bc, Q, model_id=true_model),
        "OfflineRL": accuracy_on_queries(env, offrl, Q, model_id=true_model),
        "DiagonalERM": accuracy_on_queries(env, erm, Q, model_id=true_model),
    }

    # --- Probe+LR sweeps (random and OED probe selection) ---
    acc_rand, acc_oed = [], []
    iderr_rand, iderr_oed = [], []
    for n in cfg.budgets:
        pr = ProbeLRAgent(env, ds, n_probes=n, rng=rng, use_oed=False)
        hr = pr.infer_model()
        acc_rand.append(accuracy_on_queries(env, pr, Q, model_id=true_model))
        iderr_rand.append(id_error_rate(true_model, hr))

        po = ProbeLRAgent(env, ds, n_probes=n, rng=rng, use_oed=True, k_candidates=cfg.k_candidates)
        ho = po.infer_model()
        acc_oed.append(accuracy_on_queries(env, po, Q, model_id=true_model))
        iderr_oed.append(id_error_rate(true_model, ho))

    return {
        "seed": seed,
        "true_model": true_model,
        "passive": passive,
        "acc_rand": acc_rand,
        "acc_oed": acc_oed,
        "iderr_rand": iderr_rand,
        "iderr_oed": iderr_oed,
    }


def _mean_std(rows: np.ndarray):
    return np.mean(rows, axis=0).tolist(), np.std(rows, axis=0).tolist()


def aggregate(all_seeds: List[Dict], cfg: Config) -> Dict[str, Dict]:
    budgets = cfg.budgets
    n_seeds = len(all_seeds)

    # ---- Fig 1: passive-active accuracy on Q ----
    passive_methods = ["BC", "OfflineRL", "DiagonalERM"]
    fig1 = {"budgets": budgets, "config": cfg.to_dict(), "curves": {}}
    for meth in passive_methods:
        val = float(np.mean([r["passive"][meth] for r in all_seeds]))
        std = float(np.std([r["passive"][meth] for r in all_seeds]))
        # passive is constant in n -> broadcast across budgets for a flat curve
        fig1["curves"][meth] = {
            "mean": [val] * len(budgets),
            "std": [std] * len(budgets),
        }
    for key, meth in [("acc_rand", "Probe+LR (random)"), ("acc_oed", "Probe+LR (OED)")]:
        mean, std = _mean_std(np.array([r[key] for r in all_seeds]))
        fig1["curves"][meth] = {"mean": mean, "std": std}

    # ---- Fig 2: identification error ----
    fig2 = {"budgets": budgets, "config": cfg.to_dict(), "curves": {}}
    for key, meth in [("iderr_rand", "Random probes"), ("iderr_oed", "OED probes")]:
        mean, std = _mean_std(np.array([r[key] for r in all_seeds]))
        fig2["curves"][meth] = {"mean": mean, "std": std}

    # ---- Fig 3: probe-commit tradeoff ----
    # Cost(n) = n*C_probe + C_commit + (L + C_commit) * 1{misidentified}
    # (matches E[Cost(n)] <= n C_probe + C_commit + (L + C_commit) exp(-n D_ch) in the paper)
    # We use the OED identification-error indicator, averaged over seeds.
    iderr_oed = np.array([r["iderr_oed"] for r in all_seeds])  # (seeds, budgets), 0/1
    fig3 = {"budgets": budgets, "config": cfg.to_dict(), "curves": {}}
    for cc in cfg.commit_costs:
        per_seed_cost = (
            np.array(budgets)[None, :] * cfg.C_probe
            + cc
            + (cfg.L + cc) * iderr_oed
        )  # (seeds, budgets)
        mean = per_seed_cost.mean(axis=0).tolist()
        std = per_seed_cost.std(axis=0).tolist()
        n_star = int(budgets[int(np.argmin(mean))])
        fig3["curves"][f"C_commit={int(cc)}"] = {"mean": mean, "std": std, "argmin_n": n_star}

    return {
        "fig1_passive_active": fig1,
        "fig2_identification": fig2,
        "fig3_probe_commit": fig3,
    }


def write_outputs(results: Dict[str, Dict], cfg: Config, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    for name, payload in results.items():
        with open(os.path.join(outdir, f"{name}.json"), "w") as f:
            json.dump(payload, f, indent=2)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Flat CSV summary for quick inspection / spreadsheets.
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "curve", "n", "mean", "std"])
        for fig_name, payload in results.items():
            budgets = payload["budgets"]
            for curve, series in payload["curves"].items():
                for n, mean, std in zip(budgets, series["mean"], series["std"]):
                    w.writerow([fig_name, curve, n, f"{mean:.6f}", f"{std:.6f}"])

    print(f"Saved results to: {os.path.abspath(outdir)}")
    print("  - fig1_passive_active.json, fig2_identification.json, fig3_probe_commit.json")
    print("  - config.json, summary.csv")


def main():
    ap = argparse.ArgumentParser(description="Run and save tool-role benchmark experiments.")
    ap.add_argument("--outdir", default="results", help="Directory to save results (default: results/).")
    ap.add_argument("--seeds", type=int, default=None, help="Number of seeds (default: 20).")
    ap.add_argument("--m", type=int, default=None, help="Universe size (default: 100).")
    ap.add_argument("--N", type=int, default=None, help="Offline log size (default: 50000).")
    ap.add_argument("--max-query-pairs", type=int, default=None, help="Subsample size for Q (default: 2000).")
    ap.add_argument("--quick", action="store_true", help="Small/fast config for a smoke test.")
    args = ap.parse_args()

    cfg = Config()
    if args.quick:
        cfg = Config(m=40, N=5_000, max_query_pairs=500, seeds=5)
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if args.m is not None:
        cfg.m = args.m
    if args.N is not None:
        cfg.N = args.N
    if args.max_query_pairs is not None:
        cfg.max_query_pairs = args.max_query_pairs

    print("Config:", cfg.to_dict())
    all_seeds = [run_seed(s, cfg) for s in trange(cfg.seeds, desc="seeds")]

    results = aggregate(all_seeds, cfg)
    write_outputs(results, cfg, args.outdir)

    # Quick console summary.
    f1 = results["fig1_passive_active"]["curves"]
    print("\n=== Accuracy on Q at max probe budget ===")
    for meth, series in f1.items():
        print(f"  {meth:>22s}: {series['mean'][-1]:.3f}")


if __name__ == "__main__":
    main()
