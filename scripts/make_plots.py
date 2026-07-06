"""Regenerate the three paper figures FROM SAVED results.

Reads the JSON files written by `scripts/run_experiments.py` and produces:
  - passive_active_accuracy.pdf   (Fig 1)
  - identification_error_log.pdf  (Fig 2)
  - probe_commit_tradeoff.pdf     (Fig 3)

Usage:
    python scripts/run_experiments.py     # first, to produce results/*.json
    python scripts/make_plots.py          # then, to render the PDFs

By default it writes the PDFs into --outdir (the repo root) so they line up with the
committed reference figures. Pass --outdir to write elsewhere and avoid overwriting them.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def _load(results_dir: str, name: str) -> dict:
    path = os.path.join(results_dir, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/run_experiments.py` first to generate results."
        )
    with open(path) as f:
        return json.load(f)


def plot_passive_active(results_dir: str, out_path: str) -> None:
    data = _load(results_dir, "fig1_passive_active")
    x = np.array(data["budgets"])

    plt.figure(figsize=(6.6, 4.0))
    for name, series in data["curves"].items():
        plt.errorbar(
            x, series["mean"], yerr=series["std"],
            marker="o", capsize=3, elinewidth=1, label=name,
        )
    plt.xlabel("Probe budget $n$")
    plt.ylabel("Accuracy on recombination queries $\\mathcal{Q}$")
    plt.ylim(0.45, 0.98)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True, fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_identification_error(results_dir: str, out_path: str) -> None:
    data = _load(results_dir, "fig2_identification")
    x = np.array(data["budgets"], dtype=float)

    plt.figure(figsize=(6.6, 4.0))
    for name, series in data["curves"].items():
        y = np.array(series["mean"], dtype=float)
        plt.errorbar(x, np.maximum(y, 1e-6), yerr=series["std"],
                     marker="o", capsize=3, elinewidth=1, label=name)

        # Empirical exponential guide fit on n >= 1: log y ~ a - D_hat * n
        mask = (x >= 1) & (y > 0)
        if mask.sum() >= 2:
            D_hat = -np.polyfit(x[mask], np.log(y[mask]), 1)[0]
            guide = y[mask][0] * np.exp(-D_hat * (x - x[mask][0]))
            plt.plot(x, guide, linestyle="--", linewidth=1,
                     label=f"Exp. guide ({name}), $\\hat D\\approx{D_hat:.2f}$")

    plt.yscale("log")
    plt.xlabel("Probe budget $n$")
    plt.xticks(data["budgets"])
    plt.ylabel("Identification error $\\Pr(\\hat M \\neq M)$ (log scale)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_probe_commit(results_dir: str, out_path: str) -> None:
    data = _load(results_dir, "fig3_probe_commit")
    x = np.array(data["budgets"])

    plt.figure(figsize=(6.6, 4.0))
    for name, series in data["curves"].items():
        y = np.array(series["mean"])
        plt.errorbar(x, y, yerr=series["std"], marker="o", capsize=3, elinewidth=1,
                     label=f"$C_{{commit}}={name.split('=')[-1]}$")
        n_hat = x[int(np.argmin(y))]
        plt.axvline(n_hat, linestyle=":", linewidth=1)

    plt.xlabel("Probe budget $n$")
    plt.ylabel("Expected total cost")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True, fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Render figures from saved results.")
    ap.add_argument("--results-dir", default="results", help="Where run_experiments.py wrote JSON.")
    ap.add_argument("--outdir", default=".", help="Where to write the PDFs (default: repo root).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plot_passive_active(args.results_dir, os.path.join(args.outdir, "passive_active_accuracy.pdf"))
    plot_identification_error(args.results_dir, os.path.join(args.outdir, "identification_error_log.pdf"))
    plot_probe_commit(args.results_dir, os.path.join(args.outdir, "probe_commit_tradeoff.pdf"))
    print("Saved: passive_active_accuracy.pdf, identification_error_log.pdf, probe_commit_tradeoff.pdf")


if __name__ == "__main__":
    main()
