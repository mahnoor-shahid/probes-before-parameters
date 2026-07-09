# probes-before-parameters

Reference implementation and benchmark for the paper
**"Probes Before Parameters: Identifiability–Adaptation Tradeoffs for Compositional Generalization."**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-research%20code-orange)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21284812.svg)](https://doi.org/10.5281/zenodo.21284811)

**Topics:** `compositional-generalization` · `identifiability` · `causal-inference` ·
`interventions` · `optimal-experimental-design` · `active-learning` · `offline-rl` ·
`tool-use` · `probing` · `adaptation-cost`

A minimal, fully synthetic tool-semantics environment with **hidden argument roles**. Under a
conservative logging policy the two candidate models are observationally equivalent, so no
passive learner can recover the decisive role; a small number of targeted **probes**
(interventions) identifies it at an exponential rate, exposing a **probe–commit** cost tradeoff.

Implements:
- H=2 `execute`/`reject` tool call
- models `M1` (decisive role = `u`) vs `M2` (decisive role = `v`)
- diagonal logging `pi0`: execute only if `u == v` (yields observational equivalence of M1, M2)
- passive baselines (BC, Offline-RL, Diagonal-ERM)
- probe-based identification (Probe+LR, random and OED probe selection)
- the probe/commit cost tradeoff

## Install

```bash
pip install -r requirements.txt
```

## Reproduce the figures

The pipeline is two steps: run the experiments (which **compute and save** results to
`results/`), then render the figures from those saved results.

```bash
# 1) Run experiments and save results to results/*.json (+ summary.csv)
python scripts/run_experiments.py

# 2) Render the three figures from the saved results
python scripts/make_plots.py
```

This produces:
- `passive_active_accuracy.pdf` — Fig 1: passive baselines stay near chance while probing improves with `n`
- `identification_error_log.pdf` — Fig 2: identification error decays ~`exp(-n D_ch)`
- `probe_commit_tradeoff.pdf` — Fig 3: cost-minimizing probe budget shifts up with commit cost

Useful flags:
- `python scripts/run_experiments.py --quick` — small, fast smoke-test config
- `python scripts/run_experiments.py --seeds 20 --m 100 --N 50000` — override defaults
- `python scripts/run_experiments.py --outdir results_v2` then
  `python scripts/make_plots.py --results-dir results_v2 --outdir figs_v2`

The committed `*.pdf` files at the repo root are our reference results. Re-running the
pipeline regenerates equivalent figures; exact numbers vary slightly with seeds and sampling,
but the qualitative trends (passive–active separation, exponential identification, probe–commit
shift) are stable, as described in Appendix B.

## Layout

```
tool_roles/            # library
  env.py               # H=2 environment, M1/M2 outcome models, oracle optimal action
  logging_policy.py    # conservative diagonal logging policy pi0
  data.py              # offline dataset generation, diagonal g-estimation
  probing.py           # probe (do(execute)) sampling
  metrics.py           # query sets, separating mask, accuracy, id-error
  utils.py             # Bernoulli KL / Chernoff information
  agents/              # BC, Offline-RL, Diagonal-ERM, Probe+LR, Probe-Commit
scripts/
  run_experiments.py   # runs sweeps, SAVES results to results/
  make_plots.py        # loads results/, renders the three PDFs
```

## Citation

If you use this code or benchmark, please cite the paper:

```bibtex
@article{probesbeforeparameters2026,
  title   = {Probes Before Parameters: Identifiability--Adaptation Tradeoffs
             for Compositional Generalization},
  author  = {Mahnoor Shahid & Hannes Rothe},
  year    = {2026},
  note    = {Under review. Author and venue details omitted for anonymity.}
}
```

> **Note:** The paper is currently anonymized for review. Please update the `author`,
> `journal`/`booktitle`, and DOI fields once the camera-ready version is available.

## License

Released under the [MIT License](LICENSE).
