# Reference results

These CSVs are the reference data behind the three paper figures** (the committed
`*.pdf` files at the repo root). Each value is aggregated over **S = 20 random seeds**;
we report the mean and the standard deviation across seeds. Every seed fixes its own
latent validity function `g`, ground-truth model `M ∈ {M1, M2}`, offline dataset sample,
and probe/query subsampling (see Appendix B of the paper).

Common column: `n` is the **probe budget** (number of test-time interventions
`do(execute)`), swept over `{0, 1, 2, 3, 4, 5, 6, 8, 10, 12}`.

To regenerate equivalent data yourself, run `python scripts/run_experiments.py` from the
repo root; it writes `fig*_.json` and `summary.csv` into this folder (those generated
files are git-ignored). Exact numbers will differ slightly from the frozen CSVs below
because of seed/sampling noise, but the trends are stable.

## `fig1_passive_active.csv` — Fig 1, passive–active separation

Accuracy on the recombination query set `Q = {(u, v) : u ≠ v}`.

| column | meaning |
|--------|---------|
| `n` | probe budget |
| `method` | `BC`, `OfflineRL`, `DiagonalERM`, `Probe+LR (random)`, `Probe+LR (OED)` |
| `accuracy_mean` | mean accuracy on `Q` over 20 seeds (0–1) |
| `accuracy_std` | standard deviation across seeds |

Passive baselines (`BC`, `OfflineRL`, `DiagonalERM`) are constant in `n` (they never
probe); the value is repeated across budgets to give a flat reference line.

## `fig2_identification.csv` — Fig 2, exponential role identification

Model-identification error `Pr(hat M ≠ M)` for probe-enabled agents.

| column | meaning |
|--------|---------|
| `n` | probe budget |
| `method` | `Random probes`, `OED probes` |
| `id_error_mean` | mean identification error over 20 seeds (0–1) |
| `id_error_std` | standard deviation across seeds |

Decay is approximately `exp(-n · D_ch)` (Thm. 2.16); OED probing improves the constant.

## `fig3_probe_commit.csv` — Fig 3, probe–commit tradeoff

Expected total cost `Cost(n) = n·C_probe + C_commit + (L + C_commit)·1{misidentified}`,
with per-probe cost `C_probe = 1` and misidentification loss `L = 100`.

| column | meaning |
|--------|---------|
| `n` | probe budget |
| `C_commit` | commit (persistent-write) penalty; one of `{0, 10, 50, 200}` |
| `total_cost_mean` | mean total cost over 20 seeds |
| `total_cost_std` | standard deviation across seeds |

As `C_commit` grows (more "rigid" substrate), the cost-minimizing budget `n*` shifts
upward, matching Thm. 2.18.
