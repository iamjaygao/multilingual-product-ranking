# US-locale paired bootstrap — V2 Cross-Encoder vs reproduced Amazon official baseline

Artifacts: [`us_slice_bootstrap.json`](us_slice_bootstrap.json),
[`run_us_slice_bootstrap.py`](run_us_slice_bootstrap.py)

## Why US only

The reproduced official baseline is not one model across locales: `us` is a
CrossEncoder on (query, title), `es`/`jp` are bi-encoders scoring cosine
similarity. The US slice is the only like-for-like architecture comparison.

## Method

- **Pool:** competition-aligned frozen DEV (`experiments/ranking_v2/kdd_task1_benchmark/dev_task1.parquet`,
  118,231 rows / 5,071 queries), restricted to `product_locale == "us"`.
  **TEST was never read.**
- **No model was retrained or re-run.** Both score columns were read from the
  frozen prediction parquets produced by the global run:
  - official baseline: `experiments/competition_alignment/official_baseline_dev_predictions.parquet`
  - V2: `experiments/ranking_v2/cross_encoder_task1/full_run/fulldev_epoch2_predictions.parquet`
- **Scorer:** the authoritative Task 1 scorer,
  `experiments/ranking_v2/kdd_task1_benchmark/scripts/kdd_task1_ndcg.py`
  (loaded via `reranking.cross_encoder.load_task1_scorer`). No new NDCG code.
  `metric_version = kdd_task1_ndcg@6946489c8c09dc37|gain=competition_E1_S0.1_C0.01_I0|idcg0=exclude|tie=deterministic`
  (official gain E=1.0 / S=0.1 / C=0.01 / I=0.0).
- **Slice construction:** per-query NDCG tables are computed over the full DEV
  exactly as in the global run's `report()`, queries with `idcg_full == 0` are
  dropped, then the US queries are selected. This is the same code path that
  produced the published per-locale numbers.

### Bootstrap configuration

Read from and copied verbatim from **`scripts/ca_eval_official_baseline.py:110-115`**
(the global +0.0385 run). No parameter was chosen independently.

| Parameter | Value |
|---|---|
| Resamples | 10,000 |
| RNG | `numpy.random.RandomState(42)` (seed **42**) |
| Pairing | by `query_id` — resample the vector of per-query Δ with replacement, `n = len(d)` |
| Statistic | mean of resampled per-query Δ |
| CI | percentile method, `[2.5, 97.5]` |
| p-value | `min(1, 2 · min(P(bs ≤ 0), P(bs ≥ 0)))` (two-sided) |
| W/L/T tie tolerance | `|Δ| ≤ 1e-12` |

## Sanity check — PASSED

Recomputed US NDCG@20 point estimates before running the bootstrap:

| Model | Recomputed | Reference (global run) | abs diff |
|---|---|---|---|
| Amazon official baseline | 0.834539 | 0.834539 | 2.1e-07 |
| V2 Cross-Encoder (epoch 2) | 0.855827 | 0.855827 | 2.9e-07 |

Both within the 5e-06 tolerance (residual is JSON rounding of the reference to
6 dp). Bootstrap proceeded.

## Results — US slice, n = 3,133 queries

### NDCG@20 (with CI)

| Quantity | Value |
|---|---|
| Official baseline | 0.834539 |
| V2 Cross-Encoder | 0.855827 |
| **Δ (V2 − official)** | **+0.021288** |
| **95% CI** | **[+0.017255, +0.025260]** |
| **p-value** | **< 0.0001** (0 of 10,000 resamples at or below 0) |
| **W / L / T** | **1852 / 1160 / 121** |

### Does the CI cross 0?

**No.** The 95% CI is `[+0.017255, +0.025260]`; the entire interval is strictly
positive. The lower bound is +0.0173, i.e. `ci_crosses_zero = false`.

### Point estimates at other cutoffs (no CI computed)

| Cutoff | Official | V2 | Δ |
|---|---|---|---|
| full-list | 0.859064 | 0.878200 | **+0.019136** |
| @20 | 0.834539 | 0.855827 | +0.021288 |
| @10 | 0.758205 | 0.793178 | **+0.034973** |

## Constraints honored

- No model retrained or re-scored — scores read from frozen parquets.
- `split == "test"` never read; no code path in this run touches TEST.
- No existing file modified; all output under
  `experiments/competition_alignment/us_slice_bootstrap/`.
- Existing authoritative scorer reused; no second NDCG implementation.
- Bootstrap config copied from the global run, not re-chosen.
- Seed 42, recorded in the JSON.
