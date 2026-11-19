# Résumé ground truth

Every number below comes from a saved result file in this repo. A number that
appears only in README or report prose is listed as RETRACTED. All results are
**dev-split** results. The official test split has never been scored.

## 1. What the project is

A cross-encoder reranker (`FacebookAI/xlm-roberta-base`, 4-class ESCI head,
ranked by expected gain) trained on the Amazon ESCI Task 1 data (us/es/jp). It
is compared against Amazon's official baseline recipe, rebuilt here, using the
same pool, the same content-hashed NDCG scorer (gains E1 / S0.1 / C0.01 / I0)
and the same run. The repo also includes data-integrity, reproduction and
test-split-lock gates.

**Common setup for every NDCG row (call it "DEV"):** frozen dev split with 5,071
queries and 118,231 query–product pairs. It is a query-level 85/15 carve of
official Task 1 train (`small_version==1`), and its candidate pairs are the
unmodified official pairs. `candidate_pool=task1_small_v1`,
`metric_version=kdd_task1_ndcg@6946489c…`. Training used 663,407 pairs and
28,733 queries.

## 2. Results table

| value | metric + cutoff | population / setup | source file (+ key) | status |
|---|---|---|---|---|
| 0.845605 | NDCG@20 | V2 cross-encoder, DEV all locales (n=5,071) | `artifacts/results/phase4/official_baseline_dev_results.json` → `v2.ndcg_at_20`; also `artifacts/FROZEN_V2.json` → `dev_ndcg_at_20` | VERIFIED |
| 0.807127 | NDCG@20 | Official baseline (reproduced), DEV all locales (n=5,071) | `official_baseline_dev_results.json` → `official_baseline.ndcg_at_20` | VERIFIED |
| 0.883443 / 0.852066 | NDCG full-list | V2 / official baseline, DEV all locales | same file → `v2.ndcg_full`, `official_baseline.ndcg_full` | VERIFIED |
| 0.792383 / 0.735796 | NDCG@10 | V2 / official baseline, DEV all locales | same file → `v2.ndcg_at_10`, `official_baseline.ndcg_at_10` | VERIFIED |
| +0.038478, 95% CI [0.034833, 0.042133], W/L/T 3229/1696/146 | Δ NDCG@20 (V2 − official), paired bootstrap, 10,000 resamples, seed 42 | DEV all locales, n=5,071, paired by query_id | `official_baseline_dev_results.json` → `paired_bootstrap_v2_minus_official_at20` | VERIFIED |
| +0.021288, 95% CI [0.017255, 0.025260], W/L/T 1852/1160/121 | Δ NDCG@20 (V2 − official), paired bootstrap, 10,000 resamples, seed 42 | DEV **US only**, n=3,133 (only like-for-like branch: both are cross-encoders) | `artifacts/results/phase4/us_slice_bootstrap/us_slice_bootstrap.json` → `paired_bootstrap_at20` | VERIFIED |
| 0.855827 / 0.834539 | NDCG@20 | V2 / official, DEV US (n=3,133) | `us_slice_bootstrap.json` → `point_estimates.ndcg_at_20` | VERIFIED |
| 0.848554 / 0.782330 | NDCG@20 | V2 / official, DEV ES (n=845); official ES branch is an English-only bi-encoder | `official_baseline_dev_results.json` → `*.locale.es.ndcg_at_20` | VERIFIED |
| 0.814023 / 0.747723 | NDCG@20 | V2 / official, DEV JP (n=1,093); official JP branch is an English-only bi-encoder | `official_baseline_dev_results.json` → `*.locale.jp.ndcg_at_20` | VERIFIED |
| 118,231 / 118,231 rows equal, max abs diff 0.0 | prediction regression (regenerated vs. oracle) | both models, DEV | `artifacts/manifests/phase4_reproduction.json` → `oracle_comparison.*` | VERIFIED |
| 21/21 checks, bit-for-bit at 6 dp | frozen-checkpoint metric reproduction | V2 cross-encoder, DEV | `artifacts/manifests/phase3_ce_reproduction.json` → `verdict`, `checks_passed` | VERIFIED |
| 52/52 checks | split rebuild vs. recorded oracle | train/dev rebuilt from raw ESCI | `artifacts/manifests/split_integrity_verification.json` → `checks_passed` | VERIFIED |
| 781,638 rows, 0 label mismatches, 0 IDs on either side only | example_id set equality vs. official Task 1 train | current train+dev vs. official | `artifacts/results/phase4/current_vs_official_split_audit.json` → `identity_checks` | VERIFIED |
| 13/13 cases, max abs delta 1.1e-16 | scorer parity vs. independent pure-Python reference | synthetic cases + 300 random queries × {10, 20, full} | `artifacts/results/phase4/metric_parity_results.json` → `cases_passed`, `cases[-1]` | VERIFIED |
| 47/47 and 32/32 | scorer unit tests | `kdd_task1_ndcg.py` / `official_kdd_ndcg.py` | `artifacts/manifests/scorer_unit_tests.json`, `official_kdd_scorer_unit_tests.json` → `passed` | VERIFIED |
| 2 entries, both from pool construction | test-split access log | official test split | `artifacts/manifests/test_access_log.json` → `accesses` | VERIFIED |
| 0.019–0.030 NDCG | impact of the upstream S/C gain swap | course-project pool, 4 models (Random, BM25, Two-Tower, LambdaMART) | README L33; `docs/reports/metric_authority_report.md` L85-90 only | RETRACTED |
| Recall@100 0.4598 → 0.4649; RRF@100 0.5194 → 0.5301; MRR@10 0.4712 → 0.4702 | Two-Tower V0 vs. V1 retrieval | US-only, 5,000-query full catalog | `docs/reports/two_tower_v2/FINAL_REPORT.md` §3–4 only; cited `experiments/two_tower_v2/...json` is not in the repo | RETRACTED |
| 1,014.9 vs. 18.4 qps; HNSW 3254 qps, IVFFlat 964 qps | query batching / ANN throughput | Two-Tower, 5,000 queries | `FINAL_REPORT.md` §3 only | RETRACTED |
| 93.85% fallback rate; 18.7% truncated; 0.19% collision rate | Two-Tower data audits | US positives | `FINAL_REPORT.md` §3 only | RETRACTED |
| LambdaMART 0.852064, MLP 0.851153, Two-Tower 0.831272, BM25 0.822225 | NDCG (course-project pool) | 30,969-query legacy pool | `docs/reports/official_metric_final.md` only | RETRACTED |
| 88 tests | test count | — | README L135, L162 only (no saved test-run result) | RETRACTED |

## 3. Résumé-safe sentences (VERIFIED only)

- Trained an XLM-RoBERTa cross-encoder reranker that reached NDCG@20 0.8456,
  compared with 0.8071 for Amazon's official ESCI Task 1 baseline (rebuilt
  here). Both were scored on the same 5,071-query multilingual dev split
  (us/es/jp), with the same pool and the same scorer.
- On the like-for-like US slice (3,133 dev queries, cross-encoder against
  cross-encoder), NDCG@20 improved by +0.0213 over the official baseline
  (paired bootstrap, 10,000 resamples, 95% CI [+0.0173, +0.0253]).
- Across all 5,071 dev queries, NDCG@20 improved by +0.0385 over the official
  baseline (paired bootstrap, 95% CI [+0.0348, +0.0421]; per-query
  wins/losses/ties 3229/1696/146). The ES and JP baseline branches are
  English-only bi-encoders, so most of this margin is not a like-for-like
  comparison.
- NDCG full-list on the same dev split: 0.8834 against 0.8521 for the official
  baseline. NDCG@10: 0.7924 against 0.7358.
- Built reproduction gates covering this work:
  - The regenerated predictions matched the stored oracle on 118,231 of 118,231
    rows, with max abs diff 0.0, for both models.
  - Rebuilding the split from raw ESCI data passed 52 of 52 integrity checks.
  - The candidate pool matched the official 781,638 Task 1 train rows exactly,
    with 0 label mismatches.
- Verified the NDCG scorer against an independent pure-Python reference: it
  passed 13 of 13 parity cases, with max abs delta 1.1e-16 over 300 random
  queries.
- Enforced a test-split lock in code. The access log has 2 entries, both from
  pool construction, and no model has been scored on the test split.

## 4. Do not use

- **0.019–0.030 NDCG gain-swap impact.** It appears only in README and report
  prose. It was measured on the course-project pool, which is not this repo's
  DEV, using models partly written by teammates.
- **Two-Tower V1 Recall@100 / RRF@100 / MRR@10.** The result JSON is not in
  this repo. The model is also US-only, and its baseline is a different
  benchmark from DEV.
- **"~55x", "~271x", "~80x" speedups and the qps figures.** They appear only in
  report prose and are derived ratios. The ANN speedups also cost recall.
- **The 93.85%, 18.7% and 0.19% Two-Tower audit figures.** No saved result file
  contains them.
- **LambdaMART, MLP, BM25 and Two-Tower legacy NDCGs (e.g. 0.852064).** They
  exist only in reports, on a different 30,969-query pool, and several
  components are teammates' work. Do not compare them with V2's 0.8456.
- **"88 tests".** No saved test-run output contains this count.
- **The +0.066 ES/JP NDCG@20 margins.** They are in a result file, but they
  are misleading on their own because the baseline used an English-only
  bi-encoder for ES/JP. Cite them only with that caveat.
- **"Beats the baseline on test", or any leaderboard comparison.** Every number
  here is a dev number, and test has never been scored.
- **Training wall-clock (14,099 s and 13,679 s per epoch; 590.9 s eval)** and
  **inference rows/sec.** These are in files, but they were measured on
  different hardware and architectures, so they are not comparable speed
  claims.
