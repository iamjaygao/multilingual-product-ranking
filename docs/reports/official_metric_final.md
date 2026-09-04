# ESCI Ranking V2 — Phase 1.1: Official KDD NDCG@10 Final Calibration

**Date:** 2026-09-02 · **Commit at start and end:** `9b173e9` · **Scope:** metric calibration only.
No source file, model, dataset or prior experiment artifact was modified. The clean pool was not
rebuilt. The split was not changed. No feature was touched. No model was retrained. Nothing was
tuned. All new artifacts live under `experiments/ranking_v2/official_metric_final/`.

---

# 1. Executive Summary

## THE authoritative ESCI Ranking V2 baseline

| Model | Official NDCG@10 |
|---|---:|
| BM25 | **0.8222** |
| Two-Tower | **0.8313** |
| MLP (17-feature) | **0.8512** |
| **LambdaMART (17-feature)** | **0.8521** |

Frozen pool: **638,016 rows / 30,969 queries**, 0 queries excluded.
Scorer: [`scripts/official_kdd_ndcg.py`](../../src/metrics/official_kdd_ndcg.py).
Gain: **E=1.00, S=0.10, C=0.01, I=0.00 used directly as the DCG numerator.**

Every future Ranking V2 model — including the Cross-Encoder — compares against these four numbers.

## What changed and what it means

1. **The metric now matches KDD Cup / ESCI Task 1.** The legacy scorer applied `2**relevance − 1`
   on top of the ESCI values, which silently turned S into 0.0718 and C into 0.0070. Under the
   official convention S is exactly 0.10 and C is exactly 0.01 — a **1.39× uplift for Substitutes**
   and **1.44× for Complements**.
2. **Every model gained, and none changed rank.** +0.0037 to +0.0046 absolute (+0.44% to +0.55%).
   The ordering BM25 < Two-Tower < MLP < LambdaMART is unchanged, as is every locale ordering.
3. **LambdaMART's objective is now literally the headline metric.** LightGBM uses
   `label_gain[label]` directly as the DCG numerator, so `label_gain=[0, 0.01, 0.10, 1.0]` makes
   its internal lambdarank objective the same function as the official scorer. Measured on dev:
   LightGBM **0.884869** vs official scorer **0.884778**, |diff| **9.1e-05** — tie-breaking only.
   This closes P0-1 completely.
4. **No retraining was needed.** The linear-gain LambdaMART already existed from Phase 1. Its
   configuration was audited against all six requirements and reused
   (`lambdamart_checkpoint_audit.json`).
5. **The metric change slightly narrows the model gaps.** BM25 and Two-Tower gain more
   (+0.0041, +0.0046) than MLP and LambdaMART (+0.0038, +0.0037), because the weaker rankers place
   more Substitutes in the top-10 and Substitutes just became worth 1.39× more. LambdaMART−BM25
   goes from +0.03023 to +0.02984.
6. **LambdaMART vs MLP is statistically significant but practically negligible**, and the honest
   number is the like-for-like one: **+0.00062, CI [+0.00009, +0.00115], p=0.023**, with 54.6% of
   queries exactly tied. Capacity still does not look like the binding constraint.
7. **The locale failure pattern is completely unchanged.** LambdaMART US 0.8765 / ES 0.8056 /
   JP 0.7727 (gap 0.1038). Two-Tower is still *worse than BM25* on JP (0.7280 vs 0.7487). The
   metric change was not masking or creating this.

**Status: READY FOR CROSS-ENCODER.**

---

# 2. Metric Definition

```
GAIN = { "E": 1.00,  "S": 0.10,  "C": 0.01,  "I": 0.00 }

DCG@10  = sum_i  GAIN(label_i) / log2(rank_i + 1)        # rank is 1-based
NDCG@10 = DCG@10 / IDCG@10                               # per query
report  = unweighted arithmetic mean over queries
```

**`2**relevance - 1` is NOT applied.** The ESCI label value *is* the DCG numerator.

| Label | Official gain | Legacy gain (`2**rel − 1`) | Ratio |
|---|---:|---:|---:|
| E | 1.00 | 1.000000 | 1.000× |
| S | **0.10** | 0.071773 | **1.393×** |
| C | **0.01** | 0.006956 | **1.438×** |
| I | 0.00 | 0.000000 | — |

**Documented behaviours** (all covered by unit tests):

- **candidate count < 10** — the list is simply shorter; DCG and IDCG use the same short list, so
  NDCG stays well defined and is frequently exactly 1.0. No padding, no penalty.
- **IDCG == 0** (every candidate is I) — the query is **excluded** from the mean and counted in
  `n_excluded_no_relevant`, *not* scored as 0.0. A query with nothing to find cannot distinguish a
  good ranker from a bad one; scoring it 0 would lower every model by the same constant while
  adding no signal. **On the frozen test pool this excludes 0 of 30,969 queries**, so the choice is
  currently inert — it is stated explicitly so a future pool cannot change the number silently.
- **ties** — deterministic: sort by score DESC, then `product_id` ASC, both stable mergesort.
  NDCG is invariant to input row order. Identical to the Phase-1 rule, so the legacy↔official
  comparison in §5 isolates the gain change alone.

## Unit tests

[`scorer_unit_tests.json`](../../artifacts/manifests/scorer_unit_tests.json) — **35/35 hand-derived assertions pass**.
Every expected value is computed by hand in the test docstring.

| Test | Covers | Key assertion |
|---|---|---|
| A | perfect ranking E>S>C>I | NDCG == 1.0 exactly |
| B | S ranked above E | DCG = 0.1/log2(2) + 1.0/log2(3) = 0.7309297536; IDCG = 1.0/log2(2) + 0.1/log2(3) = 1.0630929754; NDCG = **0.6875501678** |
| C | linear gains | `gain(S)==0.10` and `gain(C)==0.01`, with explicit guards that they are **not** 0.0717734625 / 0.0069555501 |
| D | candidate count < 10 | 3-candidate and 1-candidate lists scored without padding |
| E | ties | 25 random permutations of an all-tied query give bit-identical NDCG |
| F | all-I query | excluded, counted separately, mean is 1.0 not 0.5 |
| G | query weighting | unweighted mean of 1.0 and 0.0 is 0.5 |
| H | cutoff | rank-11+ items contribute nothing |
| I | legacy divergence | official 0.6875501678 ≠ legacy 0.6722605601 on the same input |

---

# 3. Frozen Benchmark

Reused unchanged from Phase 1. Not rebuilt.

| | rows | queries |
|---|---:|---:|
| `train_clean.parquet` | 1,684,002 | 84,731 |
| `dev_clean.parquet` | 299,270 | 14,953 |
| **`test_clean.parquet`** | **638,016** | **30,969** |

Confirmed for every model in the authoritative table:

| Property | Value |
|---|---|
| identical rows | 638,016 (asserted at load) |
| identical query groups | 30,969 |
| identical labels | `esci_label` from the frozen pool |
| identical gain mapping | E=1.0, S=0.1, C=0.01, I=0.0 |
| identical scorer | `official_kdd_ndcg.py` |
| identical tie-break | score DESC, `product_id` ASC |
| queries excluded | 0 |

**Parity check:** re-running the legacy scorer on this pool reproduces the Phase-1 MLP number
exactly (0.847385 vs 0.847385), confirming the pool and the MLP scores are byte-identical to
Phase 1 and that the only thing that moved is the gain convention.

---

# 4. Official Baselines

| Model | NDCG@10 | Median | Std | Rows | Queries | Provenance |
|---|---:|---:|---:|---:|---:|---|
| BM25 | 0.822225 | 0.914857 | 0.2176 | 638,016 | 30,969 | frozen `bm25_score` column, not recomputed |
| Two-Tower | 0.831272 | 0.925156 | 0.2120 | 638,016 | 30,969 | frozen `semantic_score` column, not recomputed |
| MLP | 0.851153 | 0.936750 | 0.2005 | 638,016 | 30,969 | frozen `best_advanced_reranker.pth`, inference only |
| **LambdaMART** | **0.852064** | 0.937015 | 0.2000 | 638,016 | 30,969 | reused linear-gain checkpoint, inference only |

## LambdaMART checkpoint reuse (section 5 audit)

`benchmark_repair/models/corrected_lambdamart_linear_cleanpool.txt` was audited against every
requirement before being reused. **No retraining was performed.**

| Requirement | Verified value | Result |
|---|---|---|
| checkpoint exists and loads | 183 trees, 17 features | PASS |
| `label_gain == [0.0, 0.01, 0.10, 1.0]` | `[0.0, 0.01, 0.1, 1.0]` | PASS |
| integer label map I=0, C=1, S=2, E=3 | `{'E':3,'S':2,'C':1,'I':0}` | PASS |
| same 17 features | identical to `ALL_FEATURES` and to the frozen pool's `feature_list` | PASS |
| same train/dev split | 84,731 / 14,953 queries | PASS |
| same clean pool | `benchmark_repair/data/train_clean.parquet` | PASS |
| same hyperparameters | lr 0.05, 1000 cap, 31 leaves, `min_child_samples` 20, subsample 0.8, colsample 0.8, `reg_lambda` 1.0, seed 42, early stop 50 | PASS |
| only gain differs from siblings | all three variants fit in one run, one script, identical data/features/hyperparameters; `label_gain` was the only varying argument | PASS |

**Decision: REUSE.**

A note worth recording: in Phase 1 this same checkpoint looked like the *misaligned* variant
(|LightGBM internal − scorer| = 3.04e-3) because it was being judged against the legacy
exponential scorer. Under the official metric it is the **aligned** one (9.1e-05) and the
`exact_cleanpool` variant is now the misaligned one. Nothing about the model changed — only the
yardstick.

---

# 5. Legacy vs Official Metric

`metric_comparison.csv`. Same pool, same scores, same tie-break —
**the only difference is the gain convention.**

| Model | Legacy NDCG | Official NDCG | Δ absolute | Δ relative | Legacy rank | Official rank |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.818098 | 0.822225 | +0.004127 | +0.504% | 4 | 4 |
| Two-Tower | 0.826719 | 0.831272 | +0.004553 | **+0.551%** | 3 | 3 |
| MLP | 0.847385 | 0.851153 | +0.003768 | +0.445% | 2 | 2 |
| LambdaMART | 0.848326 | 0.852064 | +0.003739 | **+0.441%** | 1 | 1 |

**1. Which model is most affected?** **Two-Tower** (+0.551% relative), then BM25 (+0.504%).
LambdaMART is least affected (+0.441%). The pattern is systematic: the metric change makes
Substitutes worth 1.39× more, so it rewards whichever ranker leaves more Substitutes in the top-10
— which is the weaker rankers. Every model gains because the change can only raise the value of
non-Exact items, never lower it.

**2. Do the rankings change?** **No.** BM25 < Two-Tower < MLP < LambdaMART under both metrics.
All four `rank_changed` flags are `False`.

**3. Does the LambdaMART−MLP gap change?** Marginally, and it narrows:

| | Legacy | Official |
|---|---:|---:|
| mean paired Δ | +0.00094 | +0.00091 |
| 95% CI | [+0.00041, +0.00148] | [+0.00040, +0.00143] |
| p | 0.0008 | 0.0008 |
| tied queries | 17,065 | 17,065 |

The conclusion is identical under both.

**4. Does the BM25 vs Two-Tower relationship change?** Two-Tower still wins, and by slightly more:
legacy +0.00862 (CI [+0.00670, +0.01053]) → official **+0.00905** (CI [+0.00719, +0.01091]),
p<0.0001 both. Consistent with Two-Tower benefiting most from the reweighting.

**5. Does the locale ranking change?** **No.** The per-locale model ordering is identical under
both metrics in all three locales. The uplift is slightly larger in the weaker locales
(JP +0.0050 to +0.0061, ES +0.0045 to +0.0052, US +0.0033 to +0.0041) — again because those
locales have more Substitutes surviving in the top-10 — but no ordering moves.

---

# 6. Locale Breakdown

`official_locale_results.csv`

| Model | us (22,458) | es (3,844) | jp (4,667) | overall (30,969) |
|---|---:|---:|---:|---:|
| BM25 | 0.846965 | 0.766913 | 0.748731 | 0.822225 |
| Two-Tower | 0.861881 | 0.777871 | **0.727965** | 0.831272 |
| MLP | 0.875944 | 0.805456 | 0.769498 | 0.851153 |
| **LambdaMART** | **0.876515** | **0.805600** | **0.772678** | **0.852064** |

**1. Is the LambdaMART US/ES/JP gap still pronounced?** **Yes, essentially unchanged.**
US − JP = **0.1038** (was 0.1050 under the legacy metric); US − ES = **0.0709** (was 0.0720).
The metric change closed roughly 1% of the gap — nothing.

**2. Is Two-Tower still below BM25 on JP?** **Yes, and by more.** Two-Tower 0.727965 vs BM25
0.748731 — a deficit of **−0.0208** (legacy: −0.0215). Meanwhile Two-Tower *beats* BM25 by +0.0149
on US. This remains the clearest signature of an encoder fine-tuned only on
`product_locale == 'us'`.

**3. Did the locale failure pattern change?** **No.** Same ordering, same magnitudes, same
Two-Tower JP inversion. The metric was neither hiding nor manufacturing the multilingual problem;
it is a genuine feature/model deficiency, exactly as the Phase 0 audit concluded.

---

# 7. LambdaMART vs MLP

`paired_comparisons.json`. Query-level paired percentile bootstrap,
n=10,000, seed=42 — the implementation audited in
`experiments/audit/task3_per_query_bootstrap.py`. n=30,969 queries.

| Comparison | Mean Δ | Median Δ | 95% CI | p | Win / Loss / Tie |
|---|---:|---:|---|---:|---|
| LambdaMART − BM25 | +0.02984 | 0.0 | [+0.02837, +0.03129] | <0.0001 | 12,368 / 7,174 / 11,427 |
| LambdaMART − Two-Tower | +0.02079 | 0.0 | [+0.01961, +0.02197] | <0.0001 | 11,298 / 7,122 / 12,549 |
| MLP − BM25 | +0.02893 | 0.0 | [+0.02740, +0.03038] | <0.0001 | 12,332 / 7,347 / 11,290 |
| MLP − Two-Tower | +0.01988 | 0.0 | [+0.01871, +0.02102] | <0.0001 | 11,270 / 7,124 / 12,575 |
| Two-Tower − BM25 | +0.00905 | 0.0 | [+0.00719, +0.01091] | <0.0001 | 11,086 / 9,708 / 10,175 |
| **LambdaMART − MLP** *(headline, confounded)* | **+0.00091** | 0.0 | [+0.00040, +0.00143] | **0.0008** | 7,142 / 6,762 / **17,065** |
| **LambdaMART (V1 model) − MLP** *(like-for-like)* | **+0.00062** | 0.0 | **[+0.00009, +0.00115]** | **0.0228** | 7,251 / 6,812 / 16,906 |

## Is LambdaMART genuinely better than the MLP?

**Statistically yes; practically no. And the headline row overstates it.**

The `LambdaMART − MLP` row is **confounded**: the authoritative LambdaMART was retrained on the
repaired benchmark with the official linear gain, while the MLP is still the untouched V1
checkpoint. That row mixes an architecture difference with a training-data-and-objective
difference. The like-for-like row — both checkpoints frozen from V1, both scored here — gives
**+0.00062, CI [+0.00009, +0.00115], p=0.023**, whose lower bound is +0.00009, i.e. nine
hundred-thousandths of an NDCG point.

Reading the evidence plainly:

- The effect is **+0.0006 to +0.0009 NDCG**, which is **0.07–0.11% relative**.
- **55.1% of queries (17,065 of 30,969) are exact ties** — the two models produce an identical
  top-10 ordering on more than half the benchmark.
- Wins barely exceed losses: 7,142 vs 6,762 (51.4% of decided queries).
- Both models sit ~0.0290 above BM25 — that gap is **32× larger** than the gap between them.

The bootstrap detects a difference because n=30,969 gives enormous power, not because the
difference matters. A listwise gradient-boosted ranker and a pairwise MLP over the same 17 features
land within a thousandth of each other. **The evidence continues to suggest the shared feature
representation, not ranker capacity, is the binding constraint** — the same conclusion Phase 0 and
Phase 1 reached, now confirmed under the official metric.

---

# 8. Non-degenerate Subset

`nondegenerate_baselines.csv`

**This is a diagnostic view of discriminative power. It is NOT the leaderboard metric.**
The official benchmark remains the full 30,969-query test set.

6,523 test queries (21.1%) have only one distinct gain level across all their candidates, so
NDCG@10 = 1.0 for *any* ranking. They contribute a constant to every model and carry zero
discriminative signal.

| Model | Full test (30,969) | Non-degenerate (24,446) | Difference |
|---|---:|---:|---:|
| BM25 | 0.822225 | 0.774788 | −0.047437 |
| Two-Tower | 0.831272 | 0.786250 | −0.045022 |
| MLP | 0.851153 | 0.811436 | −0.039717 |
| LambdaMART | 0.852064 | 0.812590 | −0.039474 |

The degenerate fifth adds ~+0.040 of pure constant. On the discriminative subset the
LambdaMART−BM25 gap widens from 0.0298 to 0.0378 — the same signal, better resolved. Model
ordering is unchanged. Useful when a V2 change produces a small full-set delta and you need to see
whether it moved anything real.

---

# 9. Final Benchmark Status

## READY FOR CROSS-ENCODER

The readiness criterion is benchmark *correctness*, not model performance. Against that criterion:

| Requirement | Status |
|---|---|
| One materialised candidate pool, exactly the judged pairs | 638,016 rows = ESCI test judged pairs, 0 duplicates |
| Locale-aware joins | `(product_id, product_locale)`; ESCI-S de-duplicated on asin |
| Frozen, persisted, auditable train/dev/test split | 84,731 / 14,953 / 30,969, identical to what V1 used internally |
| One scorer, unit-tested | `official_kdd_ndcg.py`, 35/35 hand-derived assertions |
| Official KDD gain convention | E=1.0, S=0.1, C=0.01, I=0.0 used directly; no exponentiation |
| Training objective == headline metric | LightGBM internal dev 0.884869 vs official 0.884778, \|diff\| 9.1e-05 |
| All models on identical rows/queries/labels/scorer/tie-break | verified in §3 |
| Deterministic, order-invariant metric | 25-permutation invariance test passes |
| Significance harness | paired bootstrap, n=10,000, seed=42 |
| Four authoritative baselines | BM25 0.8222 / TT 0.8313 / MLP 0.8512 / LambdaMART 0.8521 |

All three Phase-0 P0 defects are closed: objective/metric mismatch (now 9.1e-05), pool duplication
(0 duplicates), mixed-pool reporting (single shared pool).

## Known limitations a Cross-Encoder effort must be aware of

These do **not** block readiness — they are model/data quality issues, not benchmark defects — but
they are live and were deliberately out of scope for Phases 1 and 1.1 (which forbid feature changes):

- **Phase 0 P1-1/2/3 parser defects persist.** `log_review_count` is constant 0.0 on 100% of rows;
  15.9% of prices are parsed 100× too large (`"25,63€"` → 2563.0); 13.7% of star ratings are
  truncated (`"3,9"` → 3.0). A text-based Cross-Encoder routes around all three.
- **JP tokenisation gap persists.** JP is 15.1% of test queries and LambdaMART scores 0.7727 there
  vs 0.8765 on US. A Cross-Encoder will **not** route around this unless it uses a multilingual
  encoder.
- **One inherited upstream defect.** ESCI itself assigns `query_id 79706` to both train and test
  (disjoint products). Constraint 3 forbids changing the split, so it is documented rather than
  removed; it affects 1 of 30,969 test queries (0.003%).
- **Transductive imputation persists** (Phase 0 P1-7): price/stars imputation medians are computed
  within each split's own frame.

---

# 10. Future Rule

**From this experiment onward:**

> Every Ranking V2 model — LambdaMART, MLP, Cross-Encoder, and any future ranking model —
> **must** report its headline NDCG@10 through
> `experiments/ranking_v2/official_metric_final/scripts/official_kdd_ndcg.py`,
> evaluated on `experiments/ranking_v2/benchmark_repair/data/test_clean.parquet`.

The legacy exponential scorer
(`experiments/ranking_v2/benchmark_repair/scripts/official_ndcg.py`, gain = `2**relevance − 1`)
is **retained for historical reference only** and must never again be used for a headline
benchmark number. Any figure produced by it — including the Phase-1 table
(BM25 0.8181 / TT 0.8267 / MLP 0.8474 / LambdaMART 0.8485) — is superseded by §1.

Numbers predating Phase 1 (BM25 0.8188, MLP 0.8458, LambdaMART 0.8464) remain **LEGACY / NOT
DIRECTLY COMPARABLE**: they were computed on the duplicate-inflated pool under the exponential
convention.

Any new model must additionally report:
1. the full-test official NDCG@10 (the leaderboard number),
2. the non-degenerate subset value (diagnostic),
3. the per-locale breakdown, and
4. a paired bootstrap against LambdaMART 0.8521.

---

# Appendix: Deliverables

```
experiments/ranking_v2/official_metric_final/
├── REPORT.md                        this document
├── baseline_comparison.csv          THE authoritative Ranking V2 baseline
├── official_locale_results.csv      us/es/jp/overall, official + legacy + delta
├── metric_comparison.csv            legacy vs official, absolute and relative delta, rank change
├── paired_comparisons.json          7 comparisons under official + 6 under legacy, bootstrap
├── nondegenerate_baselines.csv      diagnostic subset (24,446 queries)
├── scorer_unit_tests.json           35/35 hand-derived assertions
├── lambdamart_checkpoint_audit.json section-5 reuse audit
├── per_query_scores.parquet         per-query official + legacy NDCG, locale, degeneracy flag
├── eval_run.log
└── scripts/
    ├── official_kdd_ndcg.py            <-- THE scorer for all future work
    ├── test_official_kdd_ndcg.py
    └── evaluate_official_baselines.py
```

Reused read-only and unmodified: `benchmark_repair/data/{train,dev,test}_clean.parquet`,
`benchmark_repair/models/corrected_lambdamart_linear_cleanpool.txt`,
`output/best_advanced_reranker.pth`, `output/advanced_normalization_stats.json`,
`output/lambdamart_model.txt` (reference row only).
