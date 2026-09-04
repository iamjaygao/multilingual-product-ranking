# ESCI Ranking V2 — Phase 1: Official Benchmark Repair & Baseline Freeze

**Date:** 2026-09-02 · **Commit at start and end:** `9b173e9` · **Scope:** benchmark correctness only.
No pre-existing file was modified. No feature was added, removed or redefined. No hyperparameter
was tuned. No Cross-Encoder was trained. Retrieval was untouched. Everything produced here lives
under `experiments/ranking_v2/benchmark_repair/`.

---

# 1. Executive Summary

## The authoritative baseline table

All four models scored on **one** pool (`data/test_clean.parquet`, 638,016 rows / 30,969 queries)
with **one** scorer (`scripts/official_ndcg.py`) under **one** gain convention.

| Model | NDCG@10 | Candidate Pool | Gain Convention | Status |
|---|---:|---|---|---|
| BM25 | **0.8181** | clean frozen | official | authoritative |
| Two-Tower | **0.8267** | clean frozen | official | authoritative |
| MLP (17-feature) | **0.8474** | clean frozen | official | authoritative |
| LambdaMART (17-feature, corrected gain) | **0.8485** | clean frozen | official | authoritative |

Reference row, not a V2 baseline: the frozen V1 LambdaMART re-scored on the clean pool = **0.8480**.

Non-degenerate subset (24,446 queries with more than one distinct relevance level — reported
alongside per Phase 0's P2-A): BM25 0.7696 · Two-Tower 0.7805 · MLP 0.8067 · LambdaMART 0.8081.

## Headline findings

1. **All three P0 defects are fixed.** Objective and metric are now the same function
   (residual 1.08e-4, tie-breaking only); the pool has exactly 0 duplicate rows against 638,016
   judged pairs; every model shares the same rows, queries, labels, scorer and tie-break.
2. **Repairing the benchmark barely moved the score.** Corrected LambdaMART 0.8485 vs V1
   LambdaMART on the same clean pool 0.8480 — the gain-convention fix is worth **+0.0006**
   (CI [+0.0002, +0.0009], p=0.003). This was a *correctness* problem, not a performance one.
   Anyone hoping the objective fix would unlock a jump should recalibrate.
3. **The old 0.8464 vs 0.8188 comparison overstated the gap by only ~0.3%.** On one pool the
   real LambdaMART−BM25 delta is +0.0304 (CI [+0.0289, +0.0319]) versus the +0.0309 previously
   quoted across two pools. The conclusion survived; the arithmetic didn't.
4. **LambdaMART now beats the MLP by a statistically detectable but practically trivial margin.**
   Frozen-vs-frozen (both V1 checkpoints, same clean pool): **+0.00057, CI [+0.00003, +0.00112],
   p=0.041** — marginal, with 54.6% of queries tied. Phase 0 reported p=0.126 on this comparison;
   the change comes from deterministic tie-breaking and recomputed features, not from a real
   capacity difference emerging. **Model capacity still looks non-binding.**
5. **The JP locale gap is confirmed and unchanged by the repair.** LambdaMART scores US 0.8732 /
   ES 0.8016 / JP 0.7682. This is a feature-quality problem, not a benchmark artifact.
6. **A new, quantified consequence of the repair:** removing the 44,217 duplicate rows shifted
   `log_price` on 47.3% of imputed-price rows, because the imputation median is computed within
   the frame. **Zero observed prices changed.** This is Phase 0's P1-7 (transductive imputation)
   made concrete and is now the strongest argument for fixing it in Phase 2.
7. **Feature importance conclusions are directionally intact**: `semantic_score` 51.0% /
   `bm25_score` 26.8% / `word_overlap` 11.6% of gain under the corrected model (was 55.0 / 22.7 /
   11.3). Top-3 still ≈89%.

**The benchmark is READY for V2 Cross-Encoder experiments.**

---

# 2. What was wrong with V1

**Objective / metric mismatch (P0-1).** `scripts/train_lambdamart.py:18` sets
`LABEL_GAIN = [0, 1, 3, 7]`, so training valued a Substitute at **3/7 = 0.4286** of an Exact.
`evaluation/metrics.py:9` computes gain as `2**relevance - 1` over
`{E:1.0, S:0.1, C:0.01, I:0.0}`, so the metric valued a Substitute at **0.0718** of an Exact — a
**5.97× disagreement**. Measured directly: LightGBM's own internal `ndcg@10` on dev read
**0.9162** while the reported metric on the identical predictions was **0.8821**, a gap of
**3.41e-2**.

**Locale-unaware joins (P0-2).** `evaluation/evaluate_advanced.py:42` joined ESCI-S on
`product_id` alone (11,309 duplicate asins) and `:62` joined the products table on `product_id`
alone (11,567 ids exist in more than one locale). Test pool: **682,233 rows from 638,016 judged
pairs (+44,217, +6.93%)**, with up to 9 copies of a single pair across 6,556 queries. The training
frame was inflated the same way, 2,119,685 rows from 1,983,272 pairs.

**Mixed-pool reporting (P0-3).** BM25 0.8188 came from the clean pool; LambdaMART 0.8464 came
from the inflated one. BM25 on the inflated pool is 0.8155 — so 0.0033 of the quoted 0.0309 gap
was pool difference, not model quality.

---

# 3. Official Gain Convention

Verified from code, not assumed (`official_gain_mapping.json`):

| ESCI label | Integer training label | Official relevance | `label_gain` position | `label_gain` value | V1 value |
|---|---:|---:|---:|---:|---:|
| I | 0 | 0.00 | 0 | 0.000000 | 0 |
| C | 1 | 0.01 | 1 | 0.006956 | 1 |
| S | 2 | 0.10 | 2 | 0.071773 | 3 |
| E | 3 | 1.00 | 3 | 1.000000 | 7 |

**Why the values are `2**relevance − 1` and not the raw relevance.** The brief specified
`label_gain = [0.0, 0.01, 0.10, 1.0]`. LightGBM uses `label_gain[label]` *directly* as the DCG
numerator — its own default `[0,1,3,7,15,…]` is `2**i − 1`. The repository's scorer applies
`2**relevance − 1` on top of the fractional relevance. So to make LightGBM's lambdarank objective
numerically **identical** to the reported metric, `label_gain` must be `2**relevance − 1`.

This is not a hypothetical distinction. Both were trained and measured:

| `label_gain` | S:E training ratio | \|LightGBM internal dev ndcg@10 − official scorer\| |
|---|---:|---:|
| V1 `[0, 1, 3, 7]` | 0.4286 | 3.41e-02 |
| Brief's literal `[0, 0.01, 0.10, 1.0]` | 0.1000 | 3.04e-03 |
| **`2**rel−1` `[0, 0.006956, 0.071773, 1.0]`** | **0.0718** | **1.08e-04** |

The exact form is ~28× better aligned than the brief's literal form and ~315× better than V1, so
it is the authoritative one. The residual 1.08e-4 is tie-breaking (the official scorer breaks ties
on ascending `product_id`, LightGBM on internal row order); 0 dev queries lack a relevant
candidate, so query exclusion contributes nothing. Test NDCG@10 for the two corrected variants
differs by 0.0002 (0.8485 exact vs 0.8483 linear), so nothing material rides on the choice — but
only the exact form makes "objective == metric" an unqualified YES.

---

# 4. Frozen Benchmark

Built by `scripts/build_clean_pool.py`. Anchored on the ESCI judgment table; every join is a
LEFT join asserted not to change the row count; products joined on `(product_id, product_locale)`;
ESCI-S de-duplicated on `asin` first (11,309 rows dropped).

| | train | dev | test |
|---|---:|---:|---:|
| queries | 84,731 | 14,953 | 30,969 |
| rows | 1,684,002 | 299,270 | 638,016 |
| candidates/query mean | 19.87 | 20.01 | 20.60 |
| candidates/query median | 16 | 16 | 16 |
| queries with <10 candidates | 539 | 88 | 175 |
| degenerate queries (one relevance level) | 22,113 | 3,848 | 6,523 |

**train + dev = 1,983,272 = exactly the ESCI train judged pairs. test = 638,016 = exactly the
ESCI test judged pairs. Zero duplicates.**

**Label distribution (%)**

| | E | S | C | I |
|---|---:|---:|---:|---:|
| train | 66.20 | 21.27 | 2.76 | 9.77 |
| dev | 66.82 | 21.00 | 2.79 | 9.40 |
| test | **61.65** | **24.04** | 3.27 | 11.04 |

The train→test shift found in Phase 0 (P1-8) is preserved, as it must be: test is 4.6 pp less
Exact and 2.8 pp more Substitute. This is a property of ESCI, not of the build.

**Locale distribution (queries)**

| | us | jp | es |
|---|---:|---:|---:|
| train | 63,707 | 11,375 | 9,649 |
| dev | 11,181 | 2,085 | 1,687 |
| test | 22,458 | 4,667 | 3,844 |

**On the dev split.** ESCI ships only train/test. Both V1 trainers already carve a random 15% of
train *queries* via `train_test_split(unique_queries, test_size=0.15, random_state=42)` at fit
time and never persist it. This phase **materialises that exact split** — same function, same
seed, same query ordering taken from the V1 feature frame — and reproduces V1's
**84,731 / 14,953** split exactly. The split is unchanged; it is only now auditable.

---

# 5. Authoritative Baselines

| Model | NDCG@10 | Median | Std | Queries scored | Excluded (no relevant) | Provenance |
|---|---:|---:|---:|---:|---:|---|
| BM25 | 0.818098 | 0.914857 | 0.2231 | 30,969 | 0 | `output/bm25_scores_test.csv`, frozen |
| Two-Tower | 0.826719 | 0.926622 | 0.2182 | 30,969 | 0 | `models/two_tower_finetuned`, frozen |
| MLP (17-feature) | 0.847385 | 0.935552 | 0.2063 | 30,969 | 0 | `output/best_advanced_reranker.pth`, frozen |
| **LambdaMART (corrected)** | **0.848518** | 0.935552 | 0.2058 | 30,969 | 0 | retrained, `label_gain` only |
| *LambdaMART (V1 model, V1 gain)* | *0.847957* | *0.935552* | *0.2067* | *30,969* | *0* | *reference, frozen* |

**Corrected LambdaMART training** (`models/training_results.json`): `best_iteration` 355, 355
trees, 7.3 s. Dev NDCG@10 **0.881967**. Every hyperparameter copied verbatim from
`scripts/train_lambdamart.py` — lr 0.05, 1000 estimators cap, 31 leaves, `min_child_samples` 20,
subsample 0.8, colsample 0.8, `reg_lambda` 1.0, seed 42, early stopping 50 on dev `ndcg@10`.
Only `label_gain` and the (clean) rows differ.

**MLP fidelity note.** The frozen MLP is evaluated on features recomputed over the clean pool.
`mlp_fidelity.json` measures the drift: of the 17 features, only `log_price` (34.7% of rows),
`is_dominant_category` (2.1%) and `stars_clean` (0.13%) moved at all. All 14 others — including
`bm25_score`, `semantic_score`, `word_overlap`, every `query_*` feature, `brand_match` and
`color_match` — are bit-identical. Crucially, **0 of 170,146 observed-price rows changed**; all
221,395 `log_price` changes are among the 467,870 rows whose price is *imputed* from a within-frame
category median that shifted when duplicates were removed. Net effect on the metric: MLP scores
0.847385 here vs 0.8476 when Phase 0 filtered V1 scores to clean rows — a 0.0002 difference.

---

# 6. Locale Results

`locale_results.csv`, per-query NDCG@10 averaged within locale.

| Model | us (22,458) | jp (4,667) | es (3,844) | overall (30,969) |
|---|---:|---:|---:|---:|
| BM25 | 0.8432 | 0.7434 | 0.7621 | 0.8181 |
| Two-Tower | 0.8578 | **0.7219** | 0.7726 | 0.8267 |
| MLP | 0.8726 | 0.7644 | 0.8010 | 0.8474 |
| LambdaMART (corrected) | **0.8732** | **0.7682** | **0.8016** | **0.8485** |

The US↔JP gap for LambdaMART is **0.1050 NDCG**. Note that Two-Tower is the *worst* model on JP
(0.7219, below even BM25's 0.7434) while being clearly the better of the two single signals on US
(0.8578 vs 0.8432) — exactly what Phase 0 predicted from an encoder fine-tuned only on
`product_locale=='us'`. Benchmark repair did not change this; it is a modelling problem.

---

# 7. Statistical Comparison

`paired_comparisons.json`. Query-level paired percentile bootstrap, n=10,000, seed=42 — the
implementation audited in `experiments/audit/task3_per_query_bootstrap.py`. n=30,969 queries.

| Comparison | Mean Δ | 95% CI | p | Win / Loss / Tie |
|---|---:|---|---:|---|
| LambdaMART − BM25 | +0.03042 | [+0.02891, +0.03191] | <0.0001 | 12,384 / 7,153 / 11,432 |
| LambdaMART − Two-Tower | +0.02180 | [+0.02054, +0.02305] | <0.0001 | 11,351 / 7,204 / 12,414 |
| MLP − BM25 | +0.02929 | [+0.02772, +0.03078] | <0.0001 | 12,312 / 7,367 / 11,290 |
| MLP − Two-Tower | +0.02067 | [+0.01945, +0.02184] | <0.0001 | 11,275 / 7,119 / 12,575 |
| Two-Tower − BM25 | +0.00862 | [+0.00670, +0.01053] | <0.0001 | 11,075 / 9,719 / 10,175 |
| LambdaMART − LambdaMART (V1) | +0.00056 | [+0.00020, +0.00094] | 0.0032 | 5,564 / 5,565 / 19,840 |
| **LambdaMART − MLP** *(confounded)* | +0.00113 | [+0.00059, +0.00168] | <0.0001 | 7,249 / 6,763 / 16,957 |
| **LambdaMART (V1) − MLP** *(like-for-like)* | **+0.00057** | **[+0.00003, +0.00112]** | **0.0408** | 7,249 / 6,814 / 16,906 |

## Does model capacity still look non-binding?

**Yes, with a caveat about which row you read.**

The `LambdaMART − MLP` row is **confounded**: the authoritative LambdaMART was retrained on the
repaired benchmark with the corrected objective, while the MLP is still the V1 checkpoint. That
row mixes an architecture difference with a training-data-and-objective difference.

The clean architecture test is **frozen-vs-frozen**: both V1 checkpoints, both scored on the clean
pool. That gives **+0.00057, CI [+0.00003, +0.00112], p=0.041**. Phase 0 measured +0.00043,
CI [−0.00012, +0.00098], p=0.126 on the same comparison; the shift to marginal significance comes
from deterministic tie-breaking and the recomputed imputed features, not from a newly-revealed
capacity gap.

Read plainly: at n=30,969 the bootstrap can now *detect* a difference, but the difference is
**+0.0006 NDCG (0.07% relative)** with a CI whose lower bound is +0.00003, and **54.6% of queries
are exact ties**. A gradient-boosted listwise ranker and a pairwise MLP over the same 17 features
land within a thousandth of each other. Phase 0's conclusion stands: **the shared feature
representation, not ranker capacity, is the binding constraint.** Statistical detectability at
this sample size is not practical significance.

---

# 8. Benchmark Integrity Checks

`integrity_checks.json` — **60 PASS / 0 build-induced FAIL / 1 FAIL_UPSTREAM**.

| Group | Assertion | Result |
|---|---|---|
| Row conservation | train+dev rows == 1,983,272 ESCI train judged pairs | PASS |
| | test rows == 638,016 ESCI test judged pairs | PASS |
| | no split exceeds its judged-pair count (`join_output_rows > expected` gate) | PASS |
| | test pool smaller than V1's 682,233 | PASS |
| Key uniqueness | `example_id` unique in each split, and across all three | PASS ×4 |
| | `(query_id, product_id, product_locale)` unique in each split | PASS ×3 |
| Label consistency | no conflicting label per logical example | PASS ×3 |
| | `relevance` column matches the official mapping exactly | PASS ×3 |
| | `lgb_label` is a 1:1 encoding of `esci_label` | PASS ×3 |
| Split integrity | train/dev query overlap == 0 | PASS |
| | dev/test query overlap == 0 | PASS |
| | **train/test query overlap == 0** | **FAIL_UPSTREAM** (see below) |
| | train+dev queries == ESCI train queries (99,684) | PASS |
| | test queries == ESCI test queries (30,969) | PASS |
| | dev split == V1 trainers' internal split (84,731/14,953) | PASS |
| | cross-split query_ids beyond the known upstream one | PASS (none) |
| | cross-split query shares no product between train and test | PASS |
| Features | all 17 present, order identical to `ALL_FEATURES` | PASS ×4 |
| | no NaN in features | PASS ×3 |
| | no Inf in features | PASS ×3 |
| | no NaN in `bm25_score`/`semantic_score`/`relevance`/`lgb_label` | PASS ×12 |
| Query groups | group sizes sum to row count | PASS ×3 |
| | every query has ≥1 candidate | PASS ×3 |
| | rows contiguous by `query_id` (required for LightGBM grouping) | PASS ×3 |
| Scorer | 23/23 hand-derived unit assertions (`ndcg_unit_tests.json`) | PASS |

**The one FAIL_UPSTREAM.** ESCI itself assigns `query_id 79706` ("piano") to both `train`
(3 rows) and `test` (31 rows), with **disjoint products**. Constraint 3 forbids changing the
split, so this is inherited, flagged, and quantified rather than silently deduplicated — it
affects **1 of 30,969 test queries (0.003%)**. No `example_id` is duplicated; each row lives in
exactly one split. The validator distinguishes `FAIL` (build-induced, gates the benchmark) from
`FAIL_UPSTREAM` (inherited, documented, does not gate), and exits 0 only when build-induced
failures are zero.

---

# 9. Impact on Historical Conclusions

**Is the old 0.8464 retired?** **Yes.** It was computed on the 682,233-row inflated pool by a
model whose training objective disagreed with the metric. It is recorded in
`historical_results.csv` marked `LEGACY / NOT DIRECTLY COMPARABLE` and must not be quoted
alongside any V2 number. Its replacement is **0.8485**. For attribution: the same V1 model on the
clean pool scores 0.8480, so of the 0.0021 difference, +0.0016 is the pool fix and +0.0006 is the
gain fix.

**Is BM25 0.8188 still valid?** **Almost exactly.** The new authoritative value is **0.8181**.
The 0.0007 difference is entirely the new deterministic tie-break (BM25 has the heaviest tie
structure of any scorer — Phase 0 measured 34.2% of its rows sharing a score within their query),
plus the 0.003% of queries touched by the upstream split defect. The old figure was *not* wrong;
it was simply never comparable to the model numbers it was quoted against. Use 0.8181.

**Does LambdaMART still beat MLP?** **Yes, but only just, and the practical answer is "they are
equivalent."** Like-for-like: +0.00057, CI [+0.00003, +0.00112], p=0.041, with 54.6% of queries
tied. Phase 0 called this non-significant (p=0.126); it is now marginally significant. Do not
read that as a capacity finding — it is a tie-breaking and imputation artifact operating on a
0.07% relative gap.

**Is semantic_score still a strong signal?** **Yes, and unchanged in character.** It holds
**51.0%** of gain in the corrected model (was 55.0% in V1). Two-Tower alone still beats BM25 alone
by +0.0086 (CI [+0.0067, +0.0105]). Phase 0's separate finding also stands: high gain share does
not mean high marginal value. And the JP result (Two-Tower 0.7219, *below* BM25's 0.7434)
re-confirms that the signal is strong only where the encoder was trained.

**Are previous feature importance conclusions still directionally usable?** **Yes.**

| Feature | V1 model gain % | Corrected model gain % |
|---|---:|---:|
| semantic_score | 54.99 | 51.04 |
| bm25_score | 22.73 | 26.81 |
| word_overlap | 11.28 | 11.60 |
| is_dominant_category | 3.56 | 2.18 |
| brand_match | 1.72 | 2.38 |
| **top-3 combined** | **89.01** | **89.45** |

The ordering of the top three is unchanged and their combined share is stable at ≈89%. The
corrected objective shifts weight from `semantic_score` toward `bm25_score` by ~4 pp — consistent
with a metric that now cares almost exclusively about placing Exacts first, where lexical match is
the sharper signal. `log_review_count` and `is_over_budget` remain at 0 gain.

**What is NOT fixed.** Phase 0's P1 defects are untouched, because fixing them changes feature
*values*, which constraint 4 places outside this phase: `log_review_count` is still constant 0 on
100% of rows; 15.9% of prices are still parsed 100× too large; 13.7% of stars are still truncated;
tokenisation is still whitespace-based for JP. Phase 1 also produced new evidence for P1-7
(transductive imputation): 47.3% of imputed-price rows moved when the pool changed, while 0
observed-price rows did.

---

# 10. Next Step

**The clean benchmark is READY for V2 Cross-Encoder experiments.**

It provides one materialised pool with exactly the judged pairs, an auditable persisted
train/dev/test split identical to what V1 used, a single unit-tested scorer with deterministic
tie-breaking, four authoritative baselines measured under identical conditions, and a paired
bootstrap harness for significance testing.

Two things a Cross-Encoder effort must know before starting, both stated as facts rather than
recommendations:

- The bar to clear is **0.8485** (LambdaMART) on the frozen test pool, or **0.8081** on the
  non-degenerate subset, which is the more discriminative view since 21.1% of test queries return
  NDCG=1.0 for any ranking.
- The P1 data defects listed in §9 are still live. A Cross-Encoder that reads raw text will route
  around the broken `log_review_count`/`price`/`stars` features, but it will *not* route around
  the JP tokenisation gap unless it uses a multilingual encoder.

No implementation is proposed here, per the brief.

---

# Appendix: Deliverables

```
experiments/ranking_v2/benchmark_repair/
├── REPORT.md                         this document
├── official_gain_mapping.json        Phase A, verified-from-code mappings
├── baseline_comparison.csv           Phase G, authoritative table
├── historical_results.csv            Phase G, LEGACY / NOT DIRECTLY COMPARABLE
├── nondegenerate_subset.csv          non-degenerate view of the same models
├── locale_results.csv                Phase H, us/jp/es/overall
├── paired_comparisons.json           Phase I, 8 comparisons, bootstrap n=10,000 seed=42
├── integrity_checks.json             Phase C, 61 assertions with PASS/FAIL/FAIL_UPSTREAM
├── p0_validation.json                Phase J, the three P0 defects re-checked
├── ndcg_unit_tests.json              Phase D, 23 hand-derived scorer assertions
├── mlp_fidelity.json                 feature drift of the frozen MLP's inputs
├── per_query_scores.parquet          per-query NDCG@10 for all 5 score columns
├── data/
│   ├── train_clean.parquet           1,684,002 rows / 84,731 queries
│   ├── dev_clean.parquet               299,270 rows / 14,953 queries
│   ├── test_clean.parquet              638,016 rows / 30,969 queries
│   └── build_manifest.json
├── models/
│   ├── corrected_lambdamart_exact_cleanpool.txt    AUTHORITATIVE (355 trees)
│   ├── corrected_lambdamart_linear_cleanpool.txt   brief's literal label_gain
│   ├── corrected_lambdamart_v1gain_cleanpool.txt   pool-effect control
│   ├── corrected_lambdamart_test_scores.parquet
│   ├── training_results.json
│   └── train_run.log
└── scripts/
    ├── build_clean_pool.py           Phases A + B
    ├── validate_pool.py              Phase C
    ├── official_ndcg.py              Phase D  <-- the one scorer
    ├── test_official_ndcg.py         Phase D unit tests
    ├── train_corrected_lambdamart.py Phase F
    ├── evaluate_baselines.py         Phases E, G, H, I
    └── validate_p0_fixed.py          Phase J
```

**Format note.** The brief specified `models/corrected_lambdamart.*`. Three variants are saved
under explicit suffixes (`_exact_cleanpool`, `_linear_cleanpool`, `_v1gain_cleanpool`) so that the
gain-convention effect and the pool effect can be attributed separately;
`corrected_lambdamart_exact_cleanpool.txt` is the authoritative model.
`nondegenerate_subset.csv`, `p0_validation.json`, `ndcg_unit_tests.json` and `mlp_fidelity.json`
are additions beyond the required list.
