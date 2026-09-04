# ESCI KDD Cup 2022 Task 1 — Benchmark Alignment, Baselines & Test Lock

**Date:** 2026-09-02 · **HEAD at start and end:** `9b173e9` · **Status:** `READY (PARITY UNVERIFIED)`
Gate A resolved by human decision (A2/A3 `FAIL_UPSTREAM`, operative A2′/A3′ added and both PASS).
No data was modified, filtered or de-duplicated. No pre-existing file was touched.

---

# 1. Executive Summary

## Authoritative Task 1 baselines

**Pool** `task1_small_v1` — `test_task1.parquet`, **336,373 rows / 14,496 queries**, 0 excluded.
**Metric** full-list NDCG, competition gain **E=1.0 S=0.1 C=0.01 I=0.0** used directly as the DCG
numerator. **Scorer** `scripts/kdd_task1_ndcg.py`. **Tie-break** deterministic.

| Model | **NDCG full** | @10 | @20 | non-degen | headroom covered |
|---|---:|---:|---:|---:|---:|
| Random floor | 0.750317 | 0.567026 | 0.686096 | 0.750300 | 0% |
| BM25 | 0.819842 | 0.681948 | 0.768318 | 0.819829 | 27.9% |
| Two-Tower | 0.824499 | 0.688613 | 0.773013 | 0.824487 | 29.7% |
| **LambdaMART** | **0.842878** | 0.719866 | 0.795113 | 0.842868 | **37.1%** |
| MLP | *PENDING TASK1 RETRAIN* | — | — | — | — |
| Original order | *N/A* | — | — | — | — |
| Oracle | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 100% |

`metric_version` for every row:
`kdd_task1_ndcg@<sha16>|gain=competition_E1_S0.1_C0.01_I0|idcg0=exclude|tie=deterministic`

## Why these numbers are lower than the large-version benchmark — and why that is expected

The reduced version is **the harder subset by construction**: it is what remains after the
"easy" queries are filtered out. Two measurements make this concrete.

- **Label mix.** Task 1 train is E 43.7% / S 34.2% / C 5.2% / I 16.9%. The large-version train
  pool is E 66.3% / S 21.2% / C 2.8% / I 9.7%. Task 1 has **22.6 pp fewer Exacts** and
  **13.0 pp more Substitutes**.
- **Degenerate queries.** Large-version test had 6,523 / 30,969 (**21.06%**) queries where every
  candidate carried one relevance level, so NDCG was 1.0 for any ranking. Task 1 test has
  **1 / 14,496 (0.007%)**. Those free points are gone.

JP also carries more weight: test query weights shift US .7252→.6178, ES .1241→.1668,
JP .1507→.2154, and JP is the locale every model is weakest on at @10.

**Lower absolute NDCG here is a benchmark-definition change, not a regression and not a bug.**

## Ten things worth knowing

1. **The Task 1 pool matches the official Table 1 exactly** — every locale, every split, every
   depth, to the unit: 48,300 / 1,118,011 total; 33,804 / 781,638 train; 14,496 / 336,373 test;
   depths 20.29 / 27.26 / 28.43 / 23.20. Product-title null rate **0.0**.
2. **Gate A's two red checks are one upstream defect.** `query_id 79706` ("piano") has 31
   `small_version==1` test rows and 3 large-only train rows, disjoint products, sole occurrence.
   Recorded, not repaired. The operative assertions **A2′ and A3′ both PASS**.
3. **The pre-registered prediction landed almost exactly** — predicted 0.842315 from locale
   reweighting alone, observed **0.842878** (Δ +0.000563), band **within**. §6 explains why that
   near-match is a coincidence of two large opposing effects, not a validation of the model.
4. **Two-Tower is still worse than BM25 on JP** (full 0.8060 vs 0.8243; @10 0.6262 vs 0.6596) —
   the US-only fine-tuning signature survives the benchmark change intact.
5. **But the headline JP gap collapses**, and it is the *metric*, not the population, that does
   it. LambdaMART per-locale full-list: US 0.8444 / ES 0.8413 / JP 0.8398 — a spread of **0.0046**.
   At @10 on the identical data the spread is **0.0473**. See §7.
6. **The S/C gain ambiguity is a hard caveat, quantified.** Swapping S and C moves every baseline
   by **0.0185–0.0300** — an order of magnitude above the 0.002 threshold. Any comparison against
   a historical number computed under the other convention is invalid.
7. **The IDCG=0 policy is inert here.** Zero test queries have IDCG 0, so `exclude`, `zero` and
   `one` give byte-identical results. The fallback choice costs nothing on this benchmark.
8. **Tie-breaking is safe.** Max |deterministic − random-mean| across baselines is **0.000150**
   (BM25), far below 0.002.
9. **The random floor is high — 0.750317.** Full-list NDCG over ~23 mostly-relevant candidates is
   forgiving. Absolute NDCG is a misleading progress signal here; the *headroom covered* column
   above is the honest one, and LambdaMART covers only **37.1%**.
10. **Gate B was not attempted** (no Terrier, no Java; installing a JRE was out of scope). The
    scorer is an independent implementation whose parity has **not** been checked.

---

# 2. Dataset Correction

The previously frozen benchmark (638,016 rows / 30,969 queries) corresponds to
`large_version == 1`, i.e. **KDD Cup 2022 Tasks 2/3**, not Task 1. Confirmed against Table 2:
its overall test row matches to the unit.

`experiments/ranking_v2/official_metric_final/` is **RETAINED** as the
**LARGE-VERSION INTERNAL RANKING BENCHMARK**. It was not modified or read-modified during this
phase (verified by mtime). Its conclusions remain valid on its own population. Its numbers must
never be compared with Task 1 numbers — see §13.

---

# 3. Gate A Results

## Red (upstream) and green (operative), side by side

| Check | Scope | Result | Observed |
|---|---|---|---|
| **A1** small ⊆ large | raw | **TRUE** | `large_version` values among `small_version==1` rows = `[1]`; 1,118,011/1,118,011 → `downstream_path = REUSE` |
| **A2** split unique per query | raw table | **FAIL_UPSTREAM** | 1 query_id in 2 splits: `79706` |
| **A3** Task1 test ∩ large train | raw table | **FAIL_UPSTREAM** | intersection = 1: `{79706}` |
| **A4** products PK unique | raw | **PASS** | 0 duplicate `(product_id, product_locale)` in 1,814,924 |
| **A2′** split unique per query | **`small_version==1`** | **PASS** | 0 violations across 48,300 Task 1 queries |
| **A3′** materialized arm B ∩ Task1 test | **actual materialized pool** | **PASS** | arm B = 1,983,269 rows / 99,683 queries; intersection **0** |

## The upstream defect, documented not repaired

`query_id 79706`, `"piano"`, `us`, 34 raw rows, `example_id` 1565543–1565576:

| split | `small_version` | `large_version` | rows |
|---|---:|---:|---:|
| test | 1 | 1 | **31** |
| train | **0** | 1 | **3** |

Product overlap between the two sides: **0**. Sole occurrence in the dataset. It violates the
split property stated in the official ESCI README (a query belongs to exactly one split).
**Not modified, not filtered, not de-duplicated, not excluded.**

Within the Task 1 pool the query appears only in test, which is why A2′ passes. A3 fails only
because the three offending rows are large-only, which is precisely the arm-B exposure A3 exists
to catch — and A3′ shows the shared helper closes it.

## The arm-B exclusion is a shared helper (resolution item 3)

`scripts/task1_common.py::exclude_task1_test_queries` is the single sanctioned implementation;
`build_arm_a_pool` and `build_arm_b_pool` both call it, and any future training pool must too. On
arm B it removed exactly **3 rows / 1 query** — the leak, and nothing else. Arm A calls it
defensively and asserts it is a no-op.

## Product-locale multiplicity (resolution item 5)

`task1.groupby('product_id').product_locale.nunique().max()` = **3**, with **4,359** product_ids
in more than one locale (883,716 `(product_id, locale)` pairs over 879,141 product_ids).

**Locale-prefixed `doc_id` is therefore mandatory.** This diverges from the reference helper:
`prepare_trec_eval_files.py` lines 121–126 write qrels keyed on `product_id` alone, which would
merge distinct (product, locale) judgements onto one doc_id and silently corrupt both qrels and
run file. Recorded in `terrier_parity.json`; `build_terrier_files.py` uses
`f"{product_locale}_{product_id}"` in both files.

---

# 4. Metric Definition

```
GAIN = {E: 1.00, S: 0.10, C: 0.01, I: 0.00}          # used directly as the DCG numerator
DCG@k  = sum_{i<=k} GAIN(label_i) / log2(rank_i + 1)  # rank 1-based
NDCG@k = DCG@k / IDCG@k                               # per query
report = unweighted macro average over queries
```

`2**gain - 1` is **not** applied. Three cutoffs always reported: **full (HEADLINE)**, @10, @20 —
they genuinely differ, since Task 1 averages 23.20 candidates and caps at 40.

## The S/C conflict — a known issue, independently re-confirmed

**This is not an original finding of this project.** The SQID paper already documents that the
released esci-data code swaps the S/C label-to-relevance mapping and that correcting it changes
the ESCI_baseline nDCG. What this project contributes is an independent re-confirmation on the
local repo HEAD.

Verified locally: `/Users/jaygao/WORKSPACE/projects/dataset/esci-data`, HEAD
`7916cdf6ab75a462e77f20ab40428a10923998d5`, file sha256
`688f9255…3067095`. Lines 46–51 of `ranking/prepare_trec_eval_files.py` read
`{"E": 4, "S": 2, "C": 3, "I": 1}`, and `launch-predictions-task1.sh:54` runs
`-m 'ndcg.1=0,2=0.01,3=0.1,4=1'`. Composed, that yields **E=1.0, S=0.01, C=0.1, I=0.0** — S and C
swapped versus the competition definition.

Parity therefore uses the **corrected** qrels mapping `I→1, C→2, S→3, E→4`, which with the same
gain spec gives E=1.0, S=0.1, C=0.01, I=0.0.

## Swapped-gain sensitivity — the ambiguity is a hard caveat

| Model | competition | swapped | Δ |
|---|---:|---:|---:|
| Random | 0.750317 | 0.720282 | −0.030035 |
| BM25 | 0.819842 | 0.801352 | −0.018490 |
| Two-Tower | 0.824499 | 0.802476 | −0.022022 |
| LambdaMART | 0.842878 | 0.824276 | −0.018603 |
| Oracle | 1.000000 | 0.998190 | −0.001810 |

**Max |Δ| = 0.030035, and 0.0185–0.0220 for the real models — an order of magnitude above the
0.002 threshold.** Per §8.3 this is a **hard caveat on every historical comparison**: any number
computed under the other convention is not comparable to these, and the difference is large
enough to reverse ordinary model-level conclusions.

## IDCG = 0 policy

Gate B was NOT_ATTEMPTED, so the §8.6 fallback applies: **`exclude`** (drop the query from the
mean, count it separately). `zero_idcg_policy_source: "fallback_pending_gate_b"`.

All three conventions, as required:

| Policy | Random | BM25 | Two-Tower | LambdaMART | Oracle |
|---|---:|---:|---:|---:|---:|
| exclude (applied) | 0.750317 | 0.819842 | 0.824499 | 0.842878 | 1.000000 |
| zero | 0.750317 | 0.819842 | 0.824499 | 0.842878 | 1.000000 |
| one | 0.750317 | 0.819842 | 0.824499 | 0.842878 | 1.000000 |

**Identical, because `zero_idcg_query_count = 0` on Task 1 test.** The policy choice is inert on
this benchmark and can be swapped by lookup with no effect once Gate B runs.

## Tie-breaking

Headline: score DESC → `product_id` ASC → `product_locale` ASC, stable mergesort, row-order
invariant. Random-tie-break sensitivity, 20 seeds per baseline:

| Model | deterministic | random mean | random std | \|Δ\| |
|---|---:|---:|---:|---:|
| Random | 0.750317 | 0.750317 | 0.000000 | 0.000000 |
| BM25 | 0.819842 | 0.819692 | 0.000250 | **0.000150** |
| Two-Tower | 0.824499 | 0.824526 | 0.000028 | 0.000027 |
| LambdaMART | 0.842878 | 0.842887 | 0.000030 | 0.000008 |

Max |Δ| = 0.000150, well under 0.002. Nothing to flag.

## Unit tests

`scorer_unit_tests.json` — **47/47 assertions pass**, scenarios A–K plus a 500-case property test
(all outputs in [0,1]; perfect ordering always exactly 1.0 on all three cutoffs). Every expected
value is hand-derived in the test docstring. Highlights: B and C hand-computed to 10 dp; C asserts
`gain(S)==0.10` and `gain(C)==0.01` with explicit guards that they are *not* 0.0717734625 /
0.0069555501; D shows full/@10/@20 all differ on a 25-candidate query; E shows them equal on a
short list; F verifies all three IDCG=0 conventions; H runs 60 permutations including an all-tied
query; K shows the swapped-gain mode reverses scenario C's direction.

---

# 5. Gate B — Terrier Parity

| Field | Value |
|---|---|
| **status** | **NOT_ATTEMPTED** |
| reason | No Terrier, no `trec_eval`, **no Java runtime** on this machine. Resolution item 6 forbids installing a JRE or any system-level dependency. |
| exact command | `$1/terrier trec_eval "${...}/test.qrels" "${...}/hypothesis.results" -c -J -m 'ndcg.1=0,2=0.01,3=0.1,4=1'` |
| command source | **read_from_launch_script** (`.../esci-data/ranking/launch-predictions-task1.sh:54`) |
| qrels mapping | `I→1, C→2, S→3, E→4` (**corrected**) |
| tolerance | not determined — depends on Terrier's output precision, which cannot be measured without running it |
| IDCG=0 observed | not observed → fallback `exclude` applied |

**All Gate B inputs were generated and smoke-tested**, so completing it later is a single command:

- `trec_eval_data/{synthetic,dev_random,dev_bm25}.{qrels,results}`
- run files use the reference script's rank→synthetic-score materialisation verbatim
  (`np.arange(0, 128, 128/n).round(3)[::-1][:n]`), asserted to produce no ties, so Terrier and the
  Python scorer see the identical ordering and tie-break strategy is excluded from the test
- the synthetic set contains one all-I query, one normal query and one with tied gains, and
  `trec_files_manifest.json` records the Python NDCG under all three IDCG=0 conventions so the
  policy can be read off empirically
- dev only; test is never used
- `scripts/run_terrier_parity.sh <terrier bin>` runs all three

**Wording discipline.** Because status is NOT_ATTEMPTED, this report says only *"the scorer is an
independent implementation; parity has not been verified."* It does not claim to have reproduced
Terrier, and would not claim to have reproduced the official AIcrowd scorer even on a PASS.

---

# 6. Pre-registered Predictions vs Observed

`prereg_predictions.json` was written **before** any Task 1 baseline ran.

**(a) Old-scorer aggregation check.** Weighting the old per-locale LambdaMART numbers by the
large-version test query weights: 0.7252×0.8765 + 0.1241×0.8056 + 0.1507×0.7727 = **0.852059**
against the recorded **0.852064** — matches to 5 decimal places. **Confirmed: the old scorer uses
a query-level macro average with no hidden micro weighting.**

**(b) Locale-reweight prediction.** Same per-locale numbers, Task 1 test weights:
0.6178×0.8765 + 0.1668×0.8056 + 0.2154×0.7727 = **0.842315**, i.e. −0.009749 from locale
composition alone.

**(c) Observed.** LambdaMART full-list = **0.842878**. Δ vs prediction **+0.000563**.
Band **within** [0.76, 0.87]. Action: proceed.

## The near-perfect match is a coincidence — do not read it as validation

The prediction modelled *only* locale reweighting, yet the other two factors were expected to be
large. They were, and they nearly cancelled:

| Effect | Measurement | Value |
|---|---|---:|
| large-version LambdaMART @10 | recorded | 0.852064 |
| **Task 1 LambdaMART @10** | measured, same cutoff | **0.719866** |
| → population change (locale + harder subset + recomputed features + gain convention) | at a fixed @10 cutoff | **−0.132198** |
| → cutoff change, full-list vs @10, on identical data | 0.842878 − 0.719866 | **+0.123012** |
| net | | **−0.009186** |

So the honest decomposition is: the harder Task 1 population costs about **−0.13** at a fixed
cutoff, and moving from @10 to full-list gives back about **+0.12**. Their near-cancellation to
−0.0092 happens to sit within 0.0006 of the locale-only prediction of −0.0097. Treating that as
confirmation that "only locale mattered" would be wrong by two orders of magnitude on each
component.

---

# 7. Random Floor / Oracle Ceiling

## Random floor — 100 seeds, test split

| Group | full mean | std | p5 | p95 |
|---|---:|---:|---:|---:|
| overall | 0.750283 | 0.000817 | 0.748975 | 0.751727 |
| US | 0.746736 | 0.001161 | 0.744960 | 0.748694 |
| ES | 0.738465 | 0.001727 | 0.735699 | 0.741097 |
| JP | **0.769601** | 0.001257 | 0.767589 | 0.771813 |
| non-degenerate | 0.750266 | 0.000817 | 0.748958 | 0.751710 |

Dev overall full: 0.747038 (std 0.001184). Per-cutoff figures for every group are in
`random_ranking_floor.json`.

**This is the most important context number in the report.** A random ranking already scores
0.750 full-list. The entire usable range is 0.750 → 1.000, so raw NDCG massively understates how
much of the problem is unsolved:

| Model | full | headroom covered |
|---|---:|---:|
| BM25 | 0.819842 | 27.9% |
| Two-Tower | 0.824499 | 29.7% |
| LambdaMART | 0.842878 | **37.1%** |

Note also that **JP has the highest random floor** (0.7696 vs US 0.7467) — its label mix makes
random ranking look better. Any per-locale comparison that ignores this will mis-attribute
difficulty.

## Oracle ceiling

Sorting by true gain gives full = @10 = @20 = **1.000000** exactly, after applying the IDCG=0
policy. Assertion passes.

---

# 8. Degenerate Queries

Degenerate = all candidates share one relevance level. `all_I` = the subset with IDCG = 0.

| Split | queries | degenerate | fraction | all_I | all_I fraction |
|---|---:|---:|---:|---:|---:|
| train | 28,733 | 0 | 0.000% | 0 | 0.000% |
| dev | 5,071 | 0 | 0.000% | 0 | 0.000% |
| **test** | 14,496 | **1** | **0.007%** | **0** | **0.000%** |

Test by locale: US 1/8,956; ES 0/2,417; JP 0/3,123.

**Versus the large-version benchmark: 6,523/30,969 = 21.06% degenerate.** The §11.3 expectation —
that the reduced version should have a *lower* degenerate share because "easy" queries are
filtered out — is **met, overwhelmingly**: 0.007% vs 21.06%.

Two consequences. First, `ndcg_full` and `ndcg_full_nondegenerate` are essentially identical on
this benchmark (they differ in the 5th decimal), so the non-degenerate diagnostic view adds almost
nothing here — unlike on the large-version pool where it moved the number by ~0.04. Second,
`all_I = 0` is what makes the IDCG=0 policy inert (§4).

---

# 9. Feature Pool-Dependency

Rule applied (resolution item 8): join raw per-pair scores only; recompute everything derived
from the candidate pool; when in doubt, recompute; **no STOP** unless a derived feature needs an
unavailable intermediate that would require re-running *retrieval*. **No STOP was triggered.**

**9 recomputed, 8 joined.** Full reasons in `feature_pool_dependency_audit.json`.

| Recomputed (pool-dependent) | Why |
|---|---|
| `bm25_score` | `retrieval/bm25.py` indexes **that query's own candidates**, so IDF and avgdl are candidate-set statistics — *even the raw score is pool-dependent* — and the stored value is additionally per-query min-max normalised |
| `semantic_score` | per-query min-max over the pool; the raw cosine was **never persisted**, so the frozen column cannot be inverted or re-normalised |
| `query_mean_idf`, `query_max_idf` | IDF table rebuilt from the 33,804 Task 1 train queries (the original used large-version train queries) |
| `log_price`, `stars_clean`, `log_review_count` | category-median → global-median imputation are in-pool statistics |
| `is_over_budget` | derived from the in-pool imputed price |
| `is_dominant_category` | mode of category among the query's top-20 BM25 candidates — explicitly in-pool, and depends on the recomputed BM25 ordering |

| Joined (pool-independent) | |
|---|---|
| `query_length`, `user_budget`, `cheap_intent` | functions of the query string |
| `word_overlap`, `brand_match`, `color_match` | functions of the (query, product) pair |
| `is_price_missing`, `is_rating_missing` | null indicators on raw attributes |

**Important nuance on `downstream_path = REUSE`.** Gate A established that Task 1 rows are a
strict subset of large-version rows, so per-row values *could* be joined by key. In practice
**nothing was joined from the large-version artifacts**: the only two stored per-row scores
(`output/bm25_scores_*.csv`, `output/two_tower_scores_*.csv`) are both per-query min-max
normalised over the large pool — exactly the case the rule says to recompute. The "joined"
features above are computed directly from (query, product) attributes.

**Two raw signals are now persisted for the first time**, closing a gap Phase 0 flagged as a
blocker: `bm25_raw` (un-normalised BM25 from the per-query index) and `semantic_cosine_raw`
(un-normalised cosine; mean 0.5009, std 0.1914, range [−0.2700, 1.0000]).

**Recompute cost.** Re-encoding 883,716 unique `(product_id, product_locale)` items and 48,300
queries with the **frozen** `models/two_tower_finetuned` on MPS took 4,127 s. This is inference
only — the model was not retrained, and the candidate set is fixed by the ESCI judgments, so no
retrieval was re-run.

**A deliberate correction:** `retrieval/two_tower.py:89` de-duplicates items on `item_id` alone,
which collapses products that exist in several locales with different titles. Items are keyed here
on `(product_id, product_locale)`.

**Known defect reproduced faithfully:** `log_review_count` is still identically 0 on every row,
because ESCI-S `ratings` is a string like `"1,116 ratings"` and `pd.to_numeric` yields NaN. That
is Phase 0's P1-1. Constraint 4 forbids changing feature definitions in this phase, so it is
reproduced, not fixed.

---

# 10. Baseline Results

Every number below carries `candidate_pool = task1_small_v1` and the `metric_version` string in §1.

## Overall (repeat of §1 for completeness)

| Model | full | @10 | @20 | non-degen | query_count | row_count |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.750317 | 0.567026 | 0.686096 | 0.750300 | 14,496 | 336,373 |
| BM25 | 0.819842 | 0.681948 | 0.768318 | 0.819829 | 14,496 | 336,373 |
| Two-Tower | 0.824499 | 0.688613 | 0.773013 | 0.824487 | 14,496 | 336,373 |
| LambdaMART | 0.842878 | 0.719866 | 0.795113 | 0.842868 | 14,496 | 336,373 |
| Oracle | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 14,496 | 336,373 |

**Original order — N/A.** The ESCI examples table has no explicit rank/position column
(`example_id, query, query_id, product_id, product_locale, esci_label, small_version,
large_version, split`). §12.2 forbids inferring an order from parquet row order, so this baseline
is not computable.

**MLP — PENDING TASK1 RETRAIN.** Its frozen weights and normalisation statistics were fitted on
large-version feature distributions; the Task 1 features are recomputed on a different pool, so
running the old checkpoint here would be off-distribution and not a meaningful baseline.

## LambdaMART, retrained

Retrained because §12.2 requires it: both the gain convention and the pool changed. Every
hyperparameter copied verbatim from `scripts/train_lambdamart.py`; **only `label_gain` and the
training rows differ. Nothing was tuned.**

| | |
|---|---|
| `label_gain` | `[0.0, 0.01, 0.10, 1.0]` for I, C, S, E |
| hyperparameters | lr 0.05, 1000 cap, 31 leaves, `min_child_samples` 20, subsample 0.8, colsample 0.8, `reg_lambda` 1.0, seed 42, early stopping 50 |
| best_iteration / trees | 215 / 215 (early stopping fired) |
| train time | 3.4 s |
| train / dev | 663,407 rows / 28,733 q · 118,231 rows / 5,071 q |
| dev full / @10 / @20 | 0.842991 / 0.721029 / 0.794965 |
| test full | **0.842878** |

Dev 0.842991 vs test 0.842878 — a 0.0001 gap, so no dev overfitting.

**Objective/metric alignment.** LightGBM uses `label_gain` directly as the DCG numerator, so with
the official linear gain its internal objective *is* the metric at @10: LightGBM internal dev
ndcg@10 **0.7211529** vs the scorer's dev @10 **0.7210295**, |Δ| **1.23e-04** — tie-breaking only.
Note that early stopping monitors @10 (the frozen config's `eval_at`) while the headline is
full-list; both are reported and were not changed to avoid tuning.

**Feature importance shifted toward lexical** relative to the large-version model:
`semantic_score` 39.53% (was 51–55%), `bm25_score` 31.54% (was 22–27%), `word_overlap` 7.17%,
`brand_match` 4.25%, `is_dominant_category` 4.18%.

---

# 11. Locale Breakdown

| Model | US (8,956) | ES (2,417) | JP (3,123) | Overall (14,496) |
|---|---:|---:|---:|---:|
| Random | 0.746287 | 0.741520 | **0.768683** | 0.750317 |
| BM25 | 0.819898 | 0.813825 | **0.824338** | 0.819842 |
| Two-Tower | 0.832508 | 0.818769 | 0.805964 | 0.824499 |
| **LambdaMART** | **0.844385** | **0.841310** | **0.839772** | **0.842878** |
| Oracle | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## 1. Is the LambdaMART US/ES/JP gap still pronounced? **No — at full-list it nearly vanishes.**

US − JP = **0.0046** (was 0.1050 on the large-version @10 benchmark). But this is a **metric**
effect, not a population effect, and the same table at @10 proves it:

| Model | US @10 | ES @10 | JP @10 | US − JP |
|---|---:|---:|---:|---:|
| BM25 | 0.6951 | 0.6621 | 0.6596 | 0.0355 |
| Two-Tower | 0.7163 | 0.6668 | 0.6262 | 0.0901 |
| LambdaMART | 0.7343 | 0.7089 | 0.6870 | **0.0473** |

At @10 the JP deficit is alive and well. Full-list NDCG counts every one of the ~23–40
candidates, so mistakes below rank 10 are heavily discounted but still credited — and JP's higher
random floor (0.7696 vs US 0.7467) lifts its full-list score further. **The JP problem did not go
away; the headline cutoff hides it.**

## 2. Is Two-Tower still below BM25 on JP? **Yes, at both cutoffs.**

Full-list 0.805964 vs 0.824338 (**−0.0184**); @10 0.6262 vs 0.6596 (**−0.0334**). Meanwhile
Two-Tower *beats* BM25 on US by +0.0126 full / +0.0212 @10. The US-only fine-tuning signature is
unchanged by the benchmark switch. BM25 being the **best** JP model at full-list is a new and
notable fact.

## 3. Did the locale failure pattern change? **Only in appearance, and only at full-list.**

Ordering by locale is preserved at @10 for every model. The full-list view compresses it — and
for BM25 actually inverts it (JP best). Report both cutoffs when discussing locale, or the
conclusion flips.

---

# 12. Historical Competition Reference

Three mutually non-equivalent numbers. **None is a reproduction target**; only the §6 band governs.

| Source | Value | Gain convention | Pool | Directly comparable? |
|---|---:|---|---|---|
| esci-data README baseline (BERT cross-encoder) | 0.83 | **S/C swapped** (§4) | Task 1 public test | **No** |
| Amazon Science / competition published baseline | 0.8503 | competition (presumed) | Task 1 test | **No** |
| KDD Cup Task 1 winner, private LB | 0.9043 | competition | Task 1 **private** test | **No** |

The 0.83 figure is not merely "a different model" — it is computed under a **different gain
definition**, which our own measurement shows is worth 0.0185–0.0300 NDCG. That is a more
fundamental incomparability than any modelling difference.

## Three independent non-identities that survive even perfect alignment

1. **public/private split is unrecoverable.** AIcrowd split test into public and private; the
   released `split == "test"` may be the merge. There is no way to reconstruct the private slice.
2. **train/dev re-split differs.** Competitors used all 33,804 train queries; this project holds
   out 15% (5,071 queries) as dev, training on 28,733.
3. **The scorer is an independent implementation.** Gate B is NOT_ATTEMPTED here; even a PASS
   would only establish agreement with Terrier, not with the AIcrowd scoring program — and the
   Terrier reference's own companion script carries the §4 S/C defect.

**Wording used in this report:** "competition-compatible metric and Task 1 dataset definition",
"published historical reference". **Not used:** "exact private leaderboard reproduction", "we
beat/match the winner", "reproduced the official scorer".

---

# 13. Bookkeeping Rule (supersedes Phase 1.1 §10)

Phase 1.1's rule — "all future models report through `official_kdd_ndcg.py`" — **is hereby
superseded**, because that scorer belongs to a different candidate pool.

> **Every NDCG number, wherever it appears — REPORT, CSV, JSON, terminal output — must carry
> `candidate_pool` (`large_v1` or `task1_small_v1`) and `metric_version` (scorer SHA256 + gain
> definition + IDCG=0 convention + tie-break mode). A number without both labels is invalid and
> must not be quoted.**
>
> **Numbers from the two pools may never be subtracted from or compared with each other.**

Every CSV emitted by this phase carries both columns on every row.

---

# 14. Phase 2 Training Pools (defined and validated only — nothing trained)

| | arm A `task1_only` | arm B `large_train_excl_task1_test` |
|---|---|---|
| filter | `small_version==1 AND split=='train' AND query_id in train_task1` | `large_version==1 AND split=='train' AND query_id NOT IN task1_test_queries` |
| rows | 663,407 | 1,983,269 |
| queries | 28,733 | 99,683 |
| locale (US/ES/JP) | .6179 / .1666 / .2155 | .7513 / .1137 / .1350 |
| labels E/S/C/I % | 43.66 / 34.23 / 5.17 / 16.93 | 66.29 / 21.23 / 2.76 / 9.72 |
| exclusion helper | `task1_common.exclude_task1_test_queries` (no-op, asserted) | same helper; removed **3 rows / 1 query** |
| ∩ Task 1 test | **0** | **0** |

Evaluation pool for both arms: `small_version==1 AND split=='test'`.
**`leakage_assert_passed: true`**, evaluated on the actual materialized pools.

**Distribution differences are expected and are recorded, not corrected.** Arm B contains the
"easy" queries the reduced version filters out, so it is 13.3 pp more US, 8.1 pp less JP,
22.6 pp more Exact and 13.0 pp less Substitute than arm A. Any arm-A-vs-arm-B comparison in Phase
2 must treat this as a confound, not a nuisance.

---

# 15. Known Query-Text Collisions (diagnostic; nothing filtered)

Per resolution item 4: **not quarantined, not filtered.**

15 normalised query strings appear on both sides of the Task 1 split under **different**
`query_id`s: `airpods pro, android, chromecast, gopro hero 7 black, guitar, huawei mate 20 pro,
insta360 one, ipad pro 11, oculus quest, one piece, psp, samsung s8, ssd, vape, xiaomi mi 9t pro`.

| Quantity | Value |
|---|---:|
| shared query texts | 15 |
| affected test queries | **16** of 14,496 |
| affected train queries | 15 |
| cross-split `(normalized_query, product_id, product_locale)` triples | **0** |
| label agreement among those triples | n/a (none exist) |
| **worst-case NDCG bound** | **0.001104** (16 / 14,496) |

**The zero triple count is the substantive result.** No product is judged on both sides of the
split for any shared query text, so there is **no literal pair-level transfer channel** — a
memorising model has nothing to copy. The collision is query-text only.

The 0.001104 bound is the adversarial extreme: every affected test query scoring a perfect 1.0
purely by memorising its same-text twin *and* otherwise scoring 0.0. The realistic effect is far
smaller. Query_ids are distinct, ESCI treats them as distinct queries, and the affected texts are
short head terms whose candidate sets differ.

`known_query_text_collisions.json` lists the 15 texts, both sets of query_ids, per-text locale and
row counts, and shared-product counts.

---

# 16. Benchmark Status

## `READY (PARITY UNVERIFIED)`

Per §9.6, `NOT_ATTEMPTED` Gate B downgrades the suffix only. The READY criterion is benchmark
correctness, not model performance.

| Necessary-and-sufficient condition (§17) | Status |
|---|---|
| `small_version` filter correct, matches Table 1 per locale | **PASS** — exact on every cell, all locales, all splits, all depths |
| Gate A2 / A3 / A4 | A4 PASS; A2/A3 `FAIL_UPSTREAM` by explicit human resolution, with operative **A2′/A3′ both PASS** |
| split three-way no overlap | **PASS** — train/dev 0, train/test 0, dev/test 0 |
| locale-aware two-key join, null rate < 0.001 | **PASS** — no fan-out, null rate **0.0** |
| scorer unit tests all pass (incl. property test) | **PASS** — 47/47 |
| scorer deterministic and row-order invariant | **PASS** — 60 permutations incl. all-tied |
| `feature_pool_dependency_audit.json` has no `uncertain` | **PASS** — 9 recomputed / 8 joined, no uncertain (the fixed rule removed that category) |
| `prereg_check.json` no hard STOP | **PASS** — band `within` |
| test lock manifest generated | **PASS** |

Baseline scores do not affect this judgment.

## What is NOT verified

- **Terrier parity.** Not run. The scorer is an independent implementation.
- **The upstream split defect** on `query_id 79706` remains in the data by design.
- **Phase 0 P1 data defects persist** (`log_review_count` dead, price/stars parsing, JP
  tokenisation) — out of scope under constraint 4.

---

# 17. Test Lock

`test_task1.parquet` is frozen. `test_lock_manifest.json`:

| Field | Value |
|---|---|
| `test_file_sha256` | see manifest |
| `query_count` / `row_count` | 14,496 / 336,373 |
| `locale_breakdown` | us 8,956 · jp 3,123 · es 2,417 |
| `scorer_sha256` | see manifest |
| `gain_convention` | `competition_E1_S0.1_C0.01_I0` |
| `zero_idcg_policy` | `exclude` |
| **`zero_idcg_policy_source`** | **`fallback_pending_gate_b`** |
| **`manifest_revision`** | **1** |
| `tie_break` | `deterministic_product_id_asc` |
| `terrier_parity_status` | `NOT_ATTEMPTED` |
| `degenerate_fraction` | 0.000069 |

**From Phase 2 onward: training and tuning may read only `train_task1` and `dev_task1`.
`test_task1` is run exactly once, after the model configuration is frozen.** If Gate B later
observes an IDCG=0 policy other than `exclude`, re-run `evaluate_task1_baselines.py` with it and
bump `manifest_revision` — though on this benchmark the change would be a no-op, since
`zero_idcg_query_count = 0`.

---

# 18. Deliverables

All 26 required files generated; none missing.

```
REPORT.md                            git_state_before.json     git_state_after.json
data_source_audit.json               gate_a.json               task1_dataset_counts.json
prereg_predictions.json              prereg_check.json         split_integrity.json
train_task1.parquet                  dev_task1.parquet         test_task1.parquet
feature_pool_dependency_audit.json   gain_convention_conflict.json
gain_convention_sensitivity.csv      scorer_unit_tests.json    terrier_parity.json
random_ranking_floor.json            oracle_ceiling.json       degenerate_query_analysis.json
task1_baseline_comparison.csv        locale_results.csv        benchmark_definition_comparison.md
phase2_training_pools.json           test_lock_manifest.json   known_query_text_collisions.json
per_query_scores.parquet             trec_eval_data/           _cache/
models/lambdamart_task1/{lambdamart_task1.txt, training_meta.json}
scripts/{audit_data_source, gate_a, gate_a_diagnostic, gate_a_operative, task1_common,
         dataset_counts_and_prereg, build_task1_semantic_scores, build_task1_benchmark,
         kdd_task1_ndcg, test_kdd_task1_ndcg, build_terrier_files, run_terrier_parity.sh,
         evaluate_task1_baselines, make_benchmark_comparison}
```

Additions beyond the required list: `known_query_text_collisions.json` (resolution item 4),
`per_query_scores.parquet`, `trec_eval_data/`, `_cache/` (recomputed BM25 and semantic scores,
regenerable), `gate_a_diagnostic.py`, `gate_a_operative.py`, `task1_common.py`,
`dataset_counts_and_prereg.py`, `build_task1_semantic_scores.py`, `make_benchmark_comparison.py`.

`audit_feature_pool_dependency.py` from the required script list was not created as a separate
file: the audit is produced inline by `build_task1_benchmark.py`, which is where the feature
computation lives, and emits the required `feature_pool_dependency_audit.json`. Splitting it would
have meant duplicating the feature code, which is the exact failure mode Phase 0 flagged in the
existing `advanced_features.py` / `evaluate_advanced.py` pair.
