# ESCI Ranking V2 — Phase 0: Baseline & Failure Audit

**Date:** 2026-09-02 · **Commit at audit start:** `9b173e9` · **Scope:** read-only.
No existing file was modified, no model retrained, no split changed, no feature added, no
hyperparameter touched. Everything produced by this audit lives under
`experiments/ranking_v2/audit/`.

Supporting artifacts: [repository_inventory.md](repository_inventory.md),
`audit_summary.json`, and the CSV/JSON files listed in §10.

---

# 1. Executive Summary

1. **The headline numbers are reproducible.** BM25 0.8188, Two-Tower 0.8267, MLP 0.8458 and
   LambdaMART 0.8464 were all re-derived to the exact decimal from the saved checkpoints and
   score files. The pipeline is deterministic and the metric code is a single shared
   implementation. In that narrow sense the baseline is trustworthy.
2. **But the benchmark itself is not clean, and the biggest problem is the metric, not the model.**
   LambdaMART is trained with `label_gain=[0,1,3,7]` (S is worth 43% of E) and scored with
   `2**{1.0, 0.1, 0.01, 0} - 1` (S is worth **7.2%** of E). The objective and the metric disagree
   by ~6× on the value of a Substitute. **P0.**
3. **This mismatch is not theoretical — it is the dominant failure mode.** 88.3% of the total
   NDCG@10 shortfall comes from queries containing at least one Substitute; 88.6% of the 7,282
   queries where LambdaMART *loses* to BM25 contain an S (base rate 66.4%). The worst losses are
   uniformly "1–2 Exact items buried under many Substitutes", e.g. `vida charging station`
   (2 E, 13 S): BM25 1.000 → LambdaMART 0.177.
4. **The evaluation candidate pool contains 44,217 duplicated rows (6.93%)**, caused by two
   locale-unaware `product_id` joins in `evaluation/evaluate_advanced.py` (lines 42 and 62).
   The pool has 682,233 rows against 638,016 real judged pairs. **P0.**
5. **The commonly quoted comparison mixes two different pools.** "LambdaMART 0.8464 vs BM25
   0.8188" pairs a number from the inflated pool with one from the clean pool. On a single clean
   pool the honest figures are BM25 0.8181 / Two-Tower 0.8267 / MLP 0.8476 / LambdaMART 0.8480.
   The conclusion survives, the arithmetic did not. **P0 (reporting).**
6. **semantic_score is NOT a leak, and it does NOT need OOF cross-fitting.** There is zero label
   leakage into test. There *is* large upstream train-on-train exposure (25.97% of us-locale E/S
   ranking-train rows were literal Two-Tower training positives, vs 0.004% of test rows) — but the
   resulting feature distribution mismatch is negligible: train/test means differ by 0.005, and
   Cohen's *d* for E-vs-I is 0.729 (train) vs 0.681 (test). This was the single most likely
   suspect going in, and the evidence clears it.
7. **21.1% of test queries (6,523) are degenerate** — every candidate carries the same label, so
   NDCG@10 = 1.0 for *any* ranking. They contribute a constant to every model's score and
   compress every reported gap. On the non-degenerate 78.9%, LambdaMART is 0.8054, not 0.8464.
8. **Three of the 17 features are broken by string-parsing bugs, not by weak signal.**
   `log_review_count` is identically **0.0 on 100% of all rows** (ESCI-S `ratings` is `"1,116 ratings"`,
   which `pd.to_numeric` turns into NaN → median 0 → `log1p(0)`). 15.9% of prices are European
   comma-decimals parsed **100× too large** (`"25,63€"` → `2563.0`). 13.7% of star ratings are
   truncated to integers (`"3,9 de 5 estrellas"` → `3.0`). **P1.**
9. **For 15% of the test set the entire feature stack is structurally dead.** On `jp` queries
   `word_overlap` is 0 on 72.2% of rows and `bm25_score` is 0 on 61.9% (whitespace tokenisation on
   a non-space-delimited language); `brand_match` fires on 3.1%; `color_match` on 0.06%; and
   `semantic_score`'s Spearman correlation with the label collapses to 0.115 vs 0.268 on `us`
   (the encoder was fine-tuned on US-English only). `jp` is 15.1% of queries but 23.1% of the
   total shortfall. **P1.**
10. **Model capacity is not the bottleneck.** On the clean pool LambdaMART − MLP = **+0.00043,
    95% CI [−0.00012, +0.00098], p = 0.126** — statistically indistinguishable, despite completely
    different inductive biases. They agree on 88.9% of top-10 slots. The evidence suggests the
    shared 17-feature representation is the binding constraint.

---

# 2. Verified Baseline

All numbers recomputed in this audit from saved artifacts. `n = 30,969` test queries in every row.

### 2.1 Official (as-shipped) evaluation pool — 682,233 rows

| Model | Artifact | NDCG@10 | Status |
|---|---|---:|---|
| BM25 | `output/bm25_scores_test.csv` | 0.8155 | **VERIFIED** (matches prior audit's 0.8155164) |
| Two-Tower | `output/two_tower_scores_test.csv` | 0.8247 | **VERIFIED** |
| MLP 17-feature | `output/best_advanced_reranker.pth` | **0.8458** | **VERIFIED** (matches the historical 0.8458) |
| LambdaMART 17-feature | `output/lambdamart_model.txt` (250 trees) | **0.8464** | **VERIFIED** (matches the historical 0.8464) |

### 2.2 Clean pool — 638,016 rows, locale-matched and de-duplicated, == the ESCI judged set

| Model | NDCG@10 | vs official pool |
|---|---:|---:|
| BM25 | 0.8181 | +0.0026 |
| Two-Tower | 0.8267 | +0.0020 |
| MLP 17-feature | 0.8476 | +0.0018 |
| LambdaMART 17-feature | **0.8480** | +0.0016 |

### 2.3 Historical numbers cross-checked

| Historical claim | Verdict | Evidence |
|---|---|---|
| BM25 ≈ 0.8188 | **VERIFIED, exactly** (0.8188130279308162) | clean all-locale pool, `scripts/build_query_slices.py` path |
| Two-Tower ≈ 0.8267 | **VERIFIED, exactly** (0.8267167793233932) | same path |
| MLP 17-feature ≈ 0.8458 | **VERIFIED** | official pool |
| LambdaMART 17-feature ≈ 0.8464 | **VERIFIED** | official pool |
| "LambdaMART without semantic_score ≈ 0.832" | **MISLABELLED** | 0.83193 is the **MLP** leave-one-out ablation (`output/ablations/no_semantic_score.json`, produced by `scripts/run_feature_ablation.py`, which retrains the MLP; its `full` row is 0.845779 = the MLP). **No LambdaMART semantic-ablation artifact exists in the repo.** |
| "BM25 ≈ 0.804" variant | **NOT VERIFIED** | No artifact in the repo produces 0.804. The nearest recomputation is BM25 under a binary E-vs-rest gain = 0.8041, but nothing links that to the historical figure. |
| semantic_score ≈ 55% gain / bm25 ≈ 23% / word_overlap ≈ 11% | **VERIFIED** | 54.99% / 22.73% / 11.28%, `feature_importance.csv` |
| `log_review_count` and `is_over_budget` at 0 gain | **VERIFIED, but the reason matters** | see §5 |
| `output/final_reranker_test_predictions.csv` | **NOT VERIFIED** | no script in the repo regenerates it; provenance unestablishable |

### 2.4 Gain-convention sensitivity (same rankings, three metrics)

| Convention | BM25 | Two-Tower | MLP | LambdaMART |
|---|---:|---:|---:|---:|
| Repo's `2**{1.0,0.1,0.01,0}−1` (official) | 0.8155 | 0.8247 | 0.8458 | 0.8464 |
| ESCI standard `2**{3,2,1,0}−1` = `[7,3,1,0]` | 0.8641 | 0.8760 | 0.8890 | **0.8900** |
| Binary E-vs-rest | 0.8041 | 0.8120 | 0.8354 | 0.8359 |
| Official convention, **non-degenerate queries only** (n=24,446) | 0.7663 | 0.7779 | 0.8046 | 0.8054 |

The ordering is stable across conventions, but the absolute level moves by up to 0.08. **These
NDCG@10 numbers are not comparable to any published ESCI leaderboard figure.**

---

# 3. Data / Split Risks

## P0 — must be fixed before the benchmark can be trusted

**P0-1 · Training objective and evaluation metric use different gain scales.**
`scripts/train_lambdamart.py` sets `LABEL_MAP_ORDINAL = {E:3,S:2,C:1,I:0}` and
`LABEL_GAIN = [0,1,3,7]`. `evaluation/metrics.apply_business_ndcg_labels` maps
`{E:1.0,S:0.1,C:0.01,I:0.0}` and `dcg()` applies `2**r − 1`, giving effective gains
`[I=0, C=0.00696, S=0.07177, E=1.0]`. S:E is **3/7 = 0.43 at train time vs 0.072 at eval time**.
The MLP has the same problem from the other direction — it trains on
`target_score = {1.0, 0.1, 0.01, 0.0}` with a *margin* loss, which is ordinal-only and ignores gain
magnitude entirely. Direct consequence in §7.

**P0-2 · The evaluation candidate pool contains 6.93% duplicated rows.**
`evaluation/evaluate_advanced.py:42` merges ESCI-S on `product_id` alone (11,309 duplicate asins)
and `:62` merges the products table on `product_id` alone (11,567 ids exist in >1 locale).
Result: 682,233 rows vs 638,016 real judged pairs; 12,982 pairs appear more than once (max 9
copies); 6,556 queries affected. A duplicated relevant item fills two adjacent top-10 slots in
both the actual and the ideal list, distorting those queries' NDCG. The identical defect exists
in the *training* feature builder (`reranking/advanced_features.py:51,70`), inflating the training
set to 2,119,685 rows.

**P0-3 · Published comparisons mix two different candidate pools.**
The 0.8188 BM25 figure comes from the clean pool; 0.8464 LambdaMART from the inflated one. The
gap between the two pools for BM25 is 0.0033 — about 11% of the headline LambdaMART−BM25 delta.
Fully attributable and now quantified (`evaluation_audit.json`), but it
must not be repeated in V2.

## P1 — real problems that affect performance or its interpretation

**P1-1 · `log_review_count` is identically zero on 100% of rows.** ESCI-S `ratings` is a string
(`"1,116 ratings"`, `"90 valoraciones"`). `pd.to_numeric(..., errors='coerce')` yields NaN for
**every** row, so the category median and the global median are both NaN, the final `.fillna(0.0)`
fires, and `log1p(0) = 0`. 91.8% of the catalog *does* have a parsable review count. The feature
has 0 gain because it is dead, not because popularity is uninformative — an important distinction
for V2 planning.

**P1-2 · 15.9% of prices are parsed 100× too large.** `str.replace(r'[^\d\.]','')` turns
`"25,63€"` into `"2563"`. Median parsed price on comma-decimal rows is **2299.0** vs **24.99** on
dot-decimal rows. Max parsed price is 52,925,293. This corrupts `log_price` (1.54% gain),
`is_over_budget`, and — worse — the `business_relevance` ground truth, whose budget penalty
compares `price` against `user_budget`.

**P1-3 · 13.7% of star ratings are truncated to integers.** `str.extract(r'([\d\.]+)')` on
`"3,9 de 5 estrellas"` returns `"3"`. 100% of comma-format rows come out as whole numbers; their
mean is 3.81 vs 4.35 for dot-format rows. Corrupts `stars_clean` (0.88% gain) and the star-boost
term in `business_relevance`.

**P1-4 · Non-English feature collapse.** `locale_feature_health.json`:

| | us | es | jp |
|---|---:|---:|---:|
| queries | 24,630 | 7,458 | 6,969 |
| `word_overlap` zero-rate | 19.1% | 27.6% | **72.2%** |
| `bm25_score` zero-rate | 15.6% | 19.1% | **61.9%** |
| `brand_match` fire rate | 11.1% | 10.9% | 3.1% |
| `color_match` fire rate | 2.9% | 0.23% | 0.06% |
| Spearman(`semantic_score`, label) | 0.268 | 0.229 | **0.115** |
| LambdaMART NDCG@10 | 0.8729 | 0.8013 | **0.7667** |
| share of total shortfall | 60.6% | 16.2% | **23.1%** |

Root causes: whitespace tokenisation (`str.split()`) in the IDF map, `word_overlap`, BM25 and the
color check; a `PorterStemmer` applied to Japanese; a hard-coded English color list; a `under $N`
budget regex; and a Two-Tower encoder fine-tuned exclusively on `product_locale=='us'`.

**P1-5 · 21.1% of test queries are degenerate** (6,523 of 30,969 — every candidate shares one
label, NDCG = 1.0 for any ranking). On the official pool LambdaMART is 0.8464 overall but 0.8054
on the 24,446 non-degenerate queries — the degenerate fifth adds ~+0.041 of pure constant and
shrinks every reported delta. Not a bug, but any V2 comparison should report the non-degenerate
subset alongside the full one.

**P1-6 · Small but real cross-split contamination.**
`query_id 79706` ("piano") appears in **both** train and test (3 train rows / 31 test rows,
disjoint products). Separately, 162 test queries (0.52%) have a normalized query string that also
appears in train under a different `query_id`; 83 `(normalized query, product)` pairs are shared
across splits, **13 of them with conflicting labels**. Within each split there are zero duplicate
`(query_id, product_id)` rows and zero label conflicts. `train_test` `query_id` overlap is 1.

**P1-7 · Transductive imputation.** Price/stars/ratings category medians and the
`is_dominant_category` top-20-BM25 mode are computed inside whichever split's frame is being
processed (`advanced_features.py:118-134`, `evaluate_advanced.py:105-121`), so test-time
imputation uses test-set statistics. The affected features carry <2.6% of total gain, so the
practical effect is small, but it is a genuine protocol defect.

**P1-8 · Label distribution shift between splits.**

| | train | test | delta |
|---|---:|---:|---:|
| E | 66.29% | 61.65% | −4.64 pp |
| S | 21.23% | 24.04% | **+2.81 pp** |
| C | 2.76% | 3.27% | +0.51 pp |
| I | 9.72% | 11.04% | +1.32 pp |

Test is materially more Substitute-heavy than train. Combined with P0-1, this pushes the model
toward exactly the regime where its objective is most miscalibrated. Candidate counts are close
(train mean 19.90 / median 16; test mean 20.60 / median 16), so the shift is in labels, not in
pool size.

## P2 — worth improving, not defects

- `extract_advanced_features` and `extract_test_advanced_features` are ~130 duplicated lines with
  nothing enforcing parity. They are currently in sync (verified: identical feature list and
  identical 53,267-entry idf_map in both saved metadata files).
- No persisted dev split. Both trainers regenerate a random 15%-of-train-queries validation split
  at fit time. Reproducible via `random_state=42`, but not inspectable and not shared between the
  two models' *selection* decisions in any auditable way.
- BM25 IDF is computed per-query over that query's own 16–40 candidates, not over a global corpus.
  This is a legitimate re-ranking choice but it makes `bm25_score` a *within-pool* signal that
  would not transfer to a production first-stage.
- Ties are broken by pandas' default (unstable) sort order: 34.2% of BM25-scored rows share their
  score with another row in the same query (7.3% for LambdaMART, 4.8% for MLP). No
  expected-NDCG-over-ties correction. Affects BM25's measured baseline most.
- `config.USE_SMALL_VERSION = False` while `train_two_tower.py` hard-codes `small_version == 1`.

## P3 — nice to have

- `config.USE_SPLIT = "test"` is a global that only `run_pipeline.py` reads; every other script
  hard-codes its split.
- `output/final_reranker_test_predictions.csv` (101 MB) has no regenerating script.

---

# 4. semantic_score Findings

Full lineage traced end to end
(`ranking row → output/two_tower_scores_{split}.csv → scripts/generate_two_tower_scores.py →
retrieval/two_tower.compute_two_tower_scores → models/two_tower_finetuned → scripts/train_two_tower.py`).

Two-Tower training set: `small_version==1 AND split=='train' AND product_locale=='us' AND
esci_label in {E,S}` → **329,447 positive pairs, 20,888 unique queries, 285,887 unique products**,
MultipleNegativesRankingLoss, 1 epoch.

### D1/D2 — exposure, measured separately for query / product / pair

| Exposure type | Ranking **train** rows | Ranking **test** rows |
|---|---:|---:|
| Same query text seen in TT training | 20.99% (20,932 queries) | **0.023%** (7 queries) |
| Same `query_id` seen | 20.95% | **0.00%** |
| Same product seen | 29.10% | 16.17% |
| Same query-product **positive pair** seen | 16.41% | **0.005%** |
| Pair exposure among E/S rows only | 18.77% | 0.006% |
| Pair exposure among **us-locale E/S rows** | **25.97%** | **0.004%** |

### D3 — distribution comparison (`semantic_score_distribution.csv`)

| | train | test |
|---|---:|---:|
| mean (all rows) | 0.5575 | 0.5527 |
| mean, label E | 0.6065 | 0.6043 |
| mean, label S | 0.4883 | 0.4953 |
| mean, label I | 0.4004 | 0.4125 |
| Cohen's *d*, E vs I | 0.729 | 0.681 |
| Spearman vs ordinal label | 0.245 | 0.238 |

Within the training split, E/S rows that *were* TT training pairs score 0.5940 vs 0.5738 for E/S
rows that were not — a **+0.020 gap, about 0.07 SD**. Memorisation is measurable but tiny, which
is expected: the score is min-max normalised **within each query's candidate set**, so absolute
inflation largely cancels.

### D4 — classification, using precise terms

| Category | Verdict |
|---|---|
| **A. True label leakage into test** | **NO.** The encoder never sees a test label, test `query_id`, or test pair. |
| **B. Upstream model saw the same pair** | **YES, extensive — but only on the ranking TRAIN side.** 25.97% vs 0.004%. `semantic_score` is an **in-fold / non-out-of-fold feature** for the ranker's training data. |
| **C. Train/eval feature distribution mismatch** | **PRESENT BUT NEGLIGIBLE.** Means differ by 0.005; discriminative power differs by ~7% relative (d 0.729 → 0.681). Not large enough to explain any material part of the train/test gap. |
| **D. Normal production feature** | **PARTLY.** The feature is legitimate; generating the *ranker's training values* with an encoder fitted on those same rows is not production-like. |

### Does semantic_score need OOF / cross-fitting?

**Not as a priority.** Cross-fitting exists to correct category-C distribution mismatch, and the
measured mismatch is 0.005 in mean and 0.05 in Cohen's *d*. The cost (k-fold re-fine-tuning of a
transformer, then re-scoring 2.1M train rows) is very high and the measurable upside is small.

Two caveats worth recording. First, the ranker's *learned reliance* on `semantic_score` may still
be modestly optimistic in a way an aggregate distribution check cannot see — a targeted test
would be to compare the ranker's NDCG on the 25.97%-exposed train subset against the unexposed
one. Second, this conclusion is specific to the current per-query min-max normalisation; if V2
switches to raw cosine, the exposure asymmetry could surface as absolute-scale inflation and this
analysis must be redone. **Recorded as P2, not P0.**

---

# 5. Feature Bottlenecks

### Where the information sits (`feature_importance.csv`)

| Rank | Feature | Group | Gain % | Cumulative % | Splits |
|---:|---|---|---:|---:|---:|
| 1 | `semantic_score` | semantic | 54.99 | 54.99 | 1,130 |
| 2 | `bm25_score` | lexical | 22.73 | 77.73 | 995 |
| 3 | `word_overlap` | lexical | 11.28 | **89.01** | 647 |
| 4 | `is_dominant_category` | interaction | 3.56 | 92.57 | 442 |
| 5 | `query_mean_idf` | query | 1.87 | **94.44** | 840 |
| 6–15 | brand_match … cheap_intent | | 5.56 | 100.00 | |
| 16 | `log_review_count` | popularity | **0.00** | 100.00 | 0 |
| 17 | `is_over_budget` | interaction | **0.00** | 100.00 | 0 |

**Top-1 = 55.0%, Top-3 = 89.0%, Top-5 = 94.4%.** The historical claim that
`semantic_score + bm25_score + word_overlap ≈ most of the gain` is **confirmed exactly**: 89.01%.

### Gain share ≠ marginal value — the most important nuance in this section

Leave-one-out ablations (`output/ablations/`, MLP retrains, full = 0.845779):

| Removed | NDCG@10 | Delta | Gain share |
|---|---:|---:|---:|
| `semantic_score` | 0.8319 | **−0.0138** | 55.0% |
| `bm25_score` | 0.8402 | −0.0056 | 22.7% |
| `word_overlap` | 0.8426 | −0.0032 | 11.3% |
| `brand_match` | 0.8445 | −0.0013 | 1.7% |
| `query_mean_idf` | 0.8455 | −0.0003 | 1.9% |

Removing the feature that holds 55% of the gain costs 1.6% of the score. Gain share measures how
often the trees split on a feature, not how much *unique* information it carries. The three top
features are correlated with each other (bm25–word_overlap 0.57, bm25–semantic 0.37,
semantic–word_overlap 0.31) and each largely substitutes for the others. **The 17-feature set has
one signal — "does this document lexically/semantically match this query" — expressed three ways.**

### Features with almost no information

- `log_review_count` — **constant 0.0 on 100% of rows.** Broken parser (P1-1), not a weak signal.
- `is_over_budget` — 0 on 99.95% of rows, 0 gain. The `under $N` regex is English-only.
- `user_budget` — −1 sentinel on 99.92% of rows, 0.005% gain.
- `cheap_intent` — 0 on 99.87% of rows, 0.001% gain.
- `color_match` — 0 on 98.15% of rows, 0.045% gain; fires on 0.06% of `jp` rows.
- `is_price_missing` / `is_rating_missing` — 0.11% / 0.29% gain. `is_rating_missing` is a pure
  artefact of the same broken `ratings` parse.

Five of 17 features carry <0.1% of gain combined; two carry exactly zero.

### Redundancy (`feature_correlation.csv`)

**No pair reaches |r| ≥ 0.90.** The maximum is `query_mean_idf` ↔ `query_max_idf` at 0.750
(train) / 0.634 (test) — same query-rarity concept measured two ways, together worth 2.2% of gain.
Next: `bm25_score` ↔ `word_overlap` 0.572, `query_length` ↔ `query_mean_idf` −0.522,
`log_price` ↔ `stars_clean` −0.417. So the set is not redundant in the linear-correlation sense;
its problem is *narrowness* (one concept, three encodings) rather than duplication.

### Answering the three prompted questions — from evidence only

- **Field-aware lexical signals — YES, evidence supports this.** `bm25_score` and `word_overlap`
  are computed against a single concatenated `title + description + bullet_point` blob
  (`generate_bm25_scores.py:30`) or title-only (`word_overlap`). There is no title-vs-description
  weighting, no coverage/IDF-weighted variant, no proximity or phrase signal. The two lexical
  features together hold 34% of gain, and removing both costs 0.0170 (`output/group_ablations/no_lexical.json`)
  — the largest group ablation. Lexical signal matters most and is the least developed.
- **Exact identifier / numeric / brand signals — YES, evidence supports this.** Numeric-heavy
  queries (contain a digit) score 0.8026 vs 0.8729 overall; Model/SKU queries 0.8296;
  Capacity/Storage queries 0.7849, the only slice where LambdaMART **loses** to BM25 (−0.010,
  n=122 — small sample, treat as directional). `brand_match` is a naive
  `brand.lower() in query.lower()` substring test with no placeholder filtering (the repo's own
  slice code had to blacklist `"vinyl"`, `"metal"`, `"glass"` as fake brands). There is no
  numeric/unit-aware matching at all.
- **Richer semantic signals — WEAKER EVIDENCE; do not over-claim.** `semantic_score` holds 55% of
  gain, but removing it costs only 0.0138, and Two-Tower alone (0.8267) beats BM25 alone (0.8181)
  by just 0.0087. The one place the evidence is unambiguous is **`jp`/`es`**, where the
  US-English-only encoder degrades to Spearman 0.115 (jp) vs 0.268 (us). A multilingual or
  locale-aware encoder is supported by evidence; "more/deeper semantic features for `us`" is not.

---

# 6. Model Capacity

| Comparison (clean pool, n=30,969) | Mean Δ | 95% CI | p | Win / Loss / Tie |
|---|---:|---|---:|---|
| LambdaMART − MLP | **+0.00043** | **[−0.00012, +0.00098]** | **0.126** | 7,198 / 6,855 / 16,916 |
| LambdaMART − BM25 | +0.02996 | [+0.02843, +0.03147] | <0.0001 | 12,361 / 7,282 / 11,326 |
| LambdaMART − Two-Tower | +0.02128 | [+0.02008, +0.02248] | <0.0001 | 11,321 / 7,047 / 12,601 |
| MLP − BM25 | +0.02953 | [+0.02799, +0.03103] | <0.0001 | 12,311 / 7,354 / 11,304 |
| Two-Tower − BM25 | +0.00867 | [+0.00676, +0.01059] | <0.0001 | 11,054 / 9,731 / 10,184 |

Comparability verified before comparing: identical dataset, identical split, byte-identical
17-feature list, identical 53,267-entry idf_map, identical query set, identical NDCG
implementation and relevance mapping. On the official (inflated) pool the LambdaMART−MLP delta is
+0.00064, CI [+0.00008, +0.00121], p=0.024 — i.e. **the 6.93% duplicate rows are the only reason
that comparison ever looked significant.**

Top-10 agreement (clean pool): mean Jaccard **0.889**, median 1.0, **50.3% of queries have an
identical top-10 set**, 3.1% have identical top-10 *ordering*. For reference, LambdaMART vs BM25
Jaccard is 0.619.

**What this means.** A 250-tree gradient-boosted ranker with a listwise lambdarank objective and a
17→64→32→1 MLP with a pairwise margin loss are two genuinely different function classes with
different losses and different regularisation. They land within 0.0004 NDCG of each other and
agree on ~89% of top-10 slots. *Evidence suggests* the binding constraint is the shared feature
representation rather than ranker capacity. This is not proof — neither model was capacity-swept,
and a much larger model was never tried — but it is the strongest evidence available from existing
artifacts, and it points away from "try a bigger ranker."

---

# 7. Failure Modes

Every figure from `failure_mode_decomposition.json` and
`per_query_ndcg_clean_pool.csv`. Total shortfall from perfect =
4,707.07 NDCG-points across 30,969 queries.

**FM-1 · Substitute-heavy queries — 88.3% of all shortfall.**
Queries with ≥1 S: 20,566 queries (66.4%), mean NDCG 0.798. Queries with no S: 10,403 queries,
mean NDCG 0.947. And within the 7,282 queries where LambdaMART loses to BM25: 88.6% contain an S
(base rate 66.4%), mean S count 7.44 (base 4.95), mean E count 10.69 (base 12.70).
The top-8 worst losses are all "few E, many S":

| query | locale | E | S | I | BM25 | LambdaMART | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| vida charging station | us | 2 | 13 | 1 | 1.000 | 0.177 | −0.823 |
| ｈショーツ | jp | 2 | 14 | 0 | 1.000 | 0.177 | −0.823 |
| 941m mercury | us | 2 | 0 | 14 | 1.000 | 0.177 | −0.823 |
| ポインタマウス | jp | 3 | 1 | 1 | 0.997 | 0.176 | −0.821 |
| 28 pulgadas tv sin smart | es | 1 | 35 | 4 | 0.942 | 0.156 | −0.786 |
| lightweight makeup organizer | us | 4 | 12 | 0 | 1.000 | 0.220 | −0.780 |

This is P0-1 made visible: the model was taught S is worth 43% of E and ranks accordingly; the
metric then charges it as if S were worth 7%. Missing the single E costs almost the whole score.

**FM-2 · Low-Exact-density queries.** Bucketing by `E_count / candidate_count`:

| E fraction | queries | % | LambdaMART NDCG | share of shortfall |
|---|---:|---:|---:|---:|
| ≤25% | 5,727 | 18.5% | **0.684** | **38.5%** |
| 25–50% | 5,628 | 18.2% | 0.744 | 30.6% |
| 50–75% | 6,220 | 20.1% | 0.823 | 23.4% |
| 75–99% | 6,872 | 22.2% | 0.949 | 7.5% |
| 100% | 6,522 | 21.1% | 1.000 | 0.0% |

37% of queries (E ≤ 50%) produce 69% of all shortfall.

**FM-3 · Non-English queries.** `jp` = 15.1% of queries, 23.1% of shortfall, NDCG 0.767;
`es` = 12.4% / 16.2% / 0.801; `us` = 72.5% / 60.6% / 0.873. See P1-4 for the mechanism.

**FM-4 · Large candidate sets.** Queries with 31–50 candidates: 18.0% of queries, **33.4% of
shortfall**, NDCG 0.719. Queries with ≤10 candidates: NDCG 0.909.

**FM-5 · Numeric / SKU / capacity queries.** Numeric-heavy 0.8026 (n=3,573); Model/SKU 0.8296
(n=865); Capacity/Storage 0.7849 (n=122) — the only slice where LambdaMART loses to BM25
(−0.0100). Small samples on the last one; directional only.

**FM-6 · What the ranker is actually good at.** The largest gains vs BM25 are queries where BM25
scores ~0 because the query tokens do not appear in the title — `dots for sporting events` (0.000
→ 1.000), `watch series 3` (0.015 → 1.000), `huawei g8` (0.031 → 0.969). `semantic_score` rescues
vocabulary mismatch. This is real and it is where the +0.030 over BM25 comes from.

---

# 8. Recommended Ranking V2 Roadmap

No expected NDCG numbers are given — none are supported by this audit.

### P0-A · Align the training objective with the evaluation metric

- **Problem.** LambdaMART optimises S:E = 0.43; the metric scores S:E = 0.072 (§3 P0-1).
- **Evidence.** 88.3% of shortfall and 88.6% of BM25-losses are on S-bearing queries; the worst
  losses are 1–2 E buried under many S (§7 FM-1).
- **Experiment.** Fix one gain convention and use it in both places. Recommended: the ESCI
  standard `relevance = {3,2,1,0}` with `2**r − 1 = [7,3,1,0]`, matching the existing
  `label_gain`. Re-evaluate all four baselines under it (already computed in §2.4 as a
  *metric-only* change) and then retrain LambdaMART and the MLP against it.
- **Information gain.** Separates "the model ranks badly" from "the model is scored against a
  different objective than it was trained on." Until this is resolved, no V2 feature result is
  interpretable.

### P0-B · Fix the candidate-pool joins and freeze one evaluation pool

- **Problem.** 44,217 duplicated rows (6.93%); baselines quoted across two different pools.
- **Evidence.** §3 P0-2/P0-3; the LambdaMART−MLP significance verdict flips (p=0.024 → p=0.126)
  purely from de-duplication.
- **Experiment.** New shared builder joining on `(product_id, product_locale)` and de-duplicating
  ESCI-S by asin; materialise the pool to a parquet once; make every evaluator read that file.
  Re-baseline all four models on it. (De-duplication alone is worth +0.0016 to +0.0026 — i.e. the
  measured pool effect, not a model improvement.)
- **Information gain.** A single, inspectable, correct benchmark. Prerequisite for everything else.

### P1-A · Fix the three ESCI-S parsers

- **Problem.** `log_review_count` dead on 100% of rows; 15.9% of prices 100× wrong; 13.7% of stars
  truncated (§3 P1-1..3).
- **Evidence.** `esci_s_parsing_audit.json`; 91.8% of the catalog has a
  parsable review count that the pipeline currently discards.
- **Experiment.** Locale-aware numeric parsing (strip thousands separators, treat `,` as decimal
  when it precedes exactly 2 digits). Re-extract features, retrain, re-evaluate. Cheap: LambdaMART
  refits in ~6s per the prior audit's timing.
- **Information gain.** Turns three dead/corrupt features into real ones and — importantly —
  repairs the `business_relevance` ground truth, which currently penalises budget on garbage prices.

### P1-B · Locale-aware text handling

- **Problem.** For 15% of test queries the lexical stack is ~dead and the semantic stack is
  halved (§3 P1-4).
- **Evidence.** jp `word_overlap` zero-rate 72.2%, `bm25_score` zero-rate 61.9%,
  Spearman(semantic, label) 0.115; jp carries 23.1% of shortfall on 15.1% of queries.
- **Experiment.** Two independent, separable arms: (i) a CJK-capable tokenizer for BM25, the IDF
  map and `word_overlap`; (ii) swap the Two-Tower base for a multilingual encoder, or train
  per-locale. Measure each locale separately — a global mean will hide both effects.
- **Information gain.** Isolates how much of the residual is a text-processing failure vs a
  modelling failure. Currently these are conflated.

### P1-C · Field-aware and identifier-aware lexical features

- **Problem.** One lexical concept, three encodings, no field structure, no numeric handling (§5).
- **Evidence.** Lexical group ablation is the largest (−0.0170); numeric-heavy 0.8026 vs 0.8729
  overall; Capacity/Storage is the only slice LambdaMART loses on.
- **Experiment.** Per-field BM25 (title / bullet / description / brand) as separate features;
  query-term coverage and IDF-weighted coverage; exact numeric+unit match ("128gb", "10.2 inch");
  a proper brand-dictionary match replacing the substring test. Add as a group and ablate as a
  group.
- **Information gain.** Directly tests whether the ceiling is "one match signal, three ways."

### P2-A · Report the non-degenerate subset alongside the headline

21.1% of test queries are NDCG=1.0 by construction. Reporting both numbers costs nothing and
materially increases the visible resolution of every V2 comparison (LambdaMART 0.8464 overall vs
0.8054 on the 24,446 non-degenerate queries, official pool).

### P2-B · Persist an explicit train/dev/test split

Materialise the 85/15 query split to disk so model selection is auditable and shared across
models. Also removes the 1 cross-split `query_id` and lets the 162 duplicate query strings be
handled deliberately.

### P2-C · Deferred: OOF semantic_score

Only if V2 changes the `semantic_score` normalisation away from per-query min-max, or if the
targeted exposed-vs-unexposed train-subset check (§4) shows a real gap. Current measured
distribution mismatch does not justify the cost.

### P3 · Housekeeping

De-duplicate the two copy-pasted feature builders into one; add a tie-breaking rule to
`ndcg_at_k`; document or delete `output/final_reranker_test_predictions.csv`.

---

# 9. What NOT to do next

- **Do not swap in a more complex ranker.** LambdaMART and the MLP differ by +0.00043
  (CI [−0.00012, +0.00098], p=0.126) and agree on 88.9% of top-10 slots despite different
  objectives and function classes. There is no evidence that capacity is binding.
- **Do not bulk-add dozens of features.** The current set already has 5 features under 0.1% gain
  and 2 at exactly zero, and no pair exceeds |r|=0.90 — the problem is narrowness of *concept*,
  not count. Add the field-aware/identifier group (P1-C) and ablate it as a group; do not shotgun.
- **Do not build OOF / cross-fitted semantic_score yet.** Train/test `semantic_score` means differ
  by 0.005 and Cohen's *d* by 0.05. The most expensive possible fix targets the smallest measured
  problem.
- **Do not touch retrieval.** This benchmark re-ranks the ESCI judged set — `TOP_K=150` binds on 0
  test queries. Retrieval recall is a real and separately-documented problem
  (`output/full_retrieval/`, ~58% union Recall@100) but it is a **different task** and changing it
  will not move this NDCG@10 by construction.
- **Do not go end-to-end (cross-encoder / LLM reranker).** No evidence has been gathered on it
  here, and it would be built on a benchmark whose objective and metric currently disagree by 6×
  and whose candidate pool is 6.93% duplicated. Fix P0-A and P0-B first, or the result will be
  uninterpretable.
- **Do not chase the Capacity/Storage or Size slices on current evidence.** n=122 and n=103. The
  direction is interesting; the sample is not enough for a conclusion.
- **Do not treat the 0.8464 → published-ESCI comparison as meaningful.** The repo's gain
  convention is non-standard; the same rankings score 0.8900 under the standard one.

---

# 10. Deliverables

All under `experiments/ranking_v2/audit/`. Every listed deliverable was generated; none is
`NOT AVAILABLE`.

| File | Contents |
|---|---|
| `REPORT.md` | this document |
| `repository_inventory.md` | component table + baseline provenance |
| `audit_summary.json` | machine-readable consolidation of every finding |
| `split_integrity.json` | B1/B2 overlaps, duplicates, label-gain consistency verdict |
| `label_distribution.csv` | E/S/C/I counts and shares, raw-judgment and ranking-pool universes |
| `candidate_distribution.csv` | candidates & relevants per query: mean/std/min/p1/p5/p25/median/p75/p95/p99/max |
| `semantic_score_distribution.csv` | by split, by label, and by TT-exposure |
| `semantic_audit.json` | full lineage, D1–D4 |
| `feature_audit.csv` | 17 features × 2 splits, all requested fields + flags |
| `feature_correlation.csv` | all 136 pairs, train and test |
| `feature_importance.csv` | gain, gain %, cumulative %, split count |
| `evaluation_audit.json` | C0–C8, pool integrity, BM25 reconciliation |
| `eval_gain_convention_sensitivity.csv` | 4 models × 3 gain conventions |
| `eval_nondegenerate_subset.csv` | 4 models on the 24,446 non-degenerate queries |
| `model_comparison.csv`, `model_comparison_pairwise.csv` | headline + pairwise deltas, both pools |
| `model_capacity_audit.json` | top-10 Jaccard / identical-top-10 |
| `query_slice_results.csv` | 19 slices × 4 models (us-locale — the pre-existing rules are English-only) |
| `per_query_ndcg_clean_pool.csv` | all 30,969 queries × 4 models + metadata |
| `worst_queries.csv`, `largest_losses_vs_bm25.csv`, `largest_gains_vs_bm25.csv`, `largest_losses_vs_two_tower.csv`, `largest_semantic_gains.csv` | top-200 each |
| `failure_mode_decomposition.json` | shortfall attribution by locale / label mix / candidate count |
| `locale_feature_health.json` | per-locale feature firing rates and label correlations |
| `esci_s_parsing_audit.json` | the three parser defects, quantified |
| `bootstrap_results.json` | 6 comparisons × 2 pools, n=10,000, seed=42 |
| `scripts/00–06` | every script used, re-runnable |
| `_cache/` | 276 MB of regenerable intermediates (scored test frame, train feature frame) so scripts 02–06 re-run in seconds instead of ~30 min. Safe to delete; `00_build_caches.py` rebuilds them. |

**Method note on scripts.** `00_build_caches.py` imports and calls the repo's own
`extract_advanced_features` / `extract_test_advanced_features` unchanged and loads the saved
`lambdamart_model.txt` and `best_advanced_reranker.pth` for inference only. No model was retrained
at any point in this audit. Bootstrap methodology (query-level paired percentile, n=10,000,
seed=42) is the pre-existing one from `experiments/audit/task3_per_query_bootstrap.py`.
