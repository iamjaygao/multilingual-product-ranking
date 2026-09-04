# Repository & Baseline Inventory — ESCI Ranking

Audit date: 2026-09-02 · Commit at audit start: `9b173e9`
Every entry below was read from the repository. Nothing is assumed from the task brief.

---

## 1. Component table

| Component | File / Artifact | Purpose | Status |
|---|---|---|---|
| Global config | `config.py` | `EXAMPLES_PATH`, `PRODUCTS_PATH`, `USE_SMALL_VERSION=False`, `USE_SPLIT="test"`, `TOP_K=150` | ACTIVE |
| Raw judgments | `esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet` | 2,621,288 rows; splits `train`/`test` only; locales `us`/`es`/`jp` | ACTIVE |
| Raw catalog | `esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet` | 1,814,924 rows / 1,802,772 unique `product_id` (11,567 ids in >1 locale) | ACTIVE |
| Product attributes | `esci-data/esci-s_dataset/esci_s_products.parquet` | `asin, price, stars, ratings, category`; 1,661,908 rows / 1,650,599 unique asins (11,309 dup rows) | ACTIVE — **all four columns are free-text strings, see §5** |
| BM25 scorer | `retrieval/bm25.py` → `compute_bm25_scores()` | Per-query mini-index over that query's own candidates (`bm25s`), min-max normalised in-query, truncated to `TOP_K=150` | ACTIVE |
| BM25 score generation | `scripts/generate_bm25_scores.py` | Writes `output/bm25_scores_train.csv` (1,983,161 rows) and `output/bm25_scores_test.csv` (638,016 rows) | ACTIVE |
| Two-Tower encoder | `models/two_tower_finetuned/` | SentenceTransformer, base `sentence-transformers/msmarco-distilbert-base-v3` | ACTIVE — **this is the `semantic_score` source** |
| Two-Tower training | `scripts/train_two_tower.py` | MultipleNegativesRankingLoss, 1 epoch, batch 64, filter `small_version==1 AND split=='train' AND product_locale=='us' AND esci_label in {E,S}` → **329,447 positive pairs / 20,888 unique queries / 285,887 unique products** | ACTIVE |
| Two-Tower scoring | `retrieval/two_tower.py` → `compute_two_tower_scores()` + `scripts/generate_two_tower_scores.py` | Cosine similarity, min-max normalised **within each query's candidate set**, `TOP_K=150`; writes `output/two_tower_scores_{train,test}.csv` | ACTIVE |
| Ranking feature builder (TRAIN) | `reranking/advanced_features.py` → `extract_advanced_features()` | The 17-feature set + `target_score`; `ALL_FEATURES` at line 12 is the single source of truth | ACTIVE |
| Ranking feature builder (TEST) | `evaluation/evaluate_advanced.py` → `extract_test_advanced_features()` | Hand-duplicated copy of the train builder for the test split | ACTIVE — **code duplication, see §5** |
| LambdaMART training | `scripts/train_lambdamart.py` | `LGBMRanker(objective="lambdarank", label_gain=[0,1,3,7], n_estimators=1000, lr=0.05, num_leaves=31, random_state=42)`, early stopping 50 on an internal 85/15 query split | ACTIVE |
| LambdaMART model | `output/lambdamart_model.txt` | 250 trees (early stopping fired) | ACTIVE — **official LambdaMART baseline** |
| LambdaMART feature metadata | `output/lambdamart_features.json` | `{"features": [17 names], "idf_map": {53,267 words}}` | ACTIVE |
| MLP training | `scripts/train_adv_reranker.py` | Pairwise `MarginRankingLoss(margin=1.0)`, AdamW lr=1e-4, 85/15 query split seed 42, early stopping patience 6 | ACTIVE |
| MLP model | `output/best_advanced_reranker.pth` + `reranking/advanced_model.py` (`AdvancedDeepReranker`, 17→64→32→1) | | ACTIVE — **official MLP baseline** |
| MLP normalisation stats | `output/advanced_normalization_stats.json` | `mean`, `std`, `features`, `idf_map` — verified byte-identical feature list and idf_map to `lambdamart_features.json` | ACTIVE |
| NDCG implementation | `evaluation/metrics.py` → `dcg()`, `ndcg_at_k()` | `2**relevance - 1`, grouped by `query_id`, unweighted mean, skips IDCG==0 queries | ACTIVE — **the single NDCG implementation; every evaluator calls it** |
| Relevance mapping | `evaluation/metrics.py` → `apply_business_ndcg_labels()` | `{E:1.0, S:0.1, C:0.01, I:0.0}` → `relevance`; plus a separate `business_relevance` with budget penalty + star boost | ACTIVE |
| LambdaMART evaluator | `evaluation/evaluate_lambdamart.py` | **Official LambdaMART NDCG@10 script** | ACTIVE |
| MLP evaluator | `evaluation/evaluate_advanced.py::main()` | **Official MLP NDCG@10 script** | ACTIVE |
| Older MLP evaluator | `evaluation/evaluate_reranker.py` + `reranking/features.py`, `reranking/model.py`, `output/best_esci_reranker.pth` | Earlier, smaller feature set | SUPERSEDED |
| Retrieval evaluator | `evaluation/evaluate_retrieval.py`, `scripts/evaluate_full_retrieval.py` | Full-catalog Recall@K — a **different task** from ranking | ACTIVE but out of scope |
| Feature ablation | `scripts/run_feature_ablation.py` → `output/ablations/` | Leave-one-out; **retrains the MLP**, full=0.845779 | ACTIVE — often mis-cited as LambdaMART ablations |
| Feature-group ablation | `scripts/run_group_ablation.py` → `output/group_ablations/` | Same, by feature group | ACTIVE |
| Query slicing | `scripts/build_query_slices.py` → `output/query_slices/` | Pre-registered brand/model/attribute/IDF slice rules, **`product_locale=='us'` only**, 22,458 queries | ACTIVE — reused by this audit |
| Top-k overlap | `scripts/analyze_topk_overlap.py` → `output/ranking_overlap/` | Rank-agreement between models | ACTIVE |
| Prior audit | `experiments/audit/` (`REPORT.md`, `bootstrap_results.json`, `per_query_ndcg.csv`, `feature_importance.csv`, …) | Earlier audit: feature importance, 12-feature retrain, BM25-vs-LambdaMART bootstrap, win/loss/tie, recall denominators | ACTIVE — verified and extended here |
| Two-Tower V2 experiments | `experiments/two_tower_v2/` | A separate retrieval-side effort with its own train/dev query split | **NOT part of the ranking baseline** |
| Bootstrap implementation | `experiments/audit/task3_per_query_bootstrap.py` | Query-level paired percentile bootstrap, n=10,000, seed=42 | REUSED (methodology) by this audit |

---

## 2. Direct answers to the inventory questions

**Where is the official ranking dataset?**
There is no materialised ranking dataset file. It is constructed on the fly:

```
output/bm25_scores_{split}.csv   OUTER JOIN  output/two_tower_scores_{split}.csv
    INNER JOIN  ESCI examples[split][query_id, query]
    INNER JOIN  products  ON product_id            <-- locale-unaware
        (products was itself LEFT JOINed to esci_s ON product_id)   <-- duplicate asins
    LEFT  JOIN  ESCI labels ON (query_id, product_id)
```

- Train: **2,119,685 rows / 99,684 queries** (`reranking/advanced_features.py`)
- Test: **682,233 rows / 30,969 queries** (`evaluation/evaluate_advanced.py`)

The candidate pool is effectively the full ESCI judged set — `TOP_K=150` binds on 4 train
queries and 0 test queries. This is a **re-ranking-of-judgments** benchmark, not a
re-ranking-of-retrieval-output benchmark.

**What are the 17 features?**
`reranking/advanced_features.ALL_FEATURES`, in model column order:
`query_length, query_mean_idf, query_max_idf, user_budget, cheap_intent, log_price,
is_price_missing, stars_clean, log_review_count, is_rating_missing, bm25_score,
semantic_score, word_overlap, is_dominant_category, brand_match, color_match, is_over_budget`.
Confirmed identical in `output/lambdamart_features.json` and
`output/advanced_normalization_stats.json`.

**What is the official split?**
The ESCI-shipped `split` column: `train` (99,684 queries) / `test` (30,969 queries).
**There is no dev split.** Both trainers carve a fresh random 15% of *train queries*
(`sklearn.train_test_split`, `random_state=42`) as an internal early-stopping validation
set; that split is never persisted. Model selection and reporting therefore both ultimately
reference the same two files.

**LambdaMART baseline** — config `scripts/train_lambdamart.py`, model `output/lambdamart_model.txt`
(250 trees), metadata `output/lambdamart_features.json`, evaluator `evaluation/evaluate_lambdamart.py`.

**MLP baseline** — config `scripts/train_adv_reranker.py`, weights `output/best_advanced_reranker.pth`,
architecture `reranking/advanced_model.AdvancedDeepReranker`, stats
`output/advanced_normalization_stats.json`, evaluator `evaluation/evaluate_advanced.py`.

**NDCG@10 evaluation script** — one implementation, `evaluation/metrics.ndcg_at_k`, called by
`evaluate_lambdamart.py`, `evaluate_advanced.py`, `evaluate_reranker.py`, `run_pipeline.py`.
`scripts/build_query_slices.per_query_ndcg` re-implements the same loop around the same
`dcg()` and produces identical numbers on identical input. **The metric code is consistent;
the candidate pools passed into it are not** (see REPORT.md §3).

**semantic_score source** — `models/two_tower_finetuned` → `retrieval/two_tower.compute_two_tower_scores`
→ `output/two_tower_scores_{train,test}.csv`. Per-query min-max-normalised cosine similarity.

**bm25_score source** — `retrieval/bm25.compute_bm25_scores` → `output/bm25_scores_{train,test}.csv`.
Per-query min-max-normalised BM25 from a per-query mini-index (**not** a global corpus index;
IDF is computed over the ~16–40 candidates of that one query).

---

## 3. Artifacts that exist but do NOT correspond to the current ranking baseline

| Artifact | Why not |
|---|---|
| `output/best_esci_reranker.pth`, `output/normalization_stats.json`, `reranking/features.py` | Earlier reranker generation, superseded by the "advanced" 17-feature one |
| `output/final_reranker_test_predictions.csv` | Written 2026-08-13; no script in the repo regenerates it and its provenance cannot be established from the repository → **NOT VERIFIED** |
| `experiments/two_tower_v2/**` | Retrieval-side V2 effort with its own dev split; its checkpoints are not used by `semantic_score` |
| `output/full_retrieval/**` | Full-catalog Recall@K on a 5,000-query US sample — different task |
| `checkpoints/model` | Not referenced by any ranking script |
| `analysis (not used)/` | Directory name states it |

---

## 4. Prior audit artifacts — verification status

`experiments/audit/` was re-derived independently in this audit:

| Prior claim | This audit | Status |
|---|---|---|
| LambdaMART NDCG@10 = 0.8464228 | 0.8464 (exact match) | CONFIRMED |
| BM25 on LambdaMART pool = 0.8155164 | 0.8155 (exact match) | CONFIRMED |
| BM25 all-locale clean = 0.8188130 | 0.8188130 (exact match) | CONFIRMED |
| Two-Tower all-locale clean = 0.8267168 | 0.8267168 (exact match) | CONFIRMED |
| Feature gain table (semantic 54.99%, bm25 22.73%, word_overlap 11.28%) | identical to 2 d.p. | CONFIRMED |
| LambdaMART − BM25 = +0.0309, CI [0.0293, 0.0326] | +0.03091, CI [0.02928, 0.03246] | CONFIRMED |
| win/loss/tie 12,249 / 7,248 / 11,472 | identical | CONFIRMED |
| "Pre-existing `product_id`-only join issue" flagged in `analyze_topk_overlap.py` | Quantified here for the first time: 44,217 duplicated rows (6.93%) | EXTENDED |

---

## 5. Structural issues found in the code during inventory

1. **`extract_advanced_features` and `extract_test_advanced_features` are copy-pasted**
   (~130 duplicated lines). They are currently in sync, but nothing enforces that;
   a future feature edit applied to one and not the other silently breaks train/test parity.
2. **Two locale-unaware `product_id` joins** (`evaluate_advanced.py:42` and `:62`) inflate the
   candidate pool by 6.93%.
3. **All four ESCI-S attribute columns are free-text strings** in mixed locales
   (`"$9.99"`, `"25,63€"`, `"4.3 out of 5 stars"`, `"3,9 de 5 estrellas"`, `"1,116 ratings"`).
   The parsers in the feature builders handle only the English forms — see REPORT.md §3 P1-1..P1-3.
4. **No persisted dev split** for the ranking pipeline.
5. `config.USE_SMALL_VERSION = False` while `scripts/train_two_tower.py` hard-codes
   `small_version == 1` — the encoder is fit on a subset and applied to the full dataset.
