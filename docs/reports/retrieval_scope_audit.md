# Retrieval Scope Audit — Two-Tower training scope, BM25 index scope, evaluation pool attribution

**Date:** 2026-09-03 · **Type:** static audit (code / config / persisted metadata only)
**Nothing was trained. Nothing was evaluated. No file outside this directory was modified.**
**No `split == "test"` data content was read** — only result-file metadata, config JSON, and
index/query-set manifests. The only code executed was three tokenizer calls on hand-typed
literal strings (§B.2, §A.3), which touch no dataset.

Every claim below cites `path:line`. Claims that could not be settled from code or config are
in §5 as UNKNOWN with the specific artifact needed to settle them.

---

# 1. Two-Tower training scope

## 1.1 Locale filter — the model is trained on `product_locale == 'us'` only

Two Two-Tower generations exist. **Both apply a hard US-only filter, in code, not by convention.**

**V0** — `scripts/train_two_tower.py`:

| Line | Code | Effect |
|---|---|---|
| [`scripts/train_two_tower.py:24`](../../scripts/train_two_tower.py#L24) | `LOCALE = "us"` | module constant |
| [`scripts/train_two_tower.py:39`](../../scripts/train_two_tower.py#L39) | `df = df[df["small_version"] == 1]` | Task-1 subset |
| [`scripts/train_two_tower.py:40`](../../scripts/train_two_tower.py#L40) | `df = df[df["split"] == "train"]` | train split |
| [`scripts/train_two_tower.py:41`](../../scripts/train_two_tower.py#L41) | `df = df[df["product_locale"] == LOCALE]` | **es/jp rows dropped** |
| [`scripts/train_two_tower.py:45`](../../scripts/train_two_tower.py#L45) | `df = df[df["esci_label"].isin(["E", "S"])]` | E/S positives |

**V1** — `scripts/train_two_tower_v2.py`, via the shared loader:

| Line | Code | Effect |
|---|---|---|
| [`retrieval/two_tower_training.py:40-41`](../../retrieval/two_tower_training.py#L40-L41) | `def load_positive_pairs(..., locale="us", small_version=1, split="train", labels=("E","S"))` | US-only is the **default argument** |
| [`retrieval/two_tower_training.py:48-51`](../../retrieval/two_tower_training.py#L48-L51) | `df = df[df["small_version"]==small_version]` … `df = df[df["product_locale"] == locale]` … | filter actually applied |
| [`scripts/train_two_tower_v2.py:85-86`](../../scripts/train_two_tower_v2.py#L85-L86) | `load_positive_pairs(EXAMPLES_PATH, PRODUCTS_PATH, train_queries, labels=cfg[...]["labels"])` | **`locale` is not passed** → default `"us"` is used |

Note the call at `train_two_tower_v2.py:85-86` overrides only `labels`; `locale`, `small_version`
and `split` fall through to the defaults at `two_tower_training.py:40-41`. The declared config at
[`scripts/train_two_tower_v2.py:52`](../../scripts/train_two_tower_v2.py#L52)
(`"positive_label_definition": {"labels": ["E","S"], "locale": "us", "small_version": 1, "split": "train"}`)
is **descriptive metadata only** — it is written to `config.json` but never passed into the loader.
It happens to agree with the defaults, so the recorded config is accurate; it is not what enforces
the filter.

The query universe the training pairs are drawn from is itself US-only, one layer earlier:
[`experiments/two_tower_v2/build_query_split.py:22`](../two_tower_v2/build_query_split.py#L22)
(`LOCALE = "us"`) and `:30-33` (`small_version==1`, `split=="train"`,
`product_locale==LOCALE`, `esci_label in ["E","S"]`). So es/jp queries never even enter the
train/dev query lists.

The V1 dev-time evaluator is also US-only:
[`retrieval/two_tower_training.py:97`](../../retrieval/two_tower_training.py#L97)
(`df_pr_us = df_products[df_products["product_locale"] == "us"]`) and `:109-112` (filler
distractors sampled from `title_lookup`, which is built from `df_pr_us`). Model selection —
`metric_for_best_model = "dev_cosine_recall@100"`
([`scripts/train_two_tower_v2.py:55`](../../scripts/train_two_tower_v2.py#L55)) — therefore
never saw an es or jp query either.

## 1.2 es / jp share of the training set: **0 pairs, 0.00%**

This is not an estimate. The locale filter is applied before pair construction in both
generations (`train_two_tower.py:41`; `two_tower_training.py:50`), so the count is zero by
construction.

Persisted corroboration:

- V1: [`experiments/two_tower_v2/splits/split_summary.json`](../two_tower_v2/splits/split_summary.json) —
  `"locale": "us"`, `"source_filter": {"small_version": 1, "split": "train", "product_locale": "us", "esci_label": ["E","S"]}`,
  `n_unique_queries_total: 20888`, `n_train_queries: 18800`, `n_train_pairs: 297112`,
  `n_dev_queries: 2088`, `n_dev_pairs: 32335`.
- V0: [`experiments/two_tower_v2/baseline/baseline_config.json`](../two_tower_v2/baseline/baseline_config.json) —
  `training.positive_label_definition = {"labels": ["E","S"], "locale": "us", "small_version": 1, "split": "train"}`,
  `num_training_pairs: 329447`. Its `_provenance` field states this was recorded by reading
  `scripts/train_two_tower.py` at commit `9b173e9`, not re-derived.

**Every es and jp number ever reported for Two-Tower is a zero-shot transfer measurement.**

## 1.3 Encoder identity — `sentence-transformers/msmarco-distilbert-base-v3`, monolingual English

Declared identically in both generations:
[`scripts/train_two_tower.py:22`](../../scripts/train_two_tower.py#L22) and
[`scripts/train_two_tower_v2.py:37`](../../scripts/train_two_tower_v2.py#L37) —
`MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v3"`.

Confirmed on the persisted artifact rather than from the name:

| Artifact | Field | Value |
|---|---|---|
| `models/two_tower_finetuned/README.md:10` | `base_model:` | `sentence-transformers/msmarco-distilbert-base-v3` |
| `models/two_tower_finetuned/config.json` | `model_type` / `architectures` | `distilbert` / `["DistilBertModel"]` |
| `models/two_tower_finetuned/config.json` | `vocab_size` | **30522** |
| `models/two_tower_finetuned/tokenizer_config.json` | `tokenizer_class` / `do_lower_case` | `BertTokenizer` / `true` |
| `models/two_tower_finetuned/README.md:8` | `dataset_size` | `329408` |

`vocab_size = 30522` with `BertTokenizer`/`do_lower_case=True` is the `bert-base-uncased`
English WordPiece vocabulary. It is **monolingual English**. MS MARCO, the pretraining
retrieval corpus, is English-only.

Measured consequence (tokenizer loaded from `models/two_tower_finetuned`, applied to three
hand-typed literal strings — no dataset read):

| Input | Tokens | `[UNK]` |
|---|---|---|
| `ワイヤレスイヤホン Bluetooth 高音質 ソニー` | `['[UNK]', 'blue', '##tooth', '高', '[UNK]', '[UNK]', '[UNK]']` | **4 / 7** |
| `zapatillas de running para hombre` | `['za','##pati','##llas','de','running','para','ho','##mbre']` | 0 / 8 |
| `wireless bluetooth earbuds` | `['wireless','blue','##tooth','ear','##bu','##ds']` | 0 / 6 |

Japanese text is largely **unrepresentable** in this vocabulary — most of it collapses to
`[UNK]`. Spanish survives tokenization (Latin script, WordPiece backs off to subwords) but the
encoder has no Spanish training signal at any stage. The es and jp cases are therefore *not*
the same kind of failure and should not be described with one phrase.

## 1.4 Query and product towers share weights (single-tower siamese)

One `SentenceTransformer` object is instantiated and both sides of the pair pass through it:

- V0: [`scripts/train_two_tower.py:73-74`](../../scripts/train_two_tower.py#L73-L74) —
  `model = SentenceTransformer(MODEL_NAME)`; `train_loss = losses.MultipleNegativesRankingLoss(model=model)`.
  Pairs are `InputExample(texts=[row["query"], row["item_text"]])`
  ([`:62-64`](../../scripts/train_two_tower.py#L62-L64)).
- V1: [`scripts/train_two_tower_v2.py:120-121`](../../scripts/train_two_tower_v2.py#L120-L121) —
  same construction; dataset columns `anchor`/`positive`
  ([`:91-94`](../../scripts/train_two_tower_v2.py#L91-L94)).

There is no second encoder anywhere. At serving time the same single model encodes both sides:
[`retrieval/two_tower.py:14`](../../retrieval/two_tower.py#L14)
(`MODEL_NAME = f'{ROOT_DIR}/models/two_tower_finetuned'`), `:84`, `:182`, `:236`.

**"Two-Tower" is a naming convention here, not an architecture claim** — it is one shared-weight
encoder applied twice (a siamese bi-encoder). Worth knowing before the term is used in an
interview.

## 1.5 V0 vs V1 — training locale scope is identical

| Dimension | V0 (`train_two_tower.py`) | V1 (`train_two_tower_v2.py`) | Same? |
|---|---|---|---|
| locale filter | `:24`, `:41` → `us` | `two_tower_training.py:40,50` → `us` (default) | **YES** |
| `small_version` | `:39` → 1 | `two_tower_training.py:41,48` → 1 | YES |
| split | `:40` → `train` | `two_tower_training.py:41,49` → `train` | YES |
| labels | `:45` → E/S | `train_two_tower_v2.py:86` → E/S | YES |
| base model | `:22` | `:37` | YES |
| loss | `:74` MNRL | `:121` MNRL | YES |
| batch size | `:25` → 64 | `:46` → 64 | YES |
| epochs | `:26` → 1 | `:46` → 3 | no (intended) |
| seed | none (`baseline_config.json: seed_note`) | `:51` → 42, `:77` `set_all_seeds` | no (intended) |
| warmup | `warmup_steps=10000` default, never reached | `:49` `warmup_ratio=0.10` | no (intended) |
| query-level split | none | `build_query_split.py` | no (intended) |
| dev evaluator | none (`baseline_config.json: evaluator_note`) | `:108-114` IR evaluator | no (intended) |
| **n train pairs** | 329,447 | 297,112 (10% held out as dev) | no (consequence of the split) |

**The V1 fix changed the training recipe. It did not change the training locale scope.**
The header comment at
[`scripts/train_two_tower_v2.py:8-11`](../../scripts/train_two_tower_v2.py#L8-L11) states this
explicitly ("Everything else … is held identical to V0"), and the code confirms it.

---

# 2. BM25 index scope

## 2.1 Three distinct BM25 constructions exist. None of them is per-locale.

| # | Builder | Corpus scope | On disk? | Feeds which reported number |
|---|---|---|---|---|
| **B-1** | [`retrieval/bm25.py:93-116`](../../retrieval/bm25.py#L93-L116) `build_global_bm25_index`, called by [`scripts/build_indices.py:15,23`](../../scripts/build_indices.py#L15-L23) | **ALL locales mixed**, whole products parquet, no filter | **NO** (see §2.3) | none |
| **B-2** | same function, called by [`scripts/build_full_catalog_indices.py:51,57`](../../scripts/build_full_catalog_indices.py#L51-L57) after `df_pr[df_pr['product_locale'] == LOCALE]` | **US only**, 1,215,854 products | yes, `output/full_retrieval/bm25s_index_us` | Recall@100 / RRF@100 (§3.2) |
| **B-3** | [`retrieval/bm25.py:13-46`](../../retrieval/bm25.py#L13-L46) `_score_single_group`, driven by `compute_bm25_scores` `:59-91`; and its transcription at [`experiments/ranking_v2/kdd_task1_benchmark/scripts/build_task1_benchmark.py:68-99`](../ranking_v2/kdd_task1_benchmark/scripts/build_task1_benchmark.py#L68-L99) | **one index per query**, over that query's own candidate set only | n/a (scores only) | NDCG 0.8198 (§3.1) |

**Answer to "three indices or one mixed index": neither.** The number the resume quotes
(BM25 NDCG full-list 0.8198) comes from **B-3**, which builds a fresh mini-index per query over
~23–40 candidates. `build_task1_benchmark.py:15-17` states this in its own header: *"retrieval/bm25.py
builds a BM25 index over THAT QUERY'S OWN candidate set — even the 'raw' BM25 is pool-dependent"*.
IDF and avgdl are computed within a single query's candidate set, so they carry no
collection-level statistics. Note the consequence: locale never mixes in B-3 either, because
each ESCI query's candidates are single-locale — but that is a property of the data, not of an
index design decision.

## 2.2 Analyzer — one English analyzer for all three locales, no locale-specific configuration

All three constructions call `bm25s.tokenize(...)` with **no keyword arguments**:
[`retrieval/bm25.py:23-24`](../../retrieval/bm25.py#L23-L24),
[`:110`](../../retrieval/bm25.py#L110),
[`:121`](../../retrieval/bm25.py#L121),
[`build_task1_benchmark.py:78-79`](../ranking_v2/kdd_task1_benchmark/scripts/build_task1_benchmark.py#L78-L79)
(only `show_progress=False` is passed).

The library defaults therefore apply to every locale
(`.venv/lib/python3.11/site-packages/bm25s/tokenization.py`, `def tokenize` signature):

| Parameter | Default in effect | Consequence for es / jp |
|---|---|---|
| `lower` | `True` | no-op for Japanese |
| `token_pattern` | `r"(?u)\b\w\w+\b"` | **regex word-run matching, ≥2 chars** |
| `stopwords` | `"english"` | **English stoplist applied to Spanish and Japanese** |
| `stemmer` | `None` | no stemming for any locale |

**There is no locale-specific analyzer configuration anywhere in the repo.** No `es`/`ja`
stopword list, no morphological segmenter (MeCab/Janome/Sudachi/fugashi is not in
`requirements.txt`), no CJK bigram fallback.

**Japanese tokenization — the exact behaviour, corrected.** Verified by calling
`bm25s.tokenize` on hand-typed literals:

| Input | Output tokens |
|---|---|
| `ワイヤレスイヤホン Bluetooth 高音質` | `['ワイヤレスイヤホン', 'bluetooth', '高音質']` |
| `zapatillas de running para hombre` | `['zapatillas', 'de', 'running', 'para', 'hombre']` |
| `wireless bluetooth earbuds` | `['wireless', 'bluetooth', 'earbuds']` |

Two things follow, and the second contradicts what several existing reports say:

1. **Japanese is not segmented.** `\w` matches CJK codepoints, so a whole unspaced Japanese run
   becomes exactly **one** token. `ワイヤレスイヤホン` ("wireless earphones") is a single term that
   will only ever match a title containing that identical unbroken run. This is the mechanism
   behind the observed `bm25_score` zero-rate of 61.9% on jp rows
   ([`experiments/ranking_v2/audit/REPORT.md:50`](../ranking_v2/audit/REPORT.md#L50)).
2. **It is regex-based, not whitespace-based.** Multiple reports describe BM25's jp handling as
   "whitespace tokenisation" / `str.split()` — that is inaccurate for BM25 (see the framing
   table, rows F-7 and F-8). The `str.split()` description *is* accurate for the **other**
   text features: [`reranking/advanced_features.py:92`](../../reranking/advanced_features.py#L92),
   [`:98`](../../reranking/advanced_features.py#L98),
   [`:141-142`](../../reranking/advanced_features.py#L141-L142) (`word_overlap`, IDF map, with a
   `PorterStemmer` at `:137` applied regardless of language), and
   [`:161-163`](../../reranking/advanced_features.py#L161-L163) (`color_match`). Two different
   tokenizers, two different defects, currently described as one.

3. **Spanish stopwords are not removed and Spanish is not stemmed.** `de` and `para` survive
   tokenization above. This inflates document length and dilutes IDF on es. It has not been
   quantified anywhere in the repo.

Dead code note: [`retrieval/bm25.py:48-56`](../../retrieval/bm25.py#L48-L56) defines
`simple_tokenize()`, which *is* `text.lower().split()`. `grep -rn simple_tokenize --include=*.py`
returns only its own definition — **it is never called**. It is a plausible source of the
"whitespace tokenisation" description in the reports, but it is not on any execution path.

## 2.3 Indexed product set

- **B-2 (the one that produced the Recall numbers):** US only, 1,215,854 products.
  [`scripts/build_full_catalog_indices.py:51`](../../scripts/build_full_catalog_indices.py#L51)
  `df_us = df_pr[df_pr['product_locale'] == LOCALE]`, with an assertion at `:53` that
  `product_id` is unique within the locale. Count corroborated by
  `experiments/two_tower_v2/baseline/baseline_config.json → serving.full_catalog_size_us_locale = 1215854`,
  by [`scripts/evaluate_two_tower_v2_full_catalog.py:7`](../../scripts/evaluate_two_tower_v2_full_catalog.py#L7),
  and by [`README.md:295`](../../README.md#L295).
  The stated rationale is at `build_full_catalog_indices.py:6-18`: the mixed-locale index stores
  no locale field per row, and 11,567 `product_id`s appear in more than one locale.
- **B-1 (mixed-locale, all ~1.8M products):** built by `build_indices.py`, saved by
  `retrieval/bm25.py:144` to `output/bm25s_index` + `output/bm25_ids.json`. **Neither path
  exists on disk** (`ls output/` — absent). Loaders that reference it —
  [`interactive_search.py:69`](../../interactive_search.py#L69) and
  [`tests/test_baseline_search.py:33`](../../tests/test_baseline_search.py#L33) — use the
  default path and would fail today; `scripts/mine_hard_negatives.py:57-60` explicitly overrides
  to the US index. **No reported number traces to a mixed-locale index.**
- **B-3:** no persistent index; the corpus is the query's own candidate set.

---

# 3. Evaluation scope and pool attribution

There are **three mutually incompatible pools** in this repo. The resume-relevant numbers come
from different ones and must not be stated in the same breath.

| Pool | Definition | Queries | Locale mix | Metric |
|---|---|---|---|---|
| **P1 · Task 1 (`task1_small_v1`)** | `test_task1.parquet`, `small_version==1` | 14,496 | **all three** — us 8,956 / es 2,417 / jp 3,123 | full-list NDCG |
| **P2 · large-version reference-candidate** | `small_version==0` (full ESCI), ESCI-provided candidates | 30,969 | **all three** — us 22,458 / es 3,844 / jp 4,667 | NDCG@10 / full-list |
| **P3 · full-catalog retrieval** | 1,215,854-product US catalog, open retrieval | 5,000 sampled | **US only, 100%** | Recall@100 |

## 3.1 The 0.8198 / 0.8245 / 0.8429 numbers → **P1, all three locales**

Source: [`experiments/ranking_v2/kdd_task1_benchmark/REPORT.md:11-12,17-22`](../ranking_v2/kdd_task1_benchmark/REPORT.md#L11-L22)
— pool `task1_small_v1`, `test_task1.parquet`, **336,373 rows / 14,496 queries, 0 excluded**,
full-list NDCG, gain E=1.0/S=0.1/C=0.01/I=0.0, scorer `kdd_task1_ndcg.py`, deterministic
tie-break. Locale composition is pinned in the frozen test manifest at
[`REPORT.md:634`](../ranking_v2/kdd_task1_benchmark/REPORT.md#L634):
`locale_breakdown | us 8,956 · jp 3,123 · es 2,417`, and repeated in the locale table header at
[`:458`](../ranking_v2/kdd_task1_benchmark/REPORT.md#L458).

**These are all-locale aggregate numbers.** Their per-locale decomposition
([`:458-463`](../ranking_v2/kdd_task1_benchmark/REPORT.md#L458-L463)):

| Model | US (8,956) | ES (2,417) | JP (3,123) | Overall (14,496) |
|---|---:|---:|---:|---:|
| BM25 | 0.819898 | 0.813825 | **0.824338** | 0.819842 |
| Two-Tower | 0.832508 | 0.818769 | 0.805964 | 0.824499 |
| LambdaMART | 0.844385 | 0.841310 | 0.839772 | 0.842878 |

Component provenance for those three rows:

- **BM25 `bm25_score`** — recomputed on the Task 1 pool via the per-query mini index
  ([`build_task1_benchmark.py:93-99,142-151`](../ranking_v2/kdd_task1_benchmark/scripts/build_task1_benchmark.py#L93-L151)).
- **Two-Tower `semantic_score`** — the **V0** encoder, frozen:
  [`build_task1_semantic_scores.py:46`](../ranking_v2/kdd_task1_benchmark/scripts/build_task1_semantic_scores.py#L46)
  `MODEL_PATH = os.path.join(ROOT, "models", "two_tower_finetuned")`, loaded at `:93-94`,
  recorded as `"model_frozen": True` at `:148`. Items keyed on `(product_id, product_locale)`
  at `:83-85`. **The V1 checkpoint was never scored on P1.**
- **LambdaMART** — retrained on `train_task1.parquet`
  ([`evaluate_task1_baselines.py:61,160-161`](../ranking_v2/kdd_task1_benchmark/scripts/evaluate_task1_baselines.py#L61-L161)),
  hyperparameters copied verbatim from `scripts/train_lambdamart.py`, nothing tuned (`:172`).
  **LambdaMART itself IS trained on all three locales.** Only its `semantic_score` input comes
  from a US-only encoder. This distinction matters for §4.

## 3.2 Recall@100 = 0.4649 and RRF@100 = 0.5301 → **P3, US-only, and NOT the Task 1 pool**

Source: [`experiments/two_tower_v2/phase1_correct_training/full_run/full_catalog_eval/v0_vs_v1_comparison.json`](../two_tower_v2/phase1_correct_training/full_run/full_catalog_eval/v0_vs_v1_comparison.json):

```
v0_two_tower_recall100: 0.4598   v1_two_tower_recall100: 0.4649
v0_rrf_recall100:       0.5194   v1_rrf_recall100:       0.5301
v1_checkpoint: .../full_run/checkpoints/checkpoint-9286
eval_query_sample: {size: 5000, seed: 42, locale: "us"}
```

Scope, from code:

- Catalog = **US only**, 1,215,854 products —
  [`scripts/evaluate_two_tower_v2_full_catalog.py:36,47-48`](../../scripts/evaluate_two_tower_v2_full_catalog.py#L36-L48)
  (`LOCALE = "us"`, `df_us = df_pr[df_pr['product_locale'] == LOCALE]`), header at `:7`.
- Query set = `output/full_retrieval/eval_query_ids.json`, loaded at
  [`:60-64`](../../scripts/evaluate_two_tower_v2_full_catalog.py#L60-L64).
- Ground truth = broad `{E,S,C}` — [`scripts/sample_eval_queries.py:28`](../../scripts/sample_eval_queries.py#L28),
  which is a **different relevance definition** from both the training positives (E/S,
  `train_two_tower_v2.py:52`) and the Task 1 NDCG gain vector.
- BM25 side of the RRF = the US-only index B-2
  ([`scripts/run_full_bm25_retrieval.py:35-37`](../../scripts/run_full_bm25_retrieval.py#L35-L37)).

**Confirmed: this is the large-version full-catalog pool, US-only. It is not P1.**
`0.4649` and `0.8245` are not comparable and never share a sentence.

## 3.3 The 5,000-query benchmark is 100% US

[`scripts/sample_eval_queries.py:26`](../../scripts/sample_eval_queries.py#L26) `LOCALE = "us"`;
[`:38`](../../scripts/sample_eval_queries.py#L38)
`df_test = df_ex[(df_ex['split'] == 'test') & (df_ex['product_locale'] == LOCALE)]`;
sampled at `:42-44` with `RandomState(42)`, `N_QUERIES = 5000` (`:25`).

Persisted manifest (`output/full_retrieval/eval_query_ids.json`, header fields only):
`{"seed": 42, "locale": "us", "n_requested": 5000, "n_available": 22458}`.

**Locale composition: us 5,000 / es 0 / jp 0.**

Note `n_available = 22458` exactly equals the us column of the **P2** locale table
([`official_metric_final/REPORT.md:219`](../ranking_v2/official_metric_final/REPORT.md#L219),
`us (22,458)`), and `sample_eval_queries.py:38` applies **no `small_version` filter** — so the
sample is drawn from the large-version test split, confirming P3 sits on the large-version
population.

This also means: **the query_slice_analysis findings, the retrieval-complementarity numbers, and
every Recall@100 result in this repo are US-only measurements.** They contain zero evidence
about es or jp. `experiments/query_slice_analysis/REPORT.md:3` states this correctly.

## 3.4 The 0.7280 / 0.7487 jp observation → **P2, large-version, all-locale**

[`experiments/ranking_v2/official_metric_final/REPORT.md:219-224,230-231`](../ranking_v2/official_metric_final/REPORT.md#L219-L231)
— locale table headed `us (22,458) | es (3,844) | jp (4,667) | overall (30,969)`. Two-Tower jp
0.727965 vs BM25 jp 0.748731; Two-Tower beats BM25 by +0.0149 on us. The `semantic_score` column
there is **frozen, not recomputed**
([`evaluate_official_baselines.py:229`](../ranking_v2/official_metric_final/scripts/evaluate_official_baselines.py#L229):
`"Two-Tower": "frozen semantic_score column, not recomputed"`), originating from
`output/two_tower_scores_test.csv` via
[`retrieval/two_tower.py:14,84`](../../retrieval/two_tower.py#L14) → V0 encoder.

So the jp inversion is observed on **P2**, while the Task 1 jp inversion (§3.1, 0.805964 vs
0.824338) is observed on **P1**. Same direction, different pools, different magnitudes. Reports
sometimes cite one and sometimes the other without naming the pool.

---

# 4. Framing audit

**Governing fact:** the Two-Tower encoder saw **0 es and 0 jp training pairs** (§1.1–1.2) and its
vocabulary cannot represent Japanese (§1.3). Its es/jp scores are zero-shot transfer of an
English model. Under the stated criterion, presenting "Two-Tower is weak on jp" as a *diagnostic
finding* is an over-claim; presenting it as *the expected consequence of the training scope* is
correct.

**Second-order fact:** LambdaMART **is** trained on all three locales (§3.1). Its US↔JP gap is
therefore a legitimate finding about the *feature stack* — but attributing that gap to a single
named cause ("tokenisation") is a separate, unsupported causal claim, because `semantic_score`
carries ~51% of model gain
([`benchmark_repair/REPORT.md:309`](../ranking_v2/benchmark_repair/REPORT.md#L309)) and is
itself the US-only encoder.

**Result: 18 statements reviewed — 6 OVER-CLAIM, 5 NEEDS-ANNOTATION, 7 OK.**

| # | File | Line | Original text (abridged) | Verdict | Suggested wording |
|---|---|---|---|---|---|
| F-1 | `experiments/ranking_v2/official_metric_final/REPORT.md` | 236-237 | "The metric was neither hiding nor manufacturing **the multilingual problem**; it is **a genuine feature/model deficiency**, exactly as the Phase 0 audit concluded." | **OVER-CLAIM** | "The metric was neither hiding nor manufacturing the locale gap. For Two-Tower the gap is not a deficiency finding — the encoder saw 0 es/jp training pairs (`train_two_tower.py:41`), so its es/jp scores are zero-shot English transfer and the gap is the expected result. Only the LambdaMART gap, whose model is trained on all three locales, is a genuine feature-stack finding." |
| F-2 | `experiments/ranking_v2/official_metric_final/REPORT.md` | 339-341 | "**JP tokenisation gap persists.** JP is 15.1% of test queries and LambdaMART scores 0.7727 there vs 0.8765 on US. A Cross-Encoder will **not** route around this unless it uses a multilingual encoder." | **OVER-CLAIM** | "**JP gap persists.** LambdaMART scores 0.7727 on jp vs 0.8765 on us. The causes are not separated: (a) `semantic_score` (~51% of model gain) comes from an English encoder with 0 jp training pairs; (b) BM25's tokenizer emits one token per unspaced Japanese run; (c) `word_overlap`/IDF/`color_match` use `str.split()` + PorterStemmer. Naming (b) alone as *the* cause is not supported by any ablation in this repo." |
| F-3 | `experiments/ranking_v2/benchmark_repair/REPORT.md` | 46-47 | "**The JP locale gap is confirmed and unchanged by the repair.** LambdaMART scores US 0.8732 / ES 0.8016 / JP 0.7682. **This is a feature-quality problem, not a benchmark artifact.**" | **OVER-CLAIM** | "…unchanged by the repair. It is not a benchmark artifact. It is partly a training-scope artifact: the dominant feature `semantic_score` comes from an encoder trained only on `product_locale=='us'`, so a jp deficit is expected, not diagnosed. The residual gap after that is a feature-quality question and has not been isolated." |
| F-4 | `experiments/ranking_v2/benchmark_repair/REPORT.md` | 356 | "…it will **not** route around **the JP tokenisation gap** unless it uses a multilingual encoder." | **OVER-CLAIM** | "…will not close the JP gap unless it uses a multilingual encoder — because the gap is driven primarily by an English-only encoder, not primarily by BM25 tokenisation. (Subsequently borne out: the multilingual V2 CE lifted jp @20 from 0.7477 to 0.8140, `competition_alignment/REPORT.md:175-181`.)" |
| F-5 | `experiments/ranking_v2/kdd_task1_benchmark/REPORT.md` | 479-480 | "**The JP problem did not go away; the headline cutoff hides it.**" | **OVER-CLAIM** | "The JP@10 deficit did not go away; the full-list cutoff compresses it. For Two-Tower this deficit is the expected signature of US-only training, not a discovered problem. Note also that BM25 — untrained — is the *best* jp model at full-list (`:461`), which is itself evidence that the jp deficit tracks trained components." |
| F-6 | `README.md` | 49 | "The dense encoder is based on `msmarco-distilbert-base-v3`, fine-tuned on the ESCI training split's **`small_version == 1` subset** using MultipleNegativesRankingLoss for one epoch (batch size 64) on E/S-labeled query-product pairs." | **OVER-CLAIM (omission)** | Add the locale filter, which is the single most consequential one and is currently the only filter omitted: "…fine-tuned on the ESCI training split's `small_version == 1`, **`product_locale == 'us'`** subset…". Without it, README §Setting 1 reports all-locale NDCG (`:132-133`) from a US-only-trained encoder with no disclosure. A reader cannot tell that es/jp are zero-shot. |
| F-7 | `experiments/ranking_v2/benchmark_repair/REPORT.md` | 333 | "**tokenisation is still whitespace-based for JP**" | **NEEDS ANNOTATION** (factually wrong for BM25) | "tokenisation is still locale-blind for JP: BM25 uses `bm25s.tokenize`'s default regex `(?u)\b\w\w+\b` with an **English** stoplist, which emits one token per unspaced Japanese run; `word_overlap`/IDF/`color_match` separately use `str.split()` + PorterStemmer (`advanced_features.py:92,98,141-142,161-163`). Two different tokenizers, two different defects." |
| F-8 | `experiments/ranking_v2/audit/REPORT.md` | 50 and 173-175 | ":50 `bm25_score` is 0 on 61.9% (**whitespace tokenisation** on a non-space-delimited language)"; ":173 Root causes: **whitespace tokenisation (`str.split()`) in the IDF map, `word_overlap`, BM25** and the color check" | **NEEDS ANNOTATION** (partly wrong) | The `str.split()` attribution is correct for the IDF map, `word_overlap` and `color_match`, and **incorrect for BM25**, which never calls `str.split()` — it uses the `bm25s` regex tokenizer. The 61.9% zero-rate is real; the stated mechanism for it is wrong. Also note `retrieval/bm25.py:48` `simple_tokenize()` *is* whitespace-based but is **never called** — likely the source of the confusion. |
| F-9 | `experiments/ranking_v2/kdd_task1_benchmark/REPORT.md` | 621-622 | "Phase 0 P1 data defects persist (`log_review_count` dead, price/stars parsing, **JP tokenisation**)" | **NEEDS ANNOTATION** | Same correction as F-7/F-8; and "JP tokenisation" should be split from "JP encoder coverage", which is a training-scope item, not a data defect. |
| F-10 | `experiments/ranking_v2/official_metric_final/REPORT.md` | 50-52 | "**The locale failure pattern is completely unchanged.** LambdaMART US 0.8765 / ES 0.8056 / JP 0.7727 (gap 0.1038). Two-Tower is still *worse than BM25* on JP (0.7280 vs 0.7487). The metric change was not masking or creating this." | **NEEDS ANNOTATION** | Accurate as a measurement and correctly scoped to the metric question. Add one clause so the summary is not read alone: "…(as expected: the Two-Tower encoder has no es/jp training data)." The full attribution exists 180 lines later at `:232-233` but not here, and this is the executive summary. |
| F-11 | `experiments/ranking_v2/benchmark_repair/REPORT.md` | 199-202 | "Two-Tower is the *worst* model on JP (0.7219, below even BM25's 0.7434) while being clearly the better of the two single signals on US — **exactly what Phase 0 predicted from an encoder fine-tuned only on `product_locale=='us'`**. Benchmark repair did not change this; **it is a modelling problem.**" | **NEEDS ANNOTATION** | The attribution clause is correct and well-sourced. Only the closing "it is a modelling problem" over-reaches — it is a *training-data-scope* decision, not a modelling defect. Suggest: "…it is a training-scope consequence, not a benchmark artifact and not a modelling defect." |
| F-12 | `experiments/ranking_v2/official_metric_final/REPORT.md` | 230-233 | "Two-Tower 0.727965 vs BM25 0.748731 … Meanwhile Two-Tower *beats* BM25 by +0.0149 on US. **This remains the clearest signature of an encoder fine-tuned only on `product_locale == 'us'`.**" | **OK** | No change. Explicitly names the training filter as the cause and calls it a *signature*, not a finding. This is the model the other passages should follow. |
| F-13 | `experiments/ranking_v2/kdd_task1_benchmark/REPORT.md` | 58-59, 485-487 | ":58 'Two-Tower is still worse than BM25 on JP … — **the US-only fine-tuning signature survives the benchmark change intact**'; :485 '**The US-only fine-tuning signature is unchanged** by the benchmark switch.'" | **OK** | No change. Correctly framed as a signature of training scope. |
| F-14 | `experiments/ranking_v2/benchmark_repair/REPORT.md` | 311-312 | "the JP result (Two-Tower 0.7219, *below* BM25's 0.7434) re-confirms that **the signal is strong only where the encoder was trained**." | **OK** | No change. Exactly the correct framing. |
| F-15 | `experiments/ranking_v2/audit/REPORT.md` | 53, 175, 240 | ":53 '(the encoder was fine-tuned on US-English only)'; :175 '…and a Two-Tower encoder fine-tuned exclusively on `product_locale=='us'`'; :240 'Two-Tower training set: `small_version==1 AND split=='train' AND product_locale=='us' AND …`'" | **OK** | No change. The Phase 0 audit states the training filter verbatim. It is the downstream reports that drop the qualifier. |
| F-16 | `experiments/ranking_v2/audit/REPORT.md` | 368-370 | "The one place the evidence is unambiguous is **`jp`/`es`**, where the US-English-only encoder degrades to Spearman 0.115 (jp) vs 0.268 (us). **A multilingual or locale-aware encoder is supported by evidence**" | **OK** | No change. States the cause, and frames the conclusion as a *design implication of the training scope*, not as a discovered model weakness. |
| F-17 | `README.md` | 178 | "Evaluated on a deterministic 5,000-query sample (seed=42) of the **22,458 US-locale test queries**, not the full set." | **OK** | No change. Correctly and explicitly scopes the Recall numbers to US. |
| F-18 | `experiments/query_slice_analysis/REPORT.md` | 3-4 | "Pipeline: `output/full_retrieval/*` (**5,000 US-locale ESCI test queries**, seed=42, Top-200 BM25 / Two-Tower retrieval, Recall@100 against `{E,S,C}` ESCI labels)." | **OK** | No change. Scope stated up front. |

**Pattern.** The Phase 0 audit (`ranking_v2/audit/REPORT.md`) and the two "signature" passages
(F-12, F-13, F-14) get this right. The qualifier is then **dropped in executive summaries and
next-step sections** (F-1, F-2, F-3, F-4, F-5), where the same observation is restated as a
diagnosis. F-6 is the highest-exposure instance: `README.md` is the public-facing document, it
reports all-locale NDCG, and it is the one place that lists the training filters and omits the
locale one.

**Non-framing discrepancy noted in passing** (out of scope, listed so it is not lost):
`README.md:10,174-176` reports the **V0** retrieval numbers (0.4542 / 0.4598 / 0.5194) while the
V1 numbers (0.4649 / 0.5301) are the current best
(`v0_vs_v1_comparison.json`). The README is one generation behind. Not a scope error.

---

# 5. UNKNOWN list

**5 items.** Each states exactly what artifact would settle it.

| # | Question | Why unresolved | What would settle it |
|---|---|---|---|
| U-1 | Does `models/two_tower_finetuned/` on disk correspond exactly to the `scripts/train_two_tower.py` run described in `baseline_config.json`? | `baseline_config.json` records `num_training_pairs: 329447`; the model card at `models/two_tower_finetuned/README.md:8` records `dataset_size: 329408`. **39-pair discrepancy, unexplained.** `baseline_config.json:_provenance` says it was recorded by *reading the script*, not by observing the run — so it is a re-derivation, not a log. No training log for V0 exists in the repo. | The V0 stdout/training log, or a `trainer_state.json` for that run. Neither is present (`models/two_tower_finetuned/` contains no `trainer_state.json`). Alternatively, re-running `train_two_tower.py`'s filter chain and printing `len(df)` — a data-loading operation, forbidden under this audit's no-execution constraint, and it reads only the *train* split so it would be permissible in a follow-up. |
| U-2 | Was the mixed-locale index (B-1, `output/bm25s_index`) ever built and used for any number now in a report? | The artifact does not exist on disk today, and `output/` was last written 2026-08-14. Absence now does not prove absence historically. Three call sites reference it (`interactive_search.py:69`, `tests/test_baseline_search.py:33`, and `retrieval/bm25.py:150`'s default). | `git log --diff-filter=A -- output/bm25s_index output/bm25_ids.json`, plus `output/full_retrieval/build_indices.log` (present, unread here). If B-1 was ever used for a *reported* metric, that metric mixed three locales in one index with no locale field — a correctness issue, not just a scope issue. |
| U-3 | What fraction of the jp NDCG deficit is attributable to (a) the US-only encoder vs (b) BM25 tokenisation vs (c) `str.split()`/PorterStemmer in `word_overlap`/IDF/`color_match`? | **No ablation isolating these exists anywhere in the repo.** §4 F-2/F-3 flag reports that assert one cause; none of them cite a decomposition. Gain shares (`semantic_score` ~51%) constrain but do not determine it. | A per-locale feature-ablation on the Task 1 pool: score LambdaMART on jp with (i) `semantic_score` dropped, (ii) `bm25_score` dropped, (iii) the `str.split()` features dropped, and compare deltas. `scripts/run_feature_ablation.py` and `scripts/run_group_ablation.py` exist and appear to do the general form of this; whether they support a per-locale cut was not verified. Requires execution — out of scope here. |
| U-4 | Is the Spanish degradation caused by the same mechanism as Japanese? | They are demonstrably **different** (§1.3, §2.2): Spanish tokenizes cleanly with 0 `[UNK]` under both the WordPiece and the BM25 tokenizer, whereas Japanese is 4/7 `[UNK]` and unsegmented. But Spanish still has 0 training pairs, an English stoplist applied to it, and no stemming. Which of those dominates is not established. | Per-locale zero-rate statistics for `bm25_score` and `word_overlap` on **es** specifically. `experiments/ranking_v2/audit/locale_feature_health.json` is cited at `audit/REPORT.md:160-162` with a us/es/jp table and was not opened during this audit; it likely answers this without any new computation. |
| U-5 | Do phases 2–7 of `experiments/two_tower_v2/` preserve the US-only scope? | Phase 3 (`phase3_hard_negative/REPORT.md:7`) and Phase 4 (`phase4_metadata/REPORT.md:7,38`) explicitly state US-locale. Phases 2, 5, 6, 7 and `semantic_score_ablation` returned **no `locale` match** in their REPORT.md files — silence, which under constraint 4 is not evidence of US-only scope. | Read the phase scripts themselves (not the reports) for their data-loading filters. Note these phases are ablations; none of them produced a checkpoint used by any number in §3, so the answer does not affect any reported metric — it affects only whether those phase reports are self-describing. |

---

# 6. Summary of what is established

1. **Two-Tower is US-only trained, in both V0 and V1**, enforced at
   `train_two_tower.py:41` and `two_tower_training.py:50` (default argument at `:40`,
   not overridden at `train_two_tower_v2.py:85-86`). es/jp pairs in training: **0**.
2. **The encoder is `sentence-transformers/msmarco-distilbert-base-v3` — monolingual English**,
   `vocab_size=30522` uncased WordPiece. Japanese is 4/7 `[UNK]`.
3. **Query and product towers share one set of weights** — a siamese bi-encoder, not two towers.
4. **BM25 is never per-locale.** The resume number comes from a **per-query mini index**; the
   Recall number comes from a **US-only global index**; a mixed-locale index exists in code but
   not on disk. One English analyzer (regex `\b\w\w+\b` + English stoplist, no stemmer) serves
   all three locales. Japanese unspaced runs become a single token.
5. **`0.4649` / `0.5301` belong to the US-only full-catalog pool (P3), not the Task 1 pool (P1).**
   The 5,000-query benchmark is 100% US.
6. **6 OVER-CLAIM statements, 5 needing annotation.** The dominant failure mode is a correct
   attribution stated once in the Phase 0 audit and dropped in every executive summary that
   restates the finding. `README.md:49` omits the locale filter entirely while `README.md:132-133`
   reports all-locale NDCG.
