# Prior work — results cited here, components that stayed behind

This repository is the personally-authored subset of an earlier project,
`esci-search-ranking-system`, which began as a **three-person university course
project**. The ownership boundary was established by auditing git history
(`git log --diff-filter=A` for creation, `git blame --line-porcelain` for current
line ownership) and is recorded in `MIGRATION_PLAN.md` §7.

Two kinds of thing are documented here:

1. **§3.9 DOCUMENT_ONLY results** — numbers produced in the prior project that
   are cited but whose code did not migrate. Each says who wrote the component
   that produced it.
2. **Components that stayed behind** — the modules this repository deliberately
   does not contain, and why. Reports elsewhere in `docs/` link here instead of
   to a dead path.

Nothing in this file is a claim of authorship over teammate work. Where a
result rests on a teammate's implementation, that is stated in the same
sentence as the number.

---

## Contributors

| Identity in git | Commits | Role |
|---|---:|---|
| `hyukjin17` / `jin` / `Hyuk Jin Chung` <chung.hy@…> | 65 | teammate |
| `gyuszix` <gyuszix@…> | 31 | teammate |
| `Jian Gao` / `iamjaygao` | 21 | author of this repository |

---

# Part 1 — DOCUMENT_ONLY results (§3.9)

## 1.1 The `BatchNorm` / pairwise-training diagnosis

**Result.** The 7-feature baseline reranker originally scored **NDCG@10 0.7822**
— *worse than no reranking at all* (BM25 baseline 0.8188). After the fix it
scored **0.8444**.

**Mechanism.** Positives and negatives are forwarded through the model in two
separate batches during pairwise training, so `BatchNorm` normalises each side
against its own batch statistics. The network can separate the two for free
using the normalisation itself — a trick that vanishes at `eval()`, when
`BatchNorm` switches to running population statistics. `LayerNorm` normalises
per-sample and has no train/eval discrepancy, which fixed it outright.

**Attribution — split.**

| Part | Who | Evidence |
|---|---|---|
| Original `DeepESCIReranker` implementation, including the `BatchNorm` | **gyuszix (teammate)** | created `b1e3776` (2026-03-03); `git log -S BatchNorm -- reranking/model.py` returns that commit only |
| Root-cause diagnosis and the `LayerNorm` fix | **Jian Gao** | `5210986` (2026-08-17); `git log -S LayerNorm -- reranking/model.py` returns that commit only |

The model class is a teammate's work. The diagnosis is mine. At HEAD of the
source repo the file was 16 lines gyuszix / 15 lines Jian Gao — but every one of
those 15 lines came from the fix commit, and line-count parity does not transfer
ownership of the original implementation.

**Why the code is not here.** `MIGRATION_PLAN.md` §7.3-3 decision **A**: a
7-feature pairwise MLP is not part of a multilingual ranking benchmark, and the
asset is the diagnosis, which is prose. Migrating the class would create exactly
the ownership ambiguity the boundary exists to prevent.

## 1.2 LambdaMART 0.8464 vs advanced MLP 0.8458

**Result.** On the legacy reference-candidate pool, NDCG@10: BM25 0.8188 →
17-feature MLP **0.8458** → LightGBM LambdaMART **0.8464**. A later evaluation
under the official KDD Cup gain vector confirmed the tie is a **statistical
null**: +0.00062, CI [+0.00009, +0.00115], p = 0.023, with 54.6% of queries
exactly tied.

**Attribution — split.**

| Part | Who | Evidence |
|---|---|---|
| LambdaMART reranker | **Jian Gao** | `eccfbc0` (2026-08-17), sole-authored, 100% blame |
| `AdvancedDeepReranker`, the MLP it is compared against | **hyukjin17 (teammate)** | created `0ecc033` (2026-03-04); 24 / 0 lines at HEAD |
| `advanced_features.py`, the 17-feature matrix both consume | **hyukjin17 (teammate)** | created `274ce10` (2026-03-04); 218 / 19 lines at HEAD |

The tree model is mine; the neural baseline it beat and the feature stack both
models ran on are a teammate's. The comparison is only meaningful as a pair, so
it is cited as a pair.

**Why the code is not here.** The MLP is §7.3-4 decision **A**. The legacy
LambdaMART path imports `extract_advanced_features` (218 lines of teammate code)
and so stays behind; the Task 1 LambdaMART path is Phase 6b and needs only the
17 feature *names*, which are a schema rather than an implementation.

## 1.3 Full-catalog retrieval — Recall@100 and RRF@100

**Result.** On a 5,000-query US-locale sample against a 1,215,854-product
US-only catalog, Broad Recall@100: BM25 0.4542 · Two-Tower 0.4598 · **RRF hybrid
0.5194**. A later Two-Tower V1 lifted these to 0.4649 and **0.5301**.

**Attribution — split, and the split matters.**

| Part | Who | Evidence |
|---|---|---|
| The experiment: index building, query sampling, retrieval runs, RRF fusion, evaluation | **Jian Gao** | `eccfbc0` (2026-08-17), all five scripts sole-authored, 100% blame |
| The retrieval primitives those scripts call | **hyukjin17 (teammate)** | `build_global_bm25_index` (`retrieval/bm25.py:93`), `search_bm25_global` (`:118`), `build_global_tt_index` (`retrieval/two_tower.py:177`), `search_tt_global` (`:204`) — all four hyukjin17 at HEAD |

**This is the entry most easily misread, so it is stated flatly: the
full-catalog retrieval *experiment* is my work; the retrieval *stack* it runs on
is not.** I wrote the harness — the indices' construction driver, the
deterministic query sample, the RRF fusion, the Recall@K evaluation — and called
into BM25 and Two-Tower search functions a teammate wrote. Neither half alone
produces the numbers.

**Why the code is not here.** `MIGRATION_PLAN.md` §7.4 **defer**: the four
called functions are teammate-authored, and the experiment is US-only, so it
carries no multilingual evidence. Both facts make it a poor fit for a repo named
`multilingual-product-ranking`.

**Pool caveat.** These numbers live on the large-version full-catalog pool with
`{E,S,C}` broad-relevance ground truth. They are **not comparable** to any NDCG
figure in this repository, which uses `task1_small_v1` with the official gain
vector. See `governance.md` §2.

## 1.4 Retrieval complementarity

**Result.** At the relevant-item-instance level, Top-100 retrieval ownership was
**BM25-only 12.023% · Two-Tower-only 13.074% · both 32.772% · neither 42.131%**.
The two retrievers have near-identical standalone Recall@100 because their unique
gains and losses nearly cancel, not because they return the same products.

**Attribution.** Same split as §1.3: the overlap analysis
(`scripts/analyze_topk_overlap.py`, `evaluate_full_retrieval.py`) is **Jian
Gao**, sole-authored in `eccfbc0`; the underlying retrieval calls are
**hyukjin17's**. The Two-Tower encoder producing the dense side is **gyuszix's**
V0 (see §2.4).

**Pool caveat.** Same US-only large-version pool as §1.3, and micro-averaged
over item instances rather than macro-averaged over queries, so these percentages
do not reconcile arithmetically with the macro Recall@100 values.

## 1.5 Query-slice findings

**Result.** On the same 5,000-query US sample, BM25 beat Two-Tower on SKU/model
queries (n=199, 0.5773 vs 0.5344, CI entirely below zero) and Two-Tower beat
BM25 on low-lexical-overlap queries (n=1367, 0.2805 vs 0.3149, CI entirely above
zero). Most other slices had CIs crossing zero.

**Attribution.** The slicing methodology and analysis
(`scripts/build_query_slices.py`, `analyze_query_slices.py`) are **Jian Gao**,
sole-authored in `eccfbc0`. The retrieval outputs analysed come from the stack in
§1.3, whose primitives are **hyukjin17's**, and the dense side from **gyuszix's**
Two-Tower V0.

**Why the code is not here.** §7.4 defer — it depends on `evaluation/metrics.py`
(barred outright, see §2.8) and on the deferred full-catalog outputs.

**Scope caveat.** 100% US-locale. These findings contain no multilingual evidence
whatsoever.

## 1.6 Business-NDCG evaluation

**Result.** A custom metric hard-penalising over-budget results and using star
rating as a within-label tie-break. Advanced MLP scored 0.8529 and LambdaMART
0.8536 under it, against 0.8458 / 0.8464 standard NDCG@10.

**Attribution.** Defined in `evaluation/metrics.py`, which is **91 lines
hyukjin17 / 21 lines iamjaygao** at HEAD. The metric itself
(`apply_business_ndcg_labels`) is teammate work. The business-aware framing
appeared in the course project's shared write-up.

**Why the code is not here.** `MIGRATION_PLAN.md` §11 item 14, a **hard rule**:
`evaluation/metrics.py` is the legacy scorer, it conflicts with the official gain
convention, and that conflict caused a full rework. Not one line of it may enter
this repository, and it may not be used as a reference. This repository has
exactly one scorer, `src/metrics/kdd_task1_ndcg.py`.

---

# Part 2 — Components that stayed behind

Reports in `docs/reports/` link here rather than to paths that do not exist in
this repository.

<a id="deepescireranker"></a>
## 2.1 `DeepESCIReranker` — 7-feature baseline MLP

`reranking/model.py`, `reranking/features.py`, `scripts/train_reranker.py`,
`evaluation/evaluate_reranker.py` in the source repo.

**Original author: gyuszix (teammate)**, `b1e3776` (2026-03-03).
**Jian Gao's later work:** the BatchNorm→LayerNorm diagnosis and fix, `5210986`
— see §1.1. Not migrated (§7.3-3 **A**).

<a id="advanceddeepreranker"></a>
## 2.2 `AdvancedDeepReranker` — 17-feature MLP

`reranking/advanced_model.py`, `scripts/train_adv_reranker.py`,
`evaluation/evaluate_advanced.py`.

**Original author: hyukjin17 (teammate)**, `0ecc033` (2026-03-04). 24 lines
hyukjin17 / 0 lines Jian Gao at HEAD.
**Jian Gao's later work:** used it as the subject of the feature ablation and as
the comparison baseline for LambdaMART — neither of which transfers ownership.
Not migrated (§7.3-4 **A**).

<a id="advanced_features"></a>
## 2.3 `advanced_features.py` — the 17-feature extractor

**Original author: hyukjin17 (teammate)**, `274ce10` (2026-03-04). 218 lines
hyukjin17 / 19 lines Jian Gao at HEAD.
**Jian Gao's later work:** fixed a train/serve normalisation bug (`5210986`), and
audited it — finding five defects: `log_review_count` identically zero, 15.9% of
prices parsed 100× too large, 13.7% of star ratings truncated, a `PorterStemmer`
applied to Japanese, and a hard-coded English colour list.

Not migrated (§7.3-4b decision **C**). Only the 17 feature **names**
(`ALL_FEATURES`) are re-declared in this repository, in
`legacy/features/build_feature_matrix.py`. A list of column names is a data
schema, not an implementation, and carries no authorship claim
(`MIGRATION_PLAN.md` §3.10, resolved).

<a id="two-tower-v0"></a>
## 2.4 Two-Tower V0 — dense retrieval encoder

`retrieval/two_tower.py`, `scripts/train_two_tower.py`.

**Original author: gyuszix (teammate)**, `5d9f8e6` / `cfd9f09` (2026-02-24/25).
At HEAD: `retrieval/two_tower.py` is gyuszix 141 / hyukjin17 83 / Jian Gao 16.
**Jian Gao's later work:** fixed a train/serve inconsistency (`5210986`); built
**Two-Tower V2** as a clean-room replacement (`0d62195`, sole-authored — see
§2.11); and audited the training scope,
establishing that the encoder saw **0 es and 0 jp training pairs** and that its
vocabulary (30,522-token English uncased WordPiece) cannot represent Japanese.

Not migrated (§7.3-2 **A**). One dependency survives as **data, not code**: the
`semantic_score` column of the optional Layer 2 feature matrix was produced by
this checkpoint. It is required only by the optional Phase 6b LambdaMART
reproduction, never by the minimum loop, and must never be regenerated with V2
(`MIGRATION_PLAN.md` §10.3).

<a id="two-tower-v2"></a>
### 2.4b Two-Tower V2 — migrated (Phase 6a)

`src/retrieval/two_tower_training.py`, `scripts/train_two_tower_v2.py`,
`src/data/build_query_split.py`.

**Sole author: Jian Gao** (`0d62195`, 2026-09-02, 100% blame, zero group
imports). Migrated in **Phase 6a**; the query split regenerates deterministically
to 18,800 train / 2,088 dev queries, byte-identical to the source repo's lists.

V2 is a **clean-room replacement** for V0, not a derivative of it. It shares V0's
*recipe* — same base encoder, same loss, same batch size, same positive-label
definition — deliberately, so that the V1 fix (warmup ratio, fixed seed,
query-level split, a real dev retrieval evaluator) could be attributed to that
fix alone. Sharing a recipe is not sharing code: no line of `retrieval/two_tower.py`
or `scripts/train_two_tower.py` was carried over, and neither file is in this
repository. **V0 remains gyuszix's work (§2.4); V2 is mine.**

⚠️ **V2 is trained on US-locale data only** — 0 es and 0 jp training pairs,
enforced by three separate filters (the query split's `LOCALE = "us"`, the
`locale="us"` default in `load_positive_pairs`, and a us-only dev evaluator
corpus). It is a monolingual English model. Every es/jp number reported for it is
zero-shot transfer, and a weak jp score is the expected consequence of that scope
rather than a finding about the architecture. Each of the three modules carries
this warning in its docstring; the evidence is in
[`retrieval_scope_audit.md`](reports/retrieval_scope_audit.md) §1.

The V2 **checkpoint** was not carried over — Phase 6a migrates code and the split
only. Training it is a separate, optional step.

<a id="bm25"></a>
## 2.5 `retrieval/bm25.py` — BM25 module

**Two facts, both true, and the entry is wrong without either.**

1. **The module was created by Jian Gao** — `c844e22` (2026-02-18), the
   repository's founding commit, *"Add BM25 lexical baseline module with NDCG@10
   evaluation"*.
2. **The current implementation is a teammate's rewrite.** `hyukjin17` rebuilt it
   on C-native `bm25s` across five commits on 2026-03-03 (`3d1e374`, `89d755f`,
   `9188fd3`, `4f66161`, `e359b4b`). At HEAD the file is **114 lines hyukjin17 /
   41 lines iamjaygao**, and all three functions that matter —
   `_score_single_group` (`:13`), `build_global_bm25_index` (`:93`),
   `search_bm25_global` (`:118`) — are entirely hyukjin17's.

So "BM25 initial implementation" as a *group-originated* component is correct in
the sense that it was produced during the course project and the shipped version
is majority-teammate code; what is not correct is calling the *initial* author a
teammate. `MIGRATION_PLAN.md` §7.1 records this as CONFLICT-1: reported, and the
human ownership conclusion left standing.

Not migrated (§7.3-1 decision **C**). This repository instead contains
`src/features/bm25_pool.py`, a minimal per-query scorer extracted from a
transcription Jian Gao wrote independently, carrying the required attribution
banner.

<a id="esci-s"></a>
## 2.6 ESCI-S integration

`utility/convert_esci_to_parquet.py`.

**Original author: jin (teammate)**, `f239a54` (2026-03-04). 34 lines jin / 18
lines Jian Gao at HEAD.
**Jian Gao's later work:** hardened the chunked-write schema handling
(`5210986`).

Not migrated (§7.3-5 **A**). The Cross-Encoder is text-only and needs no ESCI-S
field, so the ~3.4 GB dependency is dropped from this repository entirely.

<a id="interactive-gui"></a>
## 2.7 Interactive GUI

`interactive_search.py`.

**Original author: hyukjin17 (teammate)**, `1aa743a` (2026-03-04). 265 lines
hyukjin17 / 0 lines Jian Gao. No later work by Jian Gao.

Not migrated (§7.3-6 **A**). It belongs to the old end-to-end search product; a
ranking benchmark has no GUI surface. The MMR diversity logic inside it is
likewise teammate code and no reported metric uses it.

<a id="legacy-scorer"></a>
## 2.8 `evaluation/metrics.py` — the legacy scorer

91 lines hyukjin17 / 21 lines iamjaygao at HEAD. Created by Jian Gao in the
founding commit `c844e22`, then largely rewritten by hyukjin17.

**Barred by hard rule** (§11 item 14) — not migrated, and not usable as a
reference. It applies `2**gain - 1`, turning S into 0.0718 and C into 0.0070,
which conflicts with the official convention where S is exactly 0.10 and C
exactly 0.01. That conflict was the direct cause of the Phase 1.1→1.2 metric
rework. See `governance.md` §2.

<a id="deferred-retrieval"></a>
## 2.9 Full-catalog retrieval and ablation clusters

`scripts/{build_indices,build_full_catalog_indices,run_full_bm25_retrieval,run_full_tt_retrieval,evaluate_full_retrieval,sample_eval_queries,analyze_label_coverage,analyze_topk_overlap,mine_hard_negatives}.py`,
`scripts/{run_feature_ablation,run_group_ablation,build_query_slices,analyze_query_slices}.py`,
`experiments/query_slice_analysis/`.

**All sole-authored by Jian Gao** (`eccfbc0`, 2026-08-17). Deferred, not
disowned — see §1.3 for why: the retrieval primitives they call are teammate
code, and the whole cluster is US-only. Results cited in §1.3–§1.5.

<a id="config"></a>
## 2.10 `config.py`, `scripts/run_pipeline.py`

Created by Jian Gao in `c844e22`, since co-edited. Not migrated (§11 items 15–16):
`config.py` is a 14-line path stub, replaced here by `src/paths.py`;
`run_pipeline.py` drives the old end-to-end demo.

---

## Supporting data files

Several reports in `docs/reports/` reference sibling JSON and CSV artifacts —
`feature_importance.csv`, `locale_feature_health.json`, `paired_comparisons.json`
and similar. Those files describe the **legacy pools** and did not migrate; the
reports are preserved here as historical records of analyses run in the source
repository. Where such a reference appears without a link, the file remains in
`esci-search-ranking-system` at the path the surrounding text names.
