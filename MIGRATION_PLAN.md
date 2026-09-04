# MIGRATION_PLAN.md

Target repo: **`multilingual-product-ranking`** (new, personal).
Source repo: this one (`esci-search-ranking-system`) — originally a three-person course project.

---

# §1 Executive Summary

**Scope of this document: planning only. No file has been moved, copied, deleted or refactored.**

## The four numbers

Source code (`git ls-files '*.py'` = **109** tracked files; every path below verified to exist):

| Disposition | `.py` files | What it means |
|---|---:|---|
| **MIGRATE** (verbatim) | **29** | Zero group-code dependency; copy as-is, fix import paths only |
| **MIGRATE_AND_REFACTOR** | **8** | Personal code with a removable group import or a legacy-scorer arm to drop |
| **COPY_ARTIFACT** (output travels, code does not) | **1** | `build_task1_semantic_scores.py` — its column is **Layer 2 only** (§4.2); the producer cannot migrate |
| **KEEP_IN_OLD_REPO** | **71** | Group-originated, or personal work bound to the group feature stack (5 are `__init__.py`) |

Documents and data:

| Disposition | Count | What it means |
|---|---:|---|
| **MIGRATE** (reports/docs) | **25** `.md` | Experiment records with no code dependency |
| **REBUILT, not transferred** | **1** layer | Layer 1 core Task 1 pool — deterministic from raw ESCI (§4.1) |
| **COPY_ARTIFACT** (out-of-band) | **3** | 2 checkpoint sets (minimum loop, ≈3.2 GB) + Layer 2 feature matrix (Phase 6b only) |
| **TRACK IN GIT** (small manifests) | **5** | `test_lock_manifest.json`, `split_integrity.json`, 2 prediction parquets, split `.txt`s |
| **DOCUMENT_ONLY** | **6** | Results cited from the old repo; no code ships — §3.9 |
| **UNKNOWN** | **2** | Terrier parity scope; `supervised_bakeoff/` completeness — §3.10. (The third, `ALL_FEATURES` authorship, is **resolved**: a name list is schema, not implementation.) |

So **37 of 109 Python files travel** — roughly a third — and the 71 that stay are exactly the
group-originated feature/retrieval/GUI stack plus the personal work fused to it. Full per-file
evidence in §3.

**Python file counts are unchanged by the two-layer split in §4** — the split changes how
`build_task1_benchmark.py` is *refactored* (one file → two invocable paths), not whether it
migrates. What changed is the **data** column: the three `*_task1.parquet` files are no longer an
out-of-band dependency of the minimum loop.

## The largest migration risk

**Silent regeneration of the `semantic_score` column in `train_task1.parquet` /
`test_task1.parquet`.**

That column was produced by the **group-originated Two-Tower V0** checkpoint
(`build_task1_semantic_scores.py:46,93-94`, `model_frozen: True` at `:148`). It is an input to
every LambdaMART number this project has published, including the externally-cited **0.8429**.
The new repo will contain Two-Tower **V2**, which is 100% personal work, sole-authored, better on
its own dev metric, and sitting right there — which is precisely what makes the mistake attractive.
Regenerating with V2 changes every row of `kdd_task1_benchmark/REPORT.md` while leaving the file
names, the column name, and the report text identical. Nothing would fail; the numbers would just
quietly stop meaning what they say.

This is the one failure mode in this migration that is **irreversible without the old repo** and
**invisible in review**. §10.3 makes it a sha256-verified precondition on Phase 6b.

**But the blast radius is now contained.** Under the two-layer split in §4, `semantic_score` lives
in **Layer 2**, which only the optional Phase 6b touches. The minimum loop never reads it, so a
repo that never runs Phase 6b cannot hit this failure at all. The risk is real but *opt-in* — it
is no longer a property of the core migration.

Runner-up risks, both mitigated by hard gates: loss of the TEST-never-read property, which is an
asset the new repo inherits and can only spend once (§13.1); and cross-pool metric comparison,
which has already cost this project one full rework (§13.2). A third, newly surfaced by the
layering: **a scikit-learn version drift silently repartitions train/dev** at the same seed —
pinned and gated in §4.1 / §10.4.

## The minimum loop has zero group dependencies — of either kind

Verified by grep across all eleven minimum-loop files: **no component reads `semantic_score`,
`bm25_score`, or any of the 17 engineered features.** The Cross-Encoder and the official baseline
are both text models; the scorer reads only `esci_label`/`gain`. The only consumer of the feature
columns is LambdaMART, which §10.1 already excludes from the loop.

So the minimum loop depends on **zero group code and zero group-derived feature artifacts**. The
Layer 1 core pool is rebuilt from raw ESCI by a migrated script through a fully deterministic
query-level split (`build_task1_benchmark.py:288-299`) — nothing is transferred.

The one remaining out-of-band transfer is the **frozen checkpoints**, and it is worth being precise
about what they are for. They are not a runtime dependency of the pipeline; they are the
**regression oracle** that proves the migration did not change behaviour:

```
Public repo reproducibility:
  training config    → approximate retraining      [non-blocking, §10.2b]
  frozen checkpoint  → exact eval regression       [blocking migration test, §10.2a]
```

A third party who clones the public repo can retrain from config and land inside the tolerance band
(§10.2b) without ever receiving a checkpoint. The checkpoints exist so that *this* migration can be
proven bit-exact — a one-time validation need, not an ongoing requirement of the repo.

## What the new repo is

`multilingual-product-ranking` is a **ranking benchmark repo**, not a port of the old end-to-end
search system. The spine is the Cross-Encoder line, which §7.2 verified has **zero group-code
imports**. Full-catalog retrieval, the GUI, and the 17-feature MLP stack do not come along
(§11).

---

# §2 Entry Points

Traced by reading `if __name__ == "__main__"` blocks and following imports outward — not inferred
from filenames. "Internal Dependencies" lists first-party imports only (stdlib/third-party
omitted); `→` marks a runtime path-based load rather than a static import.

| Entry Point | Purpose | Internal Dependencies |
|---|---|---|
| `experiments/ranking_v2/kdd_task1_benchmark/scripts/build_task1_benchmark.py` | Builds the competition-aligned Task 1 pool + the 17 feature columns; contains the **self-contained per-query BM25** at `:68-99` | `task1_common`, `reranking.advanced_features` (`ALL_FEATURES` name list only, `:50`) |
| `…/scripts/build_task1_semantic_scores.py` | Produces the `semantic_score` column from the frozen **V0** encoder | none static; → `models/two_tower_finetuned` (`:46`) |
| `…/scripts/task1_common.py` | Pool/split constants, `task1_test_query_ids()`, `exclude_task1_test_queries()` | none |
| `…/scripts/kdd_task1_ndcg.py` | **The authoritative scorer.** Full-list/@20/@10 NDCG, official gain, deterministic tie-break, `metric_version()` + `candidate_pool` binding (`:201-211`) | none |
| `…/scripts/evaluate_task1_baselines.py` | Random/BM25/Two-Tower/LambdaMART baselines on Task 1; retrains LambdaMART (`:160-161`) | `task1_common`, `kdd_task1_ndcg`, `reranking.advanced_features` (`ALL_FEATURES`, `:28`) |
| `…/scripts/gate_a.py`, `gate_a_operative.py`, `dataset_counts_and_prereg.py` | Pool-integrity gates and the official Table-1 count check | `task1_common` |
| `scripts/build_cross_encoder_task1_data.py` | Builds CE training pairs; **`--split test` refused by default** (`:56`, "TEST LOCK is in force") | `reranking.cross_encoder` (`TEXT_FIELDS`) |
| `scripts/train_cross_encoder_task1.py` | Trains the V2 Cross-Encoder | `reranking.cross_encoder` |
| `scripts/evaluate_cross_encoder_task1.py` | Scores CE on the frozen DEV pool | `reranking.cross_encoder` |
| `reranking/cross_encoder.py` | CE shared library: text building, gain vector, `load_task1_scorer()` | none static; → `kdd_task1_ndcg.py` via `spec_from_file_location` (`:60,70`) |
| `scripts/ca_train_official_baseline.py` | Reproduces the Amazon official baseline (3 locale models) | none |
| `scripts/ca_eval_official_baseline.py` | Scores official baseline + V2 on frozen DEV; **global paired bootstrap** (`:110-115`) | `reranking.cross_encoder`; → `kdd_task1_ndcg.py` (`:27`) |
| `scripts/ca_metric_parity.py` | Metric-parity check against the official implementation | `reranking.cross_encoder`; → `kdd_task1_ndcg.py` (`:30`) |
| `scripts/ca_protocol_audit.py` | Split/protocol audit; TEST **counts only**, labels never used (`:97`) | none |
| `experiments/competition_alignment/us_slice_bootstrap/run_us_slice_bootstrap.py` | **US-locale paired bootstrap**; config copied from `ca_eval_official_baseline.py:110-115` | `reranking.cross_encoder`; → `kdd_task1_ndcg.py` |
| `experiments/ranking_v2/official_metric_final/scripts/evaluate_official_baselines.py` | Official-vs-legacy metric validation, per-locale breakdown | `official_kdd_ndcg`, `official_ndcg`, `reranking.advanced_features` (`ALL_FEATURES`), `reranking.advanced_model` (**group class**, `:39` — see CONFLICT-3) |
| `experiments/ranking_v2/benchmark_repair/scripts/build_clean_pool.py` | Rebuilds the de-duplicated, locale-matched clean pool | `reranking.advanced_features` (`ALL_FEATURES`, `:54`) |
| `experiments/ranking_v2/audit/scripts/0*.py` | Phase-0 ranking audit (feature health, locale slices, failure modes) | `evaluation.metrics` (`dcg`), and for `00_build_caches.py` four group implementations (`:29-32`) |
| `scripts/v3_audit_and_freeze.py`, `v3_ensemble_audit.py` | V3 zero-shot / ensemble reranker audits | `reranking.cross_encoder`; → `kdd_task1_ndcg.py` |
| `scripts/benchmark_modern_rerankers.py` | Zero-shot bakeoff: BGE / Qwen / Jina | `reranking.cross_encoder`; → `kdd_task1_ndcg.py` |

**Structural read.** Every entry point in the minimum viable loop (§10.1) depends on exactly two
first-party modules — `reranking/cross_encoder.py` and `kdd_task1_ndcg.py` — plus `task1_common.py`
for pool construction. All three are 100% personal, sole-authored, zero group imports. The
group-coupled entry points are confined to the MLP/feature-stack cluster.

---

# §3 File Migration Matrix

Per-file, evidence-based. "Imported/Used By" is from the static import graph over all 106 tracked
`.py` files plus the runtime path loads in §2. Decisions are consistent with §7.3/§7.4/§7.5;
the one divergence is CONFLICT-3, reported in §7.7 and applied below as a decision downgrade only.

New-repo root is `multilingual-product-ranking/`; `New Path` is relative to it.

## 3.1 Core library — the spine

| Old Path | Decision | New Path | Reason | Imported/Used By | Refactor Needed |
|---|---|---|---|---|---|
| `reranking/cross_encoder.py` | **MIGRATE** | `src/ranking/cross_encoder.py` | Zero internal imports (§7.2 screen). The CE spine. | 9 callers: `train/evaluate/build_cross_encoder_task1*.py`, `ca_eval_official_baseline.py`, `ca_metric_parity.py`, `v3_audit_and_freeze.py`, `v3_ensemble_audit.py`, `benchmark_modern_rerankers.py`, `run_us_slice_bootstrap.py`, `tests/test_cross_encoder.py` | Repoint `TASK1_SCORER_PATH` (`:60`) to the new scorer location |
| `experiments/ranking_v2/kdd_task1_benchmark/scripts/kdd_task1_ndcg.py` | **MIGRATE** | `src/metrics/kdd_task1_ndcg.py` | Sole authoritative scorer; no imports; sha-pinned via `scorer_sha256()` (`:68`) | `evaluate_task1_baselines.py`, `build_terrier_files.py`, `test_kdd_task1_ndcg.py`, + all `→` loaders in §2 | None to the file. §13.2 adds binding to `per_query_ndcg_table` |
| `experiments/ranking_v2/kdd_task1_benchmark/scripts/test_kdd_task1_ndcg.py` | **MIGRATE** | `tests/test_kdd_task1_ndcg.py` | Unit tests for the scorer; imports only `kdd_task1_ndcg` | — | Import path only |
| `experiments/ranking_v2/kdd_task1_benchmark/scripts/task1_common.py` | **MIGRATE** | `src/data/task1_common.py` | No internal imports; holds `OFFICIAL_GAIN` (`:25`), `task1_test_query_ids()` (`:32`), `exclude_task1_test_queries()` (`:39`) | `build_task1_benchmark.py`, `evaluate_task1_baselines.py`, `gate_a_operative.py`, `dataset_counts_and_prereg.py`, `make_benchmark_comparison.py`, `build_terrier_files.py` | Rebase `ROOT`/`BASE`/`EXAMPLES`/`PRODUCTS` (`:13-18`); §13.1 assert lands here |
| `tests/test_cross_encoder.py` | **MIGRATE** | `tests/test_cross_encoder.py` | Imports only `reranking.cross_encoder` | — | Import path only |

## 3.2 Task 1 pool construction

| Old Path | Decision | New Path | Reason | Imported/Used By | Refactor Needed |
|---|---|---|---|---|---|
| `…/kdd_task1_benchmark/scripts/build_task1_benchmark.py` | **MIGRATE_AND_REFACTOR** | `src/data/build_task1_pool.py` (core) + `legacy/features/build_feature_matrix.py` (Layer 2) | Builds the pool **and** contains the self-contained personal BM25 (`:68-99`) — the §7.3-1 **C** deliverable | `task1_common` | **Must be split into two independently invocable paths** (§4): (a) **core pool** — filter `small_version==1`, join products, query-level split (`:288-299`), emit the Layer 1 columns from `:301-303`; (b) **feature append** — Layer 2 columns from `:304` (`bm25_raw`, `semantic_cosine_raw`, `ALL_FEATURES`), Phase 6b only. Verified: the two are already separable — the split at `:292-295` reads only `query_id` + `product_locale` and touches no feature column. Also: replace `from reranking.advanced_features import ALL_FEATURES` (`:50`) with a local constant, and lift `:68-99` into `src/features/bm25_pool.py` |
| `…/kdd_task1_benchmark/scripts/build_task1_semantic_scores.py` | **COPY_ARTIFACT** (not the code) | — | Produces `semantic_score` from the **group V0** checkpoint (`:46`). Running it in the new repo would require migrating V0 — forbidden by §7.3-2 | — | N/A — the *output column* is **Layer 2 only** (§4.2), carried out-of-band under Phase 6b. Governed by §10.3 |
| `…/kdd_task1_benchmark/scripts/gate_a.py` | **MIGRATE** | `src/data/gates/gate_a.py` | Pool-integrity gate; no internal imports | — | Path rebase |
| `…/kdd_task1_benchmark/scripts/gate_a_operative.py` | **MIGRATE** | `src/data/gates/gate_a_operative.py` | Operative A2′/A3′ assertions; imports `task1_common` only | — | Path rebase |
| `…/kdd_task1_benchmark/scripts/gate_a_diagnostic.py` | **MIGRATE** | `src/data/gates/gate_a_diagnostic.py` | No internal imports | — | Path rebase |
| `…/kdd_task1_benchmark/scripts/dataset_counts_and_prereg.py` | **MIGRATE** | `src/data/gates/dataset_counts.py` | Verifies the official Table-1 counts; `task1_common` only | — | Path rebase |
| `…/kdd_task1_benchmark/scripts/audit_data_source.py` | **MIGRATE** | `src/data/gates/audit_data_source.py` | No internal imports | — | Path rebase |
| `…/kdd_task1_benchmark/scripts/make_benchmark_comparison.py` | **MIGRATE** | `scripts/make_benchmark_comparison.py` | `task1_common` only | — | Path rebase |
| `…/kdd_task1_benchmark/scripts/build_terrier_files.py` | **MIGRATE** | `scripts/build_terrier_files.py` | TREC-format export; `task1_common` + `kdd_task1_ndcg` | — | Path rebase |
| `…/kdd_task1_benchmark/scripts/evaluate_task1_baselines.py` | **MIGRATE_AND_REFACTOR** | `scripts/evaluate_task1_baselines.py` | Baseline table incl. the Task-1 LambdaMART path (`:160-161`) | `task1_common`, `kdd_task1_ndcg` | Inline `ALL_FEATURES` (`:28`). Per §10.1 the LambdaMART block becomes an **optional phase**, not part of the minimum loop |

## 3.3 Cross-Encoder pipeline

| Old Path | Decision | New Path | Reason | Imported/Used By | Refactor Needed |
|---|---|---|---|---|---|
| `scripts/build_cross_encoder_task1_data.py` | **MIGRATE** | `scripts/build_ce_data.py` | Only `reranking.cross_encoder`; already carries the TEST-lock default (`:56`) | — | Import path; §13.1 upgrades the flag to an assert |
| `scripts/train_cross_encoder_task1.py` | **MIGRATE** | `scripts/train_cross_encoder.py` | Only `reranking.cross_encoder` | — | Import path |
| `scripts/evaluate_cross_encoder_task1.py` | **MIGRATE** | `scripts/evaluate_cross_encoder.py` | Only `reranking.cross_encoder` | — | Import path |
| `experiments/ranking_v2/cross_encoder_task1/README.md` | **MIGRATE** | `docs/cross_encoder.md` | Method documentation for the spine | — | Fix internal links |
| `experiments/ranking_v2/cross_encoder_task1/full_run/prereg_prediction.md` | **MIGRATE** | `docs/prereg/ce_v2_prereg.md` | Pre-registration record — evidence for §13.1 | — | None |
| `experiments/ranking_v2/cross_encoder_task1/FROZEN_V2.json` | **MIGRATE** | `artifacts/FROZEN_V2.json` | Freeze manifest for the V2 checkpoint | — | None |

## 3.4 Official-baseline reproduction, bootstrap, locale analysis

| Old Path | Decision | New Path | Reason | Imported/Used By | Refactor Needed |
|---|---|---|---|---|---|
| `scripts/ca_train_official_baseline.py` | **MIGRATE** | `scripts/reproduce_official_baseline.py` | No internal imports; `test_split_touched: False` recorded (`:161`) | — | Path rebase |
| `scripts/ca_eval_official_baseline.py` | **MIGRATE** | `scripts/eval_official_baseline.py` | `reranking.cross_encoder` only; holds the canonical bootstrap config (`:110-115`) | `run_us_slice_bootstrap.py` cites it as config source | `sys.path.insert` for the scorer (`:27`) → package import |
| `scripts/ca_metric_parity.py` | **MIGRATE** | `scripts/metric_parity.py` | `reranking.cross_encoder` only | — | Same path fix (`:30`) |
| `scripts/ca_protocol_audit.py` | **MIGRATE** | `scripts/protocol_audit.py` | No internal imports; TEST metadata-only discipline (`:97`) | — | Path rebase |
| `experiments/competition_alignment/us_slice_bootstrap/run_us_slice_bootstrap.py` | **MIGRATE** | `scripts/us_slice_bootstrap.py` | `reranking.cross_encoder` only; blocking gate in §10.2a | — | Path rebase |
| `experiments/competition_alignment/REPORT.md` | **MIGRATE** | `docs/reports/competition_alignment.md` | Headline result document | — | Link fixes |
| `experiments/competition_alignment/us_slice_bootstrap/REPORT.md` | **MIGRATE** | `docs/reports/us_slice_bootstrap.md` | The resume-facing US number | — | Link fixes |
| `experiments/competition_alignment/{leaderboard_comparability,candidate_pool_root_cause,metric_authority_report,published_strong_baseline_plan}.md` | **MIGRATE** | `docs/reports/` | Method/provenance records with no code dependency | — | Link fixes |
| `experiments/competition_alignment/ARTIFACTS.md` | **MIGRATE** | `artifacts/ARTIFACTS.md` | The out-of-band artifact register (§7.6) | — | Update paths after copy |
| `experiments/retrieval_scope_audit/retrieval_scope_audit.md` | **MIGRATE** | `docs/reports/retrieval_scope_audit.md` | Locale-scope audit; read-only, no code dependency | — | Link fixes |

## 3.5 Two-Tower V2

| Old Path | Decision | New Path | Reason | Imported/Used By | Refactor Needed |
|---|---|---|---|---|---|
| `retrieval/two_tower_training.py` | **MIGRATE** | `src/retrieval/two_tower_training.py` | No internal imports; 100% personal | `scripts/train_two_tower_v2.py` | None |
| `scripts/train_two_tower_v2.py` | **MIGRATE** | `scripts/train_two_tower_v2.py` | `config` + own module only | — | Replace `config` import with new paths module |
| `experiments/two_tower_v2/build_query_split.py` | **MIGRATE** | `src/data/build_query_split.py` | `config` only; deterministic (seed 42) | — | Same |
| `experiments/two_tower_v2/FINAL_REPORT.md`, `baseline/README.md`, `phase*/REPORT.md` | **MIGRATE** | `docs/reports/two_tower_v2/` | Personal experiment records | — | Link fixes |
| `scripts/evaluate_two_tower_v2_full_catalog.py` | **KEEP_IN_OLD_REPO** | — | Imports `retrieval.two_tower` (group: `search_tt_global`, `build_global_tt_index` both hyukjin17) and is US-only; §7.4 defers full-catalog retrieval | — | — |

## 3.6 Benchmark audit / repair / official-metric validation

| Old Path | Decision | New Path | Reason | Imported/Used By | Refactor Needed |
|---|---|---|---|---|---|
| `experiments/ranking_v2/official_metric_final/scripts/official_kdd_ndcg.py` | **MIGRATE** | `src/metrics/official_kdd_ndcg.py` | No internal imports; the parity reference implementation | `evaluate_official_baselines.py`, `test_official_kdd_ndcg.py` | Path rebase |
| `…/official_metric_final/scripts/test_official_kdd_ndcg.py` | **MIGRATE_AND_REFACTOR** | `tests/test_official_kdd_ndcg.py` | Imports `official_ndcg` (the *legacy-comparison* scorer) alongside `official_kdd_ndcg` | — | Drop the legacy-comparison arm — §11 forbids the legacy scorer in the new repo |
| `…/official_metric_final/scripts/evaluate_official_baselines.py` | **MIGRATE_AND_REFACTOR** | `scripts/evaluate_official_metric.py` | **CONFLICT-3**: imports `AdvancedDeepReranker` (`:39`) purely to score the MLP row | — | Delete the MLP baseline row + its import; inline `ALL_FEATURES` (`:38`); drop the `official_ndcg` legacy arm |
| `…/official_metric_final/REPORT.md` | **MIGRATE** | `docs/reports/official_metric_final.md` | Metric-authority record; origin of §13.2 | — | Link fixes |
| `…/benchmark_repair/scripts/build_clean_pool.py` | **MIGRATE_AND_REFACTOR** | `src/data/build_clean_pool.py` | `ALL_FEATURES` only (`:54`) | — | Inline the list |
| `…/benchmark_repair/scripts/validate_pool.py` | **MIGRATE_AND_REFACTOR** | `src/data/validate_pool.py` | `ALL_FEATURES` (`:22`) + `official_ndcg` | — | Inline list; repoint scorer to `kdd_task1_ndcg` |
| `…/benchmark_repair/scripts/validate_p0_fixed.py` | **MIGRATE** | `src/data/validate_p0.py` | No internal imports | — | Path rebase |
| `…/benchmark_repair/scripts/official_ndcg.py` | **KEEP_IN_OLD_REPO** | — | Superseded by `kdd_task1_ndcg.py`. Two scorers in one repo is the exact condition §13.2 exists to prevent | `evaluate_baselines.py`, `validate_pool.py`, `train_corrected_lambdamart.py`, `test_official_ndcg.py` | — |
| `…/benchmark_repair/scripts/test_official_ndcg.py` | **KEEP_IN_OLD_REPO** | — | Its purpose is legacy-vs-official parity; imports `evaluation.metrics.dcg` (`:198`), which §11 bars outright | — | — |
| `…/benchmark_repair/scripts/evaluate_baselines.py` | **KEEP_IN_OLD_REPO** | — | **CONFLICT-3**; also depends on `official_ndcg` + the old clean pool. Superseded by `evaluate_task1_baselines.py` | — | — |
| `…/benchmark_repair/scripts/train_corrected_lambdamart.py` | **KEEP_IN_OLD_REPO** | — | LambdaMART on the *legacy* pool via `official_ndcg`; out of the minimum loop (§10.1) | — | — |
| `…/benchmark_repair/REPORT.md`, `…/audit/REPORT.md`, `…/audit/repository_inventory.md`, `…/kdd_task1_benchmark/REPORT.md`, `…/kdd_task1_benchmark/benchmark_definition_comparison.md` | **MIGRATE** | `docs/reports/` | Audit records; no code dependency. `kdd_task1_benchmark/REPORT.md` is the doc §10.3 protects | — | Link fixes |
| `experiments/ranking_v2/audit/scripts/00_build_caches.py` | **KEEP_IN_OLD_REPO** | — | **CONFLICT-3, unrefactorable**: its entire job is building the 17-feature MLP cache; imports 4 group implementations (`:29-32`) | `01`–`06` consume its caches | — |
| `experiments/ranking_v2/audit/scripts/01…06_*.py` | **KEEP_IN_OLD_REPO** | — | All consume `00_build_caches.py` output; `02`/`05` also import `evaluation.metrics.dcg` (`:23`, `:32`) — barred by §11 | — | — |
| `experiments/audit/task*.py` (6 files) | **KEEP_IN_OLD_REPO** | — | Phase-0 audit of the 12/17-feature MLP; `task2_train_12feature.py:20` imports `extract_advanced_features` | — | — |

## 3.7 V3 exploratory (zero-shot / ensemble rerankers)

| Old Path | Decision | New Path | Reason | Imported/Used By | Refactor Needed |
|---|---|---|---|---|---|
| `scripts/benchmark_modern_rerankers.py` | **MIGRATE_AND_REFACTOR** | `experiments/exploratory/benchmark_modern_rerankers.py` | `reranking.cross_encoder` only. **Must be labelled `exploratory`** — it loads Jina v3.5 (CC BY-NC 4.0, §6) | — | Path rebase; add the non-commercial banner required by §6 |
| `scripts/v3_audit_and_freeze.py` | **MIGRATE** | `experiments/exploratory/v3_audit_and_freeze.py` | `reranking.cross_encoder` only | — | Path rebase |
| `scripts/v3_ensemble_audit.py` | **MIGRATE_AND_REFACTOR** | `experiments/exploratory/v3_ensemble_audit.py` | `reranking.cross_encoder`; reads Jina outputs (`:67`) | — | Path rebase + `exploratory` label |
| `scripts/v3_build_train50k.py`, `v3_build_funnel_subsets.py`, `v3_train_qwen_lora.py` | **MIGRATE** | `experiments/exploratory/` | No internal imports | — | Path rebase |
| `experiments/ranking_v3/**/*.json`, `*.csv`, `*.md` | **MIGRATE** | `experiments/exploratory/results/` | Personal result records | — | Label `exploratory` per §6 |

## 3.8 Group-originated — not migrated

All rows below are **KEEP_IN_OLD_REPO**, per §7.3 (A decisions) and §11.

| Old Path | Decision | New Path | Reason | Imported/Used By | Refactor Needed |
|---|---|---|---|---|---|
| `retrieval/bm25.py` | **KEEP_IN_OLD_REPO** | — | §7.3-1 **C**: the capability ships as the personal `:68-99` transcription, not this module. Current impl. hyukjin17 (114/41) | `build_indices.py`, `build_full_catalog_indices.py`, `run_full_bm25_retrieval.py`, `generate_bm25_scores.py`, `mine_hard_negatives.py`, `interactive_search.py`, `tests/test_baseline_search.py` | — |
| `retrieval/two_tower.py` | **KEEP_IN_OLD_REPO** | — | §7.3-2 **A**; gyuszix 141 / hyukjin17 83 | 8 callers incl. `run_pipeline.py`, `benchmark_ann.py` | — |
| `scripts/train_two_tower.py` | **KEEP_IN_OLD_REPO** | — | §7.3-2 **A**; V0 training, gyuszix | — | — |
| `reranking/model.py`, `reranking/features.py`, `scripts/train_reranker.py`, `evaluation/evaluate_reranker.py` | **KEEP_IN_OLD_REPO** | — | §7.3-3 **A**; `DeepESCIReranker` cluster, gyuszix | — | — |
| `reranking/advanced_model.py` | **KEEP_IN_OLD_REPO** | — | §7.3-4 **A**; hyukjin17 24/0 | 6 callers (all also KEEP or refactored to drop it) | — |
| `reranking/advanced_features.py` | **KEEP_IN_OLD_REPO** | — | §7.3-4b **C**: only the 17-name `ALL_FEATURES` list is re-declared in the new repo. hyukjin17 218/19; 5 known defects | 11 callers | — |
| `scripts/train_adv_reranker.py`, `evaluation/evaluate_advanced.py` | **KEEP_IN_OLD_REPO** | — | Advanced-MLP training/eval; group cluster | — | — |
| `utility/convert_esci_to_parquet.py` | **KEEP_IN_OLD_REPO** | — | §7.3-5 **A**; ESCI-S dropped entirely | — | — |
| `interactive_search.py` | **KEEP_IN_OLD_REPO** | — | §7.3-6 **A**; would not run today (`output/bm25s_index` absent) | — | — |
| `evaluation/metrics.py` | **KEEP_IN_OLD_REPO** | — | §11 **hard rule** — legacy scorer, gain-convention conflict; not one line, not as reference | 9 callers | — |
| `config.py` | **KEEP_IN_OLD_REPO** | — | §11 confirmed **A**; 14-line path stub, trivially re-declared | 30+ callers | — |
| `scripts/run_pipeline.py` | **KEEP_IN_OLD_REPO** | — | §11 confirmed **A**; old end-to-end demo driver | — | — |
| `evaluation/evaluate_retrieval.py`, `evaluation/evaluate_lambdamart.py` | **KEEP_IN_OLD_REPO** | — | Both import `evaluation.metrics` (§11 hard rule) | — | — |
| `scripts/train_lambdamart.py` | **KEEP_IN_OLD_REPO** | — | §7.4: imports `extract_advanced_features` (`:11`, 218 lines teammate code). Task-1 path in `evaluate_task1_baselines.py` supersedes it | — | — |
| `scripts/{build_indices,build_full_catalog_indices,run_full_bm25_retrieval,run_full_tt_retrieval,evaluate_full_retrieval,sample_eval_queries,analyze_label_coverage,mine_hard_negatives,benchmark_ann,benchmark_query_batching}.py` | **KEEP_IN_OLD_REPO** | — | Full-catalog retrieval cluster — §7.4 **defer**. US-only, so carries no multilingual evidence; 4 of the called functions are hyukjin17's | — | — |
| `scripts/{run_feature_ablation,run_group_ablation,build_query_slices,analyze_query_slices,analyze_topk_overlap}.py` | **KEEP_IN_OLD_REPO** | — | §7.4 defer; ablation/slicing bound to `advanced_features`/`advanced_model`/`evaluation.metrics` | — | — |
| `experiments/query_slice_analysis/*.py`, `scripts/{audit_false_negatives,analyze_product_text_stats,generate_interview_figures}.py` | **KEEP_IN_OLD_REPO** | — | Operate on the old US-only full-catalog outputs | — | — |
| `tests/{test_two_tower,test_baseline_search,test_idf_scorer}.py` | **KEEP_IN_OLD_REPO** | — | Test group modules that are not migrating | — | — |
| `analysis (not used)/**` | **KEEP_IN_OLD_REPO** | — | Directory name declares it dead | — | — |

## 3.9 Results cited but not shipped

| Old Path | Decision | New Path | Reason |
|---|---|---|---|
| The `BatchNorm`→`LayerNorm` diagnosis (`README.md:94`, commit `5210986`) | **DOCUMENT_ONLY** | `docs/prior_work.md` | §7.3-3: the asset is prose. Cite this repo by commit; attribute the original model to a teammate |
| LambdaMART 0.8464 vs advanced MLP 0.8458 (`README.md:130-136`) | **DOCUMENT_ONLY** | `docs/prior_work.md` | §7.3-4: a result, not code |
| Recall@100 / RRF@100 full-catalog numbers | **DOCUMENT_ONLY** | `docs/prior_work.md` | §7.4 defer; US-only, different pool |
| Retrieval complementarity (12.023 / 13.074 / 32.772 %) | **DOCUMENT_ONLY** | `docs/prior_work.md` | Same pool caveat |
| Query-slice findings (`experiments/query_slice_analysis/REPORT.md`) | **DOCUMENT_ONLY** | `docs/prior_work.md` | US-only Recall pool |
| Business-NDCG evaluation | **DOCUMENT_ONLY** | `docs/prior_work.md` | Defined in `evaluation/metrics.py`, barred by §11 |

## 3.10 UNKNOWN — 2 items

| Item | Why unresolved | What would settle it |
|---|---|---|
| `experiments/ranking_v2/kdd_task1_benchmark/trec_eval_data/` + `run_terrier_parity.sh` | `REPORT.md:619` records Terrier parity as **not run**. Whether the new repo should carry an unexecuted parity harness, or drop it until it is actually run, is a scope call, not a code fact | A decision on whether Terrier parity is in the new repo's roadmap. If yes → MIGRATE; if no → KEEP_IN_OLD_REPO |
| `experiments/ranking_v3/supervised_bakeoff/` | Directory exists; whether its contents are complete results or an abandoned in-progress run was not determined from file inspection alone | Read the run manifests/logs inside it and check for a terminal results JSON. If incomplete → do not migrate |

### RESOLVED — `ALL_FEATURES` name list carries no authorship claim

Previously listed here as a third UNKNOWN. **Confirmed: it does not.** A list of 17 column names is
a **data schema, not an implementation** — it contains no creative expression. Re-declaring it in
the new repo raises no ownership question.

Handled as decided in §3: the 17-name constant is declared locally in
`src/data/build_task1_pool.py` and `scripts/evaluate_task1_baselines.py`, replacing
`from reranking.advanced_features import ALL_FEATURES`. The §7.3-4b **C** disclosure wording
(*minimal re-implementation; no claim over the original group implementation*) still applies to
`src/features/bm25_pool.py`, which **is** an implementation.

---

# §4 Data Dependencies

Sizes measured with `du -sh`. "Git Track?" reflects the recommendation for the **new** repo.

| Data | Source | Rebuildable | Git Track? |
|---|---|---|---|
| `shopping_queries_dataset_examples.parquet` (**49 MB**, raw) | Upstream `amazon-science/esci-data` | **No** — external upstream. Fetch via the documented download step | **No.** Document the fetch; never vendor. No LFS |
| `shopping_queries_dataset_products.parquet` (**1.0 GB**, raw) | Same | **No** — external upstream | **No.** Too large; LFS not justified for a public upstream file |
| **Layer 1 — core Task 1 pool** (`train`/`dev`/`test`, text + labels only) | `build_task1_pool.py` from raw ESCI only | **YES — fully deterministic.** Zero group code, zero group-derived columns. See §4.1 | **No** (size), but **no out-of-band transfer needed** — the new repo rebuilds it |
| **Layer 2 — legacy feature matrix** (`bm25_score`, `semantic_score`, 17 features) | `build_task1_benchmark.py` feature path + `build_task1_semantic_scores.py` | **NO.** `semantic_score`'s producer is the group-originated Two-Tower **V0** checkpoint, which does not migrate (§7.3-2) | **No.** Out-of-band, **only if Phase 6b is executed**. sha256 gate applies — §10.3. See §4.2 |
| `test_lock_manifest.json` (small, generated) | `evaluate_task1_baselines.py` | Yes, but its whole purpose is to be frozen | **YES — track it.** It is the integrity anchor: `test_file_sha256`, `scorer_sha256`, `query_count`/`row_count` 14,496/336,373, `locale_breakdown` us 8,956 · jp 3,123 · es 2,417 |
| `split_integrity.json` (small, generated) | `build_task1_benchmark.py:314-330` | Yes | **YES — track it.** The rebuild oracle for Layer 1 (§10.4): `query_counts`, `row_counts`, `locale_distribution`, `depth_distribution`, `label_distribution`, `split_method` |
| `official_baseline_dev_predictions.parquet` (**2.2 MB**, generated) | `ca_eval_official_baseline.py` | Yes — but re-running inference is not guaranteed bit-identical | **Borderline — recommend YES.** 2.2 MB is well within normal git limits, and §10.2a's blocking gate reads it. No LFS needed |
| `fulldev_epoch2_predictions.parquet` (**2.4 MB**, generated) | Cross-encoder eval scripts | Same | **Borderline — recommend YES**, same reasoning |
| `official_baseline_models/{us,es,jp}/model.safetensors` (**1.1 GB**, generated) | `ca_train_official_baseline.py` | Yes — retraining is documented; **not** bit-identical across environments | **No.** Out-of-band copy. LFS only if a hosted mirror is genuinely wanted |
| Cross-encoder V2 checkpoints (**2.1 GB**, generated) | `train_cross_encoder_task1.py` | Yes, with the same caveat | **No.** Out-of-band copy. This is the checkpoint §10.2a freezes |
| `models/two_tower_finetuned/` (**254 MB**, generated) | `scripts/train_two_tower.py` — **group-originated V0** | Yes in principle, but the producer is group code that is **not migrating** (§7.3-2) | **No — and do not migrate the weights either.** Only its *output column* travels, inside the **Layer 2** feature matrix, and only under Phase 6b. See §4.2 / §10.3 |
| `experiments/two_tower_v2/splits/{train,dev}_queries.txt` (small, generated) | `build_query_split.py`, seed 42 | **Yes, deterministically** | **YES — track.** Small, and tracking makes the split auditable |
| `esci-s` enrichment (~3.4 GB compressed) | `shuttie/esci-s` | Yes | **No — and not needed at all.** §7.3-5 drops ESCI-S from the new repo |

## 4.1 Layer 1 — Core Task 1 pool (minimum loop; rebuilt by the new repo)

**This layer is what the minimum viable loop (§10.1) consumes. Nothing in it is group-derived.**

### Columns

Verified against the actual reads in the minimum loop, not inferred:

| Column | Read by | Origin |
|---|---|---|
| `query_id`, `query`, `product_id`, `example_id` | all loop components | raw `examples.parquet` |
| `product_locale`, `esci_label`, `split` | all loop components | raw `examples.parquet` |
| `product_title` | CE text builder; official-baseline scorer (`ca_eval_official_baseline.py:47`) | raw `products.parquet` |
| `product_brand` | CE text builder (`cross_encoder.py:51`; `build_cross_encoder_task1_data.py:47-48`) | raw `products.parquet` |
| `doc_id` | `build_cross_encoder_task1_data.py:48` | derived in-pool |
| **`gain`** | `ca_eval_official_baseline.py:44`, `run_us_slice_bootstrap.py:55`, `evaluate_cross_encoder_task1.py:137,142,146` | **derived from `esci_label`** by `kdd_task1_ndcg.gain_from_labels()` (`:63-65`) — 100% personal code |

> **Note — three columns beyond the eight originally specified.** The task brief listed
> `query_id, query, product_id, example_id, product_title, product_locale, esci_label, split`.
> Inspection shows the loop additionally requires **`gain`**, **`doc_id`** and **`product_brand`**.
> Omitting `gain` alone would break `ca_eval_official_baseline.py:44`,
> `run_us_slice_bootstrap.py:55` and `evaluate_cross_encoder_task1.py:137`.
> **None of the three weakens the layering claim:** `gain` is a pure function of `esci_label`
> under the official vector (E 1.0 / S 0.1 / C 0.01 / I 0.0) and is *self-verified* at
> `evaluate_cross_encoder_task1.py:142`, which recomputes it and refuses to proceed on mismatch;
> `doc_id` is assigned in-pool; `product_brand` comes straight from the raw products table.
> All three are rebuildable with zero group code.

Note that `product_color`, `product_bullet_point` and `product_description` are **not** in Layer 1:
`build_cross_encoder_task1_data.py:49,72` joins them directly from the raw products parquet at
CE-data-build time (`FROM_PRODUCTS`), so the pool never has to carry them.

### Build path

```
shopping_queries_dataset_examples.parquet  ─┐
                                            ├─ filter small_version == 1
shopping_queries_dataset_products.parquet  ─┘  join on (product_id, product_locale)
                                            │
                                            └─ query-level train/dev split:
                                               train_test_split(test_size=0.15,
                                                                random_state=42,
                                                                stratify=product_locale)
```

Verified at `build_task1_benchmark.py:288-299`. The split operates on
`qdf[["query_id","product_locale"]].drop_duplicates("query_id").sort_values("query_id")` (`:292`)
— **it reads no feature column at all**, so the core path can be built without ever invoking
feature computation.

### Properties

- **Rebuildable: YES**, fully deterministic.
- **Git track: NO** (size) — but the *build script* migrates, which is what makes tracking unnecessary.
- **Out-of-band transfer: NOT NEEDED.**
- **Zero group-code dependency, zero group-derived feature dependency.**

### ⚠️ Version constraint — `scikit-learn==1.8.0` must be pinned

`train_test_split`'s shuffling implementation may change across scikit-learn major versions. The
same `random_state=42` can therefore yield a **different** train/dev partition on a different
version, silently changing every dev-set number — including the 0.8456 in §10.2a.

- Current environment: **scikit-learn 1.8.0** (verified: `python -c "import sklearn; print(sklearn.__version__)"`).
- Already pinned in this repo at `requirements.txt:67` (`scikit-learn==1.8.0`).
- **The new repo's dependency file must carry the same exact pin.** Not `>=`, not a range.

## 4.2 Layer 2 — Legacy feature matrix (Phase 6b only)

Columns, layered on top of Layer 1: `bm25_score`, `semantic_score`, `bm25_raw`,
`semantic_cosine_raw`, `lgb_label`, `category`, and the 17 engineered features named in
`ALL_FEATURES`.

- **Rebuildable: NO.** `semantic_score` is produced by the **group-originated Two-Tower V0**
  checkpoint (`build_task1_semantic_scores.py:46,93-94`, `"model_frozen": True` at `:148`). That
  checkpoint does not migrate (§7.3-2), so the column cannot be regenerated in the new repo.
- **Out-of-band transfer: REQUIRED — but only if Phase 6b is executed.** It is not needed to close
  the minimum loop.
- **Provenance must be recorded** in `artifacts/ARTIFACTS.md` and in the layer's own manifest:
  *`semantic_score` was generated by a group-originated Two-Tower V0 checkpoint that is not part of
  this repository.*
- **The sha256 gate applies to this layer** — §10.3.

**Total out-of-band payload for the minimum loop: ≈ 3.2 GB** — the frozen checkpoints only
(1.1 GB official baseline + 2.1 GB CE V2), and those are a *regression oracle*, not a runtime
input (§1). Layer 2 (~136 MB of parquet) is additional and applies only under Phase 6b.
`ARTIFACTS.md`'s ≈ 4.3 GB figure covers both plus the V0 weights.

---

# §5 Experiment Artifacts

| Artifact | Keep? | Destination | Reason |
|---|---|---|---|
| `experiments/competition_alignment/` reports + configs | **Yes** | `docs/reports/competition_alignment/` | The headline result: official baseline 0.8071 vs V2 0.8456 @20 |
| `experiments/competition_alignment/us_slice_bootstrap/` | **Yes** | `docs/reports/` + `scripts/` | The resume-facing US number and its blocking gate (§10.2a) |
| `experiments/ranking_v2/kdd_task1_benchmark/REPORT.md` + JSON gates | **Yes** | `docs/reports/kdd_task1/` | Benchmark definition of record; the doc §10.3 protects |
| `test_lock_manifest.json`, `gate_a.json`, `split_integrity.json`, `task1_dataset_counts.json`, `scorer_unit_tests.json` | **Yes** | `artifacts/manifests/` | Integrity anchors; small; git-tracked |
| `random_ranking_floor.json`, `oracle_ceiling.json`, `degenerate_query_analysis.json` | **Yes** | `artifacts/manifests/` | Floor/ceiling context that keeps NDCG deltas interpretable |
| `locale_results.csv`, `task1_baseline_comparison.csv` | **Yes** | `artifacts/results/` | Per-locale breakdowns cited across reports |
| `experiments/ranking_v2/official_metric_final/` | **Yes** | `docs/reports/` | Metric-authority record; the origin of §13.2 |
| `experiments/ranking_v2/benchmark_repair/REPORT.md` | **Yes** | `docs/reports/` | Documents the repair; explains why the legacy scorer is barred |
| `experiments/ranking_v2/audit/REPORT.md` + `locale_feature_health.json` | **Yes** | `docs/reports/` | Phase-0 audit; the correct-framing reference per the scope audit |
| `experiments/retrieval_scope_audit/` | **Yes** | `docs/reports/` | Locale-scope audit; source of the README locale note |
| `experiments/two_tower_v2/` reports (`FINAL_REPORT.md`, `phase*/REPORT.md`) | **Yes** | `docs/reports/two_tower_v2/` | Personal V0→V1 experiment record |
| `experiments/ranking_v3/` results | **Yes, labelled `exploratory`** | `experiments/exploratory/results/` | Zero-shot bakeoff; Jina rows are non-commercial (§6) |
| `experiments/ranking_v2/cross_encoder_task1/` train history + prereg | **Yes** | `docs/reports/` + `artifacts/` | Pre-registration is the evidence base for §13.1 |
| `experiments/ranking_v2/*/  _cache/`, `_*.log`, `*_stdout.log` | **No** | — | Already gitignored; regenerable scratch |
| `experiments/audit/` (Phase-0, 12/17-feature MLP) | **No** | — | Bound to the group feature stack (§3.6) |
| `experiments/query_slice_analysis/` | **No** | — | US-only full-catalog pool; §7.4 defer |
| `output/**` (scores, indices, ablations, figures) | **No** | — | Gitignored; belongs to the old two-setting pipeline |
| `screenshots/`, architecture diagrams | **No** | — | Old search-system product surface |
| `experiments/**/checkpoints/`, `final_model/`, `*.safetensors`, `*.faiss` | **Out-of-band only** | `artifacts/` (untracked) | §4; ≈ 4.3 GB |

---

# §6 External Dependencies / Licenses

| Dependency | Source | License | Modified? | NOTICE required? |
|---|---|---|---|---|
| **ESCI dataset** (`shopping_queries_dataset_*`) | `amazon-science/esci-data` | **Apache 2.0** (code); dataset released for research under the repo's terms | No — read-only | **Yes.** Attribute Amazon; keep the KDD Cup 2022 citation |
| **Amazon official Task-1 baseline code** (recipe reproduced by `ca_train_official_baseline.py`) | `amazon-science/esci-data` | **Apache 2.0** | Reproduced, not vendored — our script re-implements the documented recipe | **Yes.** Apache 2.0 requires attribution + license text; state clearly that ours is a *reproduction* |
| `sentence-transformers/msmarco-distilbert-base-v3` | HuggingFace | **Apache 2.0** | Fine-tuned (Two-Tower V0/V2) | Yes — model attribution |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | HuggingFace | **Apache 2.0** | Fine-tuned (official baseline US arm) | Yes |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | HuggingFace | **Apache 2.0** | Used as-is (official baseline ES/JP arms) | Yes |
| `FacebookAI/xlm-roberta-base` | HuggingFace | **MIT** | Fine-tuned (V2 Cross-Encoder base) | Yes |
| `BAAI/bge-reranker-v2-m3` | HuggingFace | **Apache 2.0** | Zero-shot, unmodified | Yes |
| `Qwen/Qwen3-Reranker-0.6B` | HuggingFace | **Apache 2.0** | Zero-shot + LoRA (`v3_train_qwen_lora.py`) | Yes |
| **`jinaai/jina-reranker-v3.5`** | HuggingFace | **⚠️ CC BY-NC 4.0 — NON-COMMERCIAL** | Zero-shot, unmodified | **Yes, plus a usage restriction** |
| `bm25s` | PyPI | MIT | Used as a library | Standard dependency attribution |
| `LightGBM` | PyPI | MIT | Library | Standard |
| `faiss` | PyPI | MIT | Library | Standard |

## ⚠️ Jina v3.5 — non-commercial, hard constraint

`jinaai/jina-reranker-v3.5` is licensed **CC BY-NC 4.0**. It appears at
`scripts/benchmark_modern_rerankers.py:56,215,248-249,296,301` and its outputs are read by
`scripts/v3_ensemble_audit.py:67`.

**Binding rules for the new repo:**

1. **Never describe any Jina-based result as commercial- or production-compatible.** It is not.
2. Every Jina experiment ships under `experiments/exploratory/` and is labelled **`exploratory`**
   in its directory, its report front-matter, and its results JSON.
3. Jina numbers may not appear in the headline results table, the README summary, or any
   production/deployment narrative.
4. `NOTICE` must state the CC BY-NC 4.0 restriction explicitly and name the affected artifacts.
5. If a commercially-usable zero-shot reranker is ever needed, BGE (Apache 2.0) and Qwen
   (Apache 2.0) are already benchmarked in the same harness and carry no such restriction.

**Deliverable:** a `NOTICE` file at the new repo root covering ESCI/Amazon (Apache 2.0), each
HuggingFace base model, and the Jina non-commercial carve-out. Also a `LICENSE` for the new repo's
own code — that choice is a human call and is **not** decided here.

---

# §7 Ownership Boundary and Migration Disposition

## 7.0 What this section is

The ownership split in §7.1 / §7.2 was **confirmed by a human against GitHub history** and is
treated here as given. This section does three things and only three:

1. **Verifies** the given split against local `git` history.
2. **Reports conflicts with evidence** where git disagrees — without overturning the human
   conclusion.
3. **Assigns a migration disposition (A / B / C)** to each group-originated component.

Method: `git log --follow --diff-filter=A` for original authorship, `git blame --line-porcelain`
at `HEAD` for current line ownership, and static import tracing for dependency direction.
All evidence is local-git only; the GitHub PR/review history was not available to this pass.

**Contributor identities** (`git shortlog -sne --all`, 117 commits total):

| Identity | Commits | Maps to |
|---|---:|---|
| `hyukjin17 <chung.hy@northeastern.edu>` + `jin` + `Hyuk Jin Chung` | 65 | teammate |
| `gyuszix <gyuszix@gmail.com>` + `<102871892+gyuszix@…>` | 31 | teammate |
| `Jian Gao <iamjaygao@hotmail.com>` + `iamjaygao <102485789+iamjaygao@…>` + `<forivey@gmail.com>` | 21 | **self** |

---

## 7.1 Verification of the group-originated list

| Component | Primary file(s) | Created by (first commit) | HEAD line ownership | Verdict |
|---|---|---|---|---|
| BM25 initial implementation | `retrieval/bm25.py` | **`iamjaygao`** · 2026-02-18 · `c844e22` | hyukjin17 114 / iamjaygao 41 | ⚠️ **CONFLICT-1** |
| Two-Tower V0 | `retrieval/two_tower.py` | `gyuszix` · 2026-02-24 · `5d9f8e6` | gyuszix 141 / hyukjin17 83 / Jian Gao 16 | ✅ consistent |
| Two-Tower V0 (training) | `scripts/train_two_tower.py` | `gyuszix` · 2026-02-25 · `cfd9f09` | gyuszix 73 / hyukjin17 11 / Jian Gao 8 | ✅ consistent |
| `DeepESCIReranker` | `reranking/model.py` | `gyuszix` · 2026-03-03 · `b1e3776` | gyuszix 16 / Jian Gao 15 | ✅ consistent (see note) |
| `DeepESCIReranker` (features) | `reranking/features.py` | `gyuszix` · 2026-03-03 · `b1e3776` | gyuszix 172 / Jian Gao 1 | ✅ consistent |
| `AdvancedDeepReranker` | `reranking/advanced_model.py` | `hyukjin17` · 2026-03-04 · `0ecc033` | hyukjin17 24 / — | ✅ consistent |
| `AdvancedDeepReranker` (features) | `reranking/advanced_features.py` | `hyukjin17` · 2026-03-04 · `274ce10` | hyukjin17 218 / Jian Gao 19 | ✅ consistent |
| ESCI-S integration | `utility/convert_esci_to_parquet.py` | `jin` · 2026-03-04 · `f239a54` | jin 34 / Jian Gao 18 | ✅ consistent |
| Interactive GUI | `interactive_search.py` | `hyukjin17` · 2026-03-04 · `1aa743a` | hyukjin17 265 / — | ✅ consistent |

**Note on `reranking/model.py` — the rule works exactly as written.** At `HEAD` the file is
nearly half Jian Gao lines (15 of 31), but every one of those lines comes from `5210986`
(2026-08-17, *"fix: two train/serve consistency bugs in reranker and Two-Tower"*), which replaced
`BatchNorm` with `LayerNorm`. `git log -S BatchNorm -- reranking/model.py` returns exactly one
commit: `b1e3776` by `gyuszix`. **Original implementation: gyuszix. Later independent diagnosis
and repair: Jian Gao.** Line-count parity does not transfer ownership, per §7.1's rule. Consistent.

### ⚠️ CONFLICT-1 — BM25 initial implementation

**Reported, not resolved. The human ownership conclusion stands.**

Local git says `retrieval/bm25.py` was **created by `iamjaygao` (Jian Gao)**, 2026-02-18, commit
`c844e22` *"Add BM25 lexical baseline module with NDCG@10 evaluation"* — the repo's founding
commit. The same commit also created `config.py`, `evaluation/metrics.py`, and
`scripts/run_pipeline.py`.

The file was then substantially rewritten by `hyukjin17` across `3d1e374`, `89d755f`, `9188fd3`,
`4f66161`, `e359b4b` (all 2026-03-03, *"use C-native bm25s to speed up inference"* and related).
At `HEAD`, the three functions that matter are entirely hyukjin17's:

| Function | Line | Blame |
|---|---:|---|
| `_score_single_group` (per-query scorer) | `retrieval/bm25.py:13` | hyukjin17 |
| `build_global_bm25_index` | `retrieval/bm25.py:93` | hyukjin17 |
| `search_bm25_global` | `retrieval/bm25.py:118` | hyukjin17 |

**How to read the conflict.** "Group-originated" in the sense of *"produced during the
three-person course project"* is correct — Feb 2026 is inside the course window, and the module
as it stands today is majority-teammate code. The label *"BM25 **initial** implementation"* is
what git contradicts: the initial implementation was Jian Gao's; the *current* implementation is
hyukjin17's rewrite.

**This does not change the disposition** (see 7.3-1) and does not create a personal-authorship
claim: nothing in the new repo should present `retrieval/bm25.py` as personal work either way.

### ⚠️ CONFLICT-2 — four founding files are unclassified

`config.py`, `evaluation/metrics.py`, `scripts/run_pipeline.py` (and `retrieval/bm25.py`) were all
created by Jian Gao in `c844e22` but appear in neither §7.1 nor §7.2. Current state:

| File | HEAD blame | Note |
|---|---|---|
| `config.py` | gyuszix 7 / iamjaygao 5 / hyukjin17 3 | 14-line path/constant file |
| `evaluation/metrics.py` | hyukjin17 91 / iamjaygao 21 | superseded by `kdd_task1_ndcg.py` |
| `scripts/run_pipeline.py` | (founding commit, later edits) | old end-to-end demo driver |

Flagged for a human classification call. Disposition below assumes **A** for all three, since
none is needed by the new repo (see 7.3-7).

---

## 7.2 Verification of the personal-work list

Every file below is **sole-authored by Jian Gao** — created by Jian Gao and 100% Jian Gao at
`HEAD` blame. No teammate line appears in any of them.

| Work item | Files verified | Created | Sole author at HEAD |
|---|---|---|---|
| Full-catalog retrieval evaluation | `scripts/{build_full_catalog_indices,sample_eval_queries,run_full_bm25_retrieval,run_full_tt_retrieval,evaluate_full_retrieval}.py` | 2026-08-17 `eccfbc0` | ✅ |
| RRF / overlap / coverage analysis | `scripts/{evaluate_full_retrieval,analyze_topk_overlap,analyze_label_coverage}.py` | 2026-08-17 `eccfbc0` | ✅ |
| LambdaMART | `scripts/train_lambdamart.py`, `evaluation/evaluate_lambdamart.py` | 2026-08-17 `eccfbc0` | ✅ |
| Feature ablation | `scripts/{run_feature_ablation,run_group_ablation}.py` | 2026-08-17 `eccfbc0` | ✅ |
| Query slicing | `scripts/{build_query_slices,analyze_query_slices}.py` | 2026-08-17 `eccfbc0` | ✅ |
| Training/serving bug diagnosis | `5210986` (8 files, +87/−34) | 2026-08-17 | ✅ (edits to group files) |
| Two-Tower V2 | `scripts/train_two_tower_v2.py`, `retrieval/two_tower_training.py`, `experiments/two_tower_v2/build_query_split.py` | 2026-09-02 `0d62195` | ✅ |
| Ranking audit | `experiments/ranking_v2/audit/scripts/*` | 2026-09-02 `769282c` | ✅ |
| Benchmark repair | `experiments/ranking_v2/benchmark_repair/scripts/*` | 2026-09-02 `4ad8fb1` | ✅ |
| KDD Task 1 alignment | `experiments/ranking_v2/kdd_task1_benchmark/scripts/*` | 2026-09-02 `8145fdf` | ✅ |
| Official metric validation | `experiments/ranking_v2/official_metric_final/scripts/*` | 2026-09-02 `8145fdf` | ✅ |
| Cross-Encoder | `reranking/cross_encoder.py`, `scripts/{train,evaluate,build_}cross_encoder_task1*.py` | 2026-09-02 `9acaf0e` | ✅ |
| Amazon official-baseline reproduction | `scripts/ca_*.py` | **uncommitted** | ⚠️ see 7.6 |
| Paired bootstrap | `experiments/competition_alignment/us_slice_bootstrap/*` | **uncommitted** | ⚠️ see 7.6 |
| Locale analysis / benchmark audit | `experiments/retrieval_scope_audit/*`, `scripts/v3_*.py` | **uncommitted** | ⚠️ see 7.6 |

**Verdict: §7.2 is fully consistent with git history.** No conflicts.

### Dependency screen for personal work

The question that decides migration cost: *does personal work import group-originated code?*

| Personal component | Imports from group code | Blocking? |
|---|---|---|
| **Cross-Encoder** | **none** | ✅ **fully independent** |
| **Two-Tower V2** | none (only `config` + own `two_tower_training.py`) | ✅ independent |
| **KDD Task 1 benchmark / audit / repair / metric validation** | `ALL_FEATURES` from `reranking/advanced_features.py` — a **17-string list**, not an implementation | ⚠️ trivial (inline it) |
| Full-catalog retrieval | `build_global_bm25_index`, `search_bm25_global` (`retrieval/bm25.py`); `build_global_tt_index`, `search_tt_global` (`retrieval/two_tower.py`) — **all hyukjin17** | ❌ real dependency |
| **LambdaMART (legacy path)** `scripts/train_lambdamart.py` | `extract_advanced_features` from `reranking/advanced_features.py` — **218-line hyukjin17 implementation** | ❌ real dependency |
| **LambdaMART (Task 1 path)** `evaluate_task1_baselines.py` | only `ALL_FEATURES`; features are pre-materialized in `train_task1.parquet` | ✅ effectively independent |

Two structural facts drive everything in 7.3:

1. **The Cross-Encoder line — the new repo's headline — has zero group-code dependency.**
2. **`build_task1_benchmark.py:68-99` already contains a self-contained per-query BM25**, written
   by Jian Gao. Its header says *"Transcribed from `retrieval/bm25.py::_score_single_group`"* — it
   does **not** import it. The Task 1 pipeline therefore already runs without `retrieval/bm25.py`.

---

## 7.3 Disposition per group-originated component

Priority applied: **A > C > B**. No component qualified for B.

| # | Component | Original owner/source | Git evidence | Personal later work | Decision | Why | Replacement / dependency plan |
|---|---|---|---|---|---|---|---|
| 1 | **BM25 initial implementation** (`retrieval/bm25.py`) | Created `iamjaygao`; current impl. `hyukjin17` (see CONFLICT-1) | `c844e22` 2026-02-18 (create); `89d755f`/`e359b4b`/`4f66161`/`9188fd3` 2026-03-03 (bm25s rewrite). HEAD: hyukjin17 114 / iamjaygao 41 | Transcribed a minimal per-query scorer into `build_task1_benchmark.py:68-99`; audited its tokenizer (`retrieval_scope_audit.md` §2) | **C** | The ranking repo genuinely needs a `bm25_score` feature, but not the module. The global-index and demo-search surfaces (`build_global_bm25_index`, `search_bm25_global`, `save/load_bm25_index`) belong to the old end-to-end system and are out of scope. | Migrate **only** the personal transcription at `build_task1_benchmark.py:68-99` (~30 lines wrapping `bm25s`) as a standalone `features/bm25_pool.py`. Do **not** migrate `retrieval/bm25.py`. Drop the dead `simple_tokenize()` (`bm25.py:48`, never called). For a *multilingual* repo the English-default tokenizer (`stopwords="english"`, no CJK segmentation) must be re-specified anyway — see `retrieval_scope_audit.md` §2.2. Document per §7.3-C. |
| 2 | **Two-Tower V0** (`retrieval/two_tower.py`, `scripts/train_two_tower.py`) | `gyuszix` | `5d9f8e6` 2026-02-24, `cfd9f09` 2026-02-25. HEAD: gyuszix 141+73 / hyukjin17 83+11 / Jian Gao 16+8 | Fixed a train/serve inconsistency (`5210986`); built V2 as a clean-room replacement; audited training scope | **A** | Two-Tower V2 (100% personal) supersedes it as *code*. The only thing V0 is still needed for is reproducing the frozen `semantic_score` column behind the published LambdaMART number — and that is an **artifact** dependency, not a code dependency. | Do not migrate either file. Carry the **frozen `train_task1.parquet` / `test_task1.parquet`** (features already materialized) as immutable benchmark artifacts, with provenance recorded: `semantic_score` was produced by a group-originated V0 checkpoint. ⚠️ **Do not silently regenerate `semantic_score` with V2** — it would change the LambdaMART baseline and break reproducibility of every number in `kdd_task1_benchmark/REPORT.md`. If regeneration is wanted, publish it as a *new* baseline row, not a restatement. |
| 3 | **`DeepESCIReranker`** (`reranking/model.py`, `features.py`, `scripts/train_reranker.py`) | `gyuszix` | `b1e3776` / `6867b85` 2026-03-03. `git log -S BatchNorm` → `b1e3776` (gyuszix) only | **Root-caused and fixed** the `BatchNorm`/pairwise-training failure — `git log -S LayerNorm` → `5210986` (Jian Gao), sole commit | **A** | A 7-feature pairwise MLP is not part of a multilingual ranking benchmark. The valuable asset is the **diagnosis**, which is prose, not code. | Do not migrate. Keep the debugging write-up (`README.md:94` in this repo) as a portfolio narrative that cites this repo by commit `5210986`. State plainly: *original model implementation by a teammate; the BatchNorm diagnosis and LayerNorm fix are mine.* Migrating the model class would create exactly the ownership ambiguity §7.1 forbids. |
| 4 | **`AdvancedDeepReranker`** (`reranking/advanced_model.py`) | `hyukjin17` | `0ecc033` 2026-03-04. HEAD: hyukjin17 24 / — (0 personal lines) | Used it as the ablation subject (`run_feature_ablation.py`); used it as the LambdaMART comparison baseline | **A** | Not needed. LambdaMART does not depend on the MLP class. The MLP-vs-tree tie is a *result* that can be cited, not code that must ship. | Do not migrate. Cite the 0.8458 vs 0.8464 comparison as a result from the prior repo. If the new repo wants a neural baseline, the Cross-Encoder already is one — and it is 100% personal. |
| 4b | **`advanced_features.py`** (the real dependency) | `hyukjin17` | `274ce10` 2026-03-04. HEAD: hyukjin17 218 / Jian Gao 19 | Fixed a train/serve normalization bug (`5210986`, +26/−7); audited it and found 5 defects | **C** | `extract_advanced_features` is what `scripts/train_lambdamart.py:11` actually imports — the single heaviest group dependency in the personal stack. It also carries known defects: `log_review_count` identically 0, 15.9% of prices parsed 100× too large, 13.7% of stars truncated, `PorterStemmer` applied to Japanese, hard-coded English color list (`ranking_v2/audit/REPORT.md` P1-1…P1-4). | Two-step: (a) for the frozen benchmark, **no rewrite needed** — features are already materialized in `train_task1.parquet`, so only the `ALL_FEATURES` **name list** (17 strings) needs inlining into the new repo; (b) if live feature extraction is ever needed, write a minimal locale-aware producer from scratch. A multilingual repo cannot ship the existing one regardless: its text handling is English-only by construction. Never migrate `extract_advanced_features` verbatim. |
| 5 | **ESCI-S integration** (`utility/convert_esci_to_parquet.py`) | `jin` | `f239a54` 2026-03-04. HEAD: jin 34 / Jian Gao 18 | Hardened chunked-write schema handling (`5210986`, +22/−4) | **A** | The Cross-Encoder line is text-only and needs no ESCI-S field. The only consumer is the 17-feature stack, and that is satisfied by the frozen parquets (row 4b). Per §7.4, migrate the artifact, not the subsystem. | Do not migrate. Drop the ~3.4 GB ESCI-S download from the new repo's setup entirely — a strict simplification. If price/stars features are ever revived, re-derive them with correct locale-aware numeric parsing (the current parser is one of the audited defects), not by porting this converter. |
| 6 | **Interactive GUI** (`interactive_search.py`) | `hyukjin17` | `1aa743a` 2026-03-04. HEAD: hyukjin17 265 / — (0 personal lines) | none | **A** | Squarely the old end-to-end search product. A ranking benchmark repo has no GUI surface. It also depends on `output/bm25s_index` (mixed-locale, **not present on disk**) and would not run today. | Do not migrate. Do not port the MMR diversity logic either — it is teammate code and no reported metric uses it (`README.md:283`). |
| 7 | *(unclassified — CONFLICT-2)* `config.py`, `evaluation/metrics.py`, `scripts/run_pipeline.py` | Created `iamjaygao`, since co-edited | `c844e22` 2026-02-18 | `metrics.py` superseded by `kdd_task1_ndcg.py` (100% personal, unit-tested, sha-pinned) | **A** (pending human confirm) | `metrics.py` is now 91/111 hyukjin17 and is already replaced by the authoritative scorer. `config.py` is a 14-line path stub — trivially rewritten. `run_pipeline.py` drives the old demo. | Do not migrate. New repo gets its own paths module and uses `kdd_task1_ndcg.py` as the single scorer. Flagged for human sign-off since these are outside both §7.1 and §7.2. |

**Summary: 6 × A, 2 × C, 0 × B.** Nothing in the group-originated set is required verbatim for
reproducibility, because the two things the personal pipeline actually needs — the Task 1 feature
matrix and the `semantic_score` column — are **frozen data artifacts**, not live code paths.

### Required documentation wording for the two C decisions

Per §7.3-C, the new repo must state, for both `features/bm25_pool.py` and the `ALL_FEATURES` list:

> The original course project contained a group implementation of this component. The version in
> this repository is a minimal re-implementation written independently for this ranking benchmark.
> No claim is made over the original group implementation.

---

## 7.4 Migration readiness of the personal components

| Personal component | Group-code dependency | Frozen-artifact dependency | Refactor needed? | Verdict |
|---|---|---|---|---|
| **Cross-Encoder** | **none** | `train/dev/test_task1.parquet` (labels + `query`/`product_title`; the CE reads raw text, not the 17 features) | none | ✅ **migrate as-is — make this the repo spine** |
| **Two-Tower V2** | none | `experiments/two_tower_v2/splits/*.txt` (regenerable: `build_query_split.py`, seed 42, deterministic) | none | ✅ migrate as-is |
| **KDD Task 1 alignment / benchmark audit / repair / official metric validation** | `ALL_FEATURES` (17 strings) | `test_task1.parquet` (test-locked, sha-pinned in `test_lock_manifest.json`) | inline one list | ✅ migrate; 1-line change |
| **Amazon official-baseline reproduction + paired bootstrap** | none | `official_baseline_dev_predictions.parquet`, `fulldev_epoch2_predictions.parquet`, 3 locale checkpoints | none | ✅ migrate; **commit first** (7.6) |
| **Locale analysis / retrieval scope audit** | none (read-only over source) | none | none | ✅ migrate as-is |
| **LambdaMART (Task 1 path)** | `ALL_FEATURES` only | `train_task1.parquet` (features pre-materialized) | inline one list | ✅ migrate |
| **LambdaMART (legacy path)** `scripts/train_lambdamart.py` | `extract_advanced_features` (218 lines, hyukjin17) | `output/{bm25,two_tower}_scores_*.csv`, ESCI-S parquet | **yes** — rewrite or drop | ⚠️ migrate the Task 1 path **only**; leave the legacy path behind |
| **Full-catalog retrieval / RRF / overlap / coverage** | `retrieval/bm25.py` + `retrieval/two_tower.py` (both hyukjin17 for the functions used) | `bm25s_index_us`, `tt_index_us.faiss` (~1.2 M products) | **yes** — heaviest coupling in the set | ⚠️ **defer**; see below |
| **Feature ablation / query slicing** | `advanced_features`, `advanced_model`, `evaluate_advanced`, `metrics` | old-pool CSVs | **yes** — 4 group imports | ⚠️ defer or cite as prior-repo results |

**On full-catalog retrieval (Setting 2).** It is genuinely personal work (`eccfbc0`, sole-authored)
and it is the source of the Recall@100 / RRF numbers. But it is also the component most tightly
bound to teammate code — the four functions it calls are all hyukjin17's — and it is **US-only**
(`retrieval_scope_audit.md` §3.2–3.3), so it carries no multilingual evidence at all. Against
§7.4's "新 repo 是 `multilingual-product-ranking`,不是旧 search system 的复制品": **defer it out
of the initial migration.** Cite Recall@100 / RRF as results from the prior repo. If it is later
wanted, rebuild the retriever interface from scratch — by then it should be multilingual anyway,
which the current US-only index cannot be.

---

## 7.5 Ownership table (consolidated — the §7.5 deliverable)

| Component | Original owner/source | Git evidence | Personal later work | Decision A/B/C | Why | Replacement / dependency plan |
|---|---|---|---|---|---|---|
| BM25 initial implementation | Created by Jian Gao; **current impl. hyukjin17** (CONFLICT-1) | `c844e22` (2026-02-18, create) → `89d755f`,`9188fd3`,`4f66161`,`e359b4b` (2026-03-03, bm25s rewrite). HEAD 114/41 | Minimal transcription in `build_task1_benchmark.py:68-99`; tokenizer audit | **C** | Need the feature, not the module; global-index surface is out of scope | Ship the personal ~30-line per-query scorer as `features/bm25_pool.py`; re-specify the tokenizer for multilingual use; drop `retrieval/bm25.py` |
| Two-Tower V0 | gyuszix | `5d9f8e6` (2026-02-24), `cfd9f09` (2026-02-25). HEAD 141/83/16 | Train/serve fix `5210986`; V2 clean-room rebuild; scope audit | **A** | V2 supersedes the code; only an artifact dependency remains | Carry frozen `*_task1.parquet` with `semantic_score` provenance noted; do not regenerate silently |
| `DeepESCIReranker` | gyuszix | `b1e3776`, `6867b85` (2026-03-03); `-S BatchNorm` → gyuszix only | BatchNorm→LayerNorm diagnosis + fix, `5210986` (`-S LayerNorm` → Jian Gao only) | **A** | Out of scope; the asset is the diagnosis, which is prose | Keep the debugging narrative citing `5210986`; explicitly attribute the original model to a teammate |
| `AdvancedDeepReranker` | hyukjin17 | `0ecc033` (2026-03-04). HEAD 24/0 | Ablation subject; LambdaMART comparison baseline | **A** | Not needed; Cross-Encoder is the personal neural baseline | Cite 0.8458 vs 0.8464 as a prior-repo result |
| `advanced_features.py` *(added row — the actual blocker)* | hyukjin17 | `274ce10` (2026-03-04). HEAD 218/19 | Normalization bug fix `5210986`; 5 defects found in audit | **C** | `train_lambdamart.py:11` imports it; carries known defects; English-only text handling | Inline the 17-name `ALL_FEATURES` list; features come pre-materialized from frozen parquets; write a locale-aware producer from scratch only if live extraction is ever needed |
| ESCI-S integration | jin | `f239a54` (2026-03-04). HEAD 34/18 | Chunked-write hardening `5210986` | **A** | Cross-Encoder is text-only; frozen parquets cover the feature path | Drop the 3.4 GB ESCI-S dependency from setup entirely |
| Interactive GUI | hyukjin17 | `1aa743a` (2026-03-04). HEAD 265/0 | none | **A** | Old search product; no GUI surface in a benchmark repo; would not run today (missing index) | Do not migrate; do not port MMR either |
| **LambdaMART** | **Jian Gao** | `eccfbc0` (2026-08-17), sole author, 100% blame | — (is the personal work) | **migrate (Task 1 path)** | Personal; Task 1 path needs only a name list | Migrate `evaluate_task1_baselines.py` path; leave `scripts/train_lambdamart.py` (imports `extract_advanced_features`) behind |
| **Two-Tower V2** | **Jian Gao** | `0d62195` (2026-09-02), sole author, 100% blame | — | **migrate as-is** | Zero group dependency | Splits regenerate deterministically (seed 42) |
| **Cross-Encoder** | **Jian Gao** | `9acaf0e` (2026-09-02), sole author, 100% blame | — | **migrate as-is** | **Zero group imports — verified** | Make it the repo spine; only needs the frozen Task 1 pairs |
| **Benchmark audit / repair / KDD alignment / official metric validation** | **Jian Gao** | `769282c`, `4ad8fb1`, `8145fdf` (2026-09-02), sole author, 100% blame | — | **migrate as-is** | Only `ALL_FEATURES` couples it | Inline the list; `kdd_task1_ndcg.py` is already the sole authoritative scorer (sha-pinned) |

---

## 7.6 Provenance — resolved

**Status: CLOSED, 2026-09-03.** The 16 untracked personal artifacts previously flagged here as a
blocking issue are all committed and pushed. `origin/main` is now `7977f59`.

| Commit | Contents |
|---|---|
| `dc86819` | `scripts/ca_*.py`, `scripts/v3_*.py`, `scripts/benchmark_modern_rerankers.py` |
| `c78abcf` | `experiments/retrieval_scope_audit/`, `experiments/ranking_v3/`, `FROZEN_V2.json` |
| `694251a` | `experiments/competition_alignment/` (incl. `us_slice_bootstrap/`) |
| `955eae8` | `experiments/competition_alignment/ARTIFACTS.md` |
| `af4f490` | Cross-encoder V2 dev metrics + `prereg_prediction.md` |
| `7977f59` | `train_history.json` (`git add -f`, bypassing `.gitignore`) |

Verified: `git rev-parse origin/main` → `7977f59891be1ffdd1c41b9f8c08af87ab1fc8db`;
`git status --porcelain` shows no untracked personal source or report files.

Every item §7.2 lists as personal work — including the Amazon official-baseline reproduction, the
paired bootstrap, the locale analysis and the benchmark audit — now carries a dated, authored
commit in this repository. **The ownership boundary in §7.1–§7.5 is backed by git provenance and
is migration-ready.**

### One caveat still stands — binary artifacts remain untracked

`.gitignore` continues to exclude model weights and prediction parquets — **≈ 4.3 GB**. These are
deliberately out of git and are **not** covered by the commits above:

| Untracked artifact | Regeneration entry point |
|---|---|
| `experiments/competition_alignment/official_baseline_models/*/model.safetensors` (1.1 GB) | `scripts/ca_train_official_baseline.py` |
| `experiments/competition_alignment/official_baseline_dev_predictions.parquet` (2.2 MB) | `scripts/ca_eval_official_baseline.py` |
| `experiments/ranking_v2/cross_encoder_task1/*/checkpoints/` (2.1 GB) | cross-encoder training scripts |
| `experiments/ranking_v2/cross_encoder_task1/full_run/fulldev_epoch2_predictions.parquet` (2.4 MB) | cross-encoder eval scripts |

The authoritative record is
[`experiments/competition_alignment/ARTIFACTS.md`](experiments/competition_alignment/ARTIFACTS.md).

**Migration requirement:** these must be carried **out-of-band** (direct file copy, not `git`) or
regenerated per `ARTIFACTS.md`. Note the ordering constraint stated there — the US-slice bootstrap
reads the frozen prediction parquets, so those must exist before it can run. Carrying them by copy
is strongly preferred over regeneration: regeneration re-runs training and cannot be guaranteed
bit-identical across environments, which would break the blocking reproduction gate in §10.2a.

---

# §7.7 Cross-reference: conflicts surfaced by the §3 file-level scan

§3 traced imports symbol-by-symbol rather than module-by-module, which surfaced one place where
§7.4's dependency screen was optimistic. **Reported, not overturned — §7 stands.**

### ⚠️ CONFLICT-3 — two experiment clusters import a group *class*, not just `ALL_FEATURES`

§7.4 records the dependency for "KDD Task 1 alignment / benchmark audit / repair / official metric
validation" as *"`ALL_FEATURES` (17 strings) — trivial (inline it)"*. That is exact for the
`kdd_task1_benchmark` cluster, but **incomplete for two others**:

| File | Line | Import | §7.4 said | Actually |
|---|---:|---|---|---|
| `experiments/ranking_v2/official_metric_final/scripts/evaluate_official_baselines.py` | 39 | `from reranking.advanced_model import AdvancedDeepReranker` | ALL_FEATURES only | + group **class** |
| `experiments/ranking_v2/benchmark_repair/scripts/evaluate_baselines.py` | 38 | `from reranking.advanced_model import AdvancedDeepReranker` | ALL_FEATURES only | + group **class** |
| `experiments/ranking_v2/audit/scripts/00_build_caches.py` | 29-32 | `extract_advanced_features`, `ALL_FEATURES`, `AdvancedDeepReranker`, `extract_test_advanced_features`, `apply_business_ndcg_labels` | ALL_FEATURES only | + **4 group implementations** |

**Why this does not change §7.** In all three cases the group symbol is imported for exactly one
purpose: to score the **MLP baseline row** in a comparison table. §7.3 row 4 already decides
`AdvancedDeepReranker` = **A (do not migrate)**. Once the MLP row is dropped, the import becomes
dead and the script migrates cleanly. The consequence is a *decision downgrade in §3* —
`MIGRATE` → `MIGRATE_AND_REFACTOR` for those files — not an ownership change.

`00_build_caches.py` is the exception: it is a cache builder whose entire job is producing the
17-feature MLP cache, so it is `KEEP_IN_OLD_REPO` rather than refactorable (§3).

---

# §8 Proposed New Repo Structure

Derived from the §3 decisions — every directory below has at least one file assigned to it in §3.
Two structural facts from §7.2 drive the shape:

- **The Cross-Encoder has zero group-code imports**, so it is the spine: `src/ranking/cross_encoder.py`
  sits at the centre and everything else is arranged around it.
- **`build_task1_benchmark.py:68-99` already contains a self-contained personal BM25**, so BM25
  enters the new repo as a small feature producer under `src/features/` — *not* as a retrieval
  subsystem. There is no `src/retrieval/bm25.py` and no global index anywhere in this tree.

A third fact, from §4, drives the top-level split: **the minimum loop needs only the Layer 1 core
pool, which the repo rebuilds itself.** Everything that depends on the group-derived Layer 2
feature matrix is quarantined under a single clearly-marked `legacy/` tree, so the main path can
be read, run and audited without it. `legacy/` is optional at every level: delete it and the
minimum loop still closes.

```
multilingual-product-ranking/
├── README.md
├── LICENSE                              # new-repo code license (human call, §6)
├── NOTICE                               # Apache 2.0 attributions + Jina CC BY-NC carve-out (§6)
├── MIGRATION_PLAN.md                    # this document
├── pyproject.toml                       # MUST pin scikit-learn==1.8.0 (§4.1) — a different
│                                        # version silently changes the train/dev split
│
├── src/
│   ├── ranking/
│   │   └── cross_encoder.py             # ← SPINE. from reranking/cross_encoder.py, 0 group imports
│   │
│   ├── metrics/
│   │   ├── kdd_task1_ndcg.py            # THE scorer. sha-pinned, sole authority (§13.2)
│   │   └── official_kdd_ndcg.py         # parity reference implementation
│   │                                    # NOTE: no legacy scorer here, by hard rule (§11)
│   ├── data/
│   │   ├── task1_common.py              # pool constants + TEST guard (§13.1 lands here)
│   │   ├── build_task1_pool.py          # ← LAYER 1 ONLY. Rebuilds the core pool from raw ESCI.
│   │   │                                #   Text + labels + gain. No feature columns, ever.
│   │   │                                #   Deterministic; requires scikit-learn==1.8.0 (§4.1)
│   │   ├── verify_split_integrity.py     # ← rebuild oracle check vs split_integrity.json (§10.4)
│   │   ├── build_query_split.py         # deterministic, seed 42
│   │   ├── build_clean_pool.py
│   │   ├── validate_pool.py
│   │   ├── validate_p0.py
│   │   └── gates/
│   │       ├── gate_a.py
│   │       ├── gate_a_operative.py
│   │       ├── gate_a_diagnostic.py
│   │       ├── dataset_counts.py
│   │       └── audit_data_source.py
│   │
│   ├── features/
│   │   └── bm25_pool.py                 # ← the personal per-query BM25, extracted from :68-99.
│   │                                    #   Minimal re-implementation; §7.3-C banner required.
│   └── retrieval/
│       └── two_tower_training.py        # Two-Tower V2 only. No V0, no global index, no FAISS.
│
├── scripts/
│   ├── build_ce_data.py                 # TEST-locked by default (§13.1)
│   ├── train_cross_encoder.py
│   ├── evaluate_cross_encoder.py
│   ├── reproduce_official_baseline.py
│   ├── eval_official_baseline.py        # canonical paired-bootstrap config
│   ├── us_slice_bootstrap.py            # blocking gate §10.2a
│   ├── metric_parity.py
│   ├── protocol_audit.py
│   ├── evaluate_task1_baselines.py      # Layer 1 baselines only (random/BM25/CE).
│   │                                    #   LambdaMART block lives in legacy/ (§10.1)
│   ├── evaluate_official_metric.py      # MLP baseline row removed (CONFLICT-3)
│   ├── train_two_tower_v2.py
│   ├── make_benchmark_comparison.py
│   └── build_terrier_files.py
│
├── legacy/                              # ⚠️ OPTIONAL — Phase 6b only. Depends on the Layer 2
│   │                                    #   feature matrix, whose `semantic_score` column was
│   │                                    #   produced by a GROUP-ORIGINATED Two-Tower V0
│   │                                    #   checkpoint that is NOT in this repository (§4.2).
│   │                                    #   Delete this directory and the minimum loop still
│   │                                    #   closes. Nothing under src/ or scripts/ imports it.
│   ├── README.md                        # states the Layer 2 provenance + the §10.3 rule
│   ├── features/
│   │   └── build_feature_matrix.py      # Layer 2 append path, split out of
│   │                                    #   build_task1_benchmark.py:304 (§3)
│   ├── train_lambdamart_task1.py        # the Phase 6b LambdaMART block
│   └── manifests/
│       └── semantic_score_sha256.json   # §10.3 precondition record
│
├── experiments/
│   └── exploratory/                     # ⚠️ non-headline. Jina rows are CC BY-NC 4.0 (§6)
│       ├── benchmark_modern_rerankers.py
│       ├── v3_audit_and_freeze.py
│       ├── v3_ensemble_audit.py
│       ├── v3_build_train50k.py
│       ├── v3_build_funnel_subsets.py
│       ├── v3_train_qwen_lora.py
│       └── results/
│
├── artifacts/                           # small tracked manifests; large binaries untracked
│   ├── ARTIFACTS.md                     # ← the out-of-band register (successor to the old repo's
│   │                                    #   experiments/competition_alignment/ARTIFACTS.md).
│   │                                    #   Lists every untracked binary, its size, its
│   │                                    #   regeneration entry point, and — for Layer 2 — its
│   │                                    #   group-originated provenance. Git-tracked.
│   ├── FROZEN_V2.json
│   ├── manifests/
│   │   ├── test_lock_manifest.json      # TEST integrity anchor
│   │   └── split_integrity.json         # ← Layer 1 rebuild oracle (§10.4)
│   │   ├── gate_a.json
│   │   ├── split_integrity.json
│   │   ├── task1_dataset_counts.json
│   │   └── scorer_unit_tests.json
│   ├── results/
│   │   ├── locale_results.csv
│   │   └── task1_baseline_comparison.csv
│   └── splits/
│       ├── train_queries.txt
│       └── dev_queries.txt
│
├── data/                                # gitignored
│   ├── raw/                             # upstream ESCI parquets — never vendored, fetched
│   ├── task1/                           # LAYER 1 core pool — REBUILT here, not transferred
│   ├── predictions/                     # frozen prediction parquets (§10.2a oracle)
│   └── legacy_features/                 # LAYER 2 — out-of-band, Phase 6b only, may be absent
│
├── docs/
│   ├── cross_encoder.md
│   ├── prior_work.md                    # ← all DOCUMENT_ONLY items (§3.9), with attribution
│   ├── governance.md                    # §13 rules, as user-facing doc
│   ├── prereg/
│   │   └── ce_v2_prereg.md
│   └── reports/
│       ├── competition_alignment/
│       ├── us_slice_bootstrap.md
│       ├── kdd_task1/
│       ├── official_metric_final.md
│       ├── benchmark_repair.md
│       ├── ranking_audit.md
│       ├── retrieval_scope_audit.md
│       └── two_tower_v2/
│
└── tests/
    ├── test_kdd_task1_ndcg.py
    ├── test_official_kdd_ndcg.py        # legacy-comparison arm removed (§11)
    ├── test_cross_encoder.py
    ├── test_split_integrity.py          # ← Layer 1 rebuild oracle (§10.4), always runs
    ├── test_legacy_lambdamart.py        # ← §10.3 sha256 gate; SKIPPED unless Layer 2 present
    └── test_governance.py               # ← NEW: asserts §13.1 and §13.2 are enforced
```

**Two-layer separation, made structural.** `src/data/build_task1_pool.py` emits Layer 1 only;
everything Layer 2 lives under `legacy/`. No module under `src/` or `scripts/` imports anything
from `legacy/` — that is smoke test #13 (§10.5). A clone with `legacy/` and
`data/legacy_features/` removed still passes §10.2a and §10.4.

**Absent by decision, not by omission:** no `interactive_search.py`, no `src/retrieval/bm25.py`,
no FAISS index builder, no `evaluation/metrics.py`, no `reranking/advanced_*.py`, no ESCI-S
converter, no `config.py`. Each is a §7.3 **A** or a §7.4 defer; §11 lists them with reasons.

---

# §9 Migration Order

Six phases, ordered by real dependency. Each phase is independently verifiable — do not start a
phase until its predecessor's exit criteria pass.

### Phase 0 — Scaffold and legal

- **Preconditions:** none.
- **Do:** create the repo, `pyproject.toml`, `.gitignore`, `LICENSE`, and the `NOTICE` file
  covering §6 (Apache 2.0 attributions + the Jina CC BY-NC carve-out). Copy `MIGRATION_PLAN.md`.
- **Produces:** an empty but legally correct repo.
- **Exit:** `NOTICE` names ESCI/Amazon, every HF base model, and the Jina restriction.

### Phase 1 — Metric authority

- **Preconditions:** Phase 0.
- **Do:** migrate `kdd_task1_ndcg.py` and `official_kdd_ndcg.py` into `src/metrics/`, plus both
  unit-test files. Drop the legacy-comparison arm from `test_official_kdd_ndcg.py` (§11).
- **Produces:** one scorer, with no second scorer available to accidentally import.
- **Exit:** `pytest tests/test_kdd_task1_ndcg.py` passes; `scorer_sha256()` recorded in
  `artifacts/manifests/`; `grep -rn "evaluation.metrics\|official_ndcg" src/ scripts/` returns nothing.

**This is first for a reason.** Every later phase's verification depends on the scorer, and §13.2
exists because a metric problem discovered late cost this project a full rework.

### Phase 2 — Data layer and the TEST lock

- **Preconditions:** Phase 1.
- **Do:** migrate `task1_common.py`, the five `gates/` scripts, `build_task1_pool.py`
  (`ALL_FEATURES` inlined), and extract `src/features/bm25_pool.py` from `:68-99` with its §7.3-C
  banner. **Implement §13.1's TEST assert now**, before any data is read in anger. Carry the three
  **Layer 1 core pool by rebuilding it from raw ESCI** (§4.1) — no parquet transfer. Pin
  `scikit-learn==1.8.0`.
- **Produces:** a reproducible pool builder and an enforced TEST boundary.
- **Exit:** Gate A passes; counts match 14,496 queries / 336,373 rows and locale split
  us 8,956 · jp 3,123 · es 2,417; **§10.4's Layer 1 rebuild check passes** against
  `split_integrity.json` (train 28,733 / dev 5,071 / test 14,496 queries); the `scikit-learn==1.8.0`
  pin is in the dependency file; a test proves the TEST loader raises without the
  pre-registration flag. **No `*_task1.parquet` was transferred** — the pool was rebuilt.

### Phase 3 — Cross-Encoder spine

- **Preconditions:** Phase 2.
- **Do:** migrate `cross_encoder.py` (repoint `TASK1_SCORER_PATH`), the three CE scripts, and
  `tests/test_cross_encoder.py`. Copy the V2 checkpoint out-of-band.
- **Produces:** the repo's headline model, runnable end to end.
- **Exit:** with the frozen checkpoint, `evaluate_cross_encoder.py` reproduces
  **0.8834 / 0.8456 / 0.7924** bit-for-bit (§10.2a).

### Phase 4 — Official baseline, bootstrap, locale slicing

- **Preconditions:** Phase 3.
- **Do:** migrate the four `ca_*.py` scripts and `run_us_slice_bootstrap.py`. Carry the three
  locale checkpoints and both prediction parquets out-of-band.
- **Produces:** **the minimum viable loop is now closed** (§10.1).
- **Exit:** official baseline reproduces **0.8521 / 0.8071 / 0.7358**; US slice (3,133 queries)
  reproduces official **0.834539** / V2 **0.855827**; the global paired bootstrap reproduces
  Δ +0.0385, CI [+0.0348, +0.0421]; the US bootstrap reproduces Δ +0.021288,
  CI [+0.017255, +0.025260], W/L/T 1852/1160/121.

### Phase 5 — Reports and governance

- **Preconditions:** Phase 4.
- **Do:** migrate all `docs/reports/` markdown, write `docs/prior_work.md` covering every
  DOCUMENT_ONLY item in §3.9 with attribution, write `docs/governance.md`, and add
  `tests/test_governance.py`.
- **Produces:** a repo whose claims are auditable and whose ownership boundary is stated.
- **Exit:** every migrated report's internal links resolve; `docs/prior_work.md` names the
  teammate-originated components it cites; `tests/test_governance.py` passes.

### Phase 6 — Optional add-ons (not part of the minimum loop)

- **Preconditions:** Phase 5. **Each item below is independently optional.**
- **6a — Two-Tower V2:** migrate `two_tower_training.py`, `train_two_tower_v2.py`,
  `build_query_split.py`. Exit: split regenerates deterministically to 18,800 / 2,088 queries.
- **6b — LambdaMART (Task 1 path only):** migrate the LambdaMART block of
  `evaluate_task1_baselines.py` into `legacy/train_lambdamart_task1.py`, and the Layer 2 feature
  append path into `legacy/features/build_feature_matrix.py`. **Explicitly outside the minimum
  loop, per §10.1.** Requires the Layer 2 feature matrix out-of-band (§4.2).
  **Precondition:** the §10.3 `semantic_score` sha256 check must pass *before* running anything.
  Exit: 0.842878 reproduces; `legacy/manifests/semantic_score_sha256.json` written.
- **6c — Exploratory rerankers:** migrate the V3 cluster under `experiments/exploratory/` with the
  §6 Jina labelling. Exit: no Jina number appears outside `experiments/exploratory/`.
- **6d — Benchmark-repair remnants:** `build_clean_pool.py`, `validate_pool.py`, `validate_p0.py`,
  `evaluate_official_metric.py` (MLP row removed). Exit: no `advanced_model` import survives.

---

# §10 Validation Checklist

## 10.1 Minimum viable loop — definition

> **Minimum loop =** ESCI raw data → Task 1 competition-aligned pool/split → Amazon official
> baseline → Cross-Encoder → NDCG (full / @20 / @10) → paired bootstrap → locale slicing.

Closed at the end of **Phase 4**.

**LambdaMART is NOT part of the minimum loop.** Even though its Task 1 path is technically
migratable — it depends only on the `ALL_FEATURES` name list (`evaluate_task1_baselines.py:28`),
with features already materialised in the Layer 2 matrix — it is handled as a **separate
optional phase (6b)**. Three reasons:

1. The legacy path imports `extract_advanced_features` (`scripts/train_lambdamart.py:11`), 218
   lines of teammate code (§7.4).
2. Under the official metric its advantage over the MLP is a **statistical null**: **+0.00062,
   CI [+0.00009, +0.00115], p = 0.023, with 54.6% of queries exactly tied**
   (`official_metric_final/REPORT.md:47-49`). A result that thin does not belong in a
   load-bearing loop.
3. It is the **only** consumer of the group-derived Layer 2 feature matrix (§4.2). Keeping it out
   of the loop is what lets the loop run on rebuilt-from-raw data with zero group artifacts.

**Layer 2 is NOT in the minimum loop.** Verified by grep across all eleven loop files: none reads
`semantic_score`, `bm25_score`, or any of the 17 features. The loop consumes **Layer 1 only**
(§4.1), which the new repo rebuilds from raw ESCI.

**Full-catalog retrieval is NOT in the minimum loop**, per the §7.4 defer: its four called
functions are teammate-authored, and it is US-only, so it carries no multilingual evidence.

## 10.2 Acceptance criteria — two tiers

### 10.2a — BLOCKING: frozen-checkpoint reproduction must be bit-exact

> **Input scope: Layer 1 core pool + frozen checkpoints. Layer 2 need not be present.**
>
> Both models under test are **text models**. The official baseline scores
> `(query, product_title)` (`ca_eval_official_baseline.py:62-68`); the Cross-Encoder scores
> query + product text assembled from the raw products table
> (`cross_encoder.py:49-54`, `build_cross_encoder_task1_data.py:49,72`). The scorer reads only
> `esci_label`/`gain` (`kdd_task1_ndcg.py`). **No component in this gate reads
> `semantic_score`, `bm25_score`, or any of the 17 engineered features** — verified by grep across
> all eleven minimum-loop files (§10.1).
>
> This gate is therefore executable with **zero group-derived artifacts** in the new repo.

Using the **frozen checkpoints** carried out-of-band, the migrated eval pipeline must reproduce
these **exactly**. This is pure scoring + metric computation — deterministic, no training, no RNG
beyond the seeded bootstrap. **Any deviation means the migration introduced a bug.**

| Model | full | @20 | @10 |
|---|---:|---:|---:|
| Amazon official baseline | **0.8521** | **0.8071** | **0.7358** |
| V2 Cross-Encoder | **0.8834** | **0.8456** | **0.7924** |

Additional blocking gates — **all Layer 1 only**; every row below is computed from
`query_id` / `product_id` / `product_locale` / `esci_label` / `gain` plus a frozen score column:

| Gate | Expected | Source |
|---|---|---|
| US slice, n queries | **3,133** | `us_slice_bootstrap.json` |
| US slice NDCG@20, official | **0.834539** | idem |
| US slice NDCG@20, V2 | **0.855827** | idem |
| US paired bootstrap Δ / CI | **+0.021288**, [+0.017255, +0.025260] | seed 42, 10,000 resamples |
| US W/L/T | **1852 / 1160 / 121** | idem |
| Global paired bootstrap Δ / CI | **+0.0385**, [+0.0348, +0.0421] | `ca_eval_official_baseline.py:110-115` |
| Global W/L/T | **3229 / 1696 / 146** | idem |
| `metric_version` string | `kdd_task1_ndcg@6946489c8c09dc37\|gain=competition_E1_S0.1_C0.01_I0\|idcg0=exclude\|tie=deterministic` | scorer |
| Task 1 pool counts | 14,496 queries / 336,373 rows; us 8,956 · jp 3,123 · es 2,417 | `test_lock_manifest.json` |
| Layer 1 rebuild integrity | matches `split_integrity.json` | **§10.4** |

Bootstraps are seeded (`RandomState(42)`, 10,000 resamples), so they are reproducible too — the
seed is part of the contract, not a nuisance parameter.

### 10.2b — NON-BLOCKING: retraining from config

Retraining from the migrated configs should land inside a self-declared tolerance band. Cross-
environment training is not deterministic (MPS/CUDA kernels, cuDNN autotuning, dataloader worker
scheduling), so an exact match is **not** required and its absence is **not** a migration failure.

Declare the band before retraining, not after. Suggested: ±0.003 NDCG@20 for the CE, ±0.005 for
the official baseline arms. A miss outside the band is a signal to investigate, not an automatic
block.

**Do not let 10.2b results overwrite 10.2a numbers in any report.** They are different
measurements: 10.2a proves the *pipeline* migrated correctly; 10.2b probes *training*
reproducibility.

## 10.3 `semantic_score` immutability — Phase 6b precondition, NOT a minimum-loop blocker

> **Applies only when the optional Phase 6b (historical LambdaMART reproduction) is executed.**
> The minimum loop does not read this column, so this check does not gate it.
>
> `semantic_score` is produced by the group-originated Two-Tower **V0** checkpoint
> (`build_task1_semantic_scores.py:46,93-94`; `"model_frozen": True` at `:148`).
>
> **Regenerating that column with V2 is forbidden.** Doing so changes every number in
> `kdd_task1_benchmark/REPORT.md`, including the externally-cited LambdaMART **0.8429**.
>
> **Before executing Phase 6b**, verify that the `semantic_score` column of the Layer 2 feature
> matrix is bit-identical to the old repo's (sha256 comparison; result recorded in the manifest).
>
> **If a V2-based semantic score is genuinely wanted**, it must be published as a **new, additional
> baseline row** — never as a rewrite of an existing number.

**Procedure, when Phase 6b is run:**

1. Compute the sha256 of the `semantic_score` column of the Layer 2 `train`/`test` feature matrices
   in the old repo (canonicalised: sort by `example_id`, fixed float serialisation).
2. Recompute in the new repo after the out-of-band copy. **They must be bit-identical.**
3. Record both in `artifacts/legacy/manifests/semantic_score_sha256.json`, alongside the producing
   checkpoint's identity and the note that its producer is group-originated and did not migrate.
4. Wire the check into `tests/test_legacy_lambdamart.py`, **skipped by default** and collected only
   when Layer 2 is present — so a repo without Layer 2 has a green test suite rather than a
   permanently failing one.

If a V2-based column is ever added, name it differently (e.g. `semantic_score_v2`) so the two can
never be silently confused by a downstream join, and keep the V0 row in the same table.

## 10.4 Layer 1 rebuild verification — BLOCKING (minimum loop)

Because the new repo **rebuilds** the core pool rather than receiving it, the rebuild itself must
be verified. Compare against the tracked `split_integrity.json` (§4), which records exactly the
right quantities.

| Check | Expected (from `split_integrity.json`) |
|---|---|
| `query_counts` | train **28,733** · dev **5,071** · test **14,496** |
| `row_counts` | train **663,407** · dev **118,231** · test **336,373** |
| `locale_distribution` train | us .6179 · es .1666 · jp .2155 |
| `locale_distribution` dev | us .6178 · es .1666 · jp .2155 |
| `locale_distribution` test | us .6178 · es .1667 · jp .2154 |
| `depth_distribution`, `label_distribution` | must match per split |
| Query-level disjointness | train∩dev, train∩test, dev∩test all **empty** |
| `split_method` string | `query-level train_test_split(test_size=0.15, random_state=42, stratify=product_locale); a query belongs to exactly one split; no row-level splitting` |

**Any mismatch means the scikit-learn version changed or the upstream ESCI data changed.** Both are
silent failures otherwise: a different partition still produces plausible-looking NDCG numbers.
Check the `scikit-learn==1.8.0` pin (§4.1) first.

This check must run **before** §10.2a, since 10.2a's dev-set numbers are defined on this partition.

## 10.5 Additional smoke tests (derived from §3)

| # | Check | Rationale |
|---|---|---|
| 1 | `grep -rn "from evaluation.metrics\|import official_ndcg" src/ scripts/ tests/` → **empty** | §11 hard rule: legacy scorer must not enter, not even as a reference |
| 2 | `grep -rn "advanced_model\|advanced_features\|AdvancedDeepReranker" src/ scripts/` → **empty** | CONFLICT-3 refactors must actually have removed the group class |
| 3 | `grep -rn "extract_advanced_features" .` → **empty** | The 218-line teammate implementation must not have travelled |
| 4 | `python -c "import src.ranking.cross_encoder"` succeeds with no `sys.path` hacks | The spine must be a real package, not path-injected (`ca_*.py:26-30`) |
| 5 | `load_task1_scorer()` resolves without `sys.path.insert` | `TASK1_SCORER_PATH` (`cross_encoder.py:60`) was correctly repointed |
| 6 | `assert_gain_agreement()` passes (`cross_encoder.py:77-83`) | Gain vector has not drifted from the scorer's |
| 7 | Every results JSON carries `candidate_pool` **and** `metric_version` | §13.2 |
| 8 | `experiments/exploratory/` is the only path containing "jina" | §6 non-commercial containment |
| 9 | `build_ce_data.py --split test` exits non-zero without the pre-registration flag | §13.1 |
| 10 | `src/features/bm25_pool.py` carries the §7.3-C attribution banner | Ownership documentation requirement |
| 11 | No file under `src/`, `scripts/`, `tests/` is authored by a teammate in the old repo's blame | Ownership boundary holds in practice |
| 12 | `docs/prior_work.md` names teammates for each DOCUMENT_ONLY item | Attribution for cited results |
| 13 | `grep -rn "legacy" src/ scripts/ --include=*.py` → **empty** | Two-layer separation (§8): nothing on the main path may import Layer 2 |
| 14 | With `legacy/` and `data/legacy_features/` deleted, §10.2a and §10.4 still pass | Proves the minimum loop has zero group-derived artifact dependency (§1) |
| 15 | `grep -rn "semantic_score\|bm25_score\|ALL_FEATURES" src/ranking/ scripts/{train,evaluate}_cross_encoder.py scripts/*official*.py scripts/us_slice_bootstrap.py` → **empty** | The premise of the layering; verified true in the old repo before migration |
| 16 | Dependency file pins `scikit-learn==1.8.0` exactly | §4.1 — a version drift silently repartitions train/dev |

---

# §11 Files Explicitly NOT Migrated

Consolidates every **A** decision from §7.3 and every deferred component from §7.4.

| # | Not migrated | Reason |
|---|---|---|
| 1 | `retrieval/two_tower.py`, `scripts/train_two_tower.py` | §7.3-2 **A**. Two-Tower V0, gyuszix-originated. V2 supersedes the code; only the frozen `semantic_score` *column* travels (§10.3) |
| 2 | `reranking/model.py`, `reranking/features.py`, `scripts/train_reranker.py`, `evaluation/evaluate_reranker.py` | §7.3-3 **A**. `DeepESCIReranker` cluster, gyuszix-originated. The BatchNorm diagnosis migrates as prose (§3.9), the model does not |
| 3 | `reranking/advanced_model.py` | §7.3-4 **A**. hyukjin17, 24/0 lines. The Cross-Encoder is the personal neural baseline |
| 4 | `reranking/advanced_features.py` | §7.3-4b **C**. Only the 17-name `ALL_FEATURES` list is re-declared. The 218-line implementation carries 5 known defects and is English-only by construction |
| 5 | `scripts/train_adv_reranker.py`, `evaluation/evaluate_advanced.py` | Advanced-MLP cluster; depends on #3/#4 |
| 6 | `utility/convert_esci_to_parquet.py` | §7.3-5 **A**. ESCI-S dropped entirely; ~3.4 GB removed from setup |
| 7 | `interactive_search.py` | §7.3-6 **A**. Old search product; would not run today (`output/bm25s_index` absent) |
| 8 | `retrieval/bm25.py` | §7.3-1 **C**. The capability ships as the personal `:68-99` transcription in `src/features/bm25_pool.py`. Global-index/demo surface is out of scope |
| 9 | Full-catalog retrieval cluster — `build_indices.py`, `build_full_catalog_indices.py`, `run_full_bm25_retrieval.py`, `run_full_tt_retrieval.py`, `evaluate_full_retrieval.py`, `sample_eval_queries.py`, `analyze_label_coverage.py`, `mine_hard_negatives.py`, `evaluate_two_tower_v2_full_catalog.py`, `benchmark_ann.py`, `benchmark_query_batching.py` | §7.4 **defer**. Personal work, but its four called functions are teammate-authored and it is US-only — no multilingual evidence. Results cited via §3.9 |
| 10 | Ablation/slicing cluster — `run_feature_ablation.py`, `run_group_ablation.py`, `build_query_slices.py`, `analyze_query_slices.py`, `analyze_topk_overlap.py`, `experiments/query_slice_analysis/*` | §7.4 **defer**. Bound to `advanced_features` / `advanced_model` / `evaluation.metrics` |
| 11 | `experiments/audit/task*.py`, `experiments/ranking_v2/audit/scripts/00-06` | Phase-0 audits of the 12/17-feature MLP. `00_build_caches.py` imports 4 group implementations and is unrefactorable (§7.7) |
| 12 | `experiments/ranking_v2/benchmark_repair/scripts/{official_ndcg,test_official_ndcg,evaluate_baselines,train_corrected_lambdamart}.py` | Legacy-scorer cluster; see #14 |
| 13 | `scripts/train_lambdamart.py` | Imports `extract_advanced_features` (`:11`). The Task-1 path in `evaluate_task1_baselines.py` supersedes it (Phase 6b) |
| 14 | **`evaluation/metrics.py`** | **A — HARD RULE. Confirmed; the "(pending human confirm)" note in §7.3-7 is hereby removed.** This is the legacy scorer. The Phase 1.1→1.2 metric-authority audit established that it conflicts with the official gain convention, and that conflict was the direct cause of that rework. The new repo uses the personally-implemented `kdd_task1_ndcg.py` as its **sole** scorer. **Not one line of the legacy implementation may enter the new repo, and it may not be used as a reference.** Consequence: `test_official_ndcg.py:198` and `02_eval_audit.py:23` / `05_model_failure_slice_bootstrap.py:32` cannot migrate |
| 15 | **`config.py`** | **A — confirmed.** 14-line path stub; the new repo declares its own paths module |
| 16 | **`scripts/run_pipeline.py`** | **A — confirmed.** Old end-to-end demo driver |
| 17 | `tests/test_two_tower.py`, `tests/test_baseline_search.py`, `tests/test_idf_scorer.py` | Test modules that are not migrating |
| 18 | `analysis (not used)/**` | Directory name declares it dead |
| 19 | `output/**`, `screenshots/**`, architecture diagrams | Old two-setting pipeline outputs and product surface |

Items 14–16 resolve CONFLICT-2 from §7.1: all three founding files are **A**, confirmed, no longer
pending.

---

# §12 HARD STOP

**The following are prohibited under this task and are not authorised by this document.**

1. **No migration execution.** No `mv`, `cp`, `rsync`, `git filter-repo`, `git subtree`,
   `cherry-pick`, `git init` of a new repo, or any file relocation.
2. **No commits, pushes, tags, or branch creation** in this repo or any other.
3. **No file deletion** anywhere.
4. **No code refactoring.** No import rewrites, no module moves, no renames — the refactors
   described in §3 are *specifications*, not actions taken.
5. **No dependency installation** and no environment changes.
6. **No training, evaluation, or scoring runs.** No number in this document was recomputed; all
   are read from committed artifacts and reports.
7. **No modification of §7.0–§7.5.** Only §7.6 was updated, plus the additive §7.7.
8. **No overturning of §7.** §3 conflicts are reported (CONFLICT-3), never resolved against §7.
9. **No writing to any file other than `MIGRATION_PLAN.md`** — within the scope of this task, this
   document is the only permitted output.

Execution requires separate, explicit authorisation.

---

# §13 Governance Rules — must be implemented as code

These two rules are the most transferable methodological output of this project. Both were learned
from real failures in this repo. **They must migrate as executable checks, not as README prose** —
a convention that depends on someone remembering it has already failed here once.

## 13.1 TEST discipline — enforced at the loader

> The new repo's data loader **must raise** when asked for `split == "test"` unless the caller
> passes an explicit pre-registration argument.

**Background.** The V2 Cross-Encoder has **never been evaluated on the official test split.** That
is an asset, not an oversight: TEST can be spent exactly once, and only after a pre-registered
prediction. This repo already has the pre-registration record
(`cross_encoder_task1/full_run/prereg_prediction.md`) and every relevant script records
`test_split_touched: False` (`ca_eval_official_baseline.py:120`,
`ca_train_official_baseline.py:161`, `ca_protocol_audit.py:172`). Today that property is held by
**convention** — `build_cross_encoder_task1_data.py:56` makes test opt-in via a CLI default, and
`task1_common.py:39` offers `exclude_task1_test_queries()` that callers must remember to call.
Convention drifts. An assert does not.

**Implementation site — specific:**

- **Module:** `src/data/task1_common.py` (migrated from
  `experiments/ranking_v2/kdd_task1_benchmark/scripts/task1_common.py`).
- **Function:** add a single chokepoint `load_split(split, *, prereg=None)` and route
  **every** pool read through it — including the existing `load_examples()` (`:28`) and
  `task1_test_query_ids()` (`:32`).
- **Behaviour:**

```python
# src/data/task1_common.py
class TestSplitLocked(RuntimeError):
    pass

def load_split(split, *, prereg=None, metadata_only=False):
    """Sole entry point for reading an ESCI split.

    split == "test" raises unless `prereg` names a committed pre-registration
    document. `metadata_only=True` permits counts/schema WITHOUT labels --
    the discipline already practised in ca_protocol_audit.py:97.
    """
    if split == "test" and not metadata_only and prereg is None:
        raise TestSplitLocked(
            "TEST is locked. Reading it requires a committed pre-registration "
            "document; pass prereg='docs/prereg/<file>.md'. See MIGRATION_PLAN.md §13.1."
        )
    ...
```

- **Additional requirements:** `prereg` must name a file that **exists and is git-tracked** — a
  bare boolean is too easy to pass. Every accepted TEST read appends an entry to
  `artifacts/manifests/test_access_log.json` (timestamp, prereg file, caller, commit sha), so the
  "spent once" property is auditable rather than asserted.
- **Test:** `tests/test_governance.py` asserts `load_split("test")` raises `TestSplitLocked`, that
  `metadata_only=True` succeeds without labels, and that a non-existent prereg path is rejected.

## 13.2 Metric binding — every NDCG number carries its pool and its metric version

> Every NDCG number must carry `candidate_pool` and `metric_version`. **Cross-pool comparison is
> prohibited.**

**Background.** Phase 1.2 discovered the entire benchmark had been built on `large_version==1`
rather than Task 1's `small_version==1`, making a batch of numbers non-comparable to the official
leaderboard. This repo now maintains three mutually incompatible pools (`task1_small_v1`, the
large-version reference-candidate pool, and the US-only full-catalog pool), and the scope audit
found reports citing one while discussing another. This rule is the product of that correction.

**Current state — partially enforced.** `kdd_task1_ndcg.evaluate()` already emits both fields
(`:201-211`, *"One-shot report. Always carries candidate_pool + metric_version"*). But
`per_query_ndcg_table()` (`:101`) returns a **bare DataFrame with no binding**, and that is the
function the bootstrap paths actually use (`ca_eval_official_baseline.py:88`,
`run_us_slice_bootstrap.py`). That is the hole.

**Implementation — code-level, not convention:**

1. **Bind at the source.** Have `per_query_ndcg_table()` return a `ScoredTable` — a thin
   `pd.DataFrame` subclass (or a dataclass wrapping the frame) carrying required
   `candidate_pool` and `metric_version` attributes, populated from the same call that computes
   the scores. A bare frame becomes impossible to obtain from the scorer.

2. **Refuse mismatched comparison.** Any function that consumes two `ScoredTable`s — paired
   bootstrap, delta computation, W/L/T — asserts first:

```python
def assert_comparable(a, b):
    if a.candidate_pool != b.candidate_pool:
        raise CrossPoolComparison(
            f"refusing to compare {a.candidate_pool} against {b.candidate_pool}; "
            "see MIGRATION_PLAN.md §13.2")
    if a.metric_version != b.metric_version:
        raise MetricVersionMismatch(f"{a.metric_version} != {b.metric_version}")
```

   Call it at the top of the bootstrap and delta helpers, so the guard cannot be forgotten by a
   caller — it lives with the operation, not with the user.

3. **Serialise, always.** A results writer (`src/metrics/report.py`) is the only sanctioned way to
   emit a results JSON, and it refuses to write a payload lacking both fields. Direct
   `json.dump` of metric payloads is banned by lint rule and by smoke test #7 (§10.5).

4. **Test:** `tests/test_governance.py` asserts that comparing two tables from different pools
   raises, that `per_query_ndcg_table()` output carries both attributes, and that the report
   writer rejects an unbound payload.

**Both rules ship in Phase 2 and Phase 5 respectively, and both are covered by
`tests/test_governance.py` so they run in CI on every commit.**
