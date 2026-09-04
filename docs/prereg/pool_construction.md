# Pre-registration — Task 1 pool construction and integrity gates

**Type:** construction access, NOT an evaluation
**Registered:** 2026-09-03 (Phase 2 of `MIGRATION_PLAN.md`)
**Authorises:** `src.data.task1_common.load_split("test", prereg=...)` for the
purposes listed below, and nothing else.

---

## Why a TEST read is required here

`MIGRATION_PLAN.md` §13.1 locks the frozen TEST split behind a committed
pre-registration. That rule exists to stop a model being *evaluated* on TEST
before its prediction is registered. It is not meant to stop the frozen
evaluation set from being **materialised** in the first place — Phase 2's whole
job is to rebuild the Layer 1 core pool, all three splits, from raw ESCI (§4.1),
and Phase 2's exit criteria require the test frame's counts (14,496 queries /
336,373 rows) and locale split (us 8,956 · jp 3,123 · es 2,417) to be verified.

Two operations need the labelled test rows and cannot use `metadata_only=True`:

| Operation | Why labels are needed |
|---|---|
| `src/data/build_task1_pool.py` | Emits `esci_label` and `gain` into `test_task1_core.parquet`. The pool *is* the labels. |
| `src/data/gates/gate_a_operative.py` (`:142-146`) | Assertion A2′ compares `esci_label` for the same `(query_id, product_id, product_locale)` triple across train and test. Label agreement is the check. |

Everything else in Phase 2 uses `metadata_only=True` and never sees a test
label — including `dataset_counts.py`, `task1_test_query_ids()`, and the whole
arm-B exclusion path.

## What this pre-registration does NOT authorise

- **No model is scored on TEST under this document.** No NDCG, no ranking, no
  prediction of any kind is computed against test labels here.
- The V2 Cross-Encoder has still never been evaluated on the official test
  split. That property is intact and this document does not spend it.
- Evaluating any model on TEST requires a **separate** pre-registration stating
  the predicted result before the result is known. This document does not
  substitute for that, and must not be passed by an evaluation script.

## Registered prediction

Not applicable — no metric is computed. The verifiable claim registered here is
the expected shape of the rebuilt pool, taken from the source repo's
`split_integrity.json` before the rebuild was run:

| Quantity | Registered expectation |
|---|---|
| query_counts | train 28,733 · dev 5,071 · test 14,496 |
| row_counts | train 663,407 · dev 118,231 · test 336,373 |
| test locale distribution | us .6178 · es .1667 · jp .2154 |
| query-level disjointness | train∩dev, train∩test, dev∩test all empty |

A mismatch means the scikit-learn version or the upstream data changed, and is
a STOP (§10.4) — not something to be tuned into agreement.

## Audit trail

Every access made under this document is appended to
`artifacts/manifests/test_access_log.json` with a timestamp, the caller, and the
commit sha, so the reads authorised here are enumerable after the fact.
