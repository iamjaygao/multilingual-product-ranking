# Governance

Two rules govern this repository. Both were learned from failures in the
predecessor project, both are enforced as **executable checks** rather than
prose, and both are covered by `tests/test_governance.py` so they run on every
commit. A convention that depends on someone remembering it has already failed
here once.

A third section states the ownership boundary.

---

# 1. TEST discipline (§13.1)

> The data loader **must raise** when asked for `split == "test"` unless the
> caller passes an explicit, committed pre-registration.

## 1.1 Why

The V2 Cross-Encoder has **never been evaluated on the official test split.**
That is an asset, not an oversight: TEST can be spent exactly once, and only
after a pre-registered prediction. Spending it casually converts a clean
held-out result into a number nobody can trust.

In the source repo the property was held by *convention* — an opt-in CLI default
in the data builder, and a helper (`exclude_task1_test_queries()`) callers had to
remember to call. Conventions drift. An assert does not.

## 1.2 Mechanism

**Single chokepoint.** `src/data/task1_common.py` defines

```python
load_split(split, *, prereg=None, metadata_only=False, columns=None, small_version=1)
```

and **every** pool read is routed through it — `load_examples()`,
`load_train()`, `task1_test_query_ids()`, `exclude_task1_test_queries()`,
`build_arm_a_pool()`, `build_arm_b_pool()`. Reading `split="test"` (or `"all"`,
which includes test rows) without a pre-registration raises `TestSplitLocked`.

**The pre-registration must be real.** `_validate_prereg()` rejects, in order:

1. a `bool` or any non-path value — *"a boolean is not a pre-registration"*;
2. a path that does not exist;
3. a path that exists but is **not tracked by git** — an untracked file can be
   written after the results are known, so it cannot serve as a pre-registration.

**Every accepted TEST read is logged.** `artifacts/manifests/test_access_log.json`
receives a timestamp, the prereg path, the calling file and line, the commit sha,
and the columns requested. "Spent once" is therefore *auditable*, not merely
asserted.

**`metadata_only=True`** permits counts and schema **without labels** — the
label columns (`esci_label`, `gain`, `lgb_label`) are dropped from the result.
This is what lets integrity gates and the arm-B exclusion run without spending
the budget. Metadata reads are not logged, because they consume nothing.

## 1.3 Current state

| | |
|---|---|
| Entries in `test_access_log.json` | **2** |
| Both from | Phase 2 pool construction — `gate_a_operative.py` (assertion A2′) and `build_task1_pool.py` |
| Both authorised by | [`docs/prereg/pool_construction.md`](prereg/pool_construction.md), git-tracked |
| Model evaluations on TEST | **0** |

**The V2 Cross-Encoder has never been scored on the official test split.** Every
number in this repository — 0.883443 / 0.845605 / 0.792383 and the whole Phase 4
gate set — is a **dev** number.

`docs/prereg/pool_construction.md` authorises *construction only* and says so
explicitly. It is not a substitute for an evaluation pre-registration, and an
evaluation script must not pass it.

## 1.4 A known boundary — read this before assuming the lock is total

**`load_split()` locks the frozen split. It does not lock the raw upstream
parquet.**

Code that opens
`$ESCI_DATA_ROOT/shopping_queries_dataset/shopping_queries_dataset_examples.parquet`
directly with `pandas.read_parquet` and filters `split == "test"` itself bypasses
the chokepoint entirely. Nothing stops it.

There is a legitimate instance of this in the repository:

> **`scripts/protocol_audit.py:88`** — `r = raw[raw.split == s]` over the raw
> examples table, computing per-split row counts, locale distribution,
> candidates-per-query, and label distribution for the protocol manifest. It
> scores nothing. The script carries its own note at `:97`:
> *"TEST: counts/metadata only. Labels were NOT used for evaluation or selection."*

**This gap is deliberate.** Closing it — routing every raw parquet read through
the lock — would block legitimate dataset auditing: verifying the official
Table-1 counts, checking the label mix, confirming the pool matches upstream.
Those tasks need to look at the raw distribution of the test split, and refusing
them would make the repository less auditable, not more honest.

What the lock guarantees is narrower and worth stating precisely:

- ✅ **No model can be scored against frozen test labels without a committed,
  git-tracked pre-registration, and every such access is logged.**
- ❌ It does **not** guarantee that no line of code anywhere ever reads a test
  row from the raw upstream file.

If you add a raw-parquet read that touches test, you are outside the lock. Use
`load_split(..., metadata_only=True)` unless you have a specific reason not to,
and say in a comment why the raw read is necessary.

---

# 2. Metric binding (§13.2)

> Every NDCG number must carry `candidate_pool` and `metric_version`.
> **Cross-pool comparison is prohibited.**

## 2.1 Why

Phase 1.2 of the predecessor project discovered the entire benchmark had been
built on `large_version==1` rather than Task 1's `small_version==1`, making a
batch of published numbers non-comparable to the official leaderboard. That cost
a full rework. Three mutually incompatible pools existed at once:

| Pool | Queries | Locales | Metric |
|---|---|---|---|
| `task1_small_v1` | 14,496 test / 5,071 dev | all three | full-list NDCG, official gain |
| large-version reference-candidate | 30,969 | all three | NDCG@10 |
| US-only full-catalog | 5,000 | us only | Recall@100 |

The retrieval scope audit later found reports citing one while discussing
another. Numbers from different rows of that table must never be compared, and
the way to make that structural is to refuse to let a number travel without its
provenance.

## 2.2 Mechanism, and why it is a wrapper

`kdd_task1_ndcg.evaluate()` already emits both fields (`:201-211`). But
`per_query_ndcg_table()` (`:101`) returns a **bare DataFrame** — and that is the
function the bootstrap paths use. That was the hole.

The scorer could not simply be edited. `src/metrics/kdd_task1_ndcg.py` hashes
**its own source** to produce `scorer_sha256()`, which produces the
`metric_version` string. One changed byte changes the hash, changes
`metric_version`, and invalidates every gate value locked in Phase 3 and Phase 4.
The scorer is a **sealed artifact**; its sha256 is
`6946489c8c09dc37cdb91b905dd5108e46e4070dbd1494b4ac4f972d20f41bc7` and a test
asserts it.

So the binding lives in a wrapper, [`src/metrics/binding.py`](../src/metrics/binding.py):

| Component | Role |
|---|---|
| `ScoredTable` | `pd.DataFrame` subclass carrying `candidate_pool` + `metric_version`. `_metadata` makes pandas propagate them through slicing and copying. |
| `bound_per_query_table(...)` | Calls the sealed scorer, wraps the result. Identical numerics; a bare frame can no longer be obtained through this path. |
| `bound_evaluate(...)` | Calls `evaluate()` and **verifies** the fields it already emits, so a regression would fail loudly. |
| `assert_comparable(a, b)` | Raises `CrossPoolComparison` / `MetricVersionMismatch`. Belongs at the top of every bootstrap, delta and W-L-T helper, so the guard lives with the operation rather than with the caller's memory. |
| `write_report(payload, path)` | The only sanctioned way to emit a results JSON; refuses an unbound payload. |

## 2.3 What is enforced

- `bound_per_query_table()` output carries both fields — tested.
- Comparing two tables from different pools raises — tested.
- Comparing a `ScoredTable` against a bare DataFrame raises — tested.
- `write_report()` refuses an unbound payload — tested.
- Every results JSON under `artifacts/` carries both fields — tested (smoke #7).
- The scorer's sha256 is unchanged — tested.

---

# 3. Ownership boundary

This repository is the personally-authored subset of `esci-search-ranking-system`,
which began as a **three-person university course project**. Components written
by teammates — the baseline and advanced MLP rerankers, the Two-Tower V0 encoder,
the current BM25 module, the ESCI-S integration, the interactive GUI, and the
legacy scorer — are **not** part of this repository.

Where a result from that prior work is cited, [`docs/prior_work.md`](prior_work.md)
names who wrote the component that produced it. Two entries there deserve
particular care and are stated flatly rather than softened:

- **Full-catalog retrieval**: the experiment is mine; the retrieval primitives it
  calls are a teammate's.
- **`retrieval/bm25.py`**: I created the module in the founding commit; a
  teammate rewrote it, and the shipped implementation is theirs.

The boundary was established from git history, not recollection, and the full
audit is in `MIGRATION_PLAN.md` §7. Enforcement: smoke test #11 asserts that no
file under `src/`, `scripts/` or `tests/` traces to teammate authorship, and
smoke test #12 asserts `prior_work.md` names a teammate for every DOCUMENT_ONLY
item.
