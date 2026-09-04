"""Shared helpers for the KDD Task 1 benchmark, and THE TEST lock.

Migrated from `experiments/ranking_v2/kdd_task1_benchmark/scripts/task1_common.py`
(source repo ffd16a6). Two changes from the source:

1. Paths come from `src.paths` (env-driven) instead of a hard-coded location
   inside the old repo (MIGRATION_PLAN.md §11 item 15).
2. **`load_split()` is added as the sole chokepoint for reading an ESCI split,
   and every existing reader is routed through it** (MIGRATION_PLAN.md §13.1).

The arm-B exclusion remains a shared helper, not an inline filter. Any future
training pool drawing on large-version data must call
`exclude_task1_test_queries()` / `build_arm_b_pool()` -- never re-implement it.

--------------------------------------------------------------------------
THE TEST LOCK (§13.1)
--------------------------------------------------------------------------
The V2 Cross-Encoder has never been evaluated on the official test split. That
is an asset: TEST can be spent exactly once, and only after a pre-registered
prediction. In the predecessor repo that property was held by convention -- an
opt-in CLI default and a helper callers had to remember to call. Convention
drifts; an assert does not.

`load_split("test")` therefore RAISES unless the caller passes `prereg=` naming
a pre-registration document that exists and is tracked by git, or asks for
`metadata_only=True` (counts and schema, no labels -- the discipline already
practised in the source repo's `ca_protocol_audit.py:97`).

Every accepted TEST read is appended to
`artifacts/manifests/test_access_log.json`, so "spent once" is auditable rather
than merely asserted.
"""
from __future__ import annotations

import datetime as _dt
import inspect
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from src import paths

# ---------------------------------------------------------------------------
# Constants (unchanged from source)
# ---------------------------------------------------------------------------

#: The sole query_id that ESCI itself assigns to both train and test.
#: Documented as FAIL_UPSTREAM in gate_a.json. NOT filtered out of any pool.
KNOWN_UPSTREAM_CROSS_SPLIT_QUERY_IDS = frozenset({79706})

LABEL_ORDINAL = {"I": 0, "C": 1, "S": 2, "E": 3}
OFFICIAL_GAIN = {"E": 1.00, "S": 0.10, "C": 0.01, "I": 0.00}

#: Columns that constitute the Layer 1 core pool (§4.1). Verified against the
#: actual reads in the minimum loop; `gain`, `doc_id` and `product_brand` are
#: required and were confirmed, not assumed.
LAYER1_COLUMNS = [
    "example_id", "query_id", "query", "product_id", "product_locale",
    "doc_id", "esci_label", "gain", "split", "product_title", "product_brand",
]

#: Label columns a metadata_only read is not allowed to see.
_LABEL_COLUMNS = frozenset({"esci_label", "gain", "lgb_label"})

TEST_ACCESS_LOG = paths.MANIFESTS / "test_access_log.json"


class TestSplitLocked(RuntimeError):
    """Raised when the frozen TEST split is read without a pre-registration."""


# ---------------------------------------------------------------------------
# §13.1 -- the chokepoint
# ---------------------------------------------------------------------------

def _git_tracked(path: Path) -> bool:
    """True iff `path` is tracked by git in this repository.

    A bare boolean flag would be too easy to pass, so the pre-registration must
    name a real, committed document. An untracked file on someone's disk is not
    a pre-registration -- it can be written after the fact.
    """
    try:
        r = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path)],
            cwd=paths.REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def _head_sha() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT,
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _validate_prereg(prereg) -> Path:
    """Resolve and validate a pre-registration reference. Raises on any doubt."""
    if isinstance(prereg, bool) or not isinstance(prereg, (str, os.PathLike)):
        raise TestSplitLocked(
            f"prereg must be a path to a committed pre-registration document, "
            f"got {type(prereg).__name__}={prereg!r}. A boolean is not a "
            f"pre-registration. See MIGRATION_PLAN.md §13.1."
        )
    p = Path(prereg)
    if not p.is_absolute():
        p = paths.REPO_ROOT / p
    if not p.exists():
        raise TestSplitLocked(
            f"pre-registration document does not exist: {p}. "
            f"See MIGRATION_PLAN.md §13.1."
        )
    if not _git_tracked(p):
        raise TestSplitLocked(
            f"pre-registration document exists but is NOT tracked by git: {p}. "
            f"An untracked file can be written after the results are known, so it "
            f"cannot serve as a pre-registration. Commit it first. "
            f"See MIGRATION_PLAN.md §13.1."
        )
    return p


def _caller() -> str:
    """Best-effort identification of the frame outside this module."""
    for fr in inspect.stack()[1:]:
        if Path(fr.filename).resolve() != Path(__file__).resolve():
            try:
                rel = Path(fr.filename).resolve().relative_to(paths.REPO_ROOT)
            except ValueError:
                rel = Path(fr.filename).name
            return f"{rel}:{fr.lineno} in {fr.function}"
    return "<unknown>"


def _log_test_access(prereg: Path, columns, metadata_only: bool) -> None:
    entry = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "prereg": str(prereg.relative_to(paths.REPO_ROOT)) if prereg else None,
        "caller": _caller(),
        "commit": _head_sha(),
        "metadata_only": metadata_only,
        "columns_requested": list(columns) if columns else "ALL",
    }
    TEST_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = {"purpose": ("Append-only record of every accepted read of the frozen "
                       "TEST split. MIGRATION_PLAN.md §13.1."),
           "accesses": []}
    if TEST_ACCESS_LOG.exists():
        try:
            log = json.loads(TEST_ACCESS_LOG.read_text())
        except json.JSONDecodeError:
            pass
    log.setdefault("accesses", []).append(entry)
    TEST_ACCESS_LOG.write_text(json.dumps(log, indent=2))


def load_split(split, *, prereg=None, metadata_only=False, columns=None,
               small_version=1):
    """Sole entry point for reading an ESCI split.

    ``split == "test"`` raises `TestSplitLocked` unless `prereg` names a
    committed pre-registration document. ``metadata_only=True`` permits
    counts/schema WITHOUT labels.

    Parameters
    ----------
    split : {"train", "test", "all"}
    prereg : str | Path | None
        Path to a git-tracked pre-registration document. Required for TEST.
    metadata_only : bool
        If True, label columns are dropped from the result. Permits auditing
        row/query counts and the schema without seeing any ground truth.
    columns : list[str] | None
        Column subset to read.
    small_version : int | None
        1 (Task 1 pool, the default), 0, or None for no filter.
    """
    if split not in {"train", "test", "all"}:
        raise ValueError(f"split must be train|test|all, got {split!r}")

    prereg_path = None
    if split in {"test", "all"}:
        if metadata_only:
            pass  # counts/schema without labels -- always permitted
        elif prereg is None:
            raise TestSplitLocked(
                f"TEST is locked (requested split={split!r}). Reading it requires a "
                "committed pre-registration document; pass "
                "prereg='docs/prereg/<file>.md', or metadata_only=True for "
                "counts and schema without labels. See MIGRATION_PLAN.md §13.1."
            )
        else:
            prereg_path = _validate_prereg(prereg)

    paths.require_raw_esci()

    read_cols = list(columns) if columns else None
    if read_cols is not None:
        need = set(read_cols) | {"split"}
        if small_version is not None:
            need.add("small_version")
        read_cols = sorted(need)

    df = pd.read_parquet(paths.EXAMPLES, columns=read_cols)
    if small_version is not None:
        df = df[df["small_version"] == small_version]
    if split != "all":
        df = df[df["split"] == split]
    df = df.copy()

    if metadata_only:
        drop = [c for c in df.columns if c in _LABEL_COLUMNS]
        if drop:
            df = df.drop(columns=drop)

    if columns:
        df = df[[c for c in columns if c in df.columns]]

    if split in {"test", "all"} and not metadata_only:
        _log_test_access(prereg_path, columns, metadata_only)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Readers -- all routed through load_split()
# ---------------------------------------------------------------------------

def load_examples(columns=None, *, prereg=None, metadata_only=False):
    """Full small_version==1 example table (train + test).

    Routed through `load_split(split="all")`, so it inherits the TEST lock: it
    touches test rows and therefore needs a pre-registration or
    `metadata_only=True`. In the source repo this function read the parquet
    directly with no guard at all.
    """
    return load_split("all", prereg=prereg, metadata_only=metadata_only,
                      columns=columns)


def load_train(columns=None):
    """Train split only. Never touches TEST, so it needs no pre-registration."""
    return load_split("train", columns=columns)


def task1_test_query_ids(examples=None):
    """The frozen Task 1 evaluation query set: small_version==1 AND split=='test'.

    Uses `metadata_only=True`: query ids are not labels, so knowing *which*
    queries are held out reveals no ground truth. This is what lets the
    arm-B exclusion below run without spending the TEST budget.
    """
    if examples is not None:
        ex = examples
        return set(ex.loc[(ex["small_version"] == 1) & (ex["split"] == "test"),
                          "query_id"].unique())
    ex = load_split("test", metadata_only=True, columns=["query_id"])
    return set(ex["query_id"].unique())


def exclude_task1_test_queries(df, examples=None, query_col="query_id"):
    """THE shared arm-B exclusion.

    Removes every row whose query_id is in the frozen Task 1 test query set.
    This is the only sanctioned way to build a training pool that draws on
    large_version data, and it is what makes assertion A3' hold even though the
    raw-data assertion A3 fails upstream (query_id 79706 carries 3 large-only
    train rows while its 31 small_version==1 rows are in Task 1 test).

    Returns (filtered_df, info_dict).
    """
    test_q = task1_test_query_ids(examples)
    before = len(df)
    mask = df[query_col].isin(test_q)
    out = df.loc[~mask].copy()
    info = {
        "rows_before": int(before),
        "rows_removed": int(mask.sum()),
        "rows_after": int(len(out)),
        "queries_removed": int(df.loc[mask, query_col].nunique()),
        "removed_query_ids_sample": sorted(df.loc[mask, query_col].unique().tolist())[:20],
        "task1_test_query_count": len(test_q),
        "helper": "src.data.task1_common.exclude_task1_test_queries",
    }
    return out, info


def build_arm_b_pool(examples=None):
    """Arm B: large_version==1 AND split=='train', with Task 1 test queries removed
    via the shared helper. Returns (pool_df, info_dict)."""
    if examples is not None:
        ex = examples
    else:
        ex = load_split("train", small_version=None)
    raw = ex[(ex["large_version"] == 1) & (ex["split"] == "train")]
    pool, info = exclude_task1_test_queries(raw, examples=examples)
    info["definition"] = ("large_version==1 AND split=='train' AND "
                          "query_id NOT IN task1_test_queries "
                          "(applied via task1_common.exclude_task1_test_queries)")
    return pool, info


def build_arm_a_pool(train_query_ids, examples=None):
    """Arm A: the Task 1 training split only (after the 85/15 query-level carve)."""
    ex = examples if examples is not None else load_train()
    raw = ex[(ex["small_version"] == 1) & (ex["split"] == "train")] \
        if "small_version" in ex.columns else ex[ex["split"] == "train"]
    pool = raw[raw["query_id"].isin(set(train_query_ids))].copy()
    # defensive: run the shared exclusion anyway; it must be a no-op here
    pool2, info = exclude_task1_test_queries(pool, examples=examples)
    assert len(pool2) == len(pool), "arm A unexpectedly intersected Task 1 test queries"
    info["definition"] = ("small_version==1 AND split=='train' AND "
                          "query_id in train_task1 "
                          "(shared exclusion applied, expected to be a no-op)")
    return pool2, info


def normalize_query_text(s):
    """Lowercase + whitespace-collapse. Matches the normalisation used by the
    upstream feature code (`str(q).lower().split()`)."""
    return s.astype(str).str.lower().str.split().str.join(" ")
