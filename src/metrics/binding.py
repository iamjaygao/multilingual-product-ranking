"""§13.2 — metric binding. Every NDCG number carries its pool and metric version.

    Every NDCG number must carry `candidate_pool` and `metric_version`.
    Cross-pool comparison is prohibited.

Background
----------
Phase 1.2 of the source project discovered that the whole benchmark had been
built on `large_version==1` rather than Task 1's `small_version==1`, making a
batch of numbers non-comparable to the official leaderboard. Three mutually
incompatible pools existed at once (`task1_small_v1`, the large-version
reference-candidate pool, and the US-only full-catalog pool), and the scope audit
found reports citing one while discussing another. This module is the product of
that correction.

Why this is a wrapper and not an edit
-------------------------------------
`kdd_task1_ndcg.evaluate()` already emits both fields (`:201-211`). But
`per_query_ndcg_table()` (`:101`) returns a **bare DataFrame with no binding**,
and that is the function the bootstrap paths actually use. That is the hole.

The obvious fix -- edit the scorer -- is forbidden. `kdd_task1_ndcg.py` hashes
its **own source** to produce `scorer_sha256()`, which in turn produces the
`metric_version` string
``kdd_task1_ndcg@6946489c8c09dc37|gain=...|idcg0=exclude|tie=deterministic``.
Changing one byte of that file changes the hash, which changes `metric_version`,
which invalidates every gate value locked in Phase 3 (0.883443 / 0.845605 /
0.792383) and Phase 4 (0.8521 / 0.8071 / 0.7358, both bootstraps). The scorer is
a sealed artifact.

So the binding lives here, outside it, and `src/metrics/kdd_task1_ndcg.py`
remains byte-identical to the source repo.

Usage
-----
    from src.metrics.binding import bound_per_query_table, assert_comparable, write_report

    a = bound_per_query_table(df_a, "score", candidate_pool="task1_small_v1")
    b = bound_per_query_table(df_b, "score", candidate_pool="task1_small_v1")
    assert_comparable(a, b)               # raises on pool or version mismatch
    delta = b["ndcg_at_20"] - a["ndcg_at_20"]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.ranking.cross_encoder import load_task1_scorer

#: The only pool this repository's Layer 1 numbers may claim.
DEFAULT_POOL = "task1_small_v1"

#: Fields every metric payload must carry.
REQUIRED_FIELDS = ("candidate_pool", "metric_version")


class CrossPoolComparison(RuntimeError):
    """Raised when two ScoredTables from different candidate pools are compared."""


class MetricVersionMismatch(RuntimeError):
    """Raised when two ScoredTables were scored by different scorer versions."""


class UnboundMetricPayload(ValueError):
    """Raised when a results payload lacks candidate_pool / metric_version."""


class ScoredTable(pd.DataFrame):
    """A per-query NDCG table that cannot be separated from its provenance.

    Subclasses DataFrame so every existing consumer keeps working unchanged --
    `.loc`, `.index`, arithmetic, everything -- while `candidate_pool` and
    `metric_version` travel with the data. `_metadata` makes pandas propagate
    them across slicing and copying.
    """

    _metadata = ["candidate_pool", "metric_version"]

    @property
    def _constructor(self):
        return ScoredTable


def current_metric_version() -> str:
    """The metric_version string of the sealed scorer."""
    return load_task1_scorer().metric_version()


def bound_per_query_table(df, score_col, *, candidate_pool=DEFAULT_POOL,
                          gain_col="gain", query_col="query_id", **kwargs):
    """`kdd_task1_ndcg.per_query_ndcg_table()`, with the binding attached.

    Identical numerics -- this calls the sealed scorer and wraps its output. The
    only difference is that a bare, unattributable frame can no longer be
    obtained through this path.
    """
    sc = load_task1_scorer()
    raw = sc.per_query_ndcg_table(df, score_col, gain_col=gain_col,
                                  query_col=query_col, **kwargs)
    out = ScoredTable(raw)
    out.candidate_pool = candidate_pool
    out.metric_version = sc.metric_version()
    return out


def bound_evaluate(df, score_col, *, candidate_pool=DEFAULT_POOL, **kwargs):
    """`kdd_task1_ndcg.evaluate()`, asserting the fields it already emits.

    `evaluate()` sets both fields itself; this verifies rather than adds them,
    so a future scorer that stopped emitting them would fail loudly here.
    """
    sc = load_task1_scorer()
    r = sc.evaluate(df, score_col, candidate_pool=candidate_pool, **kwargs)
    missing = [f for f in REQUIRED_FIELDS if not r.get(f)]
    if missing:
        raise UnboundMetricPayload(
            f"scorer.evaluate() returned a payload missing {missing}; "
            "see MIGRATION_PLAN.md §13.2")
    return r


def assert_comparable(a, b) -> None:
    """Refuse to compare two tables from different pools or scorer versions.

    Call this at the top of any bootstrap / delta / W-L-T helper, so the guard
    lives with the operation rather than depending on the caller remembering.
    """
    for name, t in (("a", a), ("b", b)):
        if not hasattr(t, "candidate_pool") or not hasattr(t, "metric_version"):
            raise UnboundMetricPayload(
                f"operand {name} is not a ScoredTable; obtain it from "
                "bound_per_query_table(). See MIGRATION_PLAN.md §13.2")
    if a.candidate_pool != b.candidate_pool:
        raise CrossPoolComparison(
            f"refusing to compare {a.candidate_pool} against {b.candidate_pool}; "
            "see MIGRATION_PLAN.md §13.2")
    if a.metric_version != b.metric_version:
        raise MetricVersionMismatch(f"{a.metric_version} != {b.metric_version}")


def bind_payload(payload: dict, *, candidate_pool=DEFAULT_POOL) -> dict:
    """Attach the two required fields to a results dict, if absent."""
    out = dict(payload)
    out.setdefault("candidate_pool", candidate_pool)
    out.setdefault("metric_version", current_metric_version())
    return out


def write_report(payload: dict, path, *, candidate_pool=DEFAULT_POOL, indent=2):
    """The only sanctioned way to emit a results JSON.

    Refuses to write a payload lacking `candidate_pool` / `metric_version`
    (§13.2 point 3). Direct `json.dump` of a metric payload is what smoke test
    #7 exists to catch.
    """
    bound = bind_payload(payload, candidate_pool=candidate_pool)
    missing = [f for f in REQUIRED_FIELDS if not bound.get(f)]
    if missing:
        raise UnboundMetricPayload(
            f"refusing to write an unbound metric payload; missing {missing}. "
            "See MIGRATION_PLAN.md §13.2")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bound, indent=indent, default=str))
    return p


def payload_is_bound(obj) -> bool:
    """True if a loaded results dict carries both required fields anywhere."""
    if isinstance(obj, dict):
        if all(obj.get(f) for f in REQUIRED_FIELDS):
            return True
        return any(payload_is_bound(v) for v in obj.values())
    if isinstance(obj, list):
        return any(payload_is_bound(v) for v in obj)
    return False
