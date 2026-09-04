"""LAYER 2 -- append the legacy engineered feature columns to the core pool.

⚠️  OPTIONAL. Phase 6b only. NOT RUN during Phase 2, and not importable from
    anything under `src/` or `scripts/` (smoke test #13, §10.5).

Split out of the source repo's `build_task1_benchmark.py` (ffd16a6), whose
`keep` list at `:301-304` emitted core and feature columns together.
`src/data/build_task1_pool.py` took the core half; this is the other half.

WHY THIS CANNOT BE FULLY REBUILT
--------------------------------
`semantic_score` is produced by a **group-originated Two-Tower V0 checkpoint**
that is not part of this repository (MIGRATION_PLAN.md §7.3-2, §4.2). It cannot
be regenerated here. The Layer 2 matrix must therefore be carried out-of-band
from the source repo, and this module can only *re-derive the parts that do not
depend on it*.

⛔ **§10.3 -- `semantic_score` immutability.** Do NOT regenerate that column
   with Two-Tower V2. Doing so changes every number in the Task 1 benchmark
   report, including the externally-cited LambdaMART 0.8429, while leaving the
   file name, the column name and the report text identical. Before running
   Phase 6b, verify that the `semantic_score` column is bit-identical to the
   source repo's (sha256, recorded in `legacy/manifests/semantic_score_sha256.json`).
   If a V2-based column is ever wanted, add it under a DIFFERENT name
   (e.g. `semantic_score_v2`) as a new baseline row -- never as a rewrite.

WHAT ELSE IS MISSING
--------------------
The full 17-feature stack additionally needs ESCI-S enrichment (price, stars,
ratings, category), which §7.3-5 drops from this repo entirely, and
`extract_advanced_features` from the group-originated `reranking/advanced_features.py`,
which §11 item 4 keeps in the old repo. Only `ALL_FEATURES` -- the 17 *names* --
is re-declared here; a list of column names is a data schema, not an
implementation (§3.10, resolved).

ATTRIBUTION (§7.3-C)
--------------------
The original course project contained a group implementation of this component.
The version in this repository is a minimal re-implementation written
independently for this ranking benchmark. No claim is made over the original
group implementation.
"""
from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# The 17 feature NAMES, re-declared locally (§3.10 resolved: a name list is a
# schema, not an implementation). Replaces
#     from reranking.advanced_features import ALL_FEATURES
# which §11 item 4 bars from this repository.
# ---------------------------------------------------------------------------
ALL_FEATURES = [
    "query_length", "query_mean_idf", "query_max_idf", "user_budget", "cheap_intent",
    "log_price", "is_price_missing", "stars_clean", "log_review_count", "is_rating_missing",
    "bm25_score", "semantic_score", "word_overlap", "is_dominant_category", "brand_match",
    "color_match", "is_over_budget",
]

#: Columns Layer 2 adds on top of the Layer 1 core pool.
LAYER2_COLUMNS = ["bm25_raw", "semantic_cosine_raw", "lgb_label", "category"] + ALL_FEATURES

#: Layer 2 columns that CANNOT be recomputed in this repository.
UNREBUILDABLE = {
    "semantic_score": "group-originated Two-Tower V0 checkpoint (§7.3-2, §10.3)",
    "semantic_cosine_raw": "same checkpoint",
    "log_price": "ESCI-S enrichment, dropped from this repo (§7.3-5)",
    "is_price_missing": "ESCI-S",
    "stars_clean": "ESCI-S",
    "log_review_count": "ESCI-S (and identically zero -- see the audit)",
    "is_rating_missing": "ESCI-S",
    "is_dominant_category": "ESCI-S category field",
    "is_over_budget": "ESCI-S price field",
}


def attach_bm25(core_df):
    """Recompute `bm25_score` / `bm25_raw` on the core pool.

    This is the one Layer 2 column this repo can regenerate, via the
    independently re-implemented `src/features/bm25_pool.py`. Note the
    multilingual tokenisation caveat documented there: the scores reproduce the
    frozen ones only because the English-default tokeniser is left unchanged.
    """
    from src.features.bm25_pool import compute_bm25

    work = core_df.copy()
    work["query_text"] = work["query"]
    work["item_id"] = work["product_id"]
    if "item_text" not in work.columns:
        raise ValueError(
            "attach_bm25 needs an `item_text` column (title + description + "
            "bullet_point). Layer 1 carries only product_title, so join the "
            "remaining text fields from the raw products parquet first.")
    bm = compute_bm25(work)
    out = core_df.merge(bm, on=["query_id", "product_id"], how="left")
    if len(out) != len(core_df):
        raise AssertionError("bm25 join fanned out")
    out["bm25_score"] = out["bm25_minmax"]
    return out


def main(argv=None):
    raise SystemExit(
        "legacy/features/build_feature_matrix.py is a Phase 6b component and is "
        "not runnable as part of the minimum loop.\n\n"
        "It requires:\n"
        "  1. the out-of-band Layer 2 feature matrix (§4.2), and\n"
        "  2. a passing §10.3 semantic_score sha256 check.\n\n"
        "See MIGRATION_PLAN.md §9 Phase 6b."
    )


if __name__ == "__main__":
    sys.exit(main())
