"""
Enriches gate_a.json with a full characterisation of the Gate A failure.
Read-only. Does not modify, filter or repair any data.
"""
import json
import os

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import paths  # noqa: E402

ROOT = str(paths.REPO_ROOT)
OUT = str(paths.MANIFESTS)



def main():
    with open(os.path.join(OUT, "gate_a.json")) as f:
        g = json.load(f)
    ex = pd.read_parquet(g["examples_path_abs"])

    offending = sorted(set(g["A2_split_consistent_per_query"]["offending_query_ids"])
                       | set(g["A3_task1_test_disjoint_from_large_train"]["offending_query_ids"]))
    diag = {"offending_query_ids": offending, "per_query": {}}

    for qid in offending:
        q = ex[ex["query_id"] == qid]
        by = (q.groupby(["split", "small_version", "large_version"])
                .size().rename("rows").reset_index().to_dict("records"))
        tr = set(q[q["split"] == "train"]["product_id"])
        te = set(q[q["split"] == "test"]["product_id"])
        small = q[q["small_version"] == 1]
        diag["per_query"][str(qid)] = {
            "query_text": q["query"].iloc[0],
            "locales": sorted(q["product_locale"].unique().tolist()),
            "total_raw_rows": int(len(q)),
            "breakdown_split_x_version": by,
            "train_products": len(tr), "test_products": len(te),
            "product_overlap_between_splits": len(tr & te),
            "small_version_1_rows_by_split": {k: int(v) for k, v in
                                              small.groupby("split").size().items()},
            "appears_in_task1_pool_on_both_sides": bool(small["split"].nunique() > 1),
            "example_id_range": [int(q["example_id"].min()), int(q["example_id"].max())],
        }

    small = ex[ex["small_version"] == 1]
    n_small_multi = int((small.groupby("query_id")["split"].nunique() > 1).sum())
    diag["scope"] = {
        "queries_with_multiple_splits_in_RAW_examples": int(
            (ex.groupby("query_id")["split"].nunique() > 1).sum()),
        "queries_with_multiple_splits_WITHIN_task1_pool_only": n_small_multi,
        "task1_pool_internally_clean": n_small_multi == 0,
        "explanation": (
            "A2 is evaluated over all raw rows. Within the small_version==1 (Task 1) pool "
            "alone the offending query appears in exactly one split, so the Task 1 "
            "train/dev/test construction of section 6 would not itself be contaminated. "
            "A3 fails because it compares Task 1 test queries against large_version==1 "
            "TRAIN queries, and the offending rows are large-only (small_version==0). "
            "The exposure is therefore to section 13 arm B (the large-train pool), which "
            "is precisely what A3 was written to protect."),
        "note_on_arm_B_filter": (
            "The arm B filter specified in section 13 is "
            "\"large_version==1 AND split=='train' AND query_id NOT IN task1_test_queries\". "
            "That filter already excludes the offending query by construction, so arm B as "
            "specified would not leak even with A3 failing. A3 asserts a stronger property "
            "of the raw data than arm B's filter requires. This is an observation, not a "
            "decision -- resolving A3 remains a human call."),
    }

    qn = ex["query"].astype(str).str.lower().str.split().str.join(" ")
    ex2 = ex.assign(_qn=qn)
    small2 = ex2[ex2["small_version"] == 1]
    t1_train_txt = set(small2[small2["split"] == "train"]["_qn"])
    t1_test_txt = set(small2[small2["split"] == "test"]["_qn"])
    diag["separate_issue_shared_query_text_across_task1_splits"] = {
        "task1_train_unique_texts": len(t1_train_txt),
        "task1_test_unique_texts": len(t1_test_txt),
        "overlap": len(t1_train_txt & t1_test_txt),
        "pct_of_task1_test_texts": round(100.0 * len(t1_train_txt & t1_test_txt)
                                         / max(len(t1_test_txt), 1), 4),
        "examples": sorted(t1_train_txt & t1_test_txt)[:15],
        "note": "distinct query_ids carrying identical normalised text; not covered by "
                "A2/A3, recorded for the human decision",
    }

    g["A_FAILURE_DIAGNOSTIC"] = diag
    with open(os.path.join(OUT, "gate_a.json"), "w") as f:
        json.dump(g, f, indent=2, default=str)
    print(json.dumps(diag, indent=2, default=str))


if __name__ == "__main__":
    main()
