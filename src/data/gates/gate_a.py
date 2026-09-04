"""
Section 4 -- GATE A, the pre-build hard checks. STOP-gate.

A1  small_version==1 rows are all large_version==1   (branch record, not an assert)
A2  split is assigned consistently per query_id       (assert)
A3  Task1 test queries disjoint from large train queries (assert)
A4  products primary key (product_id, product_locale) unique (assert)

A2/A3/A4 failure => STOP with observed values. Never repair, never dedupe.
"""
import json
import os
import sys

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import paths  # noqa: E402

ROOT = str(paths.REPO_ROOT)
OUT = str(paths.MANIFESTS)


STOPS = []


def main():
    with open(os.path.join(OUT, "data_source_audit.json")) as f:
        src = json.load(f)
    ex_path = src["examples_path_abs"]
    pr_path = src["products_path_abs"]

    ex = pd.read_parquet(ex_path, columns=[
        "example_id", "query_id", "product_id", "product_locale",
        "esci_label", "small_version", "large_version", "split"])
    pr = pd.read_parquet(pr_path, columns=["product_id", "product_locale"])

    g = {"examples_path": src["examples_path"], "products_path": src["products_path"],
         "examples_path_abs": ex_path, "products_path_abs": pr_path}

    # ---------------- A1 ----------------
    sub = ex[ex["small_version"] == 1]
    a1_vals = sorted(sub["large_version"].unique().tolist())
    subset = (a1_vals == [1])
    g["A1_small_subset_of_large"] = {
        "large_version_values_among_small_rows": a1_vals,
        "SUBSET": bool(subset),
        "small_rows": int(len(sub)),
        "small_rows_with_large_version_1": int((sub["large_version"] == 1).sum()),
        "interpretation": ("every small_version==1 row is also large_version==1, so per-row "
                           "scores/features may be reused by key" if subset else
                           "small is NOT a subset of large -- everything must be recomputed "
                           "on the Task1 pool"),
    }
    g["downstream_path"] = "REUSE" if subset else "RECOMPUTE"
    print(f"A1 small subset of large : {subset}  -> downstream_path = {g['downstream_path']}")

    # ---------------- A2 ----------------
    nsplit = ex.groupby("query_id")["split"].nunique()
    n_bad = int((nsplit > 1).sum())
    g["A2_split_consistent_per_query"] = {
        "queries_with_multiple_splits": n_bad,
        "max_splits_per_query": int(nsplit.max()),
        "PASS": n_bad == 0,
        "offending_query_ids": nsplit[nsplit > 1].index.tolist()[:20],
    }
    # also record whether small/large agree on split assignment
    s_small = sub.groupby("query_id")["split"].first()
    s_large = ex.groupby("query_id")["split"].first()
    common = s_small.index.intersection(s_large.index)
    disagree = int((s_small.loc[common] != s_large.loc[common]).sum())
    g["A2_split_agreement_small_vs_large"] = {
        "shared_query_ids": int(len(common)),
        "queries_where_small_and_large_split_disagree": disagree,
    }
    print(f"A2 split consistent      : {n_bad == 0}  (queries with >1 split = {n_bad}); "
          f"small/large split disagreement = {disagree}")
    if n_bad != 0:
        STOPS.append(f"A2 FAILED: {n_bad} query_ids appear in more than one split. "
                     f"Examples: {nsplit[nsplit > 1].index.tolist()[:20]}")

    # ---------------- A3 ----------------
    task1_test_q = set(ex[(ex["small_version"] == 1) & (ex["split"] == "test")]["query_id"])
    large_train_q = set(ex[(ex["large_version"] == 1) & (ex["split"] == "train")]["query_id"])
    inter = task1_test_q & large_train_q
    g["A3_task1_test_disjoint_from_large_train"] = {
        "task1_test_queries": len(task1_test_q),
        "large_train_queries": len(large_train_q),
        "intersection_size": len(inter),
        "PASS": len(inter) == 0,
        "offending_query_ids": sorted(inter)[:20],
    }
    print(f"A3 no train/test leak    : {len(inter) == 0}  (intersection = {len(inter)})")
    if inter:
        STOPS.append(f"A3 FAILED: {len(inter)} Task1 test query_ids also appear in the "
                     f"large train split. Examples: {sorted(inter)[:20]}")

    # ---------------- A4 ----------------
    dup_mask = pr.duplicated(subset=["product_id", "product_locale"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_keys = int(pr[dup_mask].groupby(["product_id", "product_locale"]).ngroups) if n_dup_rows else 0
    g["A4_products_pk_unique"] = {
        "products_rows": int(len(pr)),
        "duplicate_rows": n_dup_rows,
        "duplicate_keys": n_dup_keys,
        "PASS": n_dup_rows == 0,
        "sample_duplicate_keys": (pr[dup_mask].drop_duplicates(["product_id", "product_locale"])
                                  .head(10).to_dict("records") if n_dup_rows else []),
    }
    print(f"A4 products PK unique    : {n_dup_rows == 0}  (duplicate rows = {n_dup_rows})")
    if n_dup_rows:
        STOPS.append(f"A4 FAILED: products table has {n_dup_rows} rows sharing "
                     f"{n_dup_keys} (product_id, product_locale) keys. NOT de-duplicated.")

    g["STOPS"] = STOPS
    g["GATE_A_PASS"] = len(STOPS) == 0
    with open(os.path.join(OUT, "gate_a.json"), "w") as f:
        json.dump(g, f, indent=2, default=str)

    print(f"\nGATE A: {'PASS' if not STOPS else 'FAIL'}")
    if STOPS:
        print("\n=== STOP ===")
        print("Completed sections: 3 (data_source_audit), 4 (gate_a, partial)")
        for s in STOPS:
            print(" -", s)
        print("Human decision required before any data is built.")
        sys.exit(2)


if __name__ == "__main__":
    main()
