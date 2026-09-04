"""
Resolution items 2, 4, 5.

item 2: add the operative assertions A2' and A3', both must PASS, and record
        them side by side with the red upstream A2/A3 in gate_a.json.
          A2' split uniqueness scoped to small_version == 1
          A3' set(materialized arm_B.query_id) & task1_test_q == empty,
              evaluated on the ACTUAL materialized arm B pool via the shared
              helper -- not a proxy.
item 4: query-text collision diagnostic -> known_query_text_collisions.json
        (no quarantine, no filtering).
item 5: task1.groupby('product_id').product_locale.nunique().max()

No data is modified.
"""
import json
import os
import sys

import pandas as pd

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import paths  # noqa: E402
from src.data.task1_common import (  # noqa: E402
    build_arm_b_pool, load_examples, normalize_query_text, task1_test_query_ids,
)

ROOT = str(paths.REPO_ROOT)
BASE = str(paths.MANIFESTS)


def main():
    with open(os.path.join(BASE, "gate_a.json")) as f:
        g = json.load(f)

    # A2' compares esci_label across splits, so this needs the labelled test
    # rows. Authorised by the committed construction pre-registration (§13.1);
    # the access is logged to artifacts/manifests/test_access_log.json.
    ex = load_examples(prereg="docs/prereg/pool_construction.md")
    t1 = ex[ex["small_version"] == 1]
    test_q = task1_test_query_ids(ex)

    # ================= A2' : split uniqueness within the Task 1 pool =========
    n_multi = int((t1.groupby("query_id")["split"].nunique() > 1).sum())
    a2p = {
        "assertion": "within small_version==1, every query_id belongs to exactly one split",
        "scope": "small_version == 1 (Task 1 pool)",
        "queries_with_multiple_splits": n_multi,
        "task1_queries": int(t1["query_id"].nunique()),
        "PASS": n_multi == 0,
    }
    print(f"A2' split uniqueness within Task1 pool : {a2p['PASS']} ({n_multi} violations)")

    # ================= A3' : materialized arm B is disjoint from Task1 test ==
    arm_b, arm_b_info = build_arm_b_pool(ex)
    arm_b_q = set(arm_b["query_id"].unique())
    inter = arm_b_q & test_q
    a3p = {
        "assertion": "set(materialized arm_B.query_id) & task1_test_q == empty",
        "evaluated_on": "the ACTUAL materialized arm B pool (not a proxy)",
        "helper_used": "task1_common.build_arm_b_pool -> exclude_task1_test_queries",
        "arm_b_rows": int(len(arm_b)),
        "arm_b_queries": len(arm_b_q),
        "task1_test_queries": len(test_q),
        "intersection_size": len(inter),
        "PASS": len(inter) == 0,
        "exclusion_info": arm_b_info,
    }
    print(f"A3' materialized arm B disjoint        : {a3p['PASS']} "
          f"(arm B {len(arm_b)} rows / {len(arm_b_q)} queries; "
          f"removed {arm_b_info['rows_removed']} rows / "
          f"{arm_b_info['queries_removed']} queries)")

    g["OPERATIVE_ASSERTIONS"] = {
        "note": ("A2/A3 are recorded RED as FAIL_UPSTREAM: they assert a property of the raw "
                 "ESCI table that ESCI itself violates for exactly one query_id (79706). No "
                 "data was modified, filtered or de-duplicated. A2'/A3' are the GREEN "
                 "operative assertions the benchmark actually depends on, and both PASS."),
        "A2_prime_split_uniqueness_within_task1": a2p,
        "A3_prime_materialized_arm_b_disjoint": a3p,
        "BOTH_PASS": bool(a2p["PASS"] and a3p["PASS"]),
    }
    g["A2_A3_STATUS"] = "FAIL_UPSTREAM (documented, data untouched)"
    g["A2_prime_A3_prime_STATUS"] = "PASS" if (a2p["PASS"] and a3p["PASS"]) else "FAIL"
    g["GATE_A_RESOLUTION"] = (
        "Resolved by human decision: A2/A3 recorded as FAIL_UPSTREAM; operative assertions "
        "A2'/A3' added and both PASS; query_id 79706 NOT excluded and no data modified.")
    g["UPSTREAM_DEFECT_79706"] = {
        "query_id": 79706,
        "query_text": "piano",
        "locale": "us",
        "small_version_1_test_rows": 31,
        "large_only_train_rows": 3,
        "product_overlap_between_splits": 0,
        "sole_occurrence_in_dataset": True,
        "violates": ("the split property stated in the official ESCI README, namely that a "
                     "query belongs to exactly one of the train/test splits"),
        "action_taken": "documented only; not modified, not filtered, not de-duplicated",
    }

    # ================= item 5 : product_id -> locale multiplicity ============
    m = t1.groupby("product_id")["product_locale"].nunique()
    max_loc = int(m.max())
    item5 = {
        "check": "task1.groupby('product_id').product_locale.nunique().max()",
        "max_locales_per_product_id": max_loc,
        "product_ids_in_multiple_locales": int((m > 1).sum()),
        "unique_product_ids": int(len(m)),
        "unique_product_id_locale_pairs": int(len(
            t1[["product_id", "product_locale"]].drop_duplicates())),
        "locale_prefixed_doc_id_mandatory": max_loc > 1,
        "doc_id_format": "f\"{product_locale}_{product_id}\"" if max_loc > 1 else "product_id",
        "divergence_from_official_helper": (
            "amazon-science/esci-data ranking/prepare_trec_eval_files.py writes qrels keyed on "
            "product_id ALONE (col_product_id, lines 121-126) with no locale component. Because "
            f"{int((m > 1).sum())} product_ids in the Task 1 pool appear under up to {max_loc} "
            "different locales, that keying collapses distinct (product, locale) judgements onto "
            "one doc_id, which can silently corrupt both the qrels and the run file. This "
            "benchmark therefore uses a locale-prefixed doc_id, diverging from the reference "
            "helper deliberately." if max_loc > 1 else "not applicable"),
    }
    g["PRODUCT_LOCALE_MULTIPLICITY"] = item5
    print(f"item5 max locales per product_id       : {max_loc} "
          f"({item5['product_ids_in_multiple_locales']} product_ids) -> "
          f"locale-prefixed doc_id mandatory = {item5['locale_prefixed_doc_id_mandatory']}")

    with open(os.path.join(BASE, "gate_a.json"), "w") as f:
        json.dump(g, f, indent=2, default=str)

    # ================= item 4 : query-text collisions ========================
    print("\nitem4 query-text collision diagnostic ...")
    t1 = t1.copy()
    t1["qn"] = normalize_query_text(t1["query"])
    tr = t1[t1["split"] == "train"]
    te = t1[t1["split"] == "test"]
    shared_txt = set(tr["qn"]) & set(te["qn"])

    tr_s = tr[tr["qn"].isin(shared_txt)]
    te_s = te[te["qn"].isin(shared_txt)]

    # cross-split (normalized_query, product_id, product_locale) triples
    key = ["qn", "product_id", "product_locale"]
    trk = tr_s[key + ["esci_label", "query_id"]].drop_duplicates()
    tek = te_s[key + ["esci_label", "query_id"]].drop_duplicates()
    merged = trk.merge(tek, on=key, suffixes=("_train", "_test"))
    n_triples = int(len(merged))
    n_agree = int((merged["esci_label_train"] == merged["esci_label_test"]).sum()) if n_triples else 0

    affected_test_queries = sorted(te_s["query_id"].unique().tolist())
    n_affected_test_q = len(affected_test_queries)
    n_test_q = int(te["query_id"].nunique())

    # worst-case NDCG bound: if EVERY affected test query were scored 1.0 purely
    # by memorisation, the inflation is bounded by n_affected / n_test_queries
    bound = n_affected_test_q / n_test_q

    coll = {
        "policy": "NOT quarantined, NOT filtered -- diagnostic only (resolution item 4)",
        "normalization": "lowercase + whitespace-collapse",
        "task1_train_unique_query_texts": int(tr["qn"].nunique()),
        "task1_test_unique_query_texts": int(te["qn"].nunique()),
        "shared_query_texts": len(shared_txt),
        "shared_query_texts_list": sorted(shared_txt),
        "affected_test_query_ids": affected_test_queries,
        "affected_test_query_count": n_affected_test_q,
        "affected_train_query_ids": sorted(tr_s["query_id"].unique().tolist()),
        "affected_train_query_count": int(tr_s["query_id"].nunique()),
        "cross_split_query_product_locale_triples": {
            "count": n_triples,
            "with_agreeing_label": n_agree,
            "with_conflicting_label": n_triples - n_agree,
            "agreement_rate": (round(n_agree / n_triples, 4) if n_triples else None),
            "examples": merged.head(20).to_dict("records") if n_triples else [],
            "interpretation": ("These are (normalised query text, product, locale) combinations "
                               "judged on BOTH sides of the split under different query_ids. "
                               "They are the only channel through which a memorising model "
                               "could transfer a literal judgement."
                               if n_triples else
                               "No product is judged on both sides of the split for any shared "
                               "query text, so there is no literal pair-level transfer channel; "
                               "the collision is query-text only."),
        },
        "worst_case_ndcg_bound": {
            "n_affected_test_queries": n_affected_test_q,
            "n_test_queries": n_test_q,
            "bound": round(bound, 6),
            "definition": ("n_affected / 14496. This is the maximum possible absolute NDCG@10 "
                           "inflation, assuming the adversarial extreme in which every affected "
                           "test query is scored a perfect 1.0 purely by memorising its "
                           "same-text training twin, and would otherwise have scored 0.0. The "
                           "realistic effect is far smaller: the query_ids differ, the candidate "
                           "sets differ, and the models are feature-based rather than "
                           "memorising."),
        },
        "per_text_detail": [],
    }
    for t in sorted(shared_txt):
        a, b = tr[tr["qn"] == t], te[te["qn"] == t]
        coll["per_text_detail"].append({
            "normalized_query": t,
            "train_query_ids": sorted(a["query_id"].unique().tolist()),
            "test_query_ids": sorted(b["query_id"].unique().tolist()),
            "train_locales": sorted(a["product_locale"].unique().tolist()),
            "test_locales": sorted(b["product_locale"].unique().tolist()),
            "train_rows": int(len(a)), "test_rows": int(len(b)),
            "shared_products": int(len(set(a["product_id"]) & set(b["product_id"]))),
        })

    with open(os.path.join(BASE, "known_query_text_collisions.json"), "w") as f:
        json.dump(coll, f, indent=2, default=str)

    print(f"  shared query texts: {len(shared_txt)}")
    print(f"  affected test queries: {n_affected_test_q} / {n_test_q} "
          f"-> worst-case NDCG bound {bound:.6f}")
    print(f"  cross-split (qn, product_id, locale) triples: {n_triples} "
          f"(label agree {n_agree}, conflict {n_triples - n_agree})")
    print("\nwrote gate_a.json (updated) and known_query_text_collisions.json")

    if not (a2p["PASS"] and a3p["PASS"]):
        print("\nSTOP: an operative assertion failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
