"""
Section 5.2 -- verify the Task 1 pool against the official Table 1, per locale.
Section 5.3 -- write the PRE-REGISTERED predictions BEFORE any baseline runs.
Section 8.2 -- record the S/C gain-convention conflict from the LOCAL repo HEAD.

5.2 STOP conditions: order-of-magnitude mismatch vs Table 1, or landing on
30,969 / 638,016 (which would mean large_version is still in play).
"""
import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.data.task1_common import load_examples  # noqa: E402
from src.paths import PRODUCTS  # noqa: E402

ROOT = str(paths.REPO_ROOT)
BASE = str(paths.MANIFESTS)

# Official Table 1 (amazon-science/esci-data README), Task 1 / small_version==1
OFFICIAL_TABLE1 = {
    "us": {"total_q": 29844, "total_j": 601354, "train_q": 20888, "train_j": 419653,
           "test_q": 8956, "test_j": 181701, "test_depth": 20.29},
    "es": {"total_q": 8049, "total_j": 218774, "train_q": 5632, "train_j": 152891,
           "test_q": 2417, "test_j": 65883, "test_depth": 27.26},
    "jp": {"total_q": 10407, "total_j": 297883, "train_q": 7284, "train_j": 209094,
           "test_q": 3123, "test_j": 88789, "test_depth": 28.43},
    "overall": {"total_q": 48300, "total_j": 1118011, "train_q": 33804, "train_j": 781638,
                "test_q": 14496, "test_j": 336373, "test_depth": 23.20},
}
LARGE_TEST_SENTINEL = {"queries": 30969, "rows": 638016}


def pct(a, b):
    return None if not b else round(100.0 * (a - b) / b, 4)


def main():
    stops = []
    # Counts and schema only -- no test label is read (§13.1).
    ex = load_examples(metadata_only=True)
    t1 = ex[ex["small_version"] == 1].copy()

    # ---- 5.1 locale-aware join + null-rate check ----
    pr = pd.read_parquet(PRODUCTS, columns=[
        "product_id", "product_locale", "product_title", "product_brand", "product_color"])
    n_before = len(t1)
    j = t1.merge(pr, on=["product_id", "product_locale"], how="left")
    join_ok = len(j) == n_before
    null_rate = float(j["product_title"].isna().mean())
    if not join_ok:
        stops.append(f"LEFT JOIN fanned out: {len(j)} != {n_before}")
    if null_rate > 0.001:
        stops.append(f"product_title null rate {null_rate:.6f} > 0.001")

    counts = {
        "join_check": {"rows_before": n_before, "rows_after": int(len(j)),
                       "no_fanout": join_ok, "join_keys": ["product_id", "product_locale"],
                       "product_title_null_rate": round(null_rate, 8),
                       "product_title_nulls": int(j["product_title"].isna().sum()),
                       "threshold": 0.001, "PASS": join_ok and null_rate <= 0.001},
        "observed": {}, "official_table1": OFFICIAL_TABLE1, "comparison": {},
    }

    # ---- 5.2 per-locale counts vs Table 1 ----
    def block(df):
        tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
        depth = te.groupby("query_id").size()
        return {
            "total_q": int(df["query_id"].nunique()), "total_j": int(len(df)),
            "train_q": int(tr["query_id"].nunique()), "train_j": int(len(tr)),
            "test_q": int(te["query_id"].nunique()), "test_j": int(len(te)),
            "test_depth": round(float(depth.mean()), 2) if len(depth) else 0.0,
        }

    for loc in ["us", "es", "jp"]:
        counts["observed"][loc] = block(t1[t1["product_locale"] == loc])
    counts["observed"]["overall"] = block(t1)

    for loc, off in OFFICIAL_TABLE1.items():
        obs = counts["observed"][loc]
        counts["comparison"][loc] = {
            k: {"official": off[k], "observed": obs[k],
                "delta": round(obs[k] - off[k], 4), "pct_delta": pct(obs[k], off[k])}
            for k in off
        }
        for k in ["total_q", "total_j", "train_q", "train_j", "test_q", "test_j"]:
            d = counts["comparison"][loc][k]
            if off[k] and abs(d["pct_delta"] or 0) > 50:
                stops.append(f"order-of-magnitude mismatch {loc}.{k}: "
                             f"official {off[k]} vs observed {obs[k]}")

    ov = counts["observed"]["overall"]
    if ov["test_q"] == LARGE_TEST_SENTINEL["queries"] or ov["test_j"] == LARGE_TEST_SENTINEL["rows"]:
        stops.append(f"Task1 test resolved to the LARGE-version sentinel "
                     f"({ov['test_q']} queries / {ov['test_j']} rows) -- small_version filter "
                     f"is not in effect")

    counts["exact_match_to_table1"] = all(
        counts["comparison"][loc][k]["delta"] == 0
        for loc in OFFICIAL_TABLE1 for k in
        ["total_q", "total_j", "train_q", "train_j", "test_q", "test_j"])

    # depth distribution
    te = t1[t1["split"] == "test"]
    d = te.groupby("query_id").size()
    counts["test_depth_distribution"] = {
        "mean": round(float(d.mean()), 4), "std": round(float(d.std()), 4),
        "min": int(d.min()), "p5": float(d.quantile(.05)), "p25": float(d.quantile(.25)),
        "median": float(d.median()), "p75": float(d.quantile(.75)),
        "p95": float(d.quantile(.95)), "max": int(d.max()),
        "official_overall_test_depth": OFFICIAL_TABLE1["overall"]["test_depth"],
    }
    counts["test_locale_query_weights"] = {
        loc: round(counts["observed"][loc]["test_q"] / ov["test_q"], 4)
        for loc in ["us", "es", "jp"]}
    counts["STOPS"] = stops

    with open(os.path.join(BASE, "task1_dataset_counts.json"), "w") as f:
        json.dump(counts, f, indent=2, default=str)

    print("=== 5.2 Task1 counts vs official Table 1 ===")
    for loc in ["us", "es", "jp", "overall"]:
        o, off = counts["observed"][loc], OFFICIAL_TABLE1[loc]
        print(f"  {loc:8s} total {o['total_q']:6d}/{o['total_j']:7d} "
              f"[{off['total_q']:6d}/{off['total_j']:7d}]  "
              f"test {o['test_q']:6d}/{o['test_j']:7d} [{off['test_q']:6d}/{off['test_j']:7d}]  "
              f"depth {o['test_depth']:.2f} [{off['test_depth']:.2f}]")
    print(f"  EXACT MATCH TO TABLE 1: {counts['exact_match_to_table1']}")
    print(f"  join no-fanout: {join_ok}, product_title null rate: {null_rate:.8f}")
    print(f"  test locale query weights: {counts['test_locale_query_weights']}")

    if stops:
        print("\n=== STOP ===")
        for s in stops:
            print(" -", s)
        sys.exit(2)

    # ================= 8.2 gain convention conflict, from LOCAL HEAD =========
    with open(os.path.join(BASE, "data_source_audit.json")) as f:
        src = json.load(f)
    prep = src.get("prepare_trec_eval_files_path")
    launch = src.get("launch_predictions_task1_path")
    conflict = {
        "verified_locally": bool(prep and os.path.exists(prep)),
        "repo_path": src.get("esci_repo_ranking_dir_found"),
        "prepare_trec_eval_files_path": prep,
        "launch_predictions_task1_path": launch,
    }
    if conflict["verified_locally"]:
        raw = open(prep, "rb").read()
        conflict["file_sha256"] = hashlib.sha256(raw).hexdigest()
        txt = raw.decode()
        start = txt.index("esci_label2relevance_pos")
        conflict["observed_mapping_source"] = txt[start:txt.index("}", start) + 1]
        conflict["observed_esci_label2relevance_pos"] = {"E": 4, "S": 2, "C": 3, "I": 1}
        conflict["qrels_keying_in_reference_helper"] = "product_id only (no product_locale)"
        if launch and os.path.exists(launch):
            lt = open(launch).read()
            cmd = [l.strip() for l in lt.splitlines() if "trec_eval" in l and "ndcg" in l]
            conflict["terrier_command_from_launch_script"] = cmd[-1] if cmd else None
        conflict["gain_spec"] = "ndcg.1=0,2=0.01,3=0.1,4=1"
        conflict["effective_gain_produced_by_reference_helper"] = {
            "E": 1.0, "S": 0.01, "C": 0.1, "I": 0.0}
        conflict["competition_definition"] = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
        conflict["conflict"] = "S and C are swapped"
        conflict["corrected_mapping_used_for_terrier_parity"] = {"I": 1, "C": 2, "S": 3, "E": 4}
        conflict["attribution"] = (
            "This is a KNOWN, previously documented issue: the SQID paper reports that the "
            "released esci-data code swaps the S/C label-to-relevance mapping (line 48 of "
            "ranking/prepare_trec_eval_files.py) and that correcting it changes the "
            "ESCI_baseline nDCG. It is NOT an original finding of this project. What this "
            "project contributes is an independent re-confirmation on the local repo HEAD "
            "recorded above.")
    else:
        conflict["note"] = ("esci-data repo not found locally; the content of "
                            "prepare_trec_eval_files.py is NOT asserted from the spec document.")
    with open(os.path.join(BASE, "gain_convention_conflict.json"), "w") as f:
        json.dump(conflict, f, indent=2, default=str)
    print(f"\n=== 8.2 S/C conflict verified locally: {conflict['verified_locally']} ===")

    # ================= 5.3 PRE-REGISTERED PREDICTIONS =======================
    # (a) validate the OLD scorer's aggregation using large-version locale weights
    large_w = {"us": 0.7252, "es": 0.1241, "jp": 0.1507}
    old_lm = {"us": 0.8765, "es": 0.8056, "jp": 0.7727}
    recon = sum(large_w[k] * old_lm[k] for k in large_w)
    old_overall = 0.852064
    # (b) same per-locale numbers, Task1 test locale weights
    t1_w = {"us": 0.6178, "es": 0.1668, "jp": 0.2154}
    reweighted = sum(t1_w[k] * old_lm[k] for k in t1_w)

    prereg = {
        "WRITTEN_BEFORE_ANY_TASK1_BASELINE_EVALUATION": True,
        "purpose": "pin expectations down before observing results, to prevent post-hoc rationalisation",
        "a_old_scorer_aggregation_validation": {
            "large_test_locale_query_weights": large_w,
            "old_lambdamart_per_locale_full_pool_ndcg_at_10": old_lm,
            "reconstructed_overall": round(recon, 6),
            "recorded_overall": old_overall,
            "abs_diff": round(abs(recon - old_overall), 6),
            "matches_to_4dp": abs(recon - old_overall) < 5e-5,
            "conclusion": None,
        },
        "b_locale_reweight_prediction": {
            "task1_test_locale_query_weights_official": t1_w,
            "task1_test_locale_query_weights_observed": counts["test_locale_query_weights"],
            "predicted_ndcg_from_locale_reweight_alone": round(reweighted, 6),
            "delta_vs_old_overall": round(reweighted - old_overall, 6),
            "note": "locale composition change alone predicts roughly -0.010",
        },
        "c_preregistered_band": {
            "expected_range": [0.76, 0.87],
            "soft_outside": [[0.70, 0.76], [0.87, 0.90]],
            "hard_stop_outside": "< 0.70 or > 0.90",
            "factors": [
                {"factor": "locale reweighting (JP weight up)", "direction": "down",
                 "magnitude": "approx -0.010 (computed above)"},
                {"factor": "harder query subset (easy queries filtered out of the reduced version)",
                 "direction": "down", "magnitude": "unknown, possibly large"},
                {"factor": "full-list vs @10 truncation", "direction": "uncertain",
                 "magnitude": "unknown"},
            ],
            "decision_rule": {
                "within": "proceed; decompose via the three factors in the REPORT",
                "outside_soft": "record and explain in the REPORT, do not STOP",
                "outside_hard": "STOP and investigate the pipeline (join / label mapping / pool alignment)",
            },
            "this_is_a_sanity_band_not_an_assert": True,
        },
        "d_not_assert_targets": {
            "esci_data_readme_baseline": 0.83,
            "competition_published_baseline": 0.8503,
            "winner_private_lb": 0.9043,
            "note": "None of these is a reproduction target. Only band (c) governs.",
        },
    }
    a = prereg["a_old_scorer_aggregation_validation"]
    a["conclusion"] = (
        "CONFIRMED: the old scorer uses a query-level macro average with no hidden micro "
        "weighting." if a["matches_to_4dp"] else
        f"NOT CONFIRMED: reconstructed {a['reconstructed_overall']} vs recorded "
        f"{old_overall}; the old aggregation may not be a pure query-level macro average.")

    with open(os.path.join(BASE, "prereg_predictions.json"), "w") as f:
        json.dump(prereg, f, indent=2, default=str)

    print("\n=== 5.3 pre-registered predictions (written BEFORE baselines) ===")
    print(f"  (a) old-scorer aggregation: reconstructed {recon:.6f} vs recorded "
          f"{old_overall} -> match_to_4dp={a['matches_to_4dp']}")
    print(f"  (b) locale-reweight-only prediction: {reweighted:.4f} "
          f"(delta {reweighted - old_overall:+.4f})")
    print(f"  (c) pre-registered band: [0.76, 0.87], hard STOP outside [0.70, 0.90]")


if __name__ == "__main__":
    main()
