"""
Competition alignment §9 -- metric parity.

An INDEPENDENT reference NDCG implementation: pure Python, no numpy, no shared
code with the production scorer, written straight from the definition

    DCG@k  = sum over the top-k of gain_i / log2(rank_i + 1)      rank 1-based
    IDCG@k = same over the candidate set sorted by gain descending
    NDCG@k = DCG@k / IDCG@k

with the official gains from amazon-science/esci-data ranking/train.py:
    E=1.0  S=0.1  C=0.01  I=0.0

Terrier could not be run (no JRE, forbidden to install), so parity is established
against this second implementation instead of letting the production scorer
validate itself. 8 required cases + randomised agreement.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import pandas as pd

from src import paths
from src.metrics.binding import write_report
from src.ranking.cross_encoder import load_task1_scorer

ROOT = str(paths.REPO_ROOT)
OUT = str(paths.RESULTS / "phase4")
OFFICIAL_GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
TOL = 1e-10


# ----------------------------- independent reference -----------------------
def ref_ndcg(labels, scores, doc_ids, k=None):
    """Pure-Python NDCG. Returns None when IDCG == 0 (undefined).
    Ties: score desc, then doc_id asc -- matching the production tie rule."""
    items = [(scores[i], doc_ids[i], OFFICIAL_GAIN[labels[i]]) for i in range(len(labels))]
    items.sort(key=lambda t: (-t[0], t[1]))
    kk = len(items) if k is None else min(k, len(items))
    dcg = 0.0
    for pos in range(kk):
        dcg += items[pos][2] / math.log2(pos + 2)
    ideal = sorted((OFFICIAL_GAIN[l] for l in labels), reverse=True)
    idcg = 0.0
    for pos in range(min(kk, len(ideal))):
        idcg += ideal[pos] / math.log2(pos + 2)
    if idcg == 0.0:
        return None
    return dcg / idcg


def prod_ndcg(labels, scores, doc_ids, k):
    """Same query through the production scorer."""
    sc = load_task1_scorer()
    df = pd.DataFrame({"query_id": ["q"] * len(labels), "product_id": doc_ids,
                       "product_locale": ["us"] * len(labels), "esci_label": labels,
                       "score": scores})
    df["gain"] = sc.gain_from_labels(df["esci_label"])
    t = sc.per_query_ndcg_table(df, "score", gain_col="gain", query_col="query_id")
    col = "ndcg_full" if k is None else f"ndcg_at_{k}"
    v = t[col].iloc[0]
    return None if pd.isna(v) else float(v)


CASES = []


def case(name, labels, scores, k, note=""):
    ids = [f"p{i:03d}" for i in range(len(labels))]
    r, p = ref_ndcg(labels, scores, ids, k), prod_ndcg(labels, scores, ids, k)
    if r is None or p is None:
        ok = (r is None) == (p is None)
        d = None
    else:
        d = abs(r - p)
        ok = d < TOL
    CASES.append({"case": name, "k": k or "full", "n_candidates": len(labels),
                  "reference": r, "production": p,
                  "abs_delta": d, "pass": bool(ok), "note": note})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} k={str(k or 'full'):<4} "
          f"ref={r if r is None else round(r,12)} prod={p if p is None else round(p,12)} "
          f"|Δ|={'n/a' if d is None else f'{d:.2e}'}")
    return ok


def main():
    os.makedirs(OUT, exist_ok=True)
    print("METRIC PARITY -- production scorer vs independent reference\n")

    # Case 1: E>S>C>I ranked correctly
    case("1 perfect E>S>C>I", ["E", "S", "C", "I"], [4, 3, 2, 1], 20,
         "must be exactly 1.0")
    case("1 perfect E>S>C>I (full)", ["E", "S", "C", "I"], [4, 3, 2, 1], None)
    # Case 2: swap E and S
    case("2 swap E<->S", ["E", "S"], [1, 2], 20)
    # Case 3: swap S and C
    case("3 swap S<->C", ["S", "C"], [1, 2], 20)
    # Case 4: only I
    case("4 all irrelevant (IDCG=0)", ["I", "I", "I"], [3, 2, 1], 20,
         "IDCG==0 -> undefined; both must agree")
    # Case 5: fewer than 20 candidates
    case("5 fewer than 20 candidates", ["E", "S", "I"], [3, 2, 1], 20)
    # Case 6: more than 20 candidates
    lab6 = ["I"] * 5 + ["E"] * 20
    case("6 more than 20 candidates", lab6, list(range(len(lab6), 0, -1)), 20)
    case("6 more than 20 candidates (full)", lab6, list(range(len(lab6), 0, -1)), None)
    # Case 7: ties
    case("7 all scores tied", ["E", "I", "S", "C"], [1, 1, 1, 1], 20)
    # Case 8: IDCG=0 explicitly at full
    case("8 IDCG=0 at full cutoff", ["I", "I"], [2, 1], None)

    # exact hand-checked values
    hand = []
    e, s, c = 1.0, 0.1, 0.01
    d1, d2, d3 = 1 / math.log2(2), 1 / math.log2(3), 1 / math.log2(4)
    hand.append(("2 swap E<->S expected", (s * d1 + e * d2) / (e * d1 + s * d2)))
    hand.append(("3 swap S<->C expected", (c * d1 + s * d2) / (s * d1 + c * d2)))
    print("\nhand-computed cross-checks:")
    for nm, v in hand:
        print(f"  {nm} = {v:.12f}")
    ok2 = abs(CASES[2]["production"] - hand[0][1]) < TOL
    ok3 = abs(CASES[3]["production"] - hand[1][1]) < TOL
    CASES.append({"case": "2 hand-computed value", "pass": bool(ok2),
                  "expected": hand[0][1], "production": CASES[2]["production"],
                  "abs_delta": abs(CASES[2]["production"] - hand[0][1])})
    CASES.append({"case": "3 hand-computed value", "pass": bool(ok3),
                  "expected": hand[1][1], "production": CASES[3]["production"],
                  "abs_delta": abs(CASES[3]["production"] - hand[1][1])})
    print(f"  [{'PASS' if ok2 else 'FAIL'}] case 2 matches hand value")
    print(f"  [{'PASS' if ok3 else 'FAIL'}] case 3 matches hand value")

    # randomised agreement
    print("\nrandomised agreement (300 synthetic queries):")
    rnd = random.Random(42)
    worst, bad = 0.0, 0
    for _ in range(300):
        n = rnd.randint(1, 95)
        labs = [rnd.choice("ESCI") for _ in range(n)]
        scs = [rnd.gauss(0, 1) for _ in range(n)]
        ids = [f"p{i:03d}" for i in range(n)]
        for k in (10, 20, None):
            r, p = ref_ndcg(labs, scs, ids, k), prod_ndcg(labs, scs, ids, k)
            if (r is None) != (p is None):
                bad += 1
            elif r is not None:
                worst = max(worst, abs(r - p))
                if abs(r - p) >= TOL:
                    bad += 1
    print(f"  disagreements: {bad}   max |Δ| = {worst:.3e}")
    CASES.append({"case": "randomised 300 queries x {10,20,full}", "pass": bad == 0,
                  "disagreements": bad, "max_abs_delta": worst})

    n_pass = sum(1 for c in CASES if c["pass"])
    verdict = "PASS" if n_pass == len(CASES) else "FAIL"
    payload = {
        "metric_parity": verdict,
        "tolerance": TOL,
        "reference_implementation": "independent pure-Python, no shared code, no numpy",
        "terrier_status": "NOT_RUN -- no JRE on this machine and installing one was out of scope",
        "official_gain_source": ("amazon-science/esci-data ranking/train.py lines 49-53, "
                                 "esci_label2gain = {'E':1.0,'S':0.1,'C':0.01,'I':0.0}"),
        "gains_used": OFFICIAL_GAIN,
        "tie_rule": "score desc, then product_id asc (both implementations)",
        "idcg_zero_rule": "query undefined/excluded; both implementations must agree",
        "cases_passed": n_pass, "cases_total": len(CASES), "cases": CASES,
    }
    os.makedirs(OUT, exist_ok=True)
    write_report(payload, os.path.join(OUT, "metric_parity_results.json"))
    print(f"\n{n_pass}/{len(CASES)} checks passed -> metric_parity = {verdict}")
    print("wrote experiments/competition_alignment/metric_parity_results.json")
    sys.exit(0 if verdict == "PASS" else 2)


if __name__ == "__main__":
    main()
