"""
Section 10 -- scorer unit tests. STOP-gate: any failure halts the phase.

Scenarios A-K from the brief, each with at least one assertion and every
expected value derived by hand in the docstring, plus a 500-case property test.

Run: pytest tests/test_kdd_task1_ndcg.py
     PYTHONPATH=. python tests/test_kdd_task1_ndcg.py   (script mode; writes the manifest)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from src.metrics.kdd_task1_ndcg import (
    GAIN_COMPETITION, GAIN_SWAPPED, aggregate, evaluate, gain_from_labels,
    per_query_ndcg_table,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = []
gE, gS, gC, gI = 1.00, 0.10, 0.01, 0.00


def D(rank):
    """1-based discount."""
    return 1.0 / math.log2(rank + 1)


def check(name, got, want, tol=1e-12, detail=""):
    if isinstance(want, float) and isinstance(got, (int, float, np.floating)):
        ok = abs(float(got) - want) < tol
    else:
        ok = got == want
    RESULTS.append({"test": name, "expected": want, "got": got,
                    "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r} {detail}")
    # In the source repo these tests ran only as a script, where main() tallied
    # RESULTS and exited non-zero. Under pytest a recorded failure would be
    # invisible -- every test would pass vacuously -- so the check is raised
    # here as well. Script mode is unchanged on the all-pass path.
    assert ok, f"{name}: got={got!r} want={want!r} {detail}"
    return ok


def frame(rows, mapping=None):
    """rows = (query_id, product_id, product_locale, esci_label, score)"""
    df = pd.DataFrame(rows, columns=["query_id", "product_id", "product_locale",
                                     "esci_label", "score"])
    df["gain"] = gain_from_labels(df["esci_label"], mapping)
    return df


def one(df, col="score", **kw):
    return float(per_query_ndcg_table(df, col, **kw)["ndcg_full"].iloc[0])


# ------------------------------------------------------------------ A
def test_A_perfect():
    """E > S > C > I ranked exactly so. All three cutoffs must be 1.0."""
    print("\nA. perfect ranking")
    df = frame([("q", "p1", "us", "E", 4.0), ("q", "p2", "us", "S", 3.0),
                ("q", "p3", "us", "C", 2.0), ("q", "p4", "us", "I", 1.0)])
    t = per_query_ndcg_table(df, "score").iloc[0]
    check("full == 1.0", float(t.ndcg_full), 1.0)
    check("@10 == 1.0", float(t.ndcg_at_10), 1.0)
    check("@20 == 1.0", float(t.ndcg_at_20), 1.0)


# ------------------------------------------------------------------ B
def test_B_E_S_swap():
    """S placed above E, 2 candidates.
    DCG  = 0.10/log2(2) + 1.00/log2(3) = 0.1000000000 + 0.6309297536 = 0.7309297536
    IDCG = 1.00/log2(2) + 0.10/log2(3) = 1.0000000000 + 0.0630929754 = 1.0630929754
    NDCG = 0.6875501678
    """
    print("\nB. E/S order reversed")
    df = frame([("q", "pS", "us", "S", 2.0), ("q", "pE", "us", "E", 1.0)])
    dcg = gS * D(1) + gE * D(2)
    idcg = gE * D(1) + gS * D(2)
    got = one(df)
    check("hand-computed NDCG", got, dcg / idcg, detail=f"= {dcg/idcg:.10f}")
    check("drop below perfect", got < 1.0, True)
    check("drop magnitude", 1.0 - got, 1.0 - dcg / idcg,
          detail=f"= {1.0 - dcg/idcg:.10f}")


# ------------------------------------------------------------------ C
def test_C_S_C_swap():
    """C placed above S, 2 candidates.
    DCG  = 0.01/log2(2) + 0.10/log2(3) = 0.0100000000 + 0.0630929754 = 0.0730929754
    IDCG = 0.10/log2(2) + 0.01/log2(3) = 0.1000000000 + 0.0063092975 = 0.1063092975
    NDCG = 0.6875501678   (same ratio as B, by the 10x scaling of both gains)
    """
    print("\nC. S/C order reversed + gain values")
    df = frame([("q", "pC", "us", "C", 2.0), ("q", "pS", "us", "S", 1.0)])
    dcg = gC * D(1) + gS * D(2)
    idcg = gS * D(1) + gC * D(2)
    check("hand-computed NDCG", one(df), dcg / idcg, detail=f"= {dcg/idcg:.10f}")
    check("gain(S) == 0.10", GAIN_COMPETITION["S"], 0.10)
    check("gain(C) == 0.01", GAIN_COMPETITION["C"], 0.01)
    check("gain(S) is NOT 2**0.1-1", abs(0.10 - (2 ** 0.10 - 1)) > 1e-6, True,
          detail=f"exponential would be {2**0.10 - 1:.10f}")
    check("gain(C) is NOT 2**0.01-1", abs(0.01 - (2 ** 0.01 - 1)) > 1e-6, True,
          detail=f"exponential would be {2**0.01 - 1:.10f}")
    check("gain_from_labels maps E/S/C/I", gain_from_labels(
        pd.Series(["E", "S", "C", "I"])).tolist(), [1.00, 0.10, 0.01, 0.00])


# ------------------------------------------------------------------ D
def test_D_more_than_20():
    """25 candidates, all E, but with 6 I injected at the very top so that the
    three cutoffs must diverge: only ranks 1..k count for @k."""
    print("\nD. >20 candidates: full / @10 / @20 all differ")
    rows = [("q", f"i{i:02d}", "us", "I", 100.0 - i) for i in range(6)]
    rows += [("q", f"e{i:02d}", "us", "E", 50.0 - i) for i in range(19)]
    t = per_query_ndcg_table(frame(rows), "score").iloc[0]
    check("n_candidates == 25", int(t.n_candidates), 25)
    check("full != @20", abs(t.ndcg_full - t.ndcg_at_20) > 1e-9, True,
          detail=f"full={t.ndcg_full:.10f} @20={t.ndcg_at_20:.10f}")
    check("@20 != @10", abs(t.ndcg_at_20 - t.ndcg_at_10) > 1e-9, True,
          detail=f"@20={t.ndcg_at_20:.10f} @10={t.ndcg_at_10:.10f}")
    check("full != @10", abs(t.ndcg_full - t.ndcg_at_10) > 1e-9, True)
    # hand-check @10: 6 I then 4 E at ranks 7..10; ideal = 10 E at ranks 1..10
    exp10 = sum(D(r) for r in range(7, 11)) / sum(D(r) for r in range(1, 11))
    check("@10 hand-computed", float(t.ndcg_at_10), exp10, detail=f"= {exp10:.10f}")


# ------------------------------------------------------------------ E
def test_E_short_list():
    """n < 10 -> full, @10 and @20 all cover the whole list, so they are equal."""
    print("\nE. short list (n < 10): three cutoffs equal")
    df = frame([("q", "a", "us", "E", 1.0), ("q", "b", "us", "S", 3.0),
                ("q", "c", "us", "I", 2.0)])
    t = per_query_ndcg_table(df, "score").iloc[0]
    check("full == @10", float(t.ndcg_full), float(t.ndcg_at_10))
    check("@10 == @20", float(t.ndcg_at_10), float(t.ndcg_at_20))
    exp = (gS * D(1) + gI * D(2) + gE * D(3)) / (gE * D(1) + gS * D(2) + gI * D(3))
    check("value hand-computed", float(t.ndcg_full), exp, detail=f"= {exp:.10f}")


# ------------------------------------------------------------------ F
def test_F_all_I():
    """All-I query: IDCG == 0. Behaviour must match zero_idcg_policy.
    q1 scores 1.0; q2 is all-I. exclude -> 1.0, zero -> 0.5, one -> 1.0."""
    print("\nF. all-I query matches zero_idcg_policy")
    df = pd.concat([
        frame([("q1", "a", "us", "E", 2.0), ("q1", "b", "us", "I", 1.0)]),
        frame([("q2", "c", "us", "I", 2.0), ("q2", "d", "us", "I", 1.0)]),
    ], ignore_index=True)
    tbl = per_query_ndcg_table(df, "score")
    check("idcg_full == 0 for the all-I query", float(tbl.loc["q2", "idcg_full"]), 0.0)
    check("ndcg is NaN for the all-I query", bool(np.isnan(tbl.loc["q2", "ndcg_full"])), True)
    ex = aggregate(tbl, "exclude")
    check("exclude: count", ex["zero_idcg_query_count"], 1)
    check("exclude: scored", ex["n_queries_scored"], 1)
    check("exclude: mean == 1.0", ex["ndcg_full"], 1.0)
    check("zero: mean == 0.5", aggregate(tbl, "zero")["ndcg_full"], 0.5)
    check("one: mean == 1.0", aggregate(tbl, "one")["ndcg_full"], 1.0)


# ------------------------------------------------------------------ G
def test_G_ties_reproducible():
    """Identical scores -> deterministic tie-break on product_id asc.
    Order becomes pA(I), pB(E), pC(S).
    DCG  = 0 + 1.0*D(2) + 0.1*D(3) = 0.6309297536 + 0.05 = 0.6809297536
    IDCG = 1.0*D(1) + 0.1*D(2) + 0 = 1.0630929754
    NDCG = 0.6405175929
    """
    print("\nG. exact score ties, deterministic tie-break")
    rows = [("q", "pB", "us", "E", 1.0), ("q", "pA", "us", "I", 1.0),
            ("q", "pC", "us", "S", 1.0)]
    exp = (gI * D(1) + gE * D(2) + gS * D(3)) / (gE * D(1) + gS * D(2) + gI * D(3))
    check("ties break on product_id asc", one(frame(rows)), exp, detail=f"= {exp:.10f}")
    check("reproducible across calls", one(frame(rows)), one(frame(rows)))
    # locale is the third key
    rows2 = [("q", "pX", "us", "E", 1.0), ("q", "pX", "es", "I", 1.0)]
    v = one(frame(rows2))
    check("locale asc breaks equal product_id", v,
          (gI * D(1) + gE * D(2)) / (gE * D(1) + gI * D(2)),
          detail="es sorts before us")


# ------------------------------------------------------------------ H
def test_H_row_order_invariance():
    print("\nH. row-order invariance")
    rng = np.random.RandomState(3)
    base = [("q", f"p{i:02d}", "us", lab, float(s)) for i, (lab, s) in enumerate(
        zip(["E", "S", "C", "I", "E", "S", "I", "E", "C", "S", "I", "E"],
            [5, 5, 3, 3, 9, 1, 1, 7, 7, 2, 2, 0]))]
    ref = one(frame(base))
    ok = all(abs(one(frame([base[i] for i in rng.permutation(len(base))])) - ref) < 1e-15
             for _ in range(30))
    check("30 permutations give identical NDCG", ok, True, detail=f"NDCG = {ref:.12f}")
    # and with ties everywhere
    tied = [("q", f"p{i:02d}", "us", lab, 1.0) for i, lab in enumerate(
        ["E", "S", "C", "I", "E", "S", "I", "E"])]
    ref2 = one(frame(tied))
    ok2 = all(abs(one(frame([tied[i] for i in rng.permutation(len(tied))])) - ref2) < 1e-15
              for _ in range(30))
    check("30 permutations, all tied, identical NDCG", ok2, True, detail=f"NDCG = {ref2:.12f}")


# ------------------------------------------------------------------ I
def test_I_single_candidate():
    print("\nI. single-candidate query")
    check("single E -> 1.0", one(frame([("q", "a", "us", "E", 1.0)])), 1.0)
    check("single S -> 1.0", one(frame([("q", "a", "us", "S", 1.0)])), 1.0)
    tbl = per_query_ndcg_table(frame([("q", "a", "us", "I", 1.0)]), "score")
    check("single I -> idcg 0", float(tbl["idcg_full"].iloc[0]), 0.0)
    check("single I excluded", aggregate(tbl, "exclude")["n_queries_scored"], 0)


# ------------------------------------------------------------------ J
def test_J_degenerate_but_idcg_positive():
    """All candidates share one non-I label -> any ranking is ideal -> NDCG 1.0."""
    print("\nJ. degenerate query with IDCG > 0")
    for lab in ["E", "S", "C"]:
        rows = [("q", f"p{i}", "us", lab, float(9 - i)) for i in range(5)]
        check(f"all-{lab} query -> 1.0", one(frame(rows)), 1.0)
    rows = [("q", f"p{i}", "us", "S", 1.0) for i in range(5)]  # also all tied
    check("all-S and all tied -> 1.0", one(frame(rows)), 1.0)


# ------------------------------------------------------------------ K
def test_K_swapped_gain_reverses_direction():
    """Under GAIN_SWAPPED (S=0.01, C=0.1) scenario C's ranking becomes the
    IDEAL one, so its NDCG flips from 0.6875501678 to 1.0."""
    print("\nK. swapped-gain mode reverses B/C")
    rows_C = [("q", "pC", "us", "C", 2.0), ("q", "pS", "us", "S", 1.0)]
    comp = one(frame(rows_C, GAIN_COMPETITION))
    swap = one(frame(rows_C, GAIN_SWAPPED))
    check("C-above-S is suboptimal under competition gain", comp < 1.0, True,
          detail=f"= {comp:.10f}")
    check("C-above-S is OPTIMAL under swapped gain", swap, 1.0)
    check("direction reversed", swap > comp, True)
    # B is unaffected in direction (E still dominates) but changes in magnitude
    rows_B = [("q", "pS", "us", "S", 2.0), ("q", "pE", "us", "E", 1.0)]
    bc, bs = one(frame(rows_B, GAIN_COMPETITION)), one(frame(rows_B, GAIN_SWAPPED))
    check("B: swapped gain penalises the S-above-E error harder", bs < bc, True,
          detail=f"competition={bc:.10f} swapped={bs:.10f}")
    check("GAIN_SWAPPED table", GAIN_SWAPPED,
          {"E": 1.00, "S": 0.01, "C": 0.10, "I": 0.00})


# ------------------------------------------------- property test
def test_property_random():
    """500 random (labels, scores) cases: all outputs in [0,1]; the perfect
    ordering always scores exactly 1.0 on all three cutoffs."""
    print("\nProperty test: 500 random cases")
    rng = np.random.RandomState(1234)
    bad_range, bad_perfect = 0, 0
    for c in range(500):
        n = int(rng.randint(1, 41))
        labs = rng.choice(["E", "S", "C", "I"], size=n)
        rows = [("q", f"p{i:03d}", rng.choice(["us", "es", "jp"]), labs[i],
                 float(rng.randn())) for i in range(n)]
        df = frame(rows)
        t = per_query_ndcg_table(df, "score").iloc[0]
        for col in ["ndcg_full", "ndcg_at_10", "ndcg_at_20"]:
            v = t[col]
            if not np.isnan(v) and not (-1e-12 <= v <= 1 + 1e-12):
                bad_range += 1
        # perfect ordering = sort by gain desc
        dfp = df.copy()
        dfp["score"] = dfp["gain"]
        tp = per_query_ndcg_table(dfp, "score").iloc[0]
        if tp["idcg_full"] > 0:
            for col in ["ndcg_full", "ndcg_at_10", "ndcg_at_20"]:
                if abs(tp[col] - 1.0) > 1e-12:
                    bad_perfect += 1
    check("all outputs within [0,1]", bad_range, 0)
    check("perfect ordering always 1.0 on all cutoffs", bad_perfect, 0)


def main():
    print("=" * 74)
    print(" KDD Task 1 scorer unit tests (competition linear gain)")
    print("=" * 74)
    for t in [test_A_perfect, test_B_E_S_swap, test_C_S_C_swap, test_D_more_than_20,
              test_E_short_list, test_F_all_I, test_G_ties_reproducible,
              test_H_row_order_invariance, test_I_single_candidate,
              test_J_degenerate_but_idcg_positive, test_K_swapped_gain_reverses_direction,
              test_property_random]:
        t()
    n_pass = sum(r["pass"] for r in RESULTS)
    print("\n" + "=" * 74)
    print(f" {n_pass}/{len(RESULTS)} assertions passed")
    print("=" * 74)
    out = os.path.join(REPO_ROOT, "artifacts", "manifests", "scorer_unit_tests.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "scorer": "src/metrics/kdd_task1_ndcg.py",
            "gain_convention": "E=1.00, S=0.10, C=0.01, I=0.00 used directly as the DCG numerator",
            "exponentiation_applied": False,
            "cutoffs": ["full", 10, 20],
            "scenarios_covered": list("ABCDEFGHIJK") + ["property(500 random cases)"],
            "passed": n_pass, "total": len(RESULTS),
            "all_pass": n_pass == len(RESULTS),
            "assertions": RESULTS,
        }, f, indent=2, default=str)
    print(f"wrote {out}")
    if n_pass != len(RESULTS):
        print("\nSTOP: scorer unit test failure (section 10 is a STOP-gate).")
        sys.exit(2)


if __name__ == "__main__":
    main()
