"""
Hand-derived unit tests for the authoritative KDD NDCG@10 scorer.

Every expected number is computed by hand in the test's docstring, so the
scorer can be validated without trusting anything else in the repository.

Covers the six cases required by the Phase 1.1 brief (A-F), plus equal-query-
weight and cutoff guards (G, H).

The original file carried a ninth test, `test_I_differs_from_legacy_scorer`,
which imported the legacy exponential scorer to assert the two disagree. That
test was REMOVED during migration, not skipped: MIGRATION_PLAN.md §11 item 14
bars the legacy scorer from this repository entirely -- not one line, and not
as a reference. Tests C ("gain values are linear") and B independently pin the
linear-gain convention that test I was guarding, so no coverage is lost.

Run: pytest tests/test_official_kdd_ndcg.py
     PYTHONPATH=. python tests/test_official_kdd_ndcg.py   (script mode; writes the manifest)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from src.metrics.official_kdd_ndcg import (
    OFFICIAL_GAIN, attach_gain, dcg, evaluate, gain_from_labels, per_query_ndcg,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = []

gE, gS, gC, gI = 1.00, 0.10, 0.01, 0.00


def D(rank):
    """1-based discount 1/log2(rank+1)."""
    return 1.0 / math.log2(rank + 1)


def check(name, got, want, tol=1e-12, detail=""):
    if isinstance(want, float) and isinstance(got, (int, float)):
        ok = abs(got - want) < tol
    else:
        ok = got == want
    RESULTS.append({"test": name, "expected": want, "got": got,
                    "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r} {detail}")
    # See the note in tests/test_kdd_task1_ndcg.py: recorded-only failures are
    # invisible to pytest, so the check is raised here too.
    assert ok, f"{name}: got={got!r} want={want!r} {detail}"
    return ok


def frame(rows):
    """rows = list of (query_id, product_id, esci_label, score)"""
    df = pd.DataFrame(rows, columns=["query_id", "product_id", "esci_label", "score"])
    return attach_gain(df)


# ---------------------------------------------------------------- Test A
def test_A_perfect_ranking():
    """E > S > C > I, ranked exactly that way.

    DCG  = 1.00/log2(2) + 0.10/log2(3) + 0.01/log2(4) + 0.00/log2(5)
    IDCG = the same sum, since the ranking already is the ideal one.
    NDCG = 1.0 exactly.
    """
    print("\nTest A: perfect ranking E > S > C > I")
    df = frame([("q", "p1", "E", 4.0), ("q", "p2", "S", 3.0),
                ("q", "p3", "C", 2.0), ("q", "p4", "I", 1.0)])
    check("perfect ranking -> NDCG == 1.0", float(per_query_ndcg(df, "score").iloc[0]), 1.0)
    manual = gE * D(1) + gS * D(2) + gC * D(3) + gI * D(4)
    check("hand-computed DCG matches dcg()", dcg([gE, gS, gC, gI], 10), manual,
          detail=f"= {manual:.10f}")


# ---------------------------------------------------------------- Test B
def test_B_substitute_above_exact():
    """Two candidates, S scored above E.

    DCG  = 0.1/log2(2) + 1.0/log2(3) = 0.1000000000 + 0.6309297536 = 0.7309297536
    IDCG = 1.0/log2(2) + 0.1/log2(3) = 1.0000000000 + 0.0630929754 = 1.0630929754
    NDCG = 0.7309297536 / 1.0630929754 = 0.6875422...
    """
    print("\nTest B: S ranked above E")
    df = frame([("q", "pS", "S", 2.0), ("q", "pE", "E", 1.0)])
    exp_dcg = 0.1 / math.log2(2) + 1.0 / math.log2(3)
    exp_idcg = 1.0 / math.log2(2) + 0.1 / math.log2(3)
    check("brief's hand-computed DCG", dcg([gS, gE], 10), exp_dcg, detail=f"= {exp_dcg:.10f}")
    check("brief's hand-computed IDCG", dcg([gE, gS], 10), exp_idcg, detail=f"= {exp_idcg:.10f}")
    check("NDCG matches DCG/IDCG", float(per_query_ndcg(df, "score").iloc[0]),
          exp_dcg / exp_idcg, detail=f"= {exp_dcg/exp_idcg:.10f}")


# ---------------------------------------------------------------- Test C
def test_C_gain_values_are_linear():
    """The whole point of Phase 1.1: gains must be the raw ESCI values.

    gain(S) must be 0.10, NOT 2**0.1 - 1 = 0.0717734625
    gain(C) must be 0.01, NOT 2**0.01 - 1 = 0.0069555501
    """
    print("\nTest C: gains are linear, not exponential")
    check("gain(E) == 1.00", dcg([gE], 1), 1.00)
    check("gain(S) == 0.10", dcg([0.10], 1), 0.10)
    check("gain(C) == 0.01", dcg([0.01], 1), 0.01)
    check("gain(I) == 0.00", dcg([0.00], 1), 0.00)
    legacy_S, legacy_C = 2 ** 0.10 - 1, 2 ** 0.01 - 1
    check("gain(S) is NOT the legacy 0.0717734625", abs(0.10 - legacy_S) > 1e-6, True,
          detail=f"legacy would be {legacy_S:.10f}")
    check("gain(C) is NOT the legacy 0.0069555501", abs(0.01 - legacy_C) > 1e-6, True,
          detail=f"legacy would be {legacy_C:.10f}")
    check("S:E gain ratio == 0.10 exactly", 0.10 / 1.00, 0.10,
          detail="legacy ratio was 0.0717734625")
    check("C:E gain ratio == 0.01 exactly", 0.01 / 1.00, 0.01,
          detail="legacy ratio was 0.0069555501")
    m = gain_from_labels(pd.Series(["E", "S", "C", "I"])).tolist()
    check("gain_from_labels maps the four labels", m, [1.00, 0.10, 0.01, 0.00])
    check("OFFICIAL_GAIN dict is the KDD table", OFFICIAL_GAIN,
          {"E": 1.00, "S": 0.10, "C": 0.01, "I": 0.00})


# ---------------------------------------------------------------- Test D
def test_D_short_candidate_list():
    """Fewer than k candidates: use them all, no padding, no penalty.

    3 candidates E,S,I ranked perfectly -> NDCG 1.0.
    Same 3 reversed (I,S,E):
      DCG  = 0.00*D(1) + 0.10*D(2) + 1.00*D(3) = 0.0631 + 0.5000 = 0.5630929754
      IDCG = 1.00*D(1) + 0.10*D(2) + 0.00*D(3) = 1.0 + 0.0630929754 = 1.0630929754
      NDCG = 0.529675...
    """
    print("\nTest D: candidate count < k")
    good = frame([("q", "a", "E", 3.0), ("q", "b", "S", 2.0), ("q", "c", "I", 1.0)])
    r = evaluate(good, "score", k=10)
    check("3 candidates, perfect order -> 1.0", r["ndcg_at_k"], 1.0)
    check("short list is still scored", r["n_queries_scored"], 1)
    check("no query excluded", r["n_excluded_no_relevant"], 0)

    bad = frame([("q", "a", "E", 1.0), ("q", "b", "S", 2.0), ("q", "c", "I", 3.0)])
    exp = (gI * D(1) + gS * D(2) + gE * D(3)) / (gE * D(1) + gS * D(2) + gC * 0)
    exp = (gI * D(1) + gS * D(2) + gE * D(3)) / (gE * D(1) + gS * D(2) + gI * D(3))
    check("3 candidates, reversed order", float(per_query_ndcg(bad, "score").iloc[0]),
          exp, detail=f"= {exp:.10f}")

    single = frame([("q", "a", "E", 1.0)])
    check("1 candidate that is relevant -> 1.0",
          float(per_query_ndcg(single, "score").iloc[0]), 1.0)


# ---------------------------------------------------------------- Test E
def test_E_tie_determinism():
    """All three candidates share one score. Result must not depend on row order.

    Ties break on ascending product_id -> pA(I), pB(E), pC(S)
      DCG  = 0.00*D(1) + 1.00*D(2) + 0.10*D(3) = 0.6309297536 + 0.05 = 0.6809297536
      IDCG = 1.00*D(1) + 0.10*D(2) + 0.00*D(3) = 1.0630929754
      NDCG = 0.640512...
    """
    print("\nTest E: tie handling is deterministic")
    rows = [("q", "pB", "E", 1.0), ("q", "pA", "I", 1.0), ("q", "pC", "S", 1.0)]
    a = float(per_query_ndcg(frame(rows), "score").iloc[0])
    b = float(per_query_ndcg(frame(rows[::-1]), "score").iloc[0])
    c = float(per_query_ndcg(frame([rows[2], rows[0], rows[1]]), "score").iloc[0])
    check("forward vs reversed row order", a, b)
    check("forward vs shuffled row order", a, c)
    exp = (gI * D(1) + gE * D(2) + gS * D(3)) / (gE * D(1) + gS * D(2) + gI * D(3))
    check("tie order is ascending product_id", a, exp, detail=f"= {exp:.10f}")

    # a larger randomised order-invariance sweep
    rng = np.random.RandomState(7)
    base = [("q", f"p{i:02d}", lab, 5.0) for i, lab in
            enumerate(["E", "S", "C", "I", "E", "S", "I", "E", "C", "S", "I", "E"])]
    ref = float(per_query_ndcg(frame(base), "score").iloc[0])
    same = all(abs(float(per_query_ndcg(
        frame([base[i] for i in rng.permutation(len(base))]), "score").iloc[0]) - ref) < 1e-15
        for _ in range(25))
    check("25 random permutations, all-tied scores, identical NDCG", same, True,
          detail=f"NDCG = {ref:.10f}")

    pes = float(per_query_ndcg(frame(rows), "score", tie_break="pessimistic").iloc[0])
    check("pessimistic tie-break <= deterministic", pes <= a + 1e-12, True,
          detail=f"pessimistic = {pes:.10f}")


# ---------------------------------------------------------------- Test F
def test_F_all_irrelevant_query():
    """A query whose candidates are all I has IDCG == 0.

    Documented behaviour: EXCLUDED from the mean, counted separately.
    NOT scored as 0.0. Here q1 scores 1.0 and q2 is all-I, so the reported
    mean must be 1.0 (not 0.5).
    """
    print("\nTest F: all-I query (IDCG == 0)")
    df = pd.concat([
        frame([("q1", "a", "E", 2.0), ("q1", "b", "I", 1.0)]),
        frame([("q2", "c", "I", 2.0), ("q2", "d", "I", 1.0)]),
    ], ignore_index=True)
    r = evaluate(df, "score")
    check("total queries seen", r["n_queries_total"], 2)
    check("queries scored", r["n_queries_scored"], 1)
    check("queries excluded (no relevant)", r["n_excluded_no_relevant"], 1)
    check("mean is 1.0, not 0.5", r["ndcg_at_k"], 1.0,
          detail="all-I query excluded rather than counted as zero")
    check("excluded query absent from per_query_ndcg index",
          "q2" in set(per_query_ndcg(df, "score").index), False)


# ------------------------------------------------------- extra guards
def test_G_equal_query_weight():
    """q1: 2 candidates, perfect -> 1.0. q2: 40 candidates with the single E
    ranked last (slot 40, outside k=10) -> 0.0. Unweighted mean must be 0.5,
    not row-weighted (which would be far below 0.5)."""
    print("\nTest G: every query carries weight 1")
    q1 = frame([("q1", "a", "E", 2.0), ("q1", "b", "I", 1.0)])
    q2 = frame([("q2", "e", "E", 0.0)] + [("q2", f"i{i:02d}", "I", 1.0) for i in range(39)])
    got = evaluate(pd.concat([q1, q2], ignore_index=True), "score")["ndcg_at_k"]
    check("unweighted mean of 1.0 and 0.0", got, 0.5)


def test_H_cutoff_at_k():
    """12 candidates: 2 I on top, then 10 E.
    DCG  = 1.0 * sum(D(3..10))     (only ranks 1..10 count)
    IDCG = 1.0 * sum(D(1..10))
    """
    print("\nTest H: cutoff at k=10 is respected")
    rows = [("q", "i1", "I", 10.0), ("q", "i2", "I", 9.0)]
    rows += [("q", f"e{i:02d}", "E", 8.0 - i) for i in range(10)]
    exp = sum(D(r) for r in range(3, 11)) / sum(D(r) for r in range(1, 11))
    check("two irrelevant at the top", float(per_query_ndcg(frame(rows), "score").iloc[0]),
          exp, detail=f"= {exp:.10f}")


# ---------------------------------------------------------------- Test I
#
# REMOVED DURING MIGRATION -- `test_I_differs_from_legacy_scorer`.
#
# The original test imported the legacy exponential scorer
# (`experiments/ranking_v2/benchmark_repair/scripts/official_ndcg.py`) via a
# sys.path insertion and asserted that the two scorers disagree on a query
# where Substitute placement matters.
#
# MIGRATION_PLAN.md §11 item 14 is a hard rule: the legacy scorer may not
# enter this repository, not one line, and not as a reference. A test that
# imports it would drag it in. The test is therefore DELETED, not skipped --
# a skipped test would still name the import and invite someone to restore it.
#
# Coverage is not lost. What test I actually guarded was the linear-gain
# convention (that S is worth 0.10 rather than 2**0.10 - 1 = 0.0718), and that
# is pinned directly and without any legacy dependency by:
#   * test_C_gain_values_are_linear -- asserts OFFICIAL_GAIN is exactly
#     E=1.00 / S=0.10 / C=0.01 / I=0.00 and used unexponentiated
#   * test_B_substitute_above_exact -- asserts the NDCG value that only holds
#     under linear gain
#
# See MIGRATION_PLAN.md §3 (row: test_official_kdd_ndcg.py) and §11 item 14.


def main():
    print("=" * 72)
    print(" Unit tests -- authoritative KDD NDCG@10 scorer (linear gain)")
    print("=" * 72)
    for t in [test_A_perfect_ranking, test_B_substitute_above_exact,
              test_C_gain_values_are_linear, test_D_short_candidate_list,
              test_E_tie_determinism, test_F_all_irrelevant_query,
              test_G_equal_query_weight, test_H_cutoff_at_k]:
        t()
    n_pass = sum(r["pass"] for r in RESULTS)
    print("\n" + "=" * 72)
    print(f" {n_pass}/{len(RESULTS)} assertions passed")
    print("=" * 72)
    out = os.path.join(REPO_ROOT, "artifacts", "manifests",
                       "official_kdd_scorer_unit_tests.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "scorer": "src/metrics/official_kdd_ndcg.py",
            "gain_convention": "E=1.00, S=0.10, C=0.01, I=0.00 used directly as the DCG numerator",
            "exponentiation_applied": False,
            "passed": n_pass, "total": len(RESULTS),
            "all_pass": n_pass == len(RESULTS),
            "assertions": RESULTS,
        }, f, indent=2, default=str)
    print(f"wrote {out}")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
