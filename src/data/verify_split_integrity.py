"""§10.4 -- Layer 1 rebuild verification. BLOCKING for the minimum loop.

The new repo REBUILDS the core Task 1 pool from raw ESCI rather than receiving
it (§4.1), so the rebuild itself has to be verified. This compares the freshly
computed `split_integrity_rebuilt.json` against the tracked oracle
`split_integrity_reference.json`, which was produced by the source repo
(ffd16a6) and records exactly the right quantities.

**Any mismatch means the scikit-learn version changed or the upstream ESCI data
changed.** Both fail silently otherwise: a different partition still produces
plausible-looking NDCG numbers. Check the `scikit-learn==1.8.0` pin (§4.1)
first, and do NOT adjust parameters until the numbers agree -- a mismatch is a
signal, not a nuisance.

This must run BEFORE §10.2a, since 10.2a's dev-set numbers are defined on this
partition.

Usage:
    python -m src.data.verify_split_integrity
"""
from __future__ import annotations

import json
import sys

from src import paths

REFERENCE = paths.MANIFESTS / "split_integrity_reference.json"
REBUILT = paths.MANIFESTS / "split_integrity_rebuilt.json"

SPLITS = ("train", "dev", "test")
LOCALES = ("us", "es", "jp")
DEPTH_KEYS = ("mean", "median", "p5", "p95", "min", "max")
LABELS = ("E", "S", "C", "I")


def _cmp(rows, name, got, want):
    ok = got == want
    rows.append({"check": name, "expected": want, "observed": got, "PASS": ok})
    return ok


def verify(ref: dict, new: dict):
    rows: list[dict] = []

    for s in SPLITS:
        _cmp(rows, f"query_counts.{s}", new["query_counts"].get(s),
             ref["query_counts"][s])
    for s in SPLITS:
        _cmp(rows, f"row_counts.{s}", new["row_counts"].get(s),
             ref["row_counts"][s])
    for s in SPLITS:
        for loc in LOCALES:
            _cmp(rows, f"locale_distribution.{s}.{loc}",
                 new["locale_distribution"][s].get(loc),
                 ref["locale_distribution"][s][loc])
    for s in SPLITS:
        for k in DEPTH_KEYS:
            _cmp(rows, f"depth_distribution.{s}.{k}",
                 new["depth_distribution"][s].get(k),
                 ref["depth_distribution"][s][k])
    for s in SPLITS:
        for lab in LABELS:
            _cmp(rows, f"label_distribution.{s}.{lab}",
                 new["label_distribution"][s].get(lab),
                 ref["label_distribution"][s][lab])
    for pair in ("train__dev", "train__test", "dev__test"):
        _cmp(rows, f"overlap.{pair}.size", new["overlaps"][pair]["size"], 0)
    for s in SPLITS:
        _cmp(rows, f"duplicates.{s}.count", new["duplicates"][s]["count"], 0)
    _cmp(rows, "split_method", new["split_method"], ref["split_method"])

    return rows


def main(argv=None):
    for p in (REFERENCE, REBUILT):
        if not p.exists():
            raise SystemExit(
                f"missing {p}\n"
                "Run `python -m src.data.build_task1_pool --prereg "
                "docs/prereg/pool_construction.md` first.")

    ref = json.loads(REFERENCE.read_text())
    new = json.loads(REBUILT.read_text())
    rows = verify(ref, new)
    failed = [r for r in rows if not r["PASS"]]

    width = max(len(r["check"]) for r in rows)
    print(f"{'check'.ljust(width)}  {'expected':>28}  {'observed':>28}  result")
    print("-" * (width + 68))
    for r in rows:
        print(f"{r['check'].ljust(width)}  {str(r['expected'])[:28]:>28}  "
              f"{str(r['observed'])[:28]:>28}  {'PASS' if r['PASS'] else 'FAIL'}")

    summary = {
        "gate": "MIGRATION_PLAN.md §10.4 Layer 1 rebuild verification",
        "blocking": True,
        "reference": REFERENCE.name,
        "rebuilt": REBUILT.name,
        "scikit_learn_version": new.get("scikit_learn_version"),
        "checks_total": len(rows),
        "checks_passed": len(rows) - len(failed),
        "checks_failed": len(failed),
        "PASS": not failed,
        "failures": failed,
        "checks": rows,
    }
    out = paths.MANIFESTS / "split_integrity_verification.json"
    out.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{summary['checks_passed']}/{summary['checks_total']} checks passed")
    print(f"wrote {out}")
    if failed:
        print("\n=== STOP (§10.4) ===")
        for r in failed:
            print(f" - {r['check']}: expected {r['expected']!r}, got {r['observed']!r}")
        print("A mismatch means sklearn or the upstream data changed. "
              "Do NOT tune parameters to make it agree.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
