"""Build the LAYER 1 core Task 1 pool from raw ESCI.

Split out of the source repo's `build_task1_benchmark.py` (ffd16a6), which
emitted core columns and Layer 2 feature columns from one `keep` list at
`:301-304`. MIGRATION_PLAN.md §3 / §4 require the two to become independently
invocable paths. This module is the **core path only**:

    LAYER 1 (here)                       LAYER 2 (legacy/features/)
    ------------------------------       ----------------------------------
    text + labels + gain                 bm25_score, semantic_score,
    rebuilt from raw ESCI                the 17 engineered features
    deterministic, zero group deps       needs a GROUP-ORIGINATED V0
    required by the minimum loop         checkpoint; Phase 6b only

**No feature column is computed here, ever.** The minimum loop (§10.1) reads
none of them: the Cross-Encoder and the official baseline are text models and
the scorer reads only `esci_label`/`gain`. Keeping them out is what lets this
repo rebuild the pool from upstream data with no group-derived artifact.

Determinism
-----------
The train/dev carve is query-level and seeded:

    train_test_split(test_size=0.15, random_state=42, stratify=product_locale)

`train_test_split`'s shuffling may change across scikit-learn major versions,
so `scikit-learn==1.8.0` is pinned in pyproject.toml (§4.1). Verify every
rebuild with `src/data/verify_split_integrity.py` (§10.4) -- a different
partition still produces plausible-looking NDCG numbers, so it fails silently
otherwise.

Usage
-----
    python -m src.data.build_task1_pool
    python -m src.data.build_task1_pool --out data/task1
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src import paths
from src.data.task1_common import (
    LAYER1_COLUMNS, OFFICIAL_GAIN, load_split,
)

SEED = 42
DEV_FRACTION = 0.15
SPLIT_METHOD = ("query-level train_test_split(test_size=0.15, random_state=42, "
                "stratify=product_locale); a query belongs to exactly one split; "
                "no row-level splitting")

#: Columns pulled from the raw products table. Title and brand only -- colour,
#: bullets and description are joined straight from the raw parquet at
#: CE-data-build time, so the pool never carries them.
_PRODUCT_COLUMNS = ["product_id", "product_locale", "product_title", "product_brand"]


def build_core_pool(prereg=None):
    """Return {"train": df, "dev": df, "test": df} of Layer 1 columns.

    `prereg` is forwarded to the TEST read (§13.1). The pool builder legitimately
    needs the test rows -- it is materialising the frozen evaluation set, not
    evaluating on it -- so it reads them with `metadata_only=False` only when a
    pre-registration is supplied, and otherwise builds the test frame from
    label-free metadata plus a separate labelled read guarded by the caller.
    """
    paths.require_raw_esci()

    # ---- train side: no TEST involvement, no lock to satisfy ----
    print("Reading raw ESCI (train) ...", flush=True)
    tr_all = load_split("train")
    print(f"  train rows {len(tr_all)} / queries {tr_all.query_id.nunique()}", flush=True)

    # ---- test side: frozen evaluation set ----
    print("Reading raw ESCI (test) ...", flush=True)
    te = load_split("test", prereg=prereg)
    print(f"  test rows {len(te)} / queries {te.query_id.nunique()}", flush=True)

    # ---- product join (locale-aware; must not fan out) ----
    print("Joining products ...", flush=True)
    pr = pd.read_parquet(paths.PRODUCTS, columns=_PRODUCT_COLUMNS)
    assert not pr.duplicated(["product_id", "product_locale"]).any(), \
        "products table PK (product_id, product_locale) is not unique"

    frames = {}
    for name, df in (("train_all", tr_all), ("test", te)):
        n = len(df)
        merged = df.merge(pr, on=["product_id", "product_locale"], how="left")
        assert len(merged) == n, f"{name}: product join fanned out {n} -> {len(merged)}"
        frames[name] = merged

    # ---- derived columns (pure functions of what we already have) ----
    for name, df in frames.items():
        df["gain"] = df["esci_label"].map(OFFICIAL_GAIN).astype(float)
        df["doc_id"] = df["product_locale"].astype(str) + "_" + df["product_id"].astype(str)
        assert df["gain"].notna().all(), f"{name}: unmapped esci_label -> gain"

    # ---- query-level stratified train/dev carve (seeded) ----
    print("Splitting train/dev (query-level, seed 42, stratified by locale) ...", flush=True)
    tr_all = frames["train_all"]
    qdf = (tr_all[["query_id", "product_locale"]]
           .drop_duplicates("query_id")
           .sort_values("query_id"))
    train_q, dev_q = train_test_split(
        qdf["query_id"].tolist(), test_size=DEV_FRACTION, random_state=SEED,
        stratify=qdf["product_locale"].tolist())
    train_q, dev_q = set(train_q), set(dev_q)

    tr = tr_all[tr_all["query_id"].isin(train_q)].copy()
    dv = tr_all[tr_all["query_id"].isin(dev_q)].copy()
    assert len(tr) + len(dv) == len(tr_all), "train/dev partition lost rows"

    out = {}
    for name, d in (("train", tr), ("dev", dv), ("test", frames["test"])):
        missing = [c for c in LAYER1_COLUMNS if c not in d.columns]
        assert not missing, f"{name}: missing Layer 1 columns {missing}"
        out[name] = (d[LAYER1_COLUMNS]
                     .sort_values(["query_id", "example_id"])
                     .reset_index(drop=True))
    return out


def split_integrity(frames):
    """Recompute the §10.4 oracle quantities. Same definitions as the source
    repo's split-integrity gate (`build_task1_benchmark.py:314-352`), restricted
    to what Layer 1 can see."""
    qs = {k: set(v["query_id"]) for k, v in frames.items()}
    si = {
        "query_counts": {k: len(v) for k, v in qs.items()},
        "row_counts": {k: int(len(v)) for k, v in frames.items()},
        "overlaps": {}, "duplicates": {}, "locale_distribution": {},
        "depth_distribution": {}, "label_distribution": {},
    }
    stops = []
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        ov = qs[a] & qs[b]
        si["overlaps"][f"{a}__{b}"] = {"size": len(ov), "PASS": len(ov) == 0,
                                       "sample": sorted(ov)[:20]}
        if ov:
            stops.append(f"{a}/{b} query overlap = {len(ov)}")
    for k, d in frames.items():
        dup = d.duplicated(subset=["query_id", "product_id", "product_locale"], keep=False)
        n = int(dup.sum())
        si["duplicates"][k] = {"count": n, "PASS": n == 0, "sample": []}
        if n:
            stops.append(f"{k} has {n} duplicate (query_id, product_id, product_locale) rows")
        qd = d.drop_duplicates("query_id")["product_locale"].value_counts(normalize=True)
        si["locale_distribution"][k] = {loc: round(float(qd.get(loc, 0)), 4)
                                        for loc in ["us", "es", "jp"]}
        dep = d.groupby("query_id").size()
        si["depth_distribution"][k] = {
            "mean": round(float(dep.mean()), 4), "median": float(dep.median()),
            "p5": float(dep.quantile(.05)), "p95": float(dep.quantile(.95)),
            "min": int(dep.min()), "max": int(dep.max())}
        lv = d["esci_label"].value_counts(normalize=True)
        si["label_distribution"][k] = {l: round(100 * float(lv.get(l, 0)), 4)
                                       for l in ["E", "S", "C", "I"]}
    si["train_dev_locale_match"] = (
        si["locale_distribution"]["train"] == si["locale_distribution"]["dev"])
    si["split_method"] = SPLIT_METHOD
    si["STOPS"] = stops
    si["PASS"] = not stops
    return si


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(paths.DATA_TASK1),
                    help="output directory for the Layer 1 parquets")
    ap.add_argument("--prereg", default=None,
                    help="path to a git-tracked pre-registration document (§13.1); "
                         "required to materialise the labelled TEST frame")
    args = ap.parse_args(argv)

    paths.ensure_dirs()
    out_dir = paths.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = build_core_pool(prereg=args.prereg)
    for name, d in frames.items():
        p = out_dir / f"{name}_task1_core.parquet"
        d.to_parquet(p, index=False)
        print(f"  wrote {p.name}: {len(d)} rows / {d.query_id.nunique()} queries", flush=True)

    si = split_integrity(frames)
    si["layer"] = 1
    si["columns"] = LAYER1_COLUMNS
    si["scikit_learn_version"] = __import__("sklearn").__version__
    si["source"] = "rebuilt from raw ESCI; no parquet transferred"
    p = paths.MANIFESTS / "split_integrity_rebuilt.json"
    p.write_text(json.dumps(si, indent=2, default=str))
    print(f"\nwrote {p}")

    print(f"\nquery_counts: {si['query_counts']}")
    print(f"row_counts:   {si['row_counts']}")
    print(f"locale_dist:  {si['locale_distribution']}")
    if si["STOPS"]:
        print("\n=== STOP ===")
        for s in si["STOPS"]:
            print(" -", s)
        sys.exit(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
