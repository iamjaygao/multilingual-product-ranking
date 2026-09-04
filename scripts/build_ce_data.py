"""Attach the product text fields the Cross-Encoder needs to the Layer 1 core pool.

Migrated from `scripts/build_cross_encoder_task1_data.py` (source repo ffd16a6).

Why this script is necessary
----------------------------
The Layer 1 core pool carries `product_title` and `product_brand` but NOT
`product_color`, `product_bullet_point` or `product_description`. Those three
live in the raw products table and are LEFT-joined here on
(product_id, product_locale) -- locale-aware, asserted not to fan out.

This is a **rebuild, not a transfer**. The source repo's
`_cache/dev_text.parquet` (77 MB) and `train_text.parquet` (425 MB) are not
copied. Its `build_manifest.json` records that the cache draws only on:

  * `product_title`, `product_brand`  -- both Layer 1 columns (§4.1)
  * `product_color`, `product_bullet_point`, `product_description`
                                       -- all raw upstream products fields

No Layer 2 feature column and no group-originated artifact is involved, so the
cache is fully reproducible here (§4.2 does not apply).

The core pool parquets are NEVER modified, and no row is added or removed.

Usage:
    python -m scripts.build_ce_data --splits train dev
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from src import paths
from src.data.task1_common import LAYER1_COLUMNS
from src.ranking.cross_encoder import TEXT_FIELDS

OUT_DIR = paths.DATA_TASK1 / "ce_text"

#: Expected shape of each split. A mismatch is a STOP condition -- this script
#: must never silently rebuild or repair data.
EXPECTED = {"train": (663407, 28733), "dev": (118231, 5071), "test": (336373, 14496)}

KEEP = list(LAYER1_COLUMNS)
FROM_PRODUCTS = ["product_color", "product_bullet_point", "product_description"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev"],
                    choices=["train", "dev", "test"],
                    help="test is excluded by default: the TEST LOCK is in force (§13.1)")
    ap.add_argument("--prereg", default=None,
                    help="path to a git-tracked pre-registration document. REQUIRED "
                         "to build the test split (§13.1); ignored otherwise.")
    ap.add_argument("--max_chars_per_field", type=int, default=4000,
                    help="storage-only cap on each raw text field. 4000 chars is "
                         ">10x what 256 tokens can hold, so it cannot affect any "
                         "model with max_length<=256. Set 0 to disable.")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    # §13.1 -- the source repo made `test` opt-in via a CLI default. That is a
    # convention. Here it is an assert: building the test text cache requires a
    # committed pre-registration, validated by the same code path that guards
    # every other TEST read.
    if "test" in args.splits:
        from src.data.task1_common import _validate_prereg, TestSplitLocked
        if args.prereg is None:
            raise TestSplitLocked(
                "Building the TEST text cache requires a committed pre-registration "
                "document; pass --prereg docs/prereg/<file>.md. "
                "See MIGRATION_PLAN.md §13.1.")
        _validate_prereg(args.prereg)

    cap = args.max_chars_per_field or None
    if cap is not None and cap < 2560:
        raise SystemExit(f"--max_chars_per_field={cap} is too small to be storage-only; "
                         "it could truncate content a 256-token model would have seen")

    out_dir = paths.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths.require_raw_esci()

    print("Loading raw products table (locale-aware key) ...", flush=True)
    pr = pd.read_parquet(paths.PRODUCTS,
                         columns=["product_id", "product_locale"] + FROM_PRODUCTS)
    if pr.duplicated(["product_id", "product_locale"]).any():
        raise SystemExit("STOP: products table has duplicate (product_id, product_locale) keys")
    for c in FROM_PRODUCTS:
        if cap is not None:
            pr[c] = pr[c].astype("string").str.slice(0, cap)

    manifest = {"source_products": str(paths.PRODUCTS),
                "join_keys": ["product_id", "product_locale"],
                "fields_from_frozen_split": ["product_title", "product_brand"],
                "fields_joined_from_products": FROM_PRODUCTS,
                "text_field_order": [f"{lbl} <- {col}" for lbl, col in TEXT_FIELDS],
                "max_chars_per_field": cap,
                "frozen_splits_modified": False,
                "core_pool_rebuilt_not_transferred": True,
                "splits": {}}

    for split in args.splits:
        src = paths.DATA_TASK1 / f"{split}_task1_core.parquet"
        if not src.exists():
            raise SystemExit(
                f"STOP: Layer 1 core pool not found: {src}\n"
                "Run `python -m src.data.build_task1_pool --prereg "
                "docs/prereg/pool_construction.md` first (Phase 2).")
        df = pd.read_parquet(src, columns=KEEP)

        exp_rows, exp_q = EXPECTED[split]
        got_rows, got_q = len(df), df["query_id"].nunique()
        if (got_rows, got_q) != (exp_rows, exp_q):
            raise SystemExit(
                f"STOP: {split} artifact disagrees with the Task 1 REPORT.\n"
                f"  expected {exp_rows} rows / {exp_q} queries\n"
                f"  actual   {got_rows} rows / {got_q} queries\n"
                "Not rebuilding or repairing. Human decision required.")

        n0 = len(df)
        df = df.merge(pr, on=["product_id", "product_locale"], how="left")
        if len(df) != n0:
            raise SystemExit(f"STOP: products join fanned out on {split}: {len(df)} != {n0}")

        nulls = {c: int(df[c].isna().sum()) for c in FROM_PRODUCTS}
        title_null = int(df["product_title"].isna().sum())
        out = out_dir / f"{split}_text.parquet"
        df.to_parquet(out, index=False)

        manifest["splits"][split] = {
            "source": str(src.relative_to(paths.REPO_ROOT)),
            "output": str(out.relative_to(paths.REPO_ROOT)),
            "rows": int(len(df)), "queries": int(df["query_id"].nunique()),
            "rows_match_report": True,
            "null_counts_joined_fields": nulls,
            "null_count_product_title": title_null,
            "size_mb": round(os.path.getsize(out) / 1e6, 1),
        }
        print(f"  {split}: {len(df)} rows / {df['query_id'].nunique()} queries "
              f"-> {out} ({manifest['splits'][split]['size_mb']} MB); "
              f"nulls {nulls}", flush=True)

    mpath = out_dir / "build_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    print("wrote", mpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
