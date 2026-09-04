"""
Section 3 -- locate the ORIGINAL ESCI parquet files by search (no hard-coded
assumptions) and record their schema. Also look for a local checkout of
amazon-science/esci-data's ranking/ helper directory.

STOP if the original examples/products cannot be found: this phase must
rebuild from the raw files, and the version flags cannot be recovered from an
already-processed clean parquet.
"""
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import paths  # noqa: E402

ROOT = str(paths.REPO_ROOT)
OUT = str(paths.MANIFESTS)


REQUIRED_EXAMPLES = {"example_id", "query_id", "query", "product_id",
                     "product_locale", "esci_label", "small_version",
                     "large_version", "split"}
REQUIRED_PRODUCTS = {"product_id", "product_locale", "product_title",
                     "product_description", "product_bullet_point",
                     "product_brand", "product_color"}

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}


def find_parquets(root):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".parquet"):
                hits.append(os.path.join(dirpath, fn))
    return sorted(hits)


def schema_of(path):
    try:
        return {f.name: str(f.type) for f in pq.ParquetFile(path).schema_arrow}
    except Exception as e:  # noqa: BLE001
        return {"__error__": str(e)}


def find_esci_repo(root):
    """Look for amazon-science/esci-data's ranking/ helpers anywhere on the box
    that is cheap to reach: inside the project, then a few common parents."""
    found = {"esci_repo_ranking_dir_found": None,
             "prepare_trec_eval_files_path": None,
             "launch_predictions_task1_path": None,
             "searched_roots": []}
    candidates = [str(paths.ESCI_DATA_ROOT), root, os.path.dirname(root),
                  os.path.expanduser("~/WORKSPACE"),
                  os.path.expanduser("~/esci-data"), os.path.expanduser("~/data")]
    seen = set()
    for base in candidates:
        base = os.path.abspath(base)
        if base in seen or not os.path.isdir(base):
            continue
        seen.add(base)
        found["searched_roots"].append(base)
        depth_base = base.rstrip("/").count("/")
        for dirpath, dirnames, filenames in os.walk(base):
            if dirpath.rstrip("/").count("/") - depth_base > 4:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if "prepare_trec_eval_files.py" in filenames:
                found["prepare_trec_eval_files_path"] = os.path.join(
                    dirpath, "prepare_trec_eval_files.py")
                found["esci_repo_ranking_dir_found"] = dirpath
            for fn in filenames:
                if fn.startswith("launch-predictions-task1") or fn.startswith("launch_predictions_task1"):
                    found["launch_predictions_task1_path"] = os.path.join(dirpath, fn)
    return found


def main():
    os.makedirs(OUT, exist_ok=True)
    audit = {k: None for k in [
        "examples_path", "products_path", "examples_rows", "products_rows",
        "unique_query_ids", "esci_repo_ranking_dir_found",
        "prepare_trec_eval_files_path", "launch_predictions_task1_path"]}
    audit.update({"examples_schema": {}, "products_schema": {},
                  "small_version_counts": {}, "large_version_counts": {},
                  "split_counts": {}, "esci_label_values": [], "locale_values": []})

    print("Scanning for parquet files ...")
    all_pq = find_parquets(str(paths.ESCI_DATA_ROOT))
    audit["parquet_files_scanned"] = len(all_pq)

    ex_cands, pr_cands = [], []
    for p in all_pq:
        s = schema_of(p)
        cols = set(s)
        if REQUIRED_EXAMPLES.issubset(cols):
            ex_cands.append((p, s))
        if REQUIRED_PRODUCTS.issubset(cols):
            pr_cands.append((p, s))

    audit["examples_candidates"] = [p for p, _ in ex_cands]
    audit["products_candidates"] = [p for p, _ in pr_cands]
    print(f"  examples candidates: {audit['examples_candidates']}")
    print(f"  products candidates: {audit['products_candidates']}")

    if not ex_cands or not pr_cands:
        audit["STOP"] = ("Could not locate the ORIGINAL ESCI examples/products parquet with "
                         "the required version-flag columns. This phase must rebuild from raw "
                         "files; version flags cannot be recovered from a processed clean "
                         "parquet.")
        with open(os.path.join(OUT, "data_source_audit.json"), "w") as f:
            json.dump(audit, f, indent=2, default=str)
        print("\nSTOP:", audit["STOP"])
        sys.exit(2)

    # largest row count wins if several match (the canonical full dump)
    def rows(p):
        return pq.ParquetFile(p).metadata.num_rows
    ex_path, ex_schema = max(ex_cands, key=lambda t: rows(t[0]))
    pr_path, pr_schema = max(pr_cands, key=lambda t: rows(t[0]))

    audit["examples_path"] = os.path.relpath(ex_path, ROOT)
    audit["products_path"] = os.path.relpath(pr_path, ROOT)
    audit["examples_path_abs"] = os.path.abspath(ex_path)
    audit["products_path_abs"] = os.path.abspath(pr_path)
    audit["examples_schema"] = ex_schema
    audit["products_schema"] = pr_schema

    print(f"\nSelected examples: {audit['examples_path']}")
    print(f"Selected products: {audit['products_path']}")

    ex = pd.read_parquet(ex_path, columns=sorted(REQUIRED_EXAMPLES))
    audit["examples_rows"] = int(len(ex))
    audit["products_rows"] = int(rows(pr_path))
    audit["unique_query_ids"] = int(ex["query_id"].nunique())
    audit["small_version_counts"] = {str(k): int(v) for k, v in
                                     ex["small_version"].value_counts().sort_index().items()}
    audit["large_version_counts"] = {str(k): int(v) for k, v in
                                     ex["large_version"].value_counts().sort_index().items()}
    audit["split_counts"] = {str(k): int(v) for k, v in ex["split"].value_counts().items()}
    audit["esci_label_values"] = sorted(ex["esci_label"].dropna().unique().tolist())
    audit["locale_values"] = sorted(ex["product_locale"].dropna().unique().tolist())

    audit.update(find_esci_repo(str(paths.ESCI_DATA_ROOT)))

    with open(os.path.join(OUT, "data_source_audit.json"), "w") as f:
        json.dump(audit, f, indent=2, default=str)

    for k in ["examples_rows", "products_rows", "unique_query_ids",
              "small_version_counts", "large_version_counts", "split_counts",
              "esci_label_values", "locale_values", "esci_repo_ranking_dir_found",
              "prepare_trec_eval_files_path", "launch_predictions_task1_path"]:
        print(f"  {k}: {audit[k]}")
    print("\nwrote data_source_audit.json")


if __name__ == "__main__":
    main()
