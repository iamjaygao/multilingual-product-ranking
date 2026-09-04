"""Path resolution for this repository.

Replaces the predecessor repo's `config.py`, which is not migrated
(MIGRATION_PLAN.md §11 item 15). Two rules shape this module:

1. **No dependency on the old repo's layout.** The raw ESCI parquets are
   located through the ``ESCI_DATA_ROOT`` environment variable, defaulting to a
   sibling checkout of ``amazon-science/esci-data``. Nothing here reaches into
   ``esci-search-ranking-system``.
2. **Rebuilt, not transferred.** ``DATA_TASK1`` holds the Layer 1 core pool,
   which this repo *builds* from raw ESCI (§4.1). It is gitignored and starts
   empty on a fresh clone.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Repository root (the directory containing pyproject.toml).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Upstream ESCI checkout. Override with ESCI_DATA_ROOT.
#: Default: a sibling clone of https://github.com/amazon-science/esci-data
ESCI_DATA_ROOT = Path(
    os.environ.get("ESCI_DATA_ROOT", REPO_ROOT.parent / "dataset" / "esci-data")
).expanduser()

_SQD = ESCI_DATA_ROOT / "shopping_queries_dataset"

#: Raw upstream parquets. Read-only; never vendored, never written.
EXAMPLES = _SQD / "shopping_queries_dataset_examples.parquet"
PRODUCTS = _SQD / "shopping_queries_dataset_products.parquet"
SOURCES = _SQD / "shopping_queries_dataset_sources.csv"

# --- this repo's own directories -------------------------------------------
DATA = REPO_ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_TASK1 = DATA / "task1"                  # Layer 1 core pool (rebuilt)
DATA_PREDICTIONS = DATA / "predictions"
DATA_LEGACY_FEATURES = DATA / "legacy_features"   # Layer 2 (out-of-band, may be absent)

ARTIFACTS = REPO_ROOT / "artifacts"
MANIFESTS = ARTIFACTS / "manifests"
RESULTS = ARTIFACTS / "results"
SPLITS = ARTIFACTS / "splits"

DOCS = REPO_ROOT / "docs"
PREREG = DOCS / "prereg"


def require_raw_esci() -> None:
    """Fail early and legibly if the upstream parquets are not where we expect."""
    missing = [str(p) for p in (EXAMPLES, PRODUCTS) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Raw ESCI data not found:\n  "
            + "\n  ".join(missing)
            + f"\n\nESCI_DATA_ROOT is currently {ESCI_DATA_ROOT!s}.\n"
            "Clone https://github.com/amazon-science/esci-data and either place it "
            "at that path or set ESCI_DATA_ROOT to point at it."
        )


def ensure_dirs() -> None:
    for d in (DATA_TASK1, DATA_PREDICTIONS, MANIFESTS, RESULTS, SPLITS):
        d.mkdir(parents=True, exist_ok=True)
