"""
Phase 1 -- Query-level train/dev split for Two-Tower fine-tuning.
Splits by unique query_id (never by pair) so no query leaks between train
and dev. Fixed seed=42. Does not touch scripts/train_two_tower.py or any
existing model/output file.

LOCALE SCOPE — READ THIS BEFORE QUOTING ANY V2 NUMBER
-----------------------------------------------------
⚠️ **Two-Tower V2 is trained on US-locale data ONLY.** It is a MONOLINGUAL
ENGLISH model living in a repository named `multilingual-product-ranking`, so
the mismatch is called out here rather than left to be inferred.

Three independent filters enforce it, none of them incidental:

  1. `src/data/build_query_split.py` -- `LOCALE = "us"`, applied to the query
     universe before any pair is built. es/jp queries never enter train or dev.
  2. `src/retrieval/two_tower_training.py::load_positive_pairs` -- `locale="us"`
     is a DEFAULT ARGUMENT, applied at the `product_locale` filter. The caller
     in `scripts/train_two_tower_v2.py` does not override it.
  3. `src/retrieval/two_tower_training.py::build_dev_ir_evaluator` -- the dev
     corpus and its distractors are drawn from the us-locale catalog only, so
     checkpoint selection never saw a non-English query either.

Training pairs: es = 0, jp = 0. This is unchanged from V0/V1 by design -- the
V1 fix isolated the training *recipe* (warmup, seed, query-level split, real dev
evaluator), and changing the locale scope at the same time would have destroyed
the like-for-like comparison.

Consequence: every es/jp number ever reported for this model is ZERO-SHOT
TRANSFER of an English encoder, not a multilingual result. The base model,
`sentence-transformers/msmarco-distilbert-base-v3`, carries a 30,522-token
English uncased WordPiece vocabulary in which most Japanese text becomes `[UNK]`.
A weak jp score is the EXPECTED consequence of the training scope, not a
diagnostic finding about the architecture.

Full evidence: `docs/reports/retrieval_scope_audit.md` §1.
"""
import os
import json

import numpy as np
import pandas as pd

from src import paths

EXAMPLES_PATH = paths.EXAMPLES
PRODUCTS_PATH = paths.PRODUCTS
OUT_DIR = str(paths.SPLITS)

SEED = 42
DEV_RATIO = 0.10

#: ⚠️ US-LOCALE ONLY. See the module docstring -- this is the filter that makes
#: Two-Tower V2 a monolingual model, and it is deliberately unchanged from V0/V1
#: so the comparison stays like-for-like.
LOCALE = "us"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_examples = pd.read_parquet(EXAMPLES_PATH)
    df_products = pd.read_parquet(PRODUCTS_PATH)
    df = pd.merge(df_examples, df_products, on=["product_id", "product_locale"], how="left")
    df = df[df["small_version"] == 1]
    df = df[df["split"] == "train"]
    df = df[df["product_locale"] == LOCALE]
    df = df[df["esci_label"].isin(["E", "S"])]

    unique_queries = sorted(df["query_id"].astype(str).unique().tolist())
    rng = np.random.RandomState(SEED)
    shuffled = rng.permutation(unique_queries)
    n_dev = int(len(shuffled) * DEV_RATIO)
    dev_queries = sorted(shuffled[:n_dev].tolist())
    train_queries = sorted(shuffled[n_dev:].tolist())

    assert set(train_queries).isdisjoint(set(dev_queries)), "train/dev query overlap detected"

    with open(f"{OUT_DIR}/train_queries.txt", "w") as f:
        f.write("\n".join(train_queries))
    with open(f"{OUT_DIR}/dev_queries.txt", "w") as f:
        f.write("\n".join(dev_queries))

    df["query_id"] = df["query_id"].astype(str)
    n_train_pairs = int(df["query_id"].isin(set(train_queries)).sum())
    n_dev_pairs = int(df["query_id"].isin(set(dev_queries)).sum())

    summary = {
        "seed": SEED, "dev_ratio": DEV_RATIO, "locale": LOCALE,
        "n_unique_queries_total": len(unique_queries),
        "n_train_queries": len(train_queries), "n_dev_queries": len(dev_queries),
        "n_train_pairs": n_train_pairs, "n_dev_pairs": n_dev_pairs,
        "disjoint_verified": True,
        "source_filter": {"small_version": 1, "split": "train", "product_locale": LOCALE, "esci_label": ["E", "S"]},
    }
    with open(f"{OUT_DIR}/split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
