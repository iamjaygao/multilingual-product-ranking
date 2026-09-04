"""
Reusable Two-Tower training/evaluation building blocks for experiments/two_tower_v2/.
Does not modify scripts/train_two_tower.py or retrieval/two_tower.py -- this is
new, additive code used by scripts/train_two_tower_v2.py and later phase scripts.

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
import random
import json

import numpy as np
import pandas as pd
import torch

from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NOTE: we do NOT force torch.use_deterministic_algorithms(True) / cudnn
    # deterministic mode here -- per task instructions, full bit-for-bit
    # determinism can noticeably slow down training on some backends, and is
    # not required. What IS guaranteed: identical data order (DataLoader
    # shuffling), identical weight initialization RNG stream, and identical
    # numpy-based sampling (query split, dev-corpus distractor sampling).


def build_item_text(df):
    return (
        df["product_title"].fillna("") + " " +
        df["product_description"].fillna("") + " " +
        df["product_bullet_point"].fillna("")
    )


def load_positive_pairs(examples_path, products_path, query_ids, locale="us",
                          small_version=1, split="train", labels=("E", "S")):
    """Returns a DataFrame of (query_id, query, item_text) positive pairs,
    restricted to the given query_ids. Mirrors scripts/train_two_tower.py's
    filtering logic exactly (read-only reuse, not a modification of that file)."""
    df_examples = pd.read_parquet(examples_path)
    df_products = pd.read_parquet(products_path)
    df = pd.merge(df_examples, df_products, on=["product_id", "product_locale"], how="left")
    df = df[df["small_version"] == small_version]
    df = df[df["split"] == split]
    df = df[df["product_locale"] == locale]
    df = df[df["esci_label"].isin(list(labels))]
    df["query_id"] = df["query_id"].astype(str)
    df = df[df["query_id"].isin(set(query_ids))].copy()
    df["item_text"] = build_item_text(df)
    return df[["query_id", "product_id", "query", "item_text", "esci_label"]].reset_index(drop=True)


def build_dev_ir_evaluator(examples_path, products_path, dev_queries, name,
                            n_eval_queries=300, dev_corpus_size=20000, seed=42,
                            relevant_labels=("E", "S"), precision_recall_at_k=(10, 50, 100)):
    """
    Builds a fixed, reproducible InformationRetrievalEvaluator for periodic
    dev-time model selection during Two-Tower training. This is NOT the
    full-catalog Recall@100 benchmark (output/full_retrieval/*) -- it is a
    smaller, fixed proxy corpus so that a real retrieval metric (not just
    cosine-pair loss) can be computed cheaply at every checkpoint. The full,
    unmodified full-catalog benchmark is still used for the final V0-vs-V1
    comparison reported in REPORT.md.

    Construction (all fixed by seed=42, documented so every later phase reuses
    the identical corpus):
      1. Sample `n_eval_queries` queries from `dev_queries` (query-level split,
         disjoint from train).
      2. Corpus = union of ALL ESCI-labeled products (any label E/S/C/I) for
         those queries (guarantees true positives are always retrievable, and
         gives realistic same-topic distractors) + random filler products
         sampled from the catalog up to `dev_corpus_size`.
      3. relevant_docs = products labeled in `relevant_labels` for each query
         (default E/S, matching the training positive-label definition; NOT
         the full-catalog benchmark's broad E/S/C definition -- Phase 5
         studies that mismatch separately, deliberately not conflated here).
    """
    rng = np.random.RandomState(seed)
    dev_queries_sorted = sorted(dev_queries)
    n_eval_queries = min(n_eval_queries, len(dev_queries_sorted))
    eval_query_ids = sorted(rng.choice(dev_queries_sorted, size=n_eval_queries, replace=False).tolist())

    df_examples = pd.read_parquet(examples_path)
    df_products = pd.read_parquet(products_path)
    df_examples["query_id"] = df_examples["query_id"].astype(str)
    df_products["product_id"] = df_products["product_id"].astype(str)
    df_examples["product_id"] = df_examples["product_id"].astype(str)

    df_ex_eval = df_examples[df_examples["query_id"].isin(set(eval_query_ids))]
    query_text = df_ex_eval.groupby("query_id")["query"].first().to_dict()

    df_pr_us = df_products[df_products["product_locale"] == "us"].drop_duplicates("product_id").copy()
    df_pr_us["item_text"] = build_item_text(df_pr_us)
    title_lookup = df_pr_us.set_index("product_id")["item_text"].to_dict()

    # Corpus core: all labeled products for the eval queries (any label).
    core_ids = set(df_ex_eval["product_id"].unique()) & set(title_lookup)
    relevant_docs = {}
    for qid, group in df_ex_eval.groupby("query_id"):
        rel = set(group[group["esci_label"].isin(relevant_labels)]["product_id"]) & set(title_lookup)
        relevant_docs[qid] = rel
    eval_query_ids = [q for q in eval_query_ids if relevant_docs.get(q)]  # drop queries with no retrievable positive

    # Filler distractors, fixed seed, sampled from the full us-locale catalog.
    remaining_pool = np.array(sorted(set(title_lookup) - core_ids))
    n_filler = max(0, dev_corpus_size - len(core_ids))
    filler_ids = set(rng.choice(remaining_pool, size=min(n_filler, len(remaining_pool)), replace=False).tolist())
    corpus_ids = sorted(core_ids | filler_ids)
    corpus = {pid: title_lookup[pid] for pid in corpus_ids}
    queries = {qid: query_text[qid] for qid in eval_query_ids}
    relevant_docs = {qid: relevant_docs[qid] for qid in eval_query_ids}

    manifest = {
        "seed": seed, "n_eval_queries_requested": n_eval_queries, "n_eval_queries_used": len(eval_query_ids),
        "dev_corpus_size_requested": dev_corpus_size, "dev_corpus_size_actual": len(corpus),
        "n_core_labeled_products": len(core_ids), "n_filler_products": len(filler_ids),
        "relevant_labels": list(relevant_labels),
    }

    evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        name=name, batch_size=256, show_progress_bar=False,
        mrr_at_k=[10], ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        precision_recall_at_k=list(precision_recall_at_k),
        map_at_k=[100],
        score_functions={"cosine": cos_sim},
        main_score_function="cosine",
        write_predictions=False,
    )
    return evaluator, manifest
