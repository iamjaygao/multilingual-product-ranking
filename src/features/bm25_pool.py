"""Per-query BM25 scoring over a fixed candidate pool.

ATTRIBUTION
-----------
The original course project contained a group implementation of this component.
The version in this repository is a minimal re-implementation written
independently for this ranking benchmark. No claim is made over the original
group implementation.

(MIGRATION_PLAN.md §7.3-1 decision **C**; wording required by §7.3 "Required
documentation wording for the two C decisions".)

Provenance: extracted from `build_task1_benchmark.py:68-99` in the source repo
(ffd16a6), whose own header records that it was *transcribed* from
`retrieval/bm25.py::_score_single_group` rather than imported. That module --
the group-originated one -- is **not** migrated (§11 item 8): its global-index
and demo-search surface belongs to the old end-to-end search system, which is
out of scope for a ranking benchmark.

WHAT THIS IS FOR
----------------
This produces the `bm25_score` feature column, which is **Layer 2** (§4.2).
Nothing in the minimum viable loop (§10.1) imports this module: the
Cross-Encoder and the official baseline are text models and the scorer reads
only `esci_label`/`gain`. It is used by `legacy/features/build_feature_matrix.py`
under the optional Phase 6b.

The index is rebuilt per query over THAT QUERY'S OWN candidate set. IDF and
avgdl are therefore candidate-set statistics, which makes even the raw score
pool-dependent -- the property that forced BM25 to be recomputed rather than
joined when the Task 1 pool was built.

.. todo:: **Multilingual tokenisation (§7.3-1).**

   `bm25s.tokenize()` is called with library defaults, which are English-only:

     * ``token_pattern=r"(?u)\\b\\w\\w+\\b"`` -- a regex word-run matcher, not a
       segmenter. A whole unspaced Japanese run becomes exactly ONE token, so
       ``ワイヤレスイヤホン`` only ever matches a title containing that identical
       unbroken run. This is the mechanism behind the 61.9% ``bm25_score``
       zero-rate observed on jp rows.
     * ``stopwords="english"`` -- the English stoplist is applied to Spanish and
       Japanese too, so ``de``/``para`` survive and dilute IDF.
     * ``stemmer=None`` -- no stemming for any locale.

   A repo named *multilingual*-product-ranking should re-specify this. It is
   deliberately NOT changed here: `bm25_score` is an input to the frozen Layer 2
   feature matrix, and re-tokenising would change every historical LambdaMART
   number while leaving the column name identical (the §10.3 failure mode). Any
   change must ship as a NEW column alongside the old one, never as a rewrite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score_group(query_id, group):
    """BM25 for one query's candidate set: raw score plus per-query min-max.

    Parameters
    ----------
    query_id : hashable
    group : DataFrame with columns `query_text`, `item_text`, `item_id`

    Returns
    -------
    list[dict] with keys query_id, product_id, bm25_raw, bm25_minmax
    """
    import bm25s  # optional dependency; see the `legacy` extra in pyproject.toml

    query_text = group["query_text"].iloc[0]
    item_texts = group["item_text"].tolist()
    item_ids = group["item_id"].tolist()

    # NOTE: library defaults are English-only -- see the module TODO above.
    corpus_tokens = bm25s.tokenize(item_texts, show_progress=False)
    query_tokens = bm25s.tokenize([query_text], show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    doc_indices, raw_scores = retriever.retrieve(
        query_tokens, k=len(item_texts), show_progress=False)

    top_idx, top_raw = doc_indices[0], raw_scores[0]
    mn, mx = top_raw.min(), top_raw.max()
    norm = (top_raw - mn) / (mx - mn) if (mx - mn) > 1e-8 else np.zeros_like(top_raw)
    return [{"query_id": query_id, "product_id": item_ids[idx],
             "bm25_raw": float(top_raw[i]), "bm25_minmax": float(norm[i])}
            for i, idx in enumerate(top_idx)]


def compute_bm25(df, n_jobs=-1, batch_size=200, verbose=True):
    """Score every query in `df` (columns: query_id, query_text, item_id, item_text).

    Returns a DataFrame of query_id / product_id / bm25_raw / bm25_minmax.
    """
    from joblib import Parallel, delayed  # optional; `legacy` extra

    work = df[["query_id", "query_text", "item_id", "item_text"]]
    grouped = list(work.groupby("query_id"))
    if verbose:
        print(f"  BM25 over {len(grouped)} queries ...", flush=True)
    nested = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
        delayed(score_group)(qid, g) for qid, g in grouped)
    return pd.DataFrame([r for sub in nested for r in sub])
