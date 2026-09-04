"""
THE authoritative NDCG@10 scorer for ESCI Ranking V2.

From this experiment onward, every Ranking V2 model -- LambdaMART, MLP,
Cross-Encoder, anything future -- reports its headline NDCG@10 through this
module and nothing else.

Gain convention (KDD Cup / ESCI Task 1)
---------------------------------------
The ESCI label value IS the DCG numerator. There is no exponentiation.

    gain(E) = 1.00
    gain(S) = 0.10
    gain(C) = 0.01
    gain(I) = 0.00

    DCG@k  = sum_i  gain(label_i) / log2(rank_i + 1)      rank is 1-based
    NDCG@k = DCG@k / IDCG@k                                per query
    report = unweighted mean over queries

`2**gain - 1` is NOT applied. That was the legacy internal convention
(experiments/ranking_v2/benchmark_repair/scripts/official_ndcg.py), which
turned S into 0.0718 and C into 0.0070. Under this scorer S is exactly 0.10
and C is exactly 0.01.

Documented behaviours
---------------------
* candidate count < k
      The list is simply shorter. DCG and IDCG are computed over the same
      short list, so NDCG stays well defined and is frequently exactly 1.0.
      No padding, no penalty.

* IDCG == 0  (every candidate is I, so the query has no relevant item)
      The query is EXCLUDED from the mean and counted in
      `n_excluded_no_relevant`. It is NOT scored as 0.0 -- a query with
      nothing to find cannot distinguish a good ranker from a bad one, and
      scoring it 0 would drag every model's mean down by the same constant
      while adding no signal. On the frozen pool this excludes 0 of 30,969
      test queries, so the choice is currently inert; it is stated explicitly
      so that future pools cannot change the number silently.

* ties
      Deterministic and identical to the Phase-1 scorer: sort by score
      DESCENDING, then by product_id ASCENDING, both with a stable mergesort.
      NDCG is therefore invariant to the row order of the input frame.
      `tie_break="pessimistic"` instead orders tied rows worst-relevance-first,
      giving a lower bound.

No model-specific logic lives in this file.
"""
import numpy as np
import pandas as pd

K_DEFAULT = 10

#: The official KDD/ESCI Task 1 gains. Used directly as the DCG numerator.
OFFICIAL_GAIN = {"E": 1.00, "S": 0.10, "C": 0.01, "I": 0.00}

GAIN_CONVENTION_NAME = "official KDD/ESCI Task 1 linear gain (E=1.0, S=0.1, C=0.01, I=0.0)"


def gain_from_labels(labels):
    """ESCI label series -> official DCG gain. Unknown/NaN labels map to 0.0."""
    return pd.Series(labels).map(OFFICIAL_GAIN).fillna(0.0).astype(float)


def dcg(gains, k=K_DEFAULT):
    """sum over the top-k of gain / log2(rank + 1), rank 1-based.

    `gains` must already be DCG numerators, not ESCI labels and not relevance
    values awaiting exponentiation.
    """
    g = np.asarray(gains, dtype=float)[:k]
    if g.size == 0:
        return 0.0
    return float(np.sum(g / np.log2(np.arange(2, g.size + 2))))


def _order(group, score_col, tie_break):
    if tie_break == "deterministic":
        g = group.sort_values("product_id", kind="mergesort")
        return g.sort_values(score_col, ascending=False, kind="mergesort")
    if tie_break == "pessimistic":
        g = group.sort_values("gain", ascending=True, kind="mergesort")
        return g.sort_values(score_col, ascending=False, kind="mergesort")
    raise ValueError(f"unknown tie_break: {tie_break!r}")


def attach_gain(df, label_col="esci_label", out_col="gain"):
    """Returns a copy of `df` with the official gain column attached."""
    out = df.copy()
    out[out_col] = gain_from_labels(out[label_col]).values
    return out


def per_query_ndcg(df, score_col, k=K_DEFAULT, gain_col="gain",
                   query_col="query_id", tie_break="deterministic"):
    """pd.Series of NDCG@k indexed by query_id. Queries with IDCG==0 are absent.

    `df` must carry `gain_col` (see `attach_gain`), `query_col`, `product_id`
    and `score_col`.
    """
    for c in (score_col, gain_col, query_col, "product_id"):
        if c not in df.columns:
            raise ValueError(f"missing required column: {c!r}")
    if df[score_col].isna().any():
        raise ValueError(f"{score_col} contains NaN -- refusing to score")
    if df[gain_col].isna().any():
        raise ValueError(f"{gain_col} contains NaN -- refusing to score")

    out = {}
    for qid, g in df.groupby(query_col, sort=False):
        idcg = dcg(np.sort(g[gain_col].values)[::-1], k)
        if idcg <= 0:
            continue
        ranked = _order(g, score_col, tie_break)
        out[qid] = dcg(ranked[gain_col].values, k) / idcg
    return pd.Series(out, name=f"{score_col}_ndcg@{k}", dtype=float)


def evaluate(df, score_col, k=K_DEFAULT, gain_col="gain",
             query_col="query_id", tie_break="deterministic"):
    """Aggregate report for one score column on one pool."""
    s = per_query_ndcg(df, score_col, k, gain_col, query_col, tie_break)
    n_total = int(df[query_col].nunique())
    return {
        "score_col": score_col,
        "k": k,
        "gain_convention": GAIN_CONVENTION_NAME,
        "tie_break": tie_break,
        "n_rows": int(len(df)),
        "n_queries_total": n_total,
        "n_queries_scored": int(len(s)),
        "n_excluded_no_relevant": n_total - int(len(s)),
        "ndcg_at_k": float(s.mean()) if len(s) else 0.0,
        "ndcg_at_k_median": float(s.median()) if len(s) else 0.0,
        "ndcg_at_k_std": float(s.std()) if len(s) else 0.0,
    }
