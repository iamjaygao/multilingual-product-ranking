"""
THE scorer for the ESCI KDD Task 1 benchmark.

Separate from experiments/ranking_v2/official_metric_final/scripts/official_kdd_ndcg.py,
which scores the LARGE-version pool. The two coexist; numbers from them are
never comparable (see the bookkeeping rule, section 16).

Gain (competition definition, this project's primary convention)
----------------------------------------------------------------
    E = 1.00   S = 0.10   C = 0.01   I = 0.00
used DIRECTLY as the DCG numerator. `2**gain - 1` is never applied.

    DCG@k  = sum_{i<=k} gain_i / log2(rank_i + 1)     rank 1-based
    IDCG@k = same, over the candidate set sorted by gain descending
    NDCG@k = DCG@k / IDCG@k                            per query
    report = unweighted macro average over queries

Three cutoffs are always returned: full (HEADLINE), @10, @20. Task 1 averages
23.2 candidates per query and caps at 40, so full != @20 != @10.

Swapped-gain sensitivity
------------------------
`GAIN_SWAPPED` (E=1.0, S=0.01, C=0.1, I=0.0) reproduces what
amazon-science/esci-data's ranking/prepare_trec_eval_files.py actually produces
when composed with its Terrier command -- a known, third-party-documented
(SQID) S/C inversion, independently re-confirmed on this repo's local HEAD.
Provided so the ambiguity can be closed with a number rather than prose.

IDCG == 0
---------
Gate B was NOT_ATTEMPTED on this machine (no Terrier, no Java), so per section
8.6 the fallback applies: `exclude` -- the query is dropped from the mean and
counted separately. All three conventions (`exclude`, `zero`, `one`) are
computed in a single pass so the policy can be swapped by lookup once a
reference evaluation becomes available.

Tie-breaking
------------
Headline: score DESC -> product_id ASC -> product_locale ASC (deterministic,
row-order invariant). Also supports `tie_break="random"` with a seed, because
the deterministic rule is not unbiased: BM25 assigns 0.0 to many candidates and
their relative order would then be decided by product_id, which is unrelated to
relevance.

Everything here is vectorised (numpy lexsort + segmented cumulative sums) so
that 100-seed random-tie-break sweeps are cheap.
"""
import hashlib
import os

import numpy as np
import pandas as pd

GAIN_COMPETITION = {"E": 1.00, "S": 0.10, "C": 0.01, "I": 0.00}
GAIN_SWAPPED = {"E": 1.00, "S": 0.01, "C": 0.10, "I": 0.00}

CUTOFFS = ("full", 10, 20)
ZERO_IDCG_POLICIES = ("exclude", "zero", "one")
DEFAULT_ZERO_IDCG_POLICY = "exclude"
DEFAULT_ZERO_IDCG_POLICY_SOURCE = "fallback_pending_gate_b"


def gain_from_labels(labels, mapping=None):
    m = mapping or GAIN_COMPETITION
    return pd.Series(labels).map(m).fillna(0.0).astype(float).to_numpy()


def scorer_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def metric_version(gain_name="competition_E1_S0.1_C0.01_I0",
                   zero_idcg_policy=DEFAULT_ZERO_IDCG_POLICY,
                   tie_break="deterministic"):
    """Every reported NDCG must carry this string (bookkeeping rule, section 16)."""
    return (f"kdd_task1_ndcg@{scorer_sha256()[:16]}"
            f"|gain={gain_name}"
            f"|idcg0={zero_idcg_policy}"
            f"|tie={tie_break}")


def _segment_starts(codes):
    """Index of the first row of each contiguous group, given sorted group codes."""
    n = len(codes)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = codes[1:] != codes[:-1]
    return np.flatnonzero(new)


def _segment_sum(values, starts, n):
    """Sum of `values` within each contiguous segment defined by `starts`."""
    cs = np.concatenate(([0.0], np.cumsum(values)))
    ends = np.append(starts[1:], n)
    return cs[ends] - cs[starts]


def per_query_ndcg_table(df, score_col, gain_col="gain", query_col="query_id",
                         tie_break="deterministic", seed=None):
    """Vectorised per-query NDCG at full / @10 / @20.

    Returns a DataFrame indexed by query_id with columns
    ndcg_full, ndcg_at_10, ndcg_at_20, idcg_full, n_candidates.
    Rows whose idcg_full == 0 are kept here; the policy is applied in aggregate().
    """
    for c in (score_col, gain_col, query_col, "product_id"):
        if c not in df.columns:
            raise ValueError(f"missing required column: {c!r}")
    if df[score_col].isna().any():
        raise ValueError(f"{score_col} contains NaN -- refusing to score")
    if df[gain_col].isna().any():
        raise ValueError(f"{gain_col} contains NaN -- refusing to score")

    qcodes, quniq = pd.factorize(df[query_col], sort=True)
    score = df[score_col].to_numpy(dtype=float)
    gain = df[gain_col].to_numpy(dtype=float)
    n = len(df)

    if tie_break == "deterministic":
        pid = pd.factorize(df["product_id"], sort=True)[0]
        loc = (pd.factorize(df["product_locale"], sort=True)[0]
               if "product_locale" in df.columns else np.zeros(n, dtype=np.int64))
        # np.lexsort: last key is primary
        order = np.lexsort((loc, pid, -score, qcodes))
    elif tie_break == "random":
        if seed is None:
            raise ValueError("tie_break='random' requires a seed")
        r = np.random.RandomState(seed).rand(n)
        order = np.lexsort((r, -score, qcodes))
    else:
        raise ValueError(f"unknown tie_break: {tie_break!r}")

    # ---- actual ranking ----
    qc_s = qcodes[order]
    g_s = gain[order]
    starts = _segment_starts(qc_s)
    pos = np.arange(n) - np.repeat(starts, np.diff(np.append(starts, n)))  # 0-based rank
    disc = 1.0 / np.log2(pos + 2.0)
    contrib = g_s * disc

    dcg_full = _segment_sum(contrib, starts, n)
    dcg_10 = _segment_sum(np.where(pos < 10, contrib, 0.0), starts, n)
    dcg_20 = _segment_sum(np.where(pos < 20, contrib, 0.0), starts, n)

    # ---- ideal ranking (gain desc within query) ----
    iorder = np.lexsort((-gain, qcodes))
    qc_i = qcodes[iorder]
    g_i = gain[iorder]
    istarts = _segment_starts(qc_i)
    ipos = np.arange(n) - np.repeat(istarts, np.diff(np.append(istarts, n)))
    idisc = 1.0 / np.log2(ipos + 2.0)
    icontrib = g_i * idisc

    idcg_full = _segment_sum(icontrib, istarts, n)
    idcg_10 = _segment_sum(np.where(ipos < 10, icontrib, 0.0), istarts, n)
    idcg_20 = _segment_sum(np.where(ipos < 20, icontrib, 0.0), istarts, n)

    counts = np.diff(np.append(starts, n))

    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "ndcg_full": np.where(idcg_full > 0, dcg_full / idcg_full, np.nan),
            "ndcg_at_10": np.where(idcg_10 > 0, dcg_10 / idcg_10, np.nan),
            "ndcg_at_20": np.where(idcg_20 > 0, dcg_20 / idcg_20, np.nan),
            "idcg_full": idcg_full,
            "n_candidates": counts,
        }, index=pd.Index(quniq, name=query_col))
    return out


def aggregate(tbl, zero_idcg_policy=DEFAULT_ZERO_IDCG_POLICY):
    """Macro-average over queries under one IDCG==0 convention."""
    if zero_idcg_policy not in ZERO_IDCG_POLICIES:
        raise ValueError(f"unknown zero_idcg_policy: {zero_idcg_policy!r}")
    zero_mask = tbl["idcg_full"] <= 0
    res = {"zero_idcg_query_count": int(zero_mask.sum()),
           "zero_idcg_policy": zero_idcg_policy,
           "n_queries_total": int(len(tbl))}
    for col, name in [("ndcg_full", "ndcg_full"),
                      ("ndcg_at_10", "ndcg_at_10"),
                      ("ndcg_at_20", "ndcg_at_20")]:
        v = tbl[col].to_numpy(dtype=float).copy()
        if zero_idcg_policy == "exclude":
            v = v[~zero_mask.to_numpy()]
        elif zero_idcg_policy == "zero":
            v = np.nan_to_num(v, nan=0.0)
        else:  # "one"
            v = np.where(np.isnan(v), 1.0, v)
        res[name] = float(np.mean(v)) if len(v) else 0.0
    res["n_queries_scored"] = int((~zero_mask).sum()) if zero_idcg_policy == "exclude" else int(len(tbl))
    return res


def evaluate(df, score_col, gain_col="gain", query_col="query_id",
             tie_break="deterministic", seed=None,
             zero_idcg_policy=DEFAULT_ZERO_IDCG_POLICY,
             gain_name="competition_E1_S0.1_C0.01_I0",
             candidate_pool="task1_small_v1"):
    """One-shot report. Always carries candidate_pool + metric_version."""
    tbl = per_query_ndcg_table(df, score_col, gain_col, query_col, tie_break, seed)
    res = aggregate(tbl, zero_idcg_policy)
    res.update({
        "score_col": score_col,
        "tie_break": tie_break,
        "tie_break_seed": seed,
        "n_rows": int(len(df)),
        "candidate_pool": candidate_pool,
        "metric_version": metric_version(gain_name, zero_idcg_policy, tie_break),
        "all_zero_idcg_policies": {p: aggregate(tbl, p) for p in ZERO_IDCG_POLICIES},
    })
    return res
