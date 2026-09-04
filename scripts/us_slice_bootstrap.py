"""
US-locale paired bootstrap: V2 Cross-Encoder vs reproduced Amazon official baseline.

No model is re-run. Both score columns are read from the frozen DEV prediction
parquets written by scripts/ca_eval_official_baseline.py. DEV only -- TEST is
never read.

Bootstrap configuration is copied verbatim from the global run in
scripts/ca_eval_official_baseline.py:110-115 (10000 resamples, RandomState(42),
pairing by query_id, percentile [2.5, 97.5] CI, two-sided sign p-value).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from src import paths
from src.metrics.binding import write_report
from src.ranking.cross_encoder import load_task1_scorer

ROOT = str(paths.REPO_ROOT)
B = str(paths.DATA_TASK1)
CA = str(paths.RESULTS / "phase4")
OFF_PRED = os.path.join(CA, "official_baseline_dev_predictions.parquet")
V2_PRED = str(paths.RESULTS / "phase3" / "ce_v2_dev_predictions.parquet")
OUT = os.path.join(CA, "us_slice_bootstrap")

# --- bootstrap config, copied from scripts/ca_eval_official_baseline.py:110-115
N_BOOTSTRAP = 10000
SEED = 42
CI_PERCENTILES = [2.5, 97.5]
TIE_EPS = 1e-12

# --- reference values from the global run (experiments/competition_alignment/
#     official_baseline_dev_results.json -> */locale/us/ndcg_at_20)
REF_US_AT20 = {"official": 0.834539, "v2": 0.855827}
SANITY_TOL = 5e-6


def per_query(sc, df):
    """Exactly the report() path in ca_eval_official_baseline.py: full-DEV table,
    then drop idcg_full == 0 queries."""
    tbl = sc.per_query_ndcg_table(df, "score", gain_col="gain", query_col="query_id")
    return tbl[tbl.idcg_full > 0]


def main():
    sc = load_task1_scorer()

    dv = pd.read_parquet(os.path.join(B, "dev_task1_core.parquet"),
                         columns=["query_id", "product_id", "product_locale", "gain"])
    assert len(dv) == 118231 and dv.query_id.nunique() == 5071, "not the frozen DEV"
    dev_qids = set(dv.query_id)

    off = pd.read_parquet(OFF_PRED)
    assert set(off.query_id) == dev_qids, "official predictions are not the frozen DEV"
    v2 = pd.read_parquet(V2_PRED)
    v2 = v2[v2.query_id.isin(dev_qids)]

    t_off, t_v2 = per_query(sc, off), per_query(sc, v2)
    common = t_off.index.intersection(t_v2.index)

    locale = dv.drop_duplicates("query_id").set_index("query_id")["product_locale"]
    us_qids = common[locale.reindex(common).values == "us"]

    # --- sanity check against the global run's per-locale point estimates
    obs = {"official": float(t_off.loc[us_qids, "ndcg_at_20"].mean()),
           "v2": float(t_v2.loc[us_qids, "ndcg_at_20"].mean())}
    sanity = {k: {"observed": round(obs[k], 6), "reference": REF_US_AT20[k],
                  "abs_diff": abs(obs[k] - REF_US_AT20[k]),
                  "pass": bool(abs(obs[k] - REF_US_AT20[k]) <= SANITY_TOL)}
              for k in obs}
    for k, v in sanity.items():
        print(f"sanity {k:<10} observed={v['observed']:.6f} ref={v['reference']:.6f} "
              f"diff={v['abs_diff']:.2e} {'PASS' if v['pass'] else 'FAIL'}")
    if not all(v["pass"] for v in sanity.values()):
        json.dump({"sanity_check": sanity, "bootstrap_run": False},
                  open(os.path.join(OUT, "us_slice_bootstrap.json"), "w"), indent=2)
        sys.exit("SANITY CHECK FAILED -- bootstrap not run")

    # --- paired bootstrap on the US slice, @20
    d = (t_v2.loc[us_qids, "ndcg_at_20"] - t_off.loc[us_qids, "ndcg_at_20"]).values
    rng = np.random.RandomState(SEED)
    bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(N_BOOTSTRAP)])
    lo, hi = np.percentile(bs, CI_PERCENTILES)
    pv = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))

    deltas = {f"ndcg_{m}": float((t_v2.loc[us_qids, c] - t_off.loc[us_qids, c]).mean())
              for m, c in [("full", "ndcg_full"), ("at_20", "ndcg_at_20"), ("at_10", "ndcg_at_10")]}
    points = {k: {"official": float(t_off.loc[us_qids, c].mean()),
                  "v2": float(t_v2.loc[us_qids, c].mean())}
              for k, c in [("ndcg_full", "ndcg_full"), ("ndcg_at_20", "ndcg_at_20"),
                           ("ndcg_at_10", "ndcg_at_10")]}

    payload = {
        "slice": "US locale only (product_locale == 'us'), competition-aligned frozen DEV",
        "test_split_touched": False,
        "models_retrained": False,
        "score_sources": {"official_baseline": os.path.relpath(OFF_PRED, ROOT),
                          "v2": os.path.relpath(V2_PRED, ROOT)},
        "scorer": {"module": "experiments/ranking_v2/kdd_task1_benchmark/scripts/kdd_task1_ndcg.py",
                   "metric_version": sc.metric_version()},
        "n_queries_us": int(len(us_qids)),
        "n_queries_dev_total": int(len(common)),
        "sanity_check": sanity,
        "point_estimates": points,
        "delta_v2_minus_official": deltas,
        "paired_bootstrap_at20": {
            "mean_delta": float(d.mean()),
            "ci_95": [float(lo), float(hi)],
            "ci_crosses_zero": bool(lo <= 0.0 <= hi),
            "p_value": pv,
            "n_queries": int(len(d)),
            "win": int((d > TIE_EPS).sum()),
            "loss": int((d < -TIE_EPS).sum()),
            "tie": int((np.abs(d) <= TIE_EPS).sum()),
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "ci_percentiles": CI_PERCENTILES,
            "pairing": "by query_id",
            "config_source": "scripts/ca_eval_official_baseline.py:110-115"},
    }
    os.makedirs(OUT, exist_ok=True)
    # §13.2 -- the only sanctioned writer; refuses an unbound payload.
    write_report(payload, os.path.join(OUT, "us_slice_bootstrap.json"))

    print(f"\nUS queries: {len(d)}")
    print(f"@20  official {points['ndcg_at_20']['official']:.4f}  v2 {points['ndcg_at_20']['v2']:.4f}  "
          f"delta {deltas['ndcg_at_20']:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  p={pv:.4f}")
    print(f"W/L/T {(d > TIE_EPS).sum()}/{(d < -TIE_EPS).sum()}/{(np.abs(d) <= TIE_EPS).sum()}")
    print(f"full delta {deltas['ndcg_full']:+.4f}   @10 delta {deltas['ndcg_at_10']:+.4f}")
    print(f"CI crosses zero: {bool(lo <= 0.0 <= hi)}")


if __name__ == "__main__":
    main()
