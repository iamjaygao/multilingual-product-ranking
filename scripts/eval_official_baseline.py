"""
Competition alignment §12 -- score the reproduced official baseline on our
competition-aligned frozen DEV, and compare against V2 on the identical pool.

One model per locale, exactly as the official recipe and official inference.py do:
  us      CrossEncoder.predict on (query, product_title)
  es/jp   SentenceTransformer bi-encoder, cosine(query_emb, title_emb)

Input is query + product_title ONLY -- no brand/color/bullet/description. This is a
baseline reproduction, not an optimisation.

DEV only. TEST is never read.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from src import paths
from src.data.task1_common import load_split
from src.ranking.cross_encoder import load_task1_scorer

P = str(paths.ESCI_DATA_ROOT / "shopping_queries_dataset")
B = str(paths.DATA_TASK1)
OUT = str(paths.RESULTS / "phase4")
MODELS = str(paths.ARTIFACTS / "checkpoints" / "official_baseline")
V2_PRED = str(paths.RESULTS / "phase3" / "ce_v2_dev_predictions.parquet")


def main():
    dev_name = "mps" if torch.backends.mps.is_available() else "cpu"
    sc = load_task1_scorer()
    t0 = time.time()

    dv = pd.read_parquet(os.path.join(B, "dev_task1_core.parquet"),
                         columns=["example_id", "query_id", "query", "product_id",
                                  "product_locale", "esci_label", "gain"])
    assert len(dv) == 118231 and dv.query_id.nunique() == 5071, "not the frozen DEV"
    pr = pd.read_parquet(paths.PRODUCTS,
                         columns=["product_id", "product_locale", "product_title"])
    n0 = len(dv)
    dv = dv.merge(pr, on=["product_id", "product_locale"], how="left", validate="many_to_one")
    assert len(dv) == n0, "title join fanned out"
    dv["product_title"] = dv["product_title"].fillna("").astype(str)
    print(f"DEV {len(dv)} rows / {dv.query_id.nunique()} queries", flush=True)

    scores = np.full(len(dv), np.nan)
    runtime = {}
    for lc in ["us", "es", "jp"]:
        mp = os.path.join(MODELS, f"task_1_ranking_model_{lc}")
        if not os.path.exists(mp):
            print(f"  [{lc}] MISSING {mp} -- skipping"); continue
        m = (dv.product_locale == lc).values
        idx = np.flatnonzero(m)
        q = dv.loc[m, "query"].astype(str).tolist()
        t = dv.loc[m, "product_title"].tolist()
        s0 = time.time()
        if lc == "us":
            from sentence_transformers.cross_encoder import CrossEncoder
            mdl = CrossEncoder(mp, max_length=512, device=dev_name)
            out = mdl.predict(list(zip(q, t)), batch_size=128, show_progress_bar=False)
            scores[idx] = np.asarray(out, dtype=np.float64).reshape(-1)
        else:
            from sentence_transformers import SentenceTransformer
            mdl = SentenceTransformer(mp, device=dev_name)
            qe = mdl.encode(q, batch_size=256, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
            te = mdl.encode(t, batch_size=256, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
            scores[idx] = np.einsum("ij,ij->i", qe, te).astype(np.float64)
        runtime[lc] = {"rows": int(m.sum()), "seconds": round(time.time() - s0, 1),
                       "rows_per_sec": round(m.sum() / (time.time() - s0), 1),
                       "arch": "CrossEncoder" if lc == "us" else "bi-encoder cosine"}
        print(f"  [{lc}] scored {m.sum()} rows in {runtime[lc]['seconds']}s", flush=True)

    assert not np.isnan(scores).any(), "some locale was not scored"
    dv["score"] = scores

    def report(df, col, label):
        r = sc.evaluate(df, col, gain_col="gain", query_col="query_id")
        tbl = sc.per_query_ndcg_table(df, col, gain_col="gain", query_col="query_id")
        loc = df.drop_duplicates("query_id").set_index("query_id")["product_locale"]
        keep = tbl[tbl.idcg_full > 0]
        lv = loc.reindex(keep.index)
        out = {"model": label,
               "ndcg_full": round(r["ndcg_full"], 6),
               "ndcg_at_20": round(r["ndcg_at_20"], 6),
               "ndcg_at_10": round(r["ndcg_at_10"], 6),
               "queries": r["n_queries_scored"], "rows": int(len(df)),
               "candidate_pool": r["candidate_pool"], "metric_version": r["metric_version"],
               "locale": {l: {"n": int((lv == l).sum()),
                              "ndcg_full": round(float(keep[lv == l].ndcg_full.mean()), 6),
                              "ndcg_at_20": round(float(keep[lv == l].ndcg_at_20.mean()), 6),
                              "ndcg_at_10": round(float(keep[lv == l].ndcg_at_10.mean()), 6)}
                          for l in ["us", "es", "jp"]}}
        return out, tbl

    off, t_off = report(dv, "score", "Amazon official baseline (reproduced)")
    v2 = pd.read_parquet(V2_PRED)
    v2 = v2[v2.query_id.isin(set(dv.query_id))]
    v2r, t_v2 = report(v2, "score", "V2 Cross-Encoder epoch_2")

    ci = t_off.index.intersection(t_v2.index)
    d = (t_v2.loc[ci, "ndcg_at_20"] - t_off.loc[ci, "ndcg_at_20"]).values
    rng = np.random.RandomState(42)
    bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pv = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))

    payload = {
        "evaluation_pool": "competition-aligned frozen DEV (== official Task 1 pairs)",
        "rows": int(len(dv)), "queries": int(dv.query_id.nunique()),
        "test_split_touched": False,
        "input": "query + product_title only (official baseline recipe)",
        "official_baseline": off, "v2": v2r,
        "delta_v2_minus_official": {
            "ndcg_full": round(v2r["ndcg_full"] - off["ndcg_full"], 6),
            "ndcg_at_20": round(v2r["ndcg_at_20"] - off["ndcg_at_20"], 6),
            "ndcg_at_10": round(v2r["ndcg_at_10"] - off["ndcg_at_10"], 6),
            "by_locale_at_20": {l: round(v2r["locale"][l]["ndcg_at_20"]
                                         - off["locale"][l]["ndcg_at_20"], 6)
                                for l in ["us", "es", "jp"]}},
        "paired_bootstrap_v2_minus_official_at20": {
            "mean_delta": float(d.mean()), "ci_95": [float(lo), float(hi)],
            "p_value": pv, "n_queries": int(len(d)),
            "win": int((d > 1e-12).sum()), "loss": int((d < -1e-12).sum()),
            "tie": int((np.abs(d) <= 1e-12).sum()),
            "n_bootstrap": 10000, "seed": 42},
        "inference_runtime": runtime,
        "total_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(payload, open(os.path.join(OUT, "official_baseline_dev_results.json"), "w"),
              indent=2, default=str)
    dv[["query_id", "product_id", "product_locale", "esci_label", "gain", "score"]].to_parquet(
        os.path.join(OUT, "official_baseline_dev_predictions.parquet"), index=False)

    print(f"\n{'model':<40}{'full':>10}{'@20':>10}{'@10':>10}")
    for r in [off, v2r]:
        print(f"{r['model']:<40}{r['ndcg_full']:>10.4f}{r['ndcg_at_20']:>10.4f}{r['ndcg_at_10']:>10.4f}")
    print(f"\n{'locale @20':<40}{'us':>10}{'es':>10}{'jp':>10}")
    for r in [off, v2r]:
        print(f"{r['model'][:38]:<40}" + "".join(
            f"{r['locale'][l]['ndcg_at_20']:>10.4f}" for l in ["us", "es", "jp"]))
    print(f"\nV2 - official @20 = {payload['delta_v2_minus_official']['ndcg_at_20']:+.4f}  "
          f"CI[{lo:+.4f},{hi:+.4f}] p={pv:.4f}")
    print("wrote official_baseline_dev_results.json")


if __name__ == "__main__":
    main()
