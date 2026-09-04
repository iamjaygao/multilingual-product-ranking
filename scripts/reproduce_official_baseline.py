"""
Competition alignment §11-§12 -- reproduce the Amazon OFFICIAL Task 1 baseline.

Recipe transcribed verbatim from amazon-science/esci-data (HEAD 7916cdf):
  ranking/launch-experiments-task1.sh  and  ranking/train.py

  filter        small_version==1 AND split=='train' AND product_locale==<locale>
  gain          esci_label2gain = {'E':1.0,'S':0.1,'C':0.01,'I':0.0}   (train.py:49)
  input         InputExample(texts=[query, product_title], label=float(gain))
                -> query + product_title ONLY. No brand/color/bullet/description.
  target        the gain itself, as a REGRESSION target
  batch         TRAIN_BATCH_SIZE=32, shuffle=True, drop_last=True
  seed          RANDOM_STATE=42
  epochs        1
  us            CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', num_labels=1,
                max_length=512, activation=Identity), MSELoss, warmup_steps=5000, lr=7e-6
  es / jp       SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1'),
                CosineSimilarityLoss, evaluation_steps=1000, library-default optimiser

DOCUMENTED DEVIATIONS (both necessary, neither optional):

 1. TRAIN POOL. train.py trains on ALL official Task 1 TRAIN queries for the locale,
    minus its own small internal dev holdout (us 400 / es 200 / jp 200). Our frozen
    DEV queries are drawn from that same official TRAIN pool, so the official recipe
    as-written WOULD TRAIN ON OUR DEV -- the resulting DEV score would be
    train-on-eval and meaningless. We therefore exclude our 5,071 frozen DEV queries
    first, then apply the official internal dev holdout to what remains. This also
    makes the comparison strictly fairer: the official baseline and V2 then see the
    SAME 28,733 training queries and are scored on the SAME 5,071 DEV queries.

 2. API. sentence-transformers 5.2.3 renamed CrossEncoder's
    `default_activation_function` -> `activation_fn`. `fit()`, `MSELoss`,
    `CosineSimilarityLoss`, `CERerankingEvaluator` and `EmbeddingSimilarityEvaluator`
    all still exist with compatible signatures, so this is the only change.

TEST is never read.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src import paths

ROOT = str(paths.REPO_ROOT)
P = str(paths.ESCI_DATA_ROOT / "shopping_queries_dataset")
B = str(paths.DATA_TASK1)
OUT = str(paths.RESULTS / "phase4")

ESCI_LABEL2GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
N_DEV_QUERIES = {"us": 400, "es": 200, "jp": 200}
RANDOM_STATE, TRAIN_BATCH_SIZE = 42, 32
US_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
XX_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", required=True, choices=["us", "es", "jp"])
    ap.add_argument("--out_dir",
                    default=str(paths.ARTIFACTS / "checkpoints" / "official_baseline"))
    ap.add_argument("--max_train_rows", type=int, default=None, help="smoke test only")
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"task_1_ranking_model_{args.locale}")
    t0 = time.time()

    ex = pd.read_parquet(paths.EXAMPLES)
    pr = pd.read_parquet(paths.PRODUCTS,
                         columns=["product_id", "product_locale", "product_title"])
    df = pd.merge(ex, pr, how="left", left_on=["product_locale", "product_id"],
                  right_on=["product_locale", "product_id"], validate="many_to_one")
    df = df[(df.small_version == 1) & (df.split == "train") & (df.product_locale == args.locale)]
    df["gain"] = df.esci_label.map(ESCI_LABEL2GAIN)

    # ---- DEVIATION 1: remove our frozen DEV queries before anything else ----
    our_dev_q = set(pd.read_parquet(os.path.join(B, "dev_task1_core.parquet"),
                                    columns=["query_id"])["query_id"])
    n_before = len(df)
    df = df[~df.query_id.isin(our_dev_q)]
    removed = n_before - len(df)
    assert not (set(df.query_id) & our_dev_q), "DEV LEAK: our DEV queries still in train"
    print(f"[{args.locale}] official pool {n_before} rows -> {len(df)} after removing "
          f"{removed} rows belonging to our frozen DEV queries", flush=True)

    # ---- official internal dev holdout, on what remains ----
    qids = df.query_id.unique()
    dev_size = N_DEV_QUERIES[args.locale] / len(qids)
    q_tr, q_dv = train_test_split(qids, test_size=dev_size, random_state=RANDOM_STATE)
    d = df[["query_id", "query", "product_title", "gain"]]
    d_tr, d_dv = d[d.query_id.isin(set(q_tr))], d[d.query_id.isin(set(q_dv))]
    if args.max_train_rows:
        d_tr = d_tr.head(args.max_train_rows)
    print(f"[{args.locale}] train {len(d_tr)} rows / {d_tr.query_id.nunique()} q  |  "
          f"internal dev {len(d_dv)} rows / {d_dv.query_id.nunique()} q", flush=True)

    from sentence_transformers import InputExample, SentenceTransformer, losses
    train_samples = [InputExample(texts=[r["query"], str(r["product_title"])],
                                  label=float(r["gain"])) for _, r in d_tr.iterrows()]
    train_dl = DataLoader(train_samples, shuffle=True, batch_size=TRAIN_BATCH_SIZE,
                          drop_last=True)

    meta = {"locale": args.locale, "device": dev, "train_rows": int(len(d_tr)),
            "train_queries": int(d_tr.query_id.nunique()),
            "internal_dev_rows": int(len(d_dv)),
            "internal_dev_queries": int(d_dv.query_id.nunique()),
            "our_dev_rows_removed": int(removed), "batch_size": TRAIN_BATCH_SIZE,
            "seed": RANDOM_STATE, "epochs": 1, "gain_map": ESCI_LABEL2GAIN,
            "input": "query + product_title only"}

    if args.locale == "us":
        from sentence_transformers.cross_encoder import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
        dev_samples, q2i = {}, {}
        for _, r in d_dv.iterrows():
            qid = q2i.setdefault(r["query"], len(q2i))
            s = dev_samples.setdefault(qid, {"query": r["query"], "positive": set(),
                                             "negative": set()})
            (s["positive"] if r["gain"] > 0 else s["negative"]).add(str(r["product_title"]))
        # DEVIATION 2b: ST 5.2.3's CERerankingEvaluator concatenates positive+negative with
        # `+`, so it requires lists; official train.py built sets (valid in ST 2.x). Convert
        # to sorted lists -- same de-duplicated content, deterministic order. This touches
        # only the in-training monitoring evaluator, never the trained weights: the US branch
        # saves the end-of-training model via model.save(save_path), exactly as official does.
        for s in dev_samples.values():
            s["positive"] = sorted(s["positive"])
            s["negative"] = sorted(s["negative"])
        evaluator = CERerankingEvaluator(dev_samples, name="train-eval")
        model = CrossEncoder(US_MODEL, num_labels=1, max_length=512,
                             activation_fn=torch.nn.Identity(), device=dev)
        meta.update({"model_name": US_MODEL, "arch": "CrossEncoder", "num_labels": 1,
                     "max_length": 512, "loss": "MSELoss", "lr": 7e-6,
                     "warmup_steps": 5000, "evaluation_steps": 5000,
                     "activation": "Identity"})
        model.fit(train_dataloader=train_dl, loss_fct=torch.nn.MSELoss(), evaluator=evaluator,
                  epochs=1, evaluation_steps=5000, warmup_steps=5000,
                  output_path=f"{save_path}_tmp", optimizer_params={"lr": 7e-6})
        model.save(save_path)
    else:
        from sentence_transformers import evaluation
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            d_dv["query"].to_list(), d_dv["product_title"].astype(str).to_list(),
            d_dv["gain"].to_list())
        model = SentenceTransformer(XX_MODEL, device=dev)
        meta.update({"model_name": XX_MODEL, "arch": "SentenceTransformer (bi-encoder)",
                     "loss": "CosineSimilarityLoss", "evaluation_steps": 1000,
                     "lr": "library default", "max_length": model.max_seq_length})
        model.fit(train_objectives=[(train_dl, losses.CosineSimilarityLoss(model=model))],
                  evaluator=evaluator, epochs=1, evaluation_steps=1000,
                  output_path=save_path)

    meta["train_seconds"] = round(time.time() - t0, 1)
    meta["saved_to"] = os.path.relpath(save_path, ROOT)
    meta["test_split_touched"] = False
    json.dump(meta, open(os.path.join(args.out_dir, f"meta_{args.locale}.json"), "w"),
              indent=2, default=str)
    print(f"[{args.locale}] done in {meta['train_seconds']}s -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
