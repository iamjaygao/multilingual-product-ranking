"""
Evaluate a Cross-Encoder checkpoint on a frozen Task 1 split.

Pipeline: batched inference -> softmax -> expected ESCI gain -> THE authoritative
Task 1 scorer. No second NDCG implementation exists here; the scorer is imported
from src/metrics/kdd_task1_ndcg.py.

TEST LOCK (§13.1): --split test requires --prereg naming a committed
pre-registration document. The source repo used a bare `--i_have_frozen_the_config`
flag; that is a convention, and this replaces it with the same validated code path
that guards every other TEST read. Model selection and tuning must use dev only.

Example:
    python scripts/evaluate_cross_encoder_task1.py \
        --checkpoint .../smoke_test/checkpoints/epoch_1 \
        --split dev --max_eval_queries 50 --batch_size 32 \
        --output_dir experiments/ranking_v2/cross_encoder_task1/smoke_test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from src import paths
from src.ranking.cross_encoder import (
    CrossEncoderDataset, LABEL_ORDER, assert_gain_agreement, build_product_texts,
    load_task1_scorer, sample_query_ids, scores_from_logits,
)

CACHE = str(paths.DATA_TASK1 / "ce_text")
EXPECTED = {"train": (663407, 28733), "dev": (118231, 5071), "test": (336373, 14496)}
LOCALES = ("us", "es", "jp")


def pick_device(name: str) -> str:
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict(df: pd.DataFrame, checkpoint: str, max_length: int, batch_size: int,
            device: str, max_chars_per_field=None, log_every: int = 200) -> np.ndarray:
    """Returns (n, 4) logits, row-aligned with `df`."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    if model.config.num_labels != len(LABEL_ORDER):
        raise SystemExit(f"STOP: checkpoint has {model.config.num_labels} labels, "
                         f"expected {len(LABEL_ORDER)}")
    ck_order = [model.config.id2label[i] for i in range(len(LABEL_ORDER))]
    if [str(x) for x in ck_order] != list(LABEL_ORDER):
        raise SystemExit(f"STOP: checkpoint label order {ck_order} != {list(LABEL_ORDER)}")
    model.to(device).eval()

    texts = build_product_texts(df, max_chars_per_field=max_chars_per_field)
    ds = CrossEncoderDataset(df["query"].tolist(), texts, tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate)

    out = np.empty((len(df), len(LABEL_ORDER)), dtype=np.float64)
    pos = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float().cpu().numpy()
            out[pos:pos + len(logits)] = logits
            pos += len(logits)
            if (i + 1) % log_every == 0:
                print(f"  inference {pos}/{len(df)}", flush=True)
    if pos != len(df):
        raise AssertionError(f"prediction row count mismatch: {pos} != {len(df)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_eval_queries", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag", default=None, help="filename prefix, defaults to the split")
    ap.add_argument("--save_predictions", action="store_true")
    ap.add_argument("--max_chars_per_field", type=int, default=0)
    ap.add_argument("--prereg", default=None,
                    help="path to a git-tracked pre-registration document. REQUIRED "
                         "to evaluate on test (TEST LOCK, §13.1).")
    args = ap.parse_args()

    if args.split == "test":
        # §13.1 -- an assert, not a flag. The pre-registration must exist and be
        # git-tracked; the access is appended to the TEST access log.
        from src.data.task1_common import _validate_prereg, TestSplitLocked
        if args.prereg is None:
            raise TestSplitLocked(
                "STOP: TEST LOCK. test may be evaluated exactly once, and only after "
                "a pre-registered prediction has been committed. Pass "
                "--prereg docs/prereg/<file>.md. See MIGRATION_PLAN.md §13.1.")
        _validate_prereg(args.prereg)

    assert_gain_agreement()
    sc = load_task1_scorer()
    device = pick_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or args.split
    t0 = time.time()

    path = os.path.join(CACHE, f"{args.split}_text.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"STOP: {path} missing. Run `python -m scripts.build_ce_data`")
    df = pd.read_parquet(path)
    exp_rows, exp_q = EXPECTED[args.split]
    if (len(df), df["query_id"].nunique()) != (exp_rows, exp_q):
        raise SystemExit(f"STOP: {args.split} cache disagrees with the frozen benchmark: "
                         f"{len(df)}/{df['query_id'].nunique()} vs {exp_rows}/{exp_q}")
    df = sample_query_ids(df, args.max_eval_queries, seed=args.seed).reset_index(drop=True)
    print(f"Evaluating {len(df)} rows / {df['query_id'].nunique()} queries "
          f"({args.split}) on {device}", flush=True)

    logits = predict(df, args.checkpoint, args.max_length, args.batch_size,
                     device, args.max_chars_per_field or None)
    scores = scores_from_logits(logits)
    if not (np.all(scores >= -1e-12) and np.all(scores <= 1 + 1e-12)):
        raise AssertionError("expected-gain score outside [0, 1]")

    # keys are carried through positionally and re-asserted
    pred = df[["query_id", "product_id", "product_locale", "esci_label", "gain"]].copy()
    pred["score"] = scores
    assert len(pred) == len(df) and pred["score"].notna().all()
    # the frozen `gain` column must agree with the scorer's own mapping
    expected_gain_col = sc.gain_from_labels(pred["esci_label"])
    if not np.allclose(pred["gain"].to_numpy(dtype=float), expected_gain_col):
        raise AssertionError("frozen gain column disagrees with the authoritative gain map")

    # ---------------- authoritative scorer ----------------
    overall = sc.evaluate(pred, "score", gain_col="gain", query_col="query_id")
    tbl = sc.per_query_ndcg_table(pred, "score", gain_col="gain", query_col="query_id")

    metrics = {
        "ndcg_full": round(overall["ndcg_full"], 6),
        "ndcg_at_10": round(overall["ndcg_at_10"], 6),
        "ndcg_at_20": round(overall["ndcg_at_20"], 6),
        "query_count": overall["n_queries_scored"],
        "row_count": int(len(pred)),
    }

    qloc = pred.drop_duplicates("query_id").set_index("query_id")["product_locale"]
    keep = tbl[tbl["idcg_full"] > 0]
    lv = qloc.reindex(keep.index)
    locale_metrics = {}
    for lc in LOCALES:
        s = keep[lv == lc]
        locale_metrics[lc] = {
            "query_count": int(len(s)),
            "ndcg_full": round(float(s["ndcg_full"].mean()), 6) if len(s) else None,
            "ndcg_at_10": round(float(s["ndcg_at_10"].mean()), 6) if len(s) else None,
            "ndcg_at_20": round(float(s["ndcg_at_20"].mean()), 6) if len(s) else None,
        }
    locale_metrics["overall"] = {
        "query_count": metrics["query_count"], "ndcg_full": metrics["ndcg_full"],
        "ndcg_at_10": metrics["ndcg_at_10"], "ndcg_at_20": metrics["ndcg_at_20"]}

    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    detail = {
        "split": args.split, "checkpoint": args.checkpoint, "device": device,
        "max_length": args.max_length, "batch_size": args.batch_size,
        "max_eval_queries": args.max_eval_queries, "seed": args.seed,
        "scorer": os.path.relpath(sc.__file__, str(paths.REPO_ROOT)) if hasattr(sc, "__file__") else None,
        "candidate_pool": overall["candidate_pool"],
        "metric_version": overall["metric_version"],
        "zero_idcg_policy": overall["zero_idcg_policy"],
        "zero_idcg_query_count": overall["zero_idcg_query_count"],
        "score_definition": "expected ESCI gain: P(E)*1.0 + P(S)*0.1 + P(C)*0.01 + P(I)*0.0",
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "mean_predicted_prob": {l: round(float(probs[:, i].mean()), 6)
                                for i, l in enumerate(LABEL_ORDER)},
        "argmax_label_distribution": {
            l: int((probs.argmax(axis=1) == i).sum()) for i, l in enumerate(LABEL_ORDER)},
        "wall_time_seconds": round(time.time() - t0, 1),
        "is_official_test_run": args.split == "test",
    }

    with open(os.path.join(args.output_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump({**metrics, **detail}, f, indent=2, default=str)
    with open(os.path.join(args.output_dir, f"{tag}_locale_metrics.json"), "w") as f:
        json.dump(locale_metrics, f, indent=2, default=str)
    if args.save_predictions:
        pred.to_parquet(os.path.join(args.output_dir, f"{tag}_predictions.parquet"), index=False)

    print("\n" + json.dumps(metrics, indent=2))
    print("\nlocale:")
    for lc in list(LOCALES) + ["overall"]:
        m = locale_metrics[lc]
        print(f"  {lc:8s} n={m['query_count']:6d}  full={m['ndcg_full']}  "
              f"@10={m['ndcg_at_10']}  @20={m['ndcg_at_20']}")
    print(f"\npool={detail['candidate_pool']}  metric_version={detail['metric_version']}")


if __name__ == "__main__":
    main()
