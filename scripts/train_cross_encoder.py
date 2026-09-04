"""
Train the 4-class Cross-Encoder reranker on the frozen ESCI Task 1 train split.

Deliberately clean baseline: plain CrossEntropyLoss, no class weights, no focal
loss, no pairwise/listwise objective, no distillation, no ensembling, no
LambdaMART feature fusion, no early stopping. One checkpoint per epoch.

TEST LOCK: this script can only read train_task1 / dev_task1. It refuses to
touch the test split, and asserts train/dev query disjointness plus the frozen
row and query counts before doing any work.

A plain PyTorch loop is used rather than transformers.Trainer so that behaviour
does not shift with the (very new) transformers 5.x Trainer API.

Example (smoke):
    python scripts/train_cross_encoder_task1.py \
        --model_name FacebookAI/xlm-roberta-base \
        --output_dir experiments/ranking_v2/cross_encoder_task1/smoke_test/checkpoints \
        --max_train_queries 200 --max_dev_queries 50 --epochs 1 --batch_size 16
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

from src import paths
from src.ranking.cross_encoder import (
    CrossEncoderDataset, ID2LABEL, LABEL2ID, LABEL_ORDER, assert_gain_agreement,
    build_product_texts, load_model_and_tokenizer, sample_query_ids,
    token_length_stats,
)

CACHE = str(paths.DATA_TASK1 / "ce_text")
EXPECTED = {"train": (663407, 28733), "dev": (118231, 5071)}


def pick_device(name: str) -> str:
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split: str, max_queries, seed: int) -> pd.DataFrame:
    path = os.path.join(CACHE, f"{split}_text.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"STOP: {path} missing. Run scripts/build_cross_encoder_task1_data.py")
    df = pd.read_parquet(path)
    exp_rows, exp_q = EXPECTED[split]
    got = (len(df), df["query_id"].nunique())
    if got != (exp_rows, exp_q):
        raise SystemExit(
            f"STOP: {split} cache disagrees with the frozen benchmark.\n"
            f"  expected {exp_rows} rows / {exp_q} queries; actual {got[0]} / {got[1]}")
    return sample_query_ids(df, max_queries, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, required=True,
                    help="no default: depends on device memory")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_queries", type=int, default=None)
    ap.add_argument("--max_dev_queries", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--max_chars_per_field", type=int, default=0,
                    help="0 = no extra cap; the tokenizer already truncates")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    assert_gain_agreement()
    set_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    # ---------------- data ----------------
    print("Loading frozen Task 1 splits ...", flush=True)
    tr = load_split("train", args.max_train_queries, args.seed)
    dv = load_split("dev", args.max_dev_queries, args.seed)

    tr_q, dv_q = set(tr["query_id"]), set(dv["query_id"])
    assert not (tr_q & dv_q), f"train/dev query overlap: {len(tr_q & dv_q)}"
    # §13.1: query ids are not labels, so this needs no pre-registration --
    # task1_test_query_ids() reads them with metadata_only=True.
    from src.data.task1_common import task1_test_query_ids
    test_q = task1_test_query_ids()
    assert not (tr_q & test_q), "TEST LOCK VIOLATION: train contains test queries"
    assert not (dv_q & test_q), "TEST LOCK VIOLATION: dev contains test queries"
    print(f"  train {len(tr)} rows / {len(tr_q)} queries", flush=True)
    print(f"  dev   {len(dv)} rows / {len(dv_q)} queries", flush=True)
    print("  train/dev disjoint: True; neither touches the test split", flush=True)

    cap = args.max_chars_per_field or None
    tr_text = build_product_texts(tr, max_chars_per_field=cap)
    dv_text = build_product_texts(dv, max_chars_per_field=cap)
    tr_y = [LABEL2ID[l] for l in tr["esci_label"]]
    dv_y = [LABEL2ID[l] for l in dv["esci_label"]]

    # ---------------- model ----------------
    print(f"Loading {args.model_name} on {device} ...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.max_length)
    model.to(device)

    # ---------------- token stats (reported, never used to change max_length) ----
    print("Computing token-length / truncation statistics ...", flush=True)
    tok_stats = {
        "train": token_length_stats(tr["query"].tolist(), tr_text, tokenizer,
                                    args.max_length, tr["product_locale"].tolist()),
        "dev": token_length_stats(dv["query"].tolist(), dv_text, tokenizer,
                                  args.max_length, dv["product_locale"].tolist()),
        "note": ("reported for the record only; max_length was NOT tuned in response "
                 "to these numbers (this round is a baseline)"),
    }
    for sp in ["train", "dev"]:
        o = tok_stats[sp]["overall"]
        print(f"  {sp}: pair tokens mean {o['pair_tokens_mean']}, p95 {o['pair_tokens_p95']}, "
              f"truncated {100*o['truncation_fraction']:.2f}%, "
              f"query-overflow {100*o['query_overflow_fraction']:.4f}%", flush=True)

    train_ds = CrossEncoderDataset(tr["query"].tolist(), tr_text, tokenizer,
                                   args.max_length, tr_y)
    dev_ds = CrossEncoderDataset(dv["query"].tolist(), dv_text, tokenizer,
                                 args.max_length, dv_y)
    g = torch.Generator(); g.manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=train_ds.collate, num_workers=args.num_workers,
                          generator=g, drop_last=False)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=dev_ds.collate, num_workers=args.num_workers)

    # ---------------- optimiser ----------------
    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and not any(k in n for k in ["bias", "LayerNorm.weight"])]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and any(k in n for k in ["bias", "LayerNorm.weight"])]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.learning_rate)
    steps_per_epoch = (len(train_dl) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    total_steps = max(1, steps_per_epoch * args.epochs)
    sched = get_linear_schedule_with_warmup(
        optim, int(args.warmup_ratio * total_steps), total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---------------- train ----------------
    history = []
    print(f"\nTraining: {len(train_dl)} batches/epoch, {total_steps} optimiser steps", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, seen, t0 = 0.0, 0, time.time()
        optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_dl):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            (loss / args.gradient_accumulation_steps).backward()
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            run += loss.item() * labels.size(0); seen += labels.size(0)
            if (i + 1) % args.log_every == 0:
                print(f"  epoch {epoch} batch {i+1}/{len(train_dl)} "
                      f"loss {run/seen:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        train_loss = run / max(seen, 1)

        model.eval()
        drun, dseen, correct = 0.0, 0, 0
        with torch.no_grad():
            for batch in dev_dl:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                drun += loss_fn(logits, labels).item() * labels.size(0)
                correct += (logits.argmax(-1) == labels).sum().item()
                dseen += labels.size(0)
        dev_loss, dev_acc = drun / max(dseen, 1), correct / max(dseen, 1)
        secs = time.time() - t0
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "dev_loss": round(dev_loss, 6),
                        "dev_argmax_accuracy": round(dev_acc, 6),
                        "seconds": round(secs, 1)})
        print(f"  epoch {epoch}: train_loss {train_loss:.4f}  dev_loss {dev_loss:.4f}  "
              f"dev_argmax_acc {dev_acc:.4f}  ({secs:.0f}s)", flush=True)

        ck = os.path.join(args.output_dir, f"epoch_{epoch}")
        model.save_pretrained(ck); tokenizer.save_pretrained(ck)
        print(f"  saved {ck}", flush=True)

    peak_mb = None
    try:
        import torch as _t
        if device == "mps":
            peak_mb = round(_t.mps.driver_allocated_memory() / 1e6, 1)
        elif device == "cuda":
            peak_mb = round(_t.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil
        rss_mb = round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:  # noqa: BLE001
        rss_mb = None

    cfg = {
        "model_name": args.model_name, "device": device,
        "max_length": args.max_length, "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate, "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
        "seed": args.seed, "max_train_queries": args.max_train_queries,
        "max_dev_queries": args.max_dev_queries,
        "num_labels": len(LABEL_ORDER), "label_order": list(LABEL_ORDER),
        "id2label": ID2LABEL, "label2id": LABEL2ID,
        "loss": "CrossEntropyLoss (unweighted)",
        "ranking_score": "expected ESCI gain = P(E)*1.0 + P(S)*0.1 + P(C)*0.01 + P(I)*0.0",
        "early_stopping": False, "class_weights": False,
        "train_rows": int(len(tr)), "train_queries": int(len(tr_q)),
        "dev_rows": int(len(dv)), "dev_queries": int(len(dv_q)),
        "optimizer_steps": int(total_steps),
        "wall_time_seconds": round(time.time() - t_start, 1),
        "peak_device_memory_mb": peak_mb, "process_rss_mb": rss_mb,
        "test_split_touched": False,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump({"history": history, "token_stats": tok_stats}, f, indent=2, default=str)

    print(f"\nDone in {cfg['wall_time_seconds']}s. "
          f"peak_device_mem={peak_mb} MB rss={rss_mb} MB", flush=True)


if __name__ == "__main__":
    main()
