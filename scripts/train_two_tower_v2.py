"""
Phase 1 -- "Correct Training" for Two-Tower (V1). Fixes, in isolation:
  - warmup_ratio-based LR schedule (not the unfixed warmup_steps=10000 default)
  - explicit seed (all RNGs)
  - query-level train/dev split (no leakage)
  - a real retrieval evaluator (InformationRetrievalEvaluator) for checkpoint
    selection, instead of no validation at all
Everything else (base model, loss, batch size, positive-label definition,
item-text construction) is held identical to V0 (scripts/train_two_tower.py)
so any metric change is attributable to this fix alone, per the task's "one
variable at a time" requirement.

Does not modify scripts/train_two_tower.py, retrieval/two_tower.py, or any
existing model/output artifact. All outputs go under
experiments/two_tower_v2/phase1_correct_training/.

Usage:
    python scripts/train_two_tower_v2.py --sanity     # tiny end-to-end check
    python scripts/train_two_tower_v2.py               # full V1 run

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
import time
import argparse

import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses

from src import paths
from src.retrieval.two_tower_training import (
    build_dev_ir_evaluator, load_positive_pairs, set_all_seeds,
)

EXAMPLES_PATH = paths.EXAMPLES
PRODUCTS_PATH = paths.PRODUCTS

#: English-only base encoder. See the module docstring on locale scope.
MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v3"
SPLITS_DIR = str(paths.SPLITS)
OUT_DIR = str(paths.RESULTS / "two_tower_v2")


def build_config(sanity: bool):
    cfg = {
        "base_model": MODEL_NAME,
        "loss": "MultipleNegativesRankingLoss",
        "batch_size": 64,
        "epochs": 3,
        "lr": 2e-5,
        "warmup_ratio": 0.10,
        "lr_scheduler_type": "linear",
        "seed": 42,
        "positive_label_definition": {"labels": ["E", "S"], "locale": "us", "small_version": 1, "split": "train"},
        "dev_evaluator": {"n_eval_queries": 300, "dev_corpus_size": 20000, "relevant_labels": ["E", "S"],
                           "precision_recall_at_k": [10, 50, 100]},
        "metric_for_best_model": "dev_cosine_recall@100",
        "greater_is_better": True,
        "git_commit_note": "see experiments/two_tower_v2/baseline/baseline_config.json for git_commit at time of V0 audit",
    }
    if sanity:
        cfg.update({
            "epochs": 1, "sanity_max_train_pairs": 512,
            "dev_evaluator": {"n_eval_queries": 20, "dev_corpus_size": 2000, "relevant_labels": ["E", "S"],
                               "precision_recall_at_k": [10, 50, 100]},
        })
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="tiny end-to-end run to validate the pipeline")
    args = parser.parse_args()

    cfg = build_config(args.sanity)
    run_dir = f"{OUT_DIR}/sanity_run" if args.sanity else f"{OUT_DIR}/full_run"
    os.makedirs(run_dir, exist_ok=True)

    set_all_seeds(cfg["seed"])

    with open(f"{SPLITS_DIR}/train_queries.txt") as f:
        train_queries = [l.strip() for l in f if l.strip()]
    with open(f"{SPLITS_DIR}/dev_queries.txt") as f:
        dev_queries = [l.strip() for l in f if l.strip()]

    print(f"Loading positive training pairs for {len(train_queries)} train queries...")
    df_train = load_positive_pairs(EXAMPLES_PATH, PRODUCTS_PATH, train_queries,
                                    labels=cfg["positive_label_definition"]["labels"])
    if args.sanity:
        df_train = df_train.sample(n=min(cfg["sanity_max_train_pairs"], len(df_train)), random_state=cfg["seed"])
    print(f"Training pairs: {len(df_train)}")

    train_dataset = Dataset.from_dict({
        "anchor": df_train["query"].tolist(),
        "positive": df_train["item_text"].tolist(),
    })

    total_steps_per_epoch = len(train_dataset) // cfg["batch_size"]
    total_steps = total_steps_per_epoch * cfg["epochs"]
    warmup_steps_resolved = int(total_steps * cfg["warmup_ratio"])
    cfg["_resolved"] = {
        "n_train_pairs": len(df_train),
        "steps_per_epoch": total_steps_per_epoch,
        "total_training_steps": total_steps,
        "warmup_steps_resolved": warmup_steps_resolved,
    }
    print(f"steps_per_epoch={total_steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps_resolved}")

    print("Building dev retrieval evaluator...")
    evaluator, dev_manifest = build_dev_ir_evaluator(
        EXAMPLES_PATH, PRODUCTS_PATH, dev_queries, name="dev",
        n_eval_queries=cfg["dev_evaluator"]["n_eval_queries"],
        dev_corpus_size=cfg["dev_evaluator"]["dev_corpus_size"],
        seed=cfg["seed"], relevant_labels=cfg["dev_evaluator"]["relevant_labels"],
        precision_recall_at_k=cfg["dev_evaluator"]["precision_recall_at_k"],
    )
    cfg["dev_evaluator"]["manifest"] = dev_manifest
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Loading base model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"{run_dir}/checkpoints",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        logging_steps=max(1, total_steps_per_epoch // 20),
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    t0 = time.time()
    trainer.train()
    train_runtime_sec = time.time() - t0

    final_model_path = f"{run_dir}/final_model"
    model.save(final_model_path)
    print(f"Saved final (best, load_best_model_at_end=True) model to {final_model_path}")

    # ---- Dump training history (loss + lr + dev metrics per logged/eval step) ----
    log_history = trainer.state.log_history
    with open(f"{run_dir}/raw_log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)

    rows = []
    for entry in log_history:
        rows.append(entry)
    hist_df = pd.DataFrame(rows)
    hist_df.to_csv(f"{run_dir}/training_history.csv", index=False)

    best_metric = trainer.state.best_metric
    best_model_checkpoint = trainer.state.best_model_checkpoint

    results = {
        "config": cfg,
        "train_runtime_sec": train_runtime_sec,
        "best_metric": best_metric,
        "best_metric_name": cfg["metric_for_best_model"],
        "best_model_checkpoint": best_model_checkpoint,
        "final_epoch": trainer.state.epoch,
        "global_step": trainer.state.global_step,
    }
    with open(f"{run_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lr_rows = hist_df.dropna(subset=["learning_rate"]) if "learning_rate" in hist_df.columns else pd.DataFrame()
        if len(lr_rows):
            plt.figure(figsize=(6, 4))
            plt.plot(lr_rows["step"], lr_rows["learning_rate"])
            plt.xlabel("step"); plt.ylabel("learning_rate"); plt.title("Phase 1 (V1) LR schedule (warmup_ratio=0.10)")
            plt.tight_layout()
            plt.savefig(f"{run_dir}/lr_schedule.png", dpi=120)
            plt.close()

        metric_cols = [c for c in hist_df.columns if c.startswith("eval_dev_cosine_recall") or c.startswith("eval_dev_cosine_mrr")]
        eval_rows = hist_df.dropna(subset=metric_cols, how="all") if metric_cols else pd.DataFrame()
        if len(eval_rows) and metric_cols:
            plt.figure(figsize=(6, 4))
            for c in metric_cols:
                plt.plot(eval_rows["epoch"], eval_rows[c], marker="o", label=c.replace("eval_dev_cosine_", ""))
            plt.xlabel("epoch"); plt.ylabel("metric"); plt.legend(); plt.title("Phase 1 (V1) dev retrieval metrics")
            plt.tight_layout()
            plt.savefig(f"{run_dir}/dev_metrics.png", dpi=120)
            plt.close()
    except Exception as e:
        print(f"WARNING: plotting failed ({e}); training_history.csv is still saved.")

    print("Done.")


if __name__ == "__main__":
    main()
