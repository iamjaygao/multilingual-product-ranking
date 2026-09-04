# semantic_score Ablation (Fixed-Candidate LambdaMART Setting)

## Status: NOT RUN

This is a **fixed-candidate reranking** experiment (NDCG@10, LambdaMART),
not full-catalog retrieval — deliberately kept separate from the rest of
`two_tower_v2/`, per task instructions not to conflate the two pipelines.

## Why it wasn't run

Testing "raw cosine vs. per-query min-max (current) vs. per-query z-score vs.
a local-margin feature" requires **recomputing `semantic_score` for every row
already in `output/two_tower_scores_train.csv` / `two_tower_scores_test.csv`**
using the existing V0 Two-Tower model — i.e. re-running the encoder over the
same (query, candidate) pairs used for the 17-feature LambdaMART model
(99,684 train queries + 30,969 test queries' full candidate pools). This is
not free: it means re-encoding a similarly large query/candidate volume to
what `scripts/generate_two_tower_scores.py`'s own docstring already warns
"may take a while depending on hardware." It cannot be derived from the
already-saved CSVs alone, because those only store the final min-max-
normalized value per row — the raw cosine similarity and the per-query
min/max needed to invert it are not persisted anywhere (dropped candidates
outside the saved Top-150 also contributed to each query's min/max and are
permanently unavailable). Given Phase 1's training run was already occupying
this session's primary compute budget, this re-encoding pass was not
attempted.

## What would be required (scaffolding only, not written)

1. A small variant of `retrieval/two_tower.py`'s `compute_two_tower_scores()`
   (as a NEW function, not a modification of the existing one) that returns,
   per (query_id, item_id) row, **all** of: raw cosine, per-query min-max,
   per-query z-score, and (max − 2nd-max) as a local-margin feature —
   computed once so all four are logged from the same forward pass.
2. Regenerate `two_tower_scores_train_v2.csv` / `_test_v2.csv` (new files,
   not overwriting the existing ones) with all four columns.
3. Retrain the 17-feature LambdaMART model four times, swapping only the
   `semantic_score` column source each time (identical split/seed/
   hyperparameters/label_gain to the audited 17-feature model — LambdaMART
   retraining itself is cheap, ~5s per run, per `experiments/audit/task2_train_meta.json`
   from the prior audit — the bottleneck here is entirely the Two-Tower
   re-encoding step, not the LambdaMART retraining).
4. Compare NDCG@10 across the four variants.

Step 3 alone is cheap; steps 1-2 are the blocker. This is flagged as the
lowest-effort follow-up if someone wants to pick this back up, since it does
not require touching the Two-Tower model or its training at all.
