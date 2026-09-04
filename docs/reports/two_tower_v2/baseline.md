# Phase 0 — Baseline (V0, historical, unmodified)

This directory records the state of the Two-Tower training/serving pipeline
and its full-catalog retrieval results **as they exist in the repo today**,
without re-running anything. Nothing here has been recomputed; both files
are read from existing artifacts:

- `baseline_config.json` — read from `scripts/train_two_tower.py`,
  `retrieval/two_tower.py`, `scripts/build_full_catalog_indices.py`, and
  `output/full_retrieval/eval_query_ids.json`.
- `baseline_results.json` — copied verbatim from
  `output/full_retrieval/retrieval_metrics.csv`.

## Confirmed facts (vs. the task brief's assumptions)

All assumptions in the task brief's "背景" section were verified against the
current code and are **correct as stated**:

- Base model, loss, batch size, epoch count, positive-label definition: match.
- 329,447 training pairs, 5,147 training steps/epoch: match exactly.
- `scheduler`/`warmup_steps`/`optimizer_params` are not explicitly passed to
  `model.fit()`, so sentence-transformers' defaults apply
  (`WarmupLinear`, `warmup_steps=10000`, `lr=2e-5`): confirmed by inspecting
  `sentence_transformers.SentenceTransformer.fit`'s signature in the
  installed version (5.2.3) and by `grep`-ing `train_two_tower.py` for any
  of those keywords (none found).
- No seed, no evaluator, no validation monitoring, no best-checkpoint
  selection, no early stopping: confirmed, none of these appear anywhere in
  `train_two_tower.py`.

One fact not previously stated but confirmed here: **embedding normalization
is already consistent** between item and query encoding paths
(`normalize_embeddings=True` in both `encode_texts()` and
`search_tt_global()` in `retrieval/two_tower.py`), so `IndexFlatIP` is a
mathematically valid cosine-similarity index today — this is **not** a bug
to fix in Phase 7, contrary to what might be assumed.

## Historical full-catalog Recall@100 (V0)

| Retriever | Recall@100 |
|---|---:|
| BM25 | 0.4542 |
| Two-Tower | 0.4598 |
| Hybrid RRF@100 (k=60) | 0.5194 |

Full table with Recall@10/50/200, ExactRecall, and MRR@10 in
`baseline_results.json`. Eval query sample: 5,000 US-locale test queries,
seed=42 (`output/full_retrieval/eval_query_ids.json`), unchanged from all
prior experiments in this repo.

This baseline is treated as immutable for the rest of `two_tower_v2/`: no
file under `output/full_retrieval/` or `models/two_tower_finetuned/` is
modified by any later phase. All new models are saved under
`experiments/two_tower_v2/**`.
