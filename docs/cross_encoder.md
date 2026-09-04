# Phase 2A — Multilingual Cross-Encoder Reranker (Task 1 baseline)

Status: **implemented and smoke-tested only.** No full training run, no official test run.

Adds a 4-class Cross-Encoder ranking pipeline on top of the frozen, audited
`task1_small_v1` benchmark. Nothing in `experiments/ranking_v2/kdd_task1_benchmark/`
was modified, and no existing baseline was retrained or re-scored.

---

## 1. Task definition

| | |
|---|---|
| input | `(query, product_text)` tokenizer sequence pair |
| output | 4 logits in the **fixed** order `[E, S, C, I]` |
| loss | plain `CrossEntropyLoss` — no class weights, no focal loss, no pairwise/listwise objective, no distillation, no ensembling, no LambdaMART feature fusion |
| `id2label` | `{0: E, 1: S, 2: C, 3: I}` |
| `label2id` | `{E: 0, S: 1, C: 2, I: 3}` |

**Ranking never uses `argmax`.** The score is the expected ESCI gain:

```
score = P(E)*1.00 + P(S)*0.10 + P(C)*0.01 + P(I)*0.00
```

which is a convex combination of the official gains and therefore always in `[0, 1]`.
`reranking/cross_encoder.assert_gain_agreement()` fails loudly if this vector ever
drifts from `kdd_task1_ndcg.GAIN_COMPETITION`.

Why not argmax: two candidates can share an argmax of `E` while having very
different `P(E)`. Expected gain separates them; argmax cannot. Covered by
`test_ranking_is_not_argmax` and `test_expected_gain_beats_argmax_on_a_tie`.

---

## 2. Evaluation

The **authoritative Task 1 scorer is imported, never reimplemented**:

```
experiments/ranking_v2/kdd_task1_benchmark/scripts/kdd_task1_ndcg.py
```

Headline is **full-list NDCG**; `@10` and `@20` and the US/ES/JP breakdown are
always reported alongside. Every metrics file carries `candidate_pool` and
`metric_version`, per the Task 1 bookkeeping rule.

```
candidate_pool  = task1_small_v1
metric_version  = kdd_task1_ndcg@6946489c8c09dc37|gain=competition_E1_S0.1_C0.01_I0
                  |idcg0=exclude|tie=deterministic
```

Numbers from this pool must never be compared with `large_v1` numbers.

---

## 3. Data

Frozen splits, verified row-for-row against the Task 1 REPORT before every run:

| split | rows | queries |
|---|---:|---:|
| train | 663,407 | 28,733 |
| dev | 118,231 | 5,071 |
| test | 336,373 | 14,496 (**locked**) |

### A necessary enrichment step

The frozen parquets carry `product_title` and `product_brand` but **not**
`product_color`, `product_bullet_point` or `product_description` — they were
built for the 17-feature LambdaMART pipeline, which never needed them.

`scripts/build_cross_encoder_task1_data.py` LEFT-joins those three fields from the
raw products table on `(product_id, product_locale)` — locale-aware, asserted not
to fan out — into `_cache/{split}_text.parquet`. **The frozen splits are never
modified and no row is added or removed.**

Product text layout (empty fields omitted, so the 256-token budget goes to content):

```
title: ...
brand: ...
color: ...
bullet: ...
description: ...
```

`NaN`/`None`/`"nan"`/`"none"` all become `""` and are dropped — the literal string
`nan` can never reach the tokenizer (`test_missing_values_never_leak_the_string_nan`).

---

## 4. Tokenization

```python
tokenizer(query, product_text, truncation="only_second", max_length=256, padding=True)
```

`only_second` guarantees the **query is never truncated**; only the product side is
cut. Verified for Latin, Spanish-accented and Japanese queries in
`TestQueryNotTruncated`.

Measured at `max_length=256` on the smoke sample:

| split | pair tokens mean | p95 | **truncated** | query overflow |
|---|---:|---:|---:|---:|
| train | 329.8 | 926.0 | **47.88%** | 0.0000% |
| dev | 317.6 | 892.7 | **45.67%** | 0.0000% |

Roughly half of all pairs lose product text at 256 tokens. **This was recorded, not
acted on** — the brief fixes `max_length=256` for the baseline round.

---

## 5. Files

| Path | Role |
|---|---|
| `reranking/cross_encoder.py` | labels, expected gain, text builder, dataset, token stats, scorer loader |
| `scripts/build_cross_encoder_task1_data.py` | joins color/bullet/description into `_cache/` |
| `scripts/train_cross_encoder_task1.py` | training CLI |
| `scripts/evaluate_cross_encoder_task1.py` | inference → expected gain → authoritative scorer |
| `tests/test_cross_encoder.py` | 37 tests |

Gitignored (already covered, no rules added): `_cache/`, `checkpoints/`, `*.safetensors`,
`*.parquet`, `*.log`.

---

## 6. TEST LOCK

- `train_cross_encoder_task1.py` reads only train/dev and asserts neither contains a
  test query.
- `evaluate_cross_encoder_task1.py` refuses `--split test` unless
  `--i_have_frozen_the_config` is passed.
- **The test split has not been evaluated.** `is_official_test_run: false` in every
  metrics file written so far.

---

## 7. Reproducing the smoke test

```bash
python scripts/build_cross_encoder_task1_data.py --splits train dev

python scripts/train_cross_encoder_task1.py \
    --model_name FacebookAI/xlm-roberta-base \
    --output_dir experiments/ranking_v2/cross_encoder_task1/smoke_test/checkpoints \
    --max_train_queries 200 --max_dev_queries 50 \
    --epochs 1 --batch_size 16 --log_every 50

python scripts/evaluate_cross_encoder_task1.py \
    --checkpoint experiments/ranking_v2/cross_encoder_task1/smoke_test/checkpoints/epoch_1 \
    --split dev --max_eval_queries 50 --batch_size 32 \
    --output_dir experiments/ranking_v2/cross_encoder_task1/smoke_test \
    --tag smoke_dev --save_predictions
```

Sampling is **by query**, never by row, so every sampled query keeps its whole
candidate list and NDCG stays well defined.

### Smoke result — plumbing only, no business meaning

200 train queries, 1 epoch. **Do not compare to LambdaMART 0.842878.**

| | full | @10 | @20 |
|---|---:|---:|---:|
| overall (50 q, 1,167 rows) | 0.740115 | 0.546899 | 0.663842 |
| us (27 q) | 0.730791 | 0.577538 | 0.702406 |
| es (13 q) | 0.748096 | 0.523362 | 0.629890 |
| jp (10 q) | 0.754913 | 0.494772 | 0.603857 |

train_loss 1.1919 → dev_loss 1.1613, dev argmax accuracy 0.4404, 104 s on MPS.

The model is undertrained exactly as expected: argmax is `E` for all 1,167 rows, yet
expected-gain scoring still yields a continuous, usable ranking signal
(range 0.4428–0.6095). NDCG landing *below* the 0.750317 random floor is the
correct outcome for one epoch on 200 queries and is reassuring — it shows the
pipeline is not accidentally leaking labels.

---

## 8. Not done in this round

Full 663K-row training, any test-split evaluation, `xlm-roberta-large`,
Qwen3-Reranker, class weighting, hard-negative sampling, `max_length` tuning,
LambdaMART fusion. All deferred by the Phase 2A brief.
