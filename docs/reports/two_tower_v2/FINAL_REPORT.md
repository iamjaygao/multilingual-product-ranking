# Two-Tower Dense Retrieval — Systematic Optimization (V2)

Git commit at start of this work: `9b173e95307b6af4335d46518f703265a9608d57`.
No existing tracked file was modified (`git diff --stat` is empty for
tracked files); every deliverable is a new file under `experiments/two_tower_v2/`,
`retrieval/two_tower_training.py`, or `scripts/*_v2.py` / `scripts/audit_*`,
`scripts/mine_*`, `scripts/benchmark_*`, `scripts/analyze_product_text_stats.py`.

## 1. Original Problems

Confirmed by re-reading the actual code (all matched the task brief's
assumptions — see `baseline/README.md` for the point-by-point verification):

- `scheduler`/`warmup_steps`/`optimizer_params` never passed to `model.fit()`
  in `scripts/train_two_tower.py` → sentence-transformers defaults apply
  (`WarmupLinear`, `warmup_steps=10000`, `lr=2e-5`). With 329,447 positive
  pairs, batch=64, 1 epoch → 5,147 total steps < 10,000 warmup_steps, so the
  LR schedule **never leaves warmup** for the entire training run (ends at
  ~51.5% of target LR, no decay phase ever reached).
- No seed anywhere in the training script (non-reproducible run-to-run).
- `evaluator=None` → `save_best_model=True` is a no-op; no validation
  monitoring, no best-checkpoint selection, no early stopping.
- Single epoch, batch_size=64 (63 in-batch negatives per anchor for MNRL),
  no hard-negative mining.
- One additional, previously-unconfirmed fact found during this pass:
  **embedding normalization is already consistent** between item and query
  encoding (`normalize_embeddings=True` on both paths) — `IndexFlatIP` is
  already a mathematically valid cosine index today, contrary to what one
  might assume needed fixing.

## 2. Changes Made

All additive; nothing pre-existing modified or deleted.

| File | Purpose |
|---|---|
| `retrieval/two_tower_training.py` | Reusable seed-setting, positive-pair loading, and a fixed dev `InformationRetrievalEvaluator` builder |
| `scripts/train_two_tower_v2.py` | Phase 1 "correct training": seed, query-level split, `warmup_ratio`-based schedule (via `SentenceTransformerTrainer`, the modern non-deprecated training API), real dev retrieval evaluator, best-checkpoint selection |
| `experiments/two_tower_v2/build_query_split.py` | Query-level train/dev split (no pair-level leakage) |
| `scripts/audit_false_negatives.py` | Phase 2 false-negative audit (data analysis only) |
| `scripts/mine_hard_negatives.py` | Phase 3 BM25 hard-negative mining stats (existing BM25 index, read-only) |
| `scripts/analyze_product_text_stats.py` | Phase 4 text-length/truncation stats (tokenizer only, no model training) |
| `scripts/benchmark_query_batching.py` | Phase 7.1 offline batched throughput vs. existing online loop |
| `scripts/benchmark_ann.py` | Phase 7.3 FlatIP vs. HNSW vs. IVFFlat benchmark (existing embeddings, no re-encoding) |
| `scripts/evaluate_two_tower_v2_full_catalog.py` | Re-indexes the full 1.2M-product catalog with the V1 checkpoint and evaluates it on the exact same unmodified 5,000-query full-catalog benchmark as V0 — the apples-to-apples comparison in Section 3/4 |

## 3. Experiment Matrix

| Version | Change | Recall@10 | Recall@50 | Recall@100 | MRR@10 | p99 latency | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| **V0** (historical baseline) | — | 0.1656 | 0.3653 | **0.4598** | 0.4712 | 56.0 ms (online, per-query loop) | `output/full_retrieval/retrieval_metrics.csv`, unmodified |
| **V1** (correct training: seed=42, warmup_ratio=0.10, query-level split, real dev evaluator, 3 epochs) | warmup/seed/split/evaluator fix, full 297,112-pair / 3-epoch run | 0.1650 | 0.3665 | **0.4649** | 0.4702 | not benchmarked online (only offline batched: build 14,036s / search 3.3s for 5000 queries) | **COMPLETE**. Training: 24,795.5s (6.9h) wall-clock, 3 epochs, best checkpoint = `checkpoint-9286` (end of epoch 2) selected by dev-proxy `dev_cosine_recall@100`. Full-catalog re-index (1,215,854 products, V1 checkpoint) + evaluation on the exact same unmodified 5000-query benchmark: `experiments/two_tower_v2/phase1_correct_training/full_run/full_catalog_eval/v0_vs_v1_comparison.json`. **Recall@100: 0.4598 → 0.4649 (+0.0051 absolute, +1.1% relative)**. Exact_recall@100: 0.5295 → 0.5329. MRR@10: 0.4712 → 0.4702 (−0.0010, essentially flat/slightly down). Hybrid RRF@100 (BM25 + V1): 0.5194 → **0.5301** (+0.0107 absolute, +2.1% relative) |
| **V2** (false-negative-safe batching) | unique-query-per-batch sampling | NOT RUN | NOT RUN | NOT RUN | NOT RUN | — | Design only; audit shows same-query collision rate is already only 0.19% under V0's plain random batching, so expected impact is small; not prioritized given compute budget |
| **V3** (BM25 hard negatives) | positive=E/S, hard_neg=I from BM25 top-100 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | — | Mining-stats-only run revealed **93.85% fallback rate** (median 0 of 4 target hard negatives found per query) — the naive mining strategy as specified mostly does not work on this dataset; would need redesigning before a training run is worth attempting |
| **V4** (best metadata) | title vs. title+brand+category vs. +bullets vs. +description | NOT RUN | NOT RUN | NOT RUN | NOT RUN | — | Text-stats-only run found 18.7% of products' current (title+desc+bullets) combined text exceeds max_seq_length=510 and gets truncated |

**No number in the "NOT RUN" cells above is fabricated or estimated** — they
are literally absent because those training runs did not execute.

### Additional real (non-training) benchmarks completed this pass

| Experiment | Result |
|---|---|
| Offline batched query encoding (5,000 queries, one batch call) | **1,014.9 qps** vs. 18.4 qps for the existing per-query loop (~55x) |
| ANN benchmark, HNSW (M=32) vs. FlatIP | 3254 qps vs. 12 qps (~271x), but task_recall@100 0.4263 vs. 0.4598 (-7.3% relative) |
| ANN benchmark, IVFFlat (nlist=4096, nprobe=32) vs. FlatIP | 964 qps vs. 12 qps (~80x), but task_recall@100 0.4187 vs. 0.4598 (-8.9% relative) |

## 4. Best Model

**V1 is now the best model, by a modest and genuinely measured margin.**
Full-catalog Recall@100 (same unmodified 5,000-query benchmark, same
ground truth, same K=100): V0 = 0.4598, V1 = 0.4649 (+0.0051 absolute,
+1.1% relative). Hybrid RRF@100 (BM25 + Two-Tower) improves by more:
0.5194 → 0.5301 (+0.0107 absolute, +2.1% relative). MRR@10 is essentially
flat (0.4712 → 0.4702, a 0.0010 decrease).

- Base encoder: `sentence-transformers/msmarco-distilbert-base-v3` (unchanged)
- Training pairs: 297,112 (90% query-level split of the original 329,447;
  10% held out as `dev_queries.txt`, zero query overlap with train)
- Positive definition: E/S, us locale, small_version=1, split=train (unchanged from V0)
- Negative strategy: in-batch only (unchanged from V0 — no hard negatives, no safe batching)
- Batch size: 64 (unchanged)
- Loss: MultipleNegativesRankingLoss (unchanged)
- LR / scheduler: 2e-5, linear, **warmup_ratio=0.10 → resolved warmup_steps=1,392** (of 13,926 total steps) — the actual fix, replacing the old fixed warmup_steps=10,000 default
- Epochs: 3 (vs. V0's 1)
- Seed: 42 (explicit; V0 had none)
- Best checkpoint: `experiments/two_tower_v2/phase1_correct_training/full_run/checkpoints/checkpoint-9286` (end of epoch 2 of 3; epoch 3 slightly regressed on the dev proxy's recall@100, 0.864 → 0.863)
- Product representation: unchanged (title+description+bullet_point, no separators)
- Embedding dimension: 768 (unchanged)

This result **cannot cleanly attribute the improvement to any single one**
of (warmup fix / seed / longer training-3-epochs-vs-1 / having a real
validation signal) — all four changed together in V1, by design (Phase 1's
own instructions were to fix training correctness as one bundle before
isolating further variables in later phases). The honest conclusion is:
**the bundle of correctness fixes, taken together, produced a small but
real full-catalog Recall@100 gain**, not that any specific one of them did.

## 5. Retrieval System

- Index type: `faiss.IndexFlatIP` (exact), unchanged for the production
  path; `HNSW`/`IVFFlat` benchmarked as alternatives (Section 3), not
  deployed.
- Catalog size: 1,215,854 US-locale products.
- Embedding dimension: 768 (unchanged; dimension ablation not run).
- Batch vs. online encoding: existing online single-query loop unmodified;
  a batched offline alternative benchmarked (~55x throughput) but not
  substituted into `scripts/run_full_tt_retrieval.py`.
- Latency: see Section 3 / `phase7_serving/`.
- Memory: FlatIP embeddings ≈ 3.56 GB in memory for 1.2M × 768 float32
  vectors; HNSW/IVFFlat indices measured at 3.58–3.88 GB on disk (Section 7
  report).

## 6. What Actually Improved Quality

- **The Phase 1 correctness-fix bundle (warmup_ratio, seed, query-level
  split, real dev evaluator, 3 epochs instead of 1) improved full-catalog
  Recall@100 by +0.0051 absolute (+1.1% relative), and improved the
  BM25+Two-Tower RRF hybrid by +0.0107 absolute (+2.1% relative).** This is
  a measured, real result — not the size of improvement one might hope for
  given how broken the original warmup schedule was (never leaving warmup),
  but a genuine, non-zero gain. MRR@10 did not improve (essentially flat,
  slightly down).
- Safe batching (false-negative fix): **not tested with a training run** —
  the audit shows its target problem (same-query collisions) is already
  rare (0.19%), so no quality claim either way.
- Hard-negative mining: **not tested** — the mining strategy itself doesn't
  work as specified on this dataset (93.85% fallback rate), so there is no
  quality result to report, positive or negative.
- Metadata/representation changes: **not tested** — no training run
  attempted.
- ANN indices (HNSW/IVFFlat): these **reduce** task Recall@100 by 7-9%
  relative in exchange for 80-270x query throughput — a real, measured
  quality *cost*, not an improvement, reported for what it is (a
  latency/quality trade-off, not a win).

## 7. Remaining Limitations

- No click logs / real user interaction data anywhere in this project —
  everything is offline ESCI relevance judgments.
- ESCI labels are themselves incomplete (per the pre-existing
  `label_coverage_summary.md` audit): most retrieved items in the
  full-catalog setting are unlabeled, not confirmed-irrelevant.
- Offline-only; no production traffic, no A/B test, no online latency SLO
  validated against real query load.
- Domain mismatch: Two-Tower fine-tuning only ever saw `small_version==1`
  query-title-description-bullet text; the deployed/evaluated index spans
  the full 1.2M-product catalog.
- Limited compute in this session (single MPS-backed machine, no CUDA) is
  the direct cause of Phases 2/3/4/5/6 and the semantic_score ablation not
  completing full training-based comparisons — the code exists and is
  ready to run given more time/hardware (see each phase's own REPORT.md for
  the exact command and expected cost).
- Phase 3's finding (93.85% hard-negative mining fallback rate) means the
  literal mining strategy specified in the task brief needs to be redesigned
  before V3 is worth training at all.

## 8. Resume-safe Findings

Only claims backed by a file that actually exists in this repo, as of this
report:

1. Identified and root-caused a training bug where the Two-Tower dense
   retriever's LR schedule never left its warmup phase for the entire
   training run (5,147 actual steps vs. a 10,000-step default warmup),
   ending training at ~51.5% of the target learning rate with no decay
   phase — traced to unset `scheduler`/`warmup_steps` parameters in the
   training entry point (`experiments/two_tower_v2/baseline/README.md`).
2. Benchmarked three FAISS index types over a real 1.2M-product catalog and
   quantified the latency/recall trade-off: HNSW gives ~271x query
   throughput over exact search at a 7.3% relative Recall@100 cost;
   IVFFlat gives ~80x throughput at an 8.9% relative cost
   (`phase7_serving/ann_benchmark.csv`).
3. Found that switching from a per-query to a batched query-encoding path
   for the 5,000-query offline retrieval benchmark improves throughput by
   ~55x (18.4 qps → 1,014.9 qps) with no change to model weights or index
   contents (`phase7_serving/batching_benchmark.json`).
4. Audited the in-batch-negative training setup and found the previously
   assumed risk (same-query collisions) affects only 0.19% of training
   pairs, while a different, larger risk (a training-positive product also
   being a true-relevant item for a different query) affects 22.5% of pairs
   (`phase2_false_negative/false_negative_audit.json`).
5. Found that 18.7% of the product catalog's current title+description+
   bullet-point text exceeds the encoder's 510-token limit and is silently
   truncated, with bullet-point (attribute) content most at risk of being
   cut (`phase4_metadata/product_text_stats.json`).
6. Fixed the identified training-correctness issues (warmup schedule, seed,
   query-level validation split, real retrieval-based checkpoint selection)
   and measured a **full-catalog Recall@100 improvement from 0.4598 to
   0.4649** (+1.1% relative) on the same unmodified 5,000-query benchmark,
   with the BM25+Two-Tower hybrid (RRF) improving from 0.5194 to 0.5301
   (+2.1% relative) (`phase1_correct_training/full_run/full_catalog_eval/v0_vs_v1_comparison.json`).
   This improvement is attributed to the fix bundle as a whole, not to any
   single change in isolation (see Section 4).
