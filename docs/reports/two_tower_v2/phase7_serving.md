# Phase 7 — Serving / ANN Optimization

All three sub-experiments below were **actually run** (no model training
required — these use the existing V0 encoder/embeddings read-only).

## 7.1 — Batched query encoding vs. the existing single-query loop

`scripts/benchmark_query_batching.py` → `batching_benchmark.json`.

| | Method | Throughput |
|---|---|---:|
| Offline batched (this experiment) | one `model.encode(5000 queries, batch_size=256)` + one batched `faiss_index.search()` | **1,014.9 qps** (0.99ms/query effective) |
| Online single-query loop (existing, unmodified `scripts/run_full_tt_retrieval.py` / `tt_latency.json`) | `model.encode([q])` + `search()` per query, sequential | 18.4 qps (54.4ms/query mean) |

Batching the 5,000-query offline evaluation gives a **~55x** throughput
improvement over the existing per-query loop. `scripts/run_full_tt_retrieval.py`
itself was **not modified** — its output remains the "online single-query"
latency reference. The two numbers measure different things (batch
throughput vs. one-at-a-time call latency) and are reported side by side,
not divided into one "speedup" figure, per instructions not to overstate
precision.

## 7.2 — Embedding normalization audit

**Confirmed already correct, no fix needed** (see
`baseline/README.md` for detail): both `encode_texts()` and
`search_tt_global()` in `retrieval/two_tower.py` call `model.encode(...,
normalize_embeddings=True)`. `IndexFlatIP` over L2-normalized vectors is
mathematically equivalent to cosine similarity, on both the item and query
side, consistently.

## 7.3 — ANN benchmark (FlatIP exact oracle vs. HNSW vs. IVFFlat)

`scripts/benchmark_ann.py` → `ann_benchmark.csv`. Built directly from the
vectors reconstructed out of the existing `tt_index_us.faiss` (no
re-encoding of the 1.2M-product catalog).

| index_type | params | build_time_sec | index_size_mb | p50 (ms) | p99 (ms) | qps | task_recall@100 | ANN_recall_vs_flat |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IndexFlatIP (oracle) | — | 1.5 | 3562 | 85.0 | 105.0 | 12.0 | 0.4598 | 1.000 |
| HNSW | M=32, efC=40, efSearch=64 | 32.8 | 3878 | 0.26 | 1.00 | 3254.3 | 0.4263 | 0.861 |
| IVFFlat | nlist=4096, nprobe=32 | 43.2 | 3583 | 0.81 | 2.91 | 963.8 | 0.4187 | 0.927 |

Two genuinely different metrics, not conflated:
- **ANN_recall_vs_flat** (agreement with the FlatIP oracle's own top-100):
  HNSW 86.1%, IVFFlat 92.7%.
- **task_recall100** (real Recall@100 against ESCI ground truth, broad
  E/S/C): drops from 0.4598 (Flat) to 0.4263 (HNSW, -7.3% relative) and
  0.4187 (IVFFlat, -8.9% relative).

**Real trade-off, not free**: HNSW is ~332x faster per query and IVFFlat
~80x faster, but both cost several points of absolute Recall@100 at these
untuned parameter settings. Neither parameter grid was tuned for a
recall/latency Pareto frontier — these are single, reasonable default
configurations, not the best achievable operating point for either index
type. IVF-PQ (further compression) was not attempted — out of scope for this
pass.

Note on `query_p50_ms` methodology: these FlatIP numbers (85ms) are
**search-only** (pre-encoded query vectors, single-threaded FAISS inner
product over 1.2M x 768 floats), and are not directly comparable to the
official `tt_latency.json` (54.4ms mean), which times encode+search
together per call under a different threading/runtime context. Not
reconciled here — reported as measured.

## 7.4 — Embedding dimension ablation

**Not run.** Testing 768/384/256 requires either retraining with a
projection head or truncating an already-trained embedding and re-indexing
the full 1.2M-vector catalog per dimension — neither was attempted in this
pass.
