# Phase 3 — Hard Negative Mining

## What was run

`scripts/mine_hard_negatives.py` — BM25 hard-negative mining STATS ONLY, on a
fixed random sample of 2,000 / 18,800 train queries (seed=42), reusing the
existing US-locale BM25 index (`output/full_retrieval/bm25s_index_us`) via
`retrieval/bm25.py`'s `load_bm25_index`/`search_bm25_global` (read-only, not
reimplemented). Full results: `hard_negative_stats.json`.

## Finding: the naive mining strategy mostly fails on this dataset

Target: 4 hard negatives per query, sourced from `esci_label == 'I'` within
BM25's top-100 (per task instructions — `C` deliberately excluded, neither
positive nor negative).

| Metric | Value |
|---|---:|
| Queries sampled | 2,000 |
| Mean hard negatives found (of 4 target) | 0.553 |
| Median hard negatives found | 0 |
| Queries with zero hard negatives found | 1,492 / 2,000 (74.6%) |
| **Fallback rate** (< 4 found) | **93.85%** |
| Mean rank of mined negatives (when found) | 27.8 |

**This is a real, load-bearing finding, not a footnote**: for the large
majority of queries, BM25's top-100 simply does not contain 4
`I`-labeled products to mine. Catalog-wide label counts for this train
subset: E=181,819, S=147,628, C=19,090, **I=71,116** — `I` is not rare
overall, but it is rare *specifically within BM25's own top-100 for a given
query*, which makes sense: BM25 retrieves lexically-similar items, and the
ESCI labeling process is itself downstream of a retrieval system, so most
BM25-highly-ranked items were already judged relevant (E/S/C) rather than
`I`. Mining strictly "BM25 top-100 ∩ I-label" is not a workable hard-negative
source as literally specified, at least not at n=4/query.

## What was NOT run, and why

- The actual hard-negative-augmented training run (V3) was **not executed** —
  it would require redesigning the mining strategy first (given the 93.85%
  fallback rate above, e.g. widening the BM25 rank window, mining across a
  larger multi-query pool of unlabeled items with a confidence filter, or
  reconsidering what counts as a valid hard negative), and then a full
  multi-hour training run comparable in cost to Phase 1. Neither was in
  budget for this session.
- No loss/implementation choice (triplet, margin, or explicit-negative MNRL)
  was evaluated, since there is no trained V3 to report.

## Recommendation (not executed)

Before running V3, fix the mining strategy: e.g. treat *unlabeled* BM25
top-100 items (locale-consistent, not present in the ESCI judged set at all
for that query) as weak/soft negatives instead of requiring an explicit `I`
label, since requiring `I` specifically is what drives the 93.85% fallback
rate. This is a design change to validate on a small sample before any
training run, not something this pass implements.
