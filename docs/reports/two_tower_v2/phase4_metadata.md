# Phase 4 — Product Representation / Metadata Ablation

## What was run

`scripts/analyze_product_text_stats.py` — text-length/truncation statistics
only, using the model's own tokenizer (`models/two_tower_finetuned`), on a
fixed 20,000-product random sample of the US-locale catalog (seed=42). No
training. Full results: `product_text_stats.json`.

## Findings

| Field | Mean tokens | Median | P90 | P99 | % exceeding max_seq_length (510) |
|---|---:|---:|---:|---:|---:|
| title | 27.8 | 26 | 45 | 60 | 0.0% |
| description | 123.4 | 17 | 405 | 622 | 3.9% |
| bullet_point | 130.7 | 104 | 293 | 457 | 0.36% |
| **combined (current V0: title+description+bullets)** | **277.8** | **199** | **669** | **977** | **18.7%** |

Description has a highly skewed distribution (median 17 tokens, but P90=405)
— most products have short/no description, but a long tail is very long.
**18.7% of products' combined text exceeds the 510-token max_seq_length**
and gets end-truncated by the tokenizer. Since the current construction is
`title + description + bullet_point` (title first), title itself is never
truncated, but on the ~1-in-5 products where the combined text is too long,
bullet-point content (which carries concrete, often query-relevant attribute
info — size, color, capacity, etc.) is the part most likely to be cut, since
it comes last and description tends to intervene first.

## What was NOT run, and why

The actual metadata-representation ablation (V3-A title-only / V3-B
title+brand+category / V3-C +bullets / V3-D +description — Recall@10/50/100,
MRR@10) requires **four separate full training runs**, each comparable in
cost to Phase 1's ~5-hour run (this session already has one such run
in-flight). None of the four were executed. `product_brand` is available
directly on the products table; `category` is only available via the
supplementary `esci-s_dataset` (esci_s_products.parquet), which covers 92.0%
of the US-locale catalog (not 100% — some products would have `category`
missing under variant B/C/D, which the ablation script would need to handle
with a placeholder, e.g. `"unknown"`, consistent with how
`reranking/advanced_features.py` already treats missing category elsewhere
in this repo).

## Recommendation (not executed)

Given the truncation finding above, a lower-cost first experiment than the
full 4-way ablation would be: keep `title + bullet_point` but move
`description` to the END of the concatenation (or drop it) so that
attribute-bearing bullet content is truncated less often. This is a
one-line change but still requires a training run to validate — not run
here.
