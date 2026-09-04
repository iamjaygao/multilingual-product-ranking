# Phase 2 — False-Negative Audit + Safe Batching

## What was run

`scripts/audit_false_negatives.py` — pure data analysis, no model loading, no
training. Full results: `false_negative_audit.json`.

## Findings

- 99.1% of the 20,888 train-split positive-having queries have **more than
  one** E/S-labeled positive product (mean 15.8, median 15, p95 37, max 130
  positives per query).
- **Same-query in-batch collision rate under the current (V0) random
  batching is only 0.19%** (measured over 5 independent random shuffles at
  batch_size=64) — i.e. despite most queries having many positives, the
  chance that two pairs from the *same* query land in the *same* random
  64-item batch is low, because 64 is small relative to the 20,888-query
  universe each batch draws from. This is the risk Phase 2's "safe batching"
  fix directly targets, and the audit shows its current real-world impact is
  small.
- **Cross-query semantic false-negative risk is larger**: 22.5% of training
  pairs have their positive product ALSO ESCI-labeled E/S/C for at least one
  *different* query (mean 0.48 other relevant queries per such product).
  This is a distinct problem from same-query collisions — it means a
  meaningfully large fraction of "in-batch negatives" are, for some other
  query in the batch, not actually a valid negative. Unique-query-per-batch
  sampling (below) does **not** address this; it only eliminates the
  same-query case.

## Safe batching (design, not benchmarked with a full training run)

Simplest fix per task instructions: one positive pair per unique query per
epoch (if a query has >1 positive, pick one uniformly at random, fixed seed,
independently each epoch so different positives get used across epochs).
This guarantees zero same-query collisions by construction (each query
contributes exactly one pair to the epoch's dataset). Not implemented as a
custom PyTorch sampler in this pass.

## What was NOT run, and why

A full V1-vs-V2 (unique-query-safe) training comparison was **not run**.
Given the measured same-query collision rate is only 0.19% (i.e. the problem
this fix targets is already small), and given Phase 1's full training run
already consumes this session's compute/time budget (see
`phase1_correct_training/full_run/`, in progress, ETA ~5h), running a second
multi-hour training job to chase a fix expected to move the needle by a
fraction of a percent was not prioritized. **No results.json with trained
metrics is provided for Phase 2** — only the audit. If pursued later, the
one-line change is a custom `Sampler`/pre-epoch pair-selection step in
`scripts/train_two_tower_v2.py`, gated by `cfg["safe_batching"]`.

## Answer to the phase's core question

Same-query false negatives: **not currently a significant practical
problem** (0.19% collision rate). Cross-query semantic false negatives are
**more prevalent** (22.5% of pairs) but require a different fix (e.g.
deduping relevant products across the whole batch, not just within a query)
than what "unique query per batch" provides. This nuance is the main
deliverable of this phase.
