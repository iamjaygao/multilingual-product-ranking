# Phase 6 — Batch / Loss Ablation

## Status: NOT RUN

Nothing in this phase was executed — no sanity check, no reduced-scale
proxy. This phase explicitly depends on Phase 1/2/3 being "stable" first
(per task instructions: "只有前面稳定以后再做"), and Phase 1's own full run
had not finished within this session's compute budget (see
`phase1_correct_training/full_run/`, in progress). Running batch-size sweeps
(64/128/256) on top of an unvalidated V1 would confound "did batch size
help" with "was V1 even trained correctly."

## What would be required (scaffolding only, not written)

- 3 training runs (batch=64/128/256), each ~comparable cost to Phase 1
  (more, at batch=256, due to larger per-step compute and MPS memory
  pressure — untested whether 256 even fits in memory on this hardware).
- A clear write-up of "effective negative pool" (batch_size - 1 per anchor
  for plain MNRL) vs GPU/unified-memory usage vs wall-clock training time vs
  resulting Recall@100 — explicitly NOT conflating plain gradient
  accumulation (which does not increase the in-batch negative pool) with
  larger physical batch size (which does), per task instructions.
- No code for this exists yet in `retrieval/two_tower_training.py` beyond
  what Phase 1 already built (`per_device_train_batch_size` is already a
  config knob in `scripts/train_two_tower_v2.py`, so the batch-size sweep
  itself is a one-line config change once time/compute is available — the
  gap is compute budget, not code).
