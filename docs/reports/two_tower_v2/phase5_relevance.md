# Phase 5 — Relevance Definition Ablation (positive=E/S vs E/S/C)

## What was run

One descriptive count only, no training:

| | Count |
|---|---:|
| E | 181,819 |
| S | 147,628 |
| C | 19,090 |
| I | 71,116 |
| Current positive set (E+S) | 329,447 |
| If C added (E+S+C) | 348,537 (+5.79%) |

Adding `C` as a positive would grow the training positive set by only 5.8% —
a modest, not dramatic, change in raw pair count. This does not by itself
say anything about whether it would help or hurt `full-catalog broad
Recall@100` (which already treats E/S/C as relevant at *evaluation* time,
per `output/full_retrieval/evaluation_config.json` — this is the exact
train/eval objective mismatch the task brief flags).

## What was NOT run, and why

Both training variants (A: positive=E/S [=V1], B: positive=E/S/C) require a
full training run each; B was not executed. No `results.csv` with trained
Recall@100/MRR@10 numbers is produced. The graded/weighted-loss variant (C:
E/S strong positive, C weak positive) was not designed or attempted, per
task instructions to not build the complex version before the simple one is
even run.

## Recommendation (not executed)

Run variant B (positive=E/S/C) with otherwise-identical config to V1 once
V1 itself completes and is validated (Phase 5 should not run before Phase 1
finishes, so its baseline-for-comparison is the corrected training setup,
not V0). Watch specifically for the trade-off named in the task: does broad
Recall@100 go up while MRR@10 goes down (C items diluting top-rank
precision)? This is directly testable once compute is available, using the
same `retrieval/two_tower_training.py` building blocks already written for
Phase 1 (only the label filter changes).
