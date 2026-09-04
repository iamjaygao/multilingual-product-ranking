# Pre-registered prediction — full_run (xlm-roberta-base, 28,733 q, 2 epochs)

Written **2026-09-03, while epoch 1 is still at batch ~31,500/41,463**, i.e. before
any full_run checkpoint exists and before any full_run dev evaluation has been run.
Recorded here so the outcome cannot be rationalised after the fact — the same
discipline Phase 1.1 used, and the same trap it fell into (a prediction that landed
within 0.0006 of the observed value purely because two large opposing effects
cancelled).

## Evidence this is built on

Same dev set (5,071 queries / 118,231 rows), same authoritative scorer,
`candidate_pool=task1_small_v1`:

| model | full | @10 | @20 |
|---|---:|---:|---:|
| LambdaMART (frozen Task 1 baseline) | 0.8430 | 0.7210 | 0.7950 |
| CE `medium_run`: 2,000 q (7% of train), 1 epoch | 0.8607 | 0.7517 | 0.8165 |
| delta | +0.0177 | +0.0307 | +0.0216 |

`full_run` scales training data 14.4× (2,000 → 28,733 queries) and adds a second epoch.

## Prediction (dev, full_run best checkpoint)

| metric | point estimate | 80% band |
|---|---:|---|
| **NDCG@10** | **≈ 0.775** | **0.762 – 0.787** |
| NDCG full-list (headline) | ≈ 0.877 | 0.868 – 0.886 |
| NDCG@20 | ≈ 0.833 | 0.824 – 0.842 |

Secondary predictions:
- epoch 2 ≥ epoch 1, but by a small margin (≤ +0.005 @10). Mild overfitting risk at
  lr 2e-5 on 2 epochs is real; if epoch 2 is *worse*, epoch 1 is the pick.
- CE beats LambdaMART on all three cutoffs and in all three locales.
- The JP full-list ≥ US full-list inversion seen in `medium_run`
  (jp 0.8661 vs us 0.8591) persists, while JP stays worst at @10.

## Reasoning

1. The 2,000-query model is **already well-formed**, not undertrained: its predicted
   class mix (E 48%, S 37%, C 2.6%, I 12%) is close to the true train marginal
   (E 43.7%, S 34.2%, C 5.2%, I 16.9%), and its scores span 0.081–0.894. So the
   remaining gain from 14× more data is refinement, not a phase change — this is
   the flat part of the curve, not the steep part.
2. Published anchors bracket it: the competition's own published Task 1 baseline is
   0.8503 full-list, and the esci-data README cross-encoder baseline is 0.83 under
   the *swapped* S/C gain, which our measured swapped-gain delta (+0.0186 to +0.0220)
   maps to roughly 0.849 under competition gain. A well-trained xlm-roberta-base
   should sit above both but well below the 0.9043 winner (a much larger ensemble).
3. Better models show a smaller full↔@10 gap on this benchmark (LambdaMART 0.122,
   CE medium 0.109). Projecting full ≈ 0.877 with a gap ≈ 0.103 gives @10 ≈ 0.774.

## Why the band is wide

**I have exactly one usable scaling point.** `smoke_test` (200 q) is unusable as a
second point: it was evaluated on only 50 dev queries and was degenerate
(argmax = E on all 1,167 rows, NDCG below the random floor). So the slope is a prior,
not a fit.

## Known caps on the upside

- **48.4% of train pairs are truncated** at `max_length=256`. Roughly half the product
  text never reaches the model. This is the single most likely reason the result lands
  at the low end of the band.
- xlm-roberta-base is small; no hard negatives, no class weighting, no length bucketing.

## Falsification

If the observed @10 lands **outside 0.74 – 0.80**, this prediction was wrong for a
reason worth finding — check for a train/eval mismatch, a checkpoint-selection bug, or
an unnoticed change in the eval pool, before accepting the number.
