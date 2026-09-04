# Published strong-baseline reproduction plan (§14)

**PLAN ONLY. Nothing here was trained or executed.** Awaiting confirmation.

---

## Candidates surveyed

| team / paper | Task 1 public | Task 1 private | code released | notes |
|---|---:|---:|---|---|
| ZhichunRoad (winner) | 0.9057 | **0.9043** | partial | multi-task pretraining + large ensemble |
| "A Boring-yet-effective Approach" (9th) | 0.9012 | 0.9007 | described in paper | mMARCO pretrain → task finetune; single-model numbers reported |
| "Semantic Alignment System" (arXiv 2208.02958) | — | — | partial | primarily Task 2/3 retrieval-oriented |
| ChengHSUHSU/KDD_Cup2022 (3rd-party) | — | — | yes | Tier 3; unverified provenance |

## Recommendation: the 9th-place "Boring-yet-effective" recipe

**Not the winner.** Deliberately.

### 1. Which is easiest to reproduce at high fidelity

The 9th-place paper is the best fidelity-per-unit-effort by a wide margin:

- It reports a **clean single-model ablation ladder**, which is exactly what a
  reproduction needs as a checkpoint: base multilingual model **0.864** → after task
  finetuning **0.890** → best submission **0.9012** (nDCG@20). We can verify our
  reproduction at each rung instead of only at the end.
- It is only **0.0036 behind first place**, so it captures essentially all of the
  achievable signal.
- Its pipeline is two stages (mMARCO pretrain → Task 1/2 finetune) with no large
  ensemble, no multi-task loss juggling, no undisclosed internal data.
- The winner, by contrast, gets its final margin from **multi-task pretraining plus a
  model ensemble** (their own numbers: ensemble lifts 0.9022 → 0.9057 public,
  0.9015 → 0.9043 private). Reproducing an ensemble means reproducing several models and
  their blend weights — high cost, low incremental insight, and the single-model
  contribution is roughly the same as 9th place.

### 2. Published scores

| | nDCG@20 |
|---|---:|
| base multilingual model, zero-shot | 0.864 |
| + Task 1/2 finetuning | 0.890 |
| best submission (single system) | **0.9012** public / 0.9007 private |

**All three are on the competition public/private split and are therefore NOT comparable
to our DEV numbers** — see `leaderboard_comparability.md`. They are useful only as
*internal rungs* to check that our reproduction behaves like the paper's.

### 3. Cost estimate

Two stages. Stage 1 is the expensive one.

| stage | data | rough cost |
|---|---|---|
| mMARCO multilingual pretrain | ~millions of pairs | **the blocker** — days on a single GPU; the paper leans on published mMARCO checkpoints |
| Task 1 finetune | 781,638 official Task 1 train rows, 1–2 epochs | ~4–10 h on one A100/L40S |

**Strong recommendation: skip stage 1 by starting from an already-published mMARCO
multilingual cross-encoder checkpoint** rather than pretraining. This is what makes the
recipe tractable at all, and it is what the paper itself effectively does. If we
pretrain from scratch the project becomes GPU-weeks, not GPU-hours.

### 4. Can it run on this Mac?

**Stage 2: marginally. Stage 1: no.**

Grounded in measured numbers from this machine (M4 Max, 68.7 GB unified):
- Our V2 (xlm-roberta-base, 278M, max_length 256) trained 663K rows in **7.75 h** at
  ~66 rows/s.
- A strong recipe wants a **large** backbone (~560M) at **max_length 512** — call it 4–6×
  the per-row cost of V2, so **30–45 h per epoch** on 781K rows. Two epochs ≈ 3 days.
- MPS also has no FlashAttention and no working bf16 training path we have validated.

So: technically possible, practically not.

### 5. Hardware recommendation

| option | VRAM | fit | est. stage-2 time | why |
|---|---|---|---:|---|
| **L40S 48 GB** | 48 GB | yes, with grad checkpointing | **~6–10 h / epoch** | best value; ample for a 560M cross-encoder at 512 tokens |
| A100 80 GB | 80 GB | comfortable | ~4–7 h / epoch | only needed for larger batch or a >1B backbone |
| H100 80 GB | 80 GB | comfortable | ~2–4 h / epoch | fastest; justified only if we iterate repeatedly |

**Recommended: a single L40S 48 GB.** Estimated 1–2 epochs → **~8–20 GPU-hours** for
stage 2. At typical on-demand rates (~$1–2/GPU-h) that is roughly **$10–40** for one full
reproduction run — cheap enough that the decision should turn on whether the *result* is
worth having, not on the cost.

### 6. Honest assessment of what this buys us

Given `leaderboard_comparability.md`, reproducing a strong 2022 recipe **cannot** give us
a leaderboard-comparable number — the private split is unrecoverable. What it *would*
give us:

1. **A credible strong reference on our own frozen DEV**, next to the official baseline
   and V2, all on one pool with one scorer.
2. **A validity check on our whole pipeline**: if a published 0.90-class recipe lands far
   below expectation on our DEV, that indicts our setup, not the recipe.
3. **A realistic ceiling** for judging whether further modern-reranker work is worth it.

What it will **not** give us: any right to say "we beat the KDD Cup winner."

### 7. Open question for you

There is a cheaper path to most of benefit #3. Instead of reproducing a 2022 recipe, take
the strongest *current* multilingual cross-encoder (e.g. BGE-reranker-v2-m3, already
benchmarked zero-shot here at 0.8223 @20 on the 300-subset) and fine-tune **it** on Task 1.
A 2026 backbone finetuned on in-domain data will very likely exceed a 2022 recipe, and it
sits on the path we actually care about, whereas the 2022 reproduction is a
historical-validation exercise.

**Two coherent choices:**
- **(a) Validation first** — reproduce the 9th-place recipe to prove the pipeline can
  reach ~0.90-class quality, then move to modern models. Costs ~$10–40 and ~1 day.
- **(b) Skip to modern** — accept the official baseline + V2 as sufficient reference
  points and put the GPU budget straight into fine-tuning a modern reranker.

My recommendation is **(b)**, because the reproduction's main deliverable (a
leaderboard-comparable number) is already ruled out by the split finding, and its
secondary deliverable (a pipeline sanity check) is largely satisfied by the fact that our
candidate pool is bit-identical to official and our NDCG matches an independent
implementation to 1.1e-16.

Awaiting your decision. Nothing will be launched either way without it.
