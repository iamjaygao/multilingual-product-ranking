```
==================================================
COMPETITION ALIGNMENT STATUS
==================================================

Dataset:                PASS
Task1 small_version:    PASS
Candidate pool:         PASS
Query grouping:         PASS
Split:                  PASS
Gain:                   PASS
NDCG implementation:    PASS
Official baseline:      PASS  (reproduced, 3 locales)
TEST untouched:         TRUE

Overall:
    YELLOW
==================================================
```

**Gate 0 = PASS on all ten items.** The benchmark is real: our candidate pool is
bit-identical to official Task 1, our gain matches the official training script, and our
NDCG matches an independent implementation to 1.1e-16.

**Overall is YELLOW, not GREEN, for one reason only:** the competition's private
leaderboard split was never released and is unrecoverable, so no number we produce can
ever be compared to 0.9043. That is a property of the released data, not a defect we can
fix.

---

# Headline results — official baseline reproduced on our competition-aligned DEV

Same 5,071 queries / 118,231 rows, same official candidate pairs, same scorer, same gain.

| Model | full | **NDCG@20** | @10 |
|---|---:|---:|---:|
| Amazon official baseline (reproduced, 3 locales) | 0.8521 | **0.8071** | 0.7358 |
| V2 Cross-Encoder epoch_2 | 0.8834 | **0.8456** | 0.7924 |
| **V2 − official** | +0.0313 | **+0.0385** | +0.0566 |

Paired bootstrap @20 (n=10,000, seed=42): **Δ +0.0385, 95% CI [+0.0348, +0.0421],
p < 0.0001, W/L/T 3229/1696/146.**

### Per locale (@20)

| Model | us (3,133) | es (845) | jp (1,093) |
|---|---:|---:|---:|
| Official baseline | 0.8345 | 0.7823 | 0.7477 |
| V2 | 0.8558 | 0.8486 | 0.8140 |
| **V2 − official** | **+0.0213** | **+0.0662** | **+0.0663** |

### The US cross-encoder branch, listed separately as you asked

The official baseline is **not** three comparable models. Only US uses a cross-encoder;
ES and JP use `sentence-transformers/multi-qa-mpnet-base-dot-v1`, which is an
**English-only bi-encoder** (`multi-qa` = trained on multiple QA datasets, *not*
multilingual). Applying an English bi-encoder to Spanish and Japanese is the single
biggest reason the baseline is weak.

| branch | architecture | @20 | V2 − branch |
|---|---|---:|---:|
| **US only** | `cross-encoder/ms-marco-MiniLM-L-12-v2`, MSE regression on gain | **0.8345** | +0.0213 |
| ES | English bi-encoder, CosineSimilarityLoss | 0.7823 | +0.0662 |
| JP | English bi-encoder, CosineSimilarityLoss | 0.7477 | +0.0663 |

On the one architecturally sensible branch, the gap narrows from +0.0385 to **+0.0213**.
The protocol was **not** modified to make the baseline look stronger; this is the same run,
sliced.

---

# The eight required answers

### 1. Why does the current pool show max = 94?

**It is not a defect.** The official Task 1 data itself contains queries with far more
than 40 judged candidates. Measured from the raw parquet with only `small_version == 1`:

```
candidates/query   mean 23.15   p50 16   p90 40   p95 40   p99 48   MAX 188
train MAX 188      test MAX 95
queries with >40 candidates: 1,219 / 48,300 = 2.52%
```

Our DEV's 94 is simply the largest value among the 5,071 DEV queries drawn from a TRAIN
pool whose true max is 188. The "up to 40" figure is prose in the esci-data README
(*"a list of up to 40 potentially relevant results"*); it holds for 97.5% of queries and
matches p90/p95 exactly, but it is not enforced in the released data.

All four hypotheses were tested and refuted: `query_id` never spans locales (0 cases,
so `groupby(query_id)` ≡ `groupby([query_id, locale])`); zero duplicates on either key;
the products table PK is unique and the merge is `validate='many_to_one'`; and no
BM25/Two-Tower/retrieval candidates were ever added. **No silent repair was applied** —
truncating to 40 would have discarded 1,219 official queries' judgments and inflated NDCG
by shrinking IDCG denominators. Full detail: `candidate_pool_root_cause.md`.

### 2. Is `task1_small_v1` the same as the official Task 1 pool?

**Yes — bit-identical.**

```
current train+dev              781,638 rows / 33,804 queries
official Task 1 TRAIN          781,638 rows / 33,804 queries
example_id sets identical      True   (0 in either direction)
(query_id, locale, product_id) identical      True
label mismatches / 781,638     0
rows with small_version != 1   0
extra pairs beyond official    0
```

DEV specifically: its 118,231 pairs *are* the official pairs for those 5,071 query_ids,
unmodified. No reconstruction was needed and no `competition_dev_v1` is required.

### 3. Is V2's 0.845605 a competition-comparable score?

**No — and it never can be.** It is a legitimate, reproducible number on the official
candidate pool with the official gain, but it is measured on **a 15% query-level holdout
carved from official TRAIN**, not on any competition evaluation set. The correct phrasing
is: *"0.8456 NDCG@20 on our frozen 5,071-query DEV holdout, official Task 1 pool,
competition gain."* It is not a leaderboard score.

### 4. Is our scorer numerically identical to the official evaluator?

**Identical to the official gain definition and to an independent reference; NOT verified
against the official evaluator.**

`metric_parity = PASS` — 13/13 cases, **max |Δ| 1.1e-16** over 900 randomised
comparisons against a pure-Python reference sharing no code with production (tolerance
1e-10). All eight required cases pass, including IDCG=0 and ties.

The complication, and it is real: **the official repo contains two contradictory gain
maps.** `ranking/train.py:49` gives E=1.0/S=0.1/C=0.01/I=0.0 (matches the competition
spec). `ranking/prepare_trec_eval_files.py` composed with its Terrier flags gives
E=1.0/**S=0.01/C=0.1**/I=0.0 — S and C swapped. We follow `train.py`, because that is the
script the baseline's weights were produced with and it preserves the ESCI ordering
(a Substitute must outrank a Complement). The swap is worth **0.019–0.030 NDCG** on our
own models, so this is not academic.

**UNKNOWN:** which of the two the live AIcrowd scorer used. Not determinable from
available materials. Terrier itself was **NOT RUN** (no JRE; out of scope).

### 5. What is the official baseline's exact architecture/config?

Transcribed from `amazon-science/esci-data` HEAD `7916cdf`:

| | US | ES / JP |
|---|---|---|
| model | `cross-encoder/ms-marco-MiniLM-L-12-v2` | `sentence-transformers/multi-qa-mpnet-base-dot-v1` |
| architecture | CrossEncoder, `num_labels=1`, Identity activation | SentenceTransformer **bi-encoder** |
| loss | `MSELoss` | `CosineSimilarityLoss` |
| lr | 7e-6 | library default |
| warmup | 5,000 steps | — |
| eval steps | 5,000 | 1,000 |
| max_length | 512 | model default |

Shared: **input = query + product_title only** (no brand/color/bullet/description),
target = the gain itself as a **regression** target, `epochs=1`, `batch=32`,
`seed=42`, filter `small_version==1 AND split=='train' AND product_locale==<locale>`,
internal dev holdout us 400 / es 200 / jp 200 queries.

Three documented deviations, all in `official_baseline_config.json`:
1. **Train pool** — we removed our 5,071 DEV queries (118,231 rows: us 63,280 + es 23,160
   + jp 31,791) *before* training. The verbatim recipe would have trained on our DEV and
   produced a meaningless comparison. This also makes it fairer: baseline and V2 now see
   the same 28,733 train queries.
2. `default_activation_function` → `activation_fn` (ST 5.2.3 rename).
3. `CERerankingEvaluator` dev_samples sets → sorted lists (ST 5.2.3 needs lists).
   Monitoring only; US saves end-of-training weights via `model.save()`, as official does.

Training: US 747s / ES 873s / JP 1,784s on M4 Max MPS.

### 6. Official baseline on our competition DEV?

**NDCG@20 = 0.8071** (full 0.8521, @10 0.7358). Per locale: us 0.8345 / es 0.7823 /
jp 0.7477.

### 7. V2 on the same DEV?

**NDCG@20 = 0.8456** (full 0.8834, @10 0.7924). Per locale: us 0.8558 / es 0.8486 /
jp 0.8140. **V2 − official = +0.0385**, CI [+0.0348, +0.0421], p < 0.0001.

### 8. May we compare future results to 0.9043?

**No. RED.** The Shopping Queries Dataset paper states the data was *"stratified by
queries in three splits: training, public test, and private test, at 70%, 15%, and 15%."*
The release has only two splits, and Task 1 measures **69.99% train / 30.01% test** —
matching 70% / (15%+15%) to 0.01%. **The released `test` split is public ∪ private,
merged, with no column marking the boundary.** The private partition (~7,248 queries) is
**permanently unrecoverable**.

0.9043 is the winner's *private* LB score (public 0.9057). Even unlocking our TEST split
would give a number on public ∪ private — neither leaderboard. Compounding this, it is
UNKNOWN which gain convention the AIcrowd scorer used.

Details and source citations: `leaderboard_comparability.md`.

---

# Status of prior work

Per §3 and §15, everything is retained and reclassified, nothing deleted or modified:

| artifact | classification |
|---|---|
| V2 CE (0.845605 @20), LambdaMART, BM25, Two-Tower | **legacy / current-project benchmark** — now *validated* as running on the official Task 1 pool with the official gain |
| Jina v3.5, Qwen3-Reranker-0.6B, BGE-v2-m3 zero-shot | **exploratory_zero_shot_results** — paused |
| Qwen 50K LoRA | **halted mid-run**, not resumed |
| frozen 50K train subset, ensemble audit | retained; ensemble verdict was *stop* (holdout700 Δ +0.0005) |

One upgrade this audit earns them: because the pool is bit-identical to official and the
gain/NDCG are verified, these numbers are **competition-protocol-aligned** — they are
just not leaderboard-comparable. Those are different claims.

---

# Deliverables

```
experiments/competition_alignment/
├── REPORT.md                            this file
├── protocol_manifest.json               Gate 0 record + verified provenance
├── candidate_pool_root_cause.md         the max=94 answer
├── competition_task1_manifest.json      official pool, hashes, distributions
├── current_vs_official_split_audit.json bit-identity proof
├── metric_authority_report.md           the two conflicting gain maps
├── metric_parity_results.json           13/13, max |Δ| 1.1e-16
├── official_baseline_config.json        exact recipe + 3 documented deviations
├── official_baseline_dev_results.json   baseline vs V2 + bootstrap
├── official_baseline_dev_predictions.parquet
├── leaderboard_comparability.md         YELLOW / RED, with sources
├── published_strong_baseline_plan.md    plan only, nothing trained
└── official_baseline_models/            3 locale models + per-locale meta
```

# STOPPED

Per §17. Not started: 2022 strong-baseline training, Qwen/Jina fine-tuning, any TEST
evaluation, any leaderboard claim.

Awaiting your decision on whether to reproduce a 2022 strong baseline. My recommendation
in `published_strong_baseline_plan.md` is **not to** — its main deliverable was a
leaderboard-comparable number, which question 8 has now ruled out, and its secondary
deliverable (pipeline validation) is already satisfied by the bit-identical pool and the
1.1e-16 metric parity.
