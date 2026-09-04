# Metric authority report — which gain does ESCI Task 1 actually use?

**Answer: E=1.0, S=0.1, C=0.01, I=0.0.** Our scorer is correct.

The complication is real and worth stating plainly: **the official Amazon repository
contains two mutually inconsistent gain definitions.** One matches the competition
specification; the other silently swaps S and C. We did not guess between them — the
tie is broken by which script the baseline was actually built with.

---

## 1. What each source says

### Tier 1 — `amazon-science/esci-data`, local HEAD `7916cdf6ab75a462e77f20ab40428a10923998d5` (2024-10-07)

**Source A — `ranking/train.py`, lines 49–53** (the script that trains the official baseline):

```python
esci_label2gain = {
    'E' : 1.0,
    'S' : 0.1,
    'C' : 0.01,
    'I' : 0.0,
}
```

Used at line 69 to create the regression target and at line 82 as the `InputExample`
label. **Effective gain: E=1.0, S=0.1, C=0.01, I=0.0.**

**Source B — `ranking/prepare_trec_eval_files.py`, lines 46–51**
(sha256 `688f9255e1b94ea2a55fa20e1e06cbb27257702c85c9f6460e980133b3067095`):

```python
esci_label2relevance_pos = {
    "E" : 4,
    "S" : 2,
    "C" : 3,
    "I" : 1,
}
```

composed with the Terrier command at `launch-predictions-task1.sh:54`:

```
terrier trec_eval ${QRELS} ${RES} -c -J -m 'ndcg.1=0,2=0.01,3=0.1,4=1'
```

Resolving position → gain: `I→1→0`, `S→2→0.01`, `C→3→0.1`, `E→4→1`.
**Effective gain: E=1.0, S=0.01, C=0.1, I=0.0 — S and C are swapped.**

### Competition specification

E=1.0, S=0.1, C=0.01, I=0.0 — which agrees with Source A and disagrees with Source B.

### Tier 3 — third-party

The SQID paper independently documents this exact defect: the released esci-data code
swaps the S/C label-to-relevance mapping in `prepare_trec_eval_files.py`, and correcting
it changes the reported ESCI_baseline nDCG. Used here only as corroboration; the
conclusion rests on Tier 1.

---

## 2. Why Source A wins

1. **Semantic ordering.** ESCI is an ordered scale: Exact > Substitute > Complement >
   Irrelevant. A Substitute (a product that would satisfy the need) must be worth more
   than a Complement (an accessory to it). Source B makes a Complement worth 10× a
   Substitute, inverting the taxonomy. Source A preserves it.
2. **Source A is the training script.** The official baseline's weights were produced
   using `esci_label2gain`. Source B is a downstream evaluation *helper*. If the two
   disagree, the definition the model was optimised against is the operative one.
3. **Source A matches the published competition gain.** Source B does not.
4. **Source B's defect is independently documented** (SQID), and its position→gain
   indirection through Terrier's `-m` flag is exactly the kind of two-step mapping where
   a transposition survives review.

## 3. Magnitude — this is not academic

Measured on our own models over the frozen DEV pool, switching to the swapped
convention moves scores by far more than any model difference we have observed:

| model | competition gain | swapped gain | Δ |
|---|---:|---:|---:|
| Random | 0.750317 | 0.720282 | −0.030035 |
| BM25 | 0.819842 | 0.801352 | −0.018490 |
| Two-Tower | 0.824499 | 0.802476 | −0.022022 |
| LambdaMART | 0.842878 | 0.824276 | −0.018603 |

**0.019–0.030 NDCG.** Any historical ESCI number must be checked for which convention it
used before being compared to ours. In particular, the `0.83` figure reported in the
esci-data README was produced through Source B and is therefore **not** directly
comparable to competition-gain numbers.

## 4. What our scorer does

`experiments/ranking_v2/kdd_task1_benchmark/scripts/kdd_task1_ndcg.py`

```
GAIN_COMPETITION = {"E": 1.00, "S": 0.10, "C": 0.01, "I": 0.00}
```

used **directly** as the DCG numerator (no `2**gain - 1`), matching Source A. The
`GAIN_SWAPPED` constant exists solely to quantify the ambiguity above, and is never used
for a headline number.

The historical `[7, 3, 1, 0]` LightGBM `label_gain` is retired and is not used anywhere
in the current pipeline.

## 5. Terrier

**Not run.** No JRE on this machine and installing one was out of scope. Parity was
therefore established against an independent second implementation rather than allowing
the production scorer to validate itself — see `metric_parity_results.json`.

Consequence for wording: we may state that our scorer matches the official
**gain definition** (Source A) and matches an independent reference implementation to
1.1e-16. We may **not** state that it has been verified against the official Terrier
evaluator or against the AIcrowd scoring program.

## 6. Verdict

```
gain                = PASS   (E=1.0, S=0.1, C=0.01, I=0.0; Tier-1 train.py)
NDCG implementation = PASS   (13/13 cases, max |Δ| 1.1e-16 vs independent reference)
Terrier parity      = NOT RUN  (no JRE; out of scope)
```

Recorded as **UNKNOWN**: whether the AIcrowd 2022 scoring program used Source A or
Source B. This cannot be determined from the materials available locally, and it matters
for interpreting the published leaderboard — see `leaderboard_comparability.md`.
