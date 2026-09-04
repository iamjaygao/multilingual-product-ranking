# Leaderboard comparability — can we compare our numbers to 0.9043?

## COMPARABILITY STATUS: **YELLOW** overall — and **RED** specifically against 0.9043

Same task, same candidate-pool construction rule, same gain definition, verified NDCG
arithmetic. **But a different evaluation split, and the competition's private split is
permanently unrecoverable.**

---

## 1. The decisive finding: the released `test` split is public ∪ private, merged

The Shopping Queries Dataset paper states the data was

> "stratified by queries in three splits: training, public test, and private test, at
> 70%, 15%, and 15%, respectively."

The **released** parquet has only two `split` values: `train` and `test`. Measured on the
Task 1 filter (`small_version == 1`):

| split | queries | share |
|---|---:|---:|
| train | 33,804 | **69.99%** |
| test | 14,496 | **30.01%** |
| total | 48,300 | 100% |

70% / 30% matches 70% / (15% + 15%) to within 0.01%. The released `test` split is
therefore the **union of the competition's public and private test sets**, concatenated
with no marker distinguishing them. The `examples` table has exactly these columns:

```
example_id, query, query_id, product_id, product_locale,
esci_label, small_version, large_version, split
```

There is **no** `public`/`private` field, no leaderboard-phase column, nothing from which
the 15/15 partition could be reconstructed.

*(For reference, the Task 2/3 `large_version` filter splits 76.30% / 23.70%, a different
ratio — consistent with the two tasks having had different release scopes.)*

## 2. Answers to the four §4 questions

**Q1. Is the released `test` split the competition public test?**
**No — it is larger.** It is public ∪ private (30% of queries), whereas the public
leaderboard scored only the public half (15%).

**Q2. Does it contain labels that were hidden during the competition?**
**Yes.** Competitors could not see labels for either test half; the released `test` split
carries full `esci_label` values for all 336,373 Task 1 test rows. Anyone evaluating on
the released test split today is using labels that were hidden in 2022.

**Q3. Relationship to the 2022 public/private leaderboard?**
`released test  ==  public LB set  ∪  private LB set`, with the boundary erased.
Evaluating on the released test set produces a number that corresponds to **neither**
leaderboard — it is an average over both halves.

**Q4. Is the private leaderboard set reproducible?**
**No. UNRECOVERABLE.** Reconstructing it would require the original 15/15 query
assignment, which was never published and is not derivable from any released field. No
amount of care in our pipeline can recover it.

## 3. The three published numbers, each with its own split

| Source | Value | Metric | Split | Gain convention | Comparable to us? |
|---|---:|---|---|---|---|
| esci-data README, Task 1 baseline | **0.83** | nDCG | **UNSPECIFIED** in the README | produced via `prepare_trec_eval_files.py` → **S/C swapped** | **NO** — wrong gain *and* unknown split |
| KDD Cup 2022 winner (ZhichunRoad), public LB | 0.9057 | nDCG@20 | public test (15% of queries) | competition (presumed) | **NO** — split unrecoverable |
| KDD Cup 2022 winner, **private LB** | **0.9043** | nDCG@20 | **private test (15%), unrecoverable** | competition (presumed) | **NO** |
| 9th place (arXiv 2208.06264) | 0.9012 / 0.9007 | nDCG@20 | public / private | competition (presumed) | NO |

Two further cautions on the 0.83 baseline figure:
- The README does not say which split or which dataset version it used.
- It was computed through the helper script whose gain map swaps S and C
  (see `metric_authority_report.md`). Our own measurements show that swap is worth
  0.019–0.030 NDCG, so 0.83 is not on our scale at all.

## 4. What our own numbers are, and are not

Our frozen DEV is a **15% query-level holdout carved from official Task 1 TRAIN**. It is
not the competition test set, not the public LB set, and not the private LB set.

| property | ours | competition private LB |
|---|---|---|
| task | Task 1 query-product ranking | same |
| candidate pool | official judged pairs, bit-identical | same rule |
| gain | E1.0 / S0.1 / C0.01 / I0.0 (`train.py:49`) | presumed same |
| NDCG math | verified to 1.1e-16 vs independent reference | UNKNOWN evaluator |
| **evaluation queries** | **5,071, carved from TRAIN** | **~7,248, hidden, unrecoverable** |
| labels seen in 2022? | yes (train labels were always public) | no |

## 5. Two residual UNKNOWNs

1. **Which gain convention the AIcrowd 2022 scoring program used.** The official repo
   contains both (`train.py` → S=0.1; `prepare_trec_eval_files.py` → S=0.01). If the
   live scorer used the swapped map, then 0.9043 is on a *different metric* from ours,
   compounding the split problem. Not determinable from available materials.
2. **Whether the released test labels are byte-identical to the 2022 judgements**, or
   were revised before release. No statement found either way.

## 6. Wording rules that follow

**Permitted:**
- "competition-aligned candidate pool and gain convention"
- "our own frozen DEV holdout, carved from official Task 1 TRAIN"
- "published historical reference: the KDD Cup 2022 winner reported 0.9043 on a private
  leaderboard split that is not recoverable from the released data"

**Forbidden — and this does not become permitted by improving our models:**
- "surpassed the KDD Cup winner"
- "beat / matched 0.9043"
- "above / below the official baseline" when citing the 0.83 figure
- any subtraction between our number and 0.9043, 0.9057, or 0.83

## 7. What would it take to reach GREEN?

Nothing available to us. GREEN would require the original 15/15 public/private query
assignment plus confirmation of the AIcrowd scorer's gain map. Neither was released.

The strongest defensible claim any future model in this project can make is:

> *On the official Task 1 candidate pool with the official gain convention, evaluated on
> our own frozen 5,071-query DEV holdout, model X scores Y.*

That is a legitimate, reproducible, internally-consistent benchmark. It is **not** a
leaderboard number, and it can never be converted into one.

---

## Sources

- [amazon-science/esci-data](https://github.com/amazon-science/esci-data) — README (Task 1 baseline 0.83; "up to 40 potentially relevant results"); `ranking/train.py`, `ranking/prepare_trec_eval_files.py`, `ranking/launch-*.sh` at local HEAD `7916cdf`
- [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://arxiv.org/pdf/2206.06588) — the 70% / 15% / 15% train / public-test / private-test stratification
- [ZhichunRoad at Amazon KDD Cup 2022: MultiTask Pre-Training for E-Commerce Product Search](https://arxiv.org/pdf/2301.13455) — winner, public 0.9057 / private 0.9043
- [A Boring-yet-effective Approach for the Product Ranking Task of the Amazon KDD Cup 2022](https://arxiv.org/html/2208.06264) — 9th place, public 0.9012 / private 0.9007; "the test is also decomposed into public and private where the latter one is unseen to us"
- [Amazon product query competition draws more than 9,200 submissions](https://www.amazon.science/blog/amazon-product-query-competition-draws-more-than-9-200-submissions) — competition overview
