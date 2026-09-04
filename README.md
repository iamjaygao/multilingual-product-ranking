# multilingual-product-ranking

Auditing and improving a public Amazon product-search benchmark.

Amazon released a public dataset for measuring how well a search engine orders
product results. While working with it I found a bug in the scoring script the
dataset ships with: two of its relevance grades are swapped relative to the
training code in the same repository, so the same model gets a different score
depending on which of the two files you use. I reported it upstream and
published a minimal reproduction.

I then rebuilt Amazon's own baseline model from their published recipe and
trained a replacement that ranks results better than it does across all three
languages in the dataset. Every number below can be regenerated from this
repository, and the held-out evaluation set has deliberately never been touched.

---

## What I found in the benchmark

`amazon-science/esci-data` contains two mutually inconsistent relevance
definitions:

| Source | Mapping | Effective gain |
|---|---|---|
| `ranking/train.py:49-53` — trains the official baseline | `E 1.0, S 0.1, C 0.01, I 0.0` | matches the competition spec |
| `ranking/prepare_trec_eval_files.py:46-51` — prepares evaluation files | `E 4, S 2, C 3, I 1` → `ndcg.1=0,2=0.01,3=0.1,4=1` | **S and C swapped** |

A *Substitute* (a reasonable alternative product) and a *Complement* (an
accessory bought alongside) end up weighted 0.01 and 0.1 — the inverse of the
specification, and of what the training script in the same repository uses.

**Measured impact: 0.019–0.030 NDCG**, on my own candidate pool and train/dev
split across four models (Random, BM25, Two-Tower, LambdaMART) — an order of
magnitude, not a constant. That is roughly ten times the differences between the
models being compared, so any historical ESCI figure has to be checked for which
convention produced it before it can be compared against anything.

- Upstream PR: <https://github.com/amazon-science/esci-data/pull/26>
- Minimal reproduction: <https://github.com/iamjaygao/esci-gain-mapping-repro>
- Full analysis: [`docs/reports/metric_authority_report.md`](docs/reports/metric_authority_report.md)

This repository uses `E 1.0, S 0.1, C 0.01, I 0.0` — the competition definition,
matching `train.py`. The scorer is content-hashed so the convention cannot drift
silently.

## What I built

A multilingual cross-encoder reranker: `FacebookAI/xlm-roberta-base`, 4-class
head over `[E, S, C, I]`, `max_length=256`, plain cross-entropy. Ranking never
uses argmax — the score is the expected ESCI gain under the predicted
distribution, `P(E)·1.0 + P(S)·0.1 + P(C)·0.01 + P(I)·0.0`, so the model is
ranked on the quantity the metric rewards.

Trained on 663,407 pairs / 28,733 queries, and compared against Amazon's
baseline recipe rebuilt from `ranking/train.py` — same scorer, same pool, same
run.

| dev, all locales | official recipe | V2 cross-encoder |
|---|---:|---:|
| NDCG full-list | 0.852066 | **0.883443** |
| NDCG@20 | 0.807127 | **0.845605** |
| NDCG@10 | 0.735796 | **0.792383** |

Per-locale NDCG@20:

| | us (n=3,133) | es (n=845) | jp (n=1,093) |
|---|---:|---:|---:|
| official recipe | 0.834539 | 0.782330 | 0.747723 |
| V2 | **0.855827** | **0.848554** | **0.814023** |

Paired bootstrap, 10,000 resamples, seed 42, paired by `query_id`, percentile CI:

| slice | Δ (V2 − official) | 95% CI | W/L/T |
|---|---:|---|---|
| all locales, n=5,071 | +0.038478 | [+0.034833, +0.042133] | 3229 / 1696 / 146 |
| us only, n=3,133 | +0.021288 | [+0.017255, +0.025260] | 1852 / 1160 / 121 |

**Two things this table does not say.**

*The control is not three comparable models.* Only the US branch of Amazon's
baseline is a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`, MSE
regression on gain). ES and JP use
`sentence-transformers/multi-qa-mpnet-base-dot-v1` — an **English-only
bi-encoder** (`multi-qa` means multiple QA datasets, not multilingual) — applied
to Spanish and Japanese. So the +0.066 margins on ES and JP mostly measure a
configuration problem in the other recipe, not multilingual ability in mine.
**The like-for-like comparison is the US branch: +0.021288, CI [+0.017255,
+0.025260].** That is the number I would defend. Same run, sliced; the protocol
was not changed to make the baseline look weaker.

*Every number here is a dev number.* The official test split has never been
evaluated — see below.

Method notes: [`docs/cross_encoder.md`](docs/cross_encoder.md) ·
[`docs/reports/competition_alignment.md`](docs/reports/competition_alignment.md)

## How it's verified

**The candidate pool is provably the official one.** Two-way `example_id` set
equality against official Task 1 train: 781,638 rows, 0 IDs on either side only,
**0 label mismatches**, 0 rows added.

**The data is rebuilt from raw, not carried over.** The core pool is
reconstructed from the upstream ESCI parquets by a query-level split (seed 42,
stratified by locale) and checked against a recorded oracle: **52/52** quantities
match — row and query counts, per-split locale/depth/label distributions, split
disjointness. This is what frees the main pipeline from any code or
model-generated feature inherited from the course project it grew out of.

**The migration itself was regression-tested.** Predictions were regenerated
here from the frozen checkpoints and compared against the previous repository's
outputs: **118,231 / 118,231 rows exactly equal, max|diff| = 0.0**, for both
models. Metrics were computed from the regenerated predictions, not the reference
files — otherwise only the scorer would have been verified, not inference.

**TEST discipline is enforced in code, not by convention.** `load_split()` is
the single entry point for reading a split. Requesting `test` raises unless the
caller passes a pre-registration document that exists **and is tracked by git** —
a boolean flag is rejected explicitly, because an untracked file can be written
after the results are known. Every accepted access is appended to
[`artifacts/manifests/test_access_log.json`](artifacts/manifests/test_access_log.json),
which currently holds **2 entries, both from pool construction**. **The V2
cross-encoder has never been scored on the official test split.** Every figure in
this README is a dev figure, and spending the test split would require a
committed prediction first.

**Every NDCG number carries its provenance.** Results are emitted through a
writer that refuses payloads lacking `candidate_pool` and `metric_version`, and
comparing two score tables from different pools raises. The scorer is sealed:
its `metric_version` embeds a hash of its own source
(`kdd_task1_ndcg@6946489c8c09dc37|gain=competition_E1_S0.1_C0.01_I0|idcg0=exclude|tie=deterministic`),
so editing it invalidates every recorded gate value, and a test asserts the hash.

**88 tests**, including one that hides the optional legacy layer and re-verifies
that the main gates still pass — direct evidence that the core pipeline has no
dependency on inherited artifacts.

**Known limits, stated rather than glossed:**

- `load_split()` locks the *frozen split*. Code that opens the raw upstream
  parquet directly and filters `split == "test"` itself bypasses it. One
  legitimate instance exists (`scripts/protocol_audit.py:88`, counts and label
  distributions, no scoring). The gap is deliberate — closing it would block
  honest dataset auditing — and is documented in
  [`docs/governance.md`](docs/governance.md) §1.4.
- Metric binding is enforced at the serialisation boundary and by the wrapper
  API, not at the scorer function itself, which is sealed and cannot be edited.

## Repository layout

```
src/metrics/      the sealed NDCG scorer + the provenance-binding wrapper
src/data/         Task 1 pool construction, the TEST lock, integrity gates
src/ranking/      cross-encoder library (text building, scoring, dataset)
src/features/     minimal per-query BM25 feature producer
src/retrieval/    Two-Tower V2 training (optional; US-locale only)
scripts/          runnable entry points: build, train, evaluate, bootstrap
legacy/           optional; needs artifacts not in this repo. Deletable.
artifacts/        tracked manifests and results; large binaries out-of-band
docs/             method notes, governance, prior-work attribution, reports
tests/            88 tests, incl. governance and reproduction gates
```

## Reproducing this

**Rebuildable from upstream data** — no transfer needed:

```bash
pip install -e .                                   # pins scikit-learn==1.8.0
export ESCI_DATA_ROOT=/path/to/esci-data           # amazon-science/esci-data

python -m src.data.build_task1_pool --prereg docs/prereg/pool_construction.md
python -m src.data.verify_split_integrity          # must report 52/52
python -m scripts.build_ce_data --splits dev
```

The `scikit-learn==1.8.0` pin is load-bearing, not hygiene.
`train_test_split`'s shuffling can change across major versions, so the same
`random_state=42` may produce a **different** train/dev partition on a different
version — silently, because a wrong partition still yields plausible-looking
NDCG. `verify_split_integrity` is what catches it.

**Requires out-of-band transfer** — model weights and reference predictions are
not in git: the V2 checkpoint (1.1 GB), the three official baseline locale models
(965 MB), and the oracle predictions (4.6 MB). Sizes, hashes, and how to obtain
each are in [`artifacts/ARTIFACTS.md`](artifacts/ARTIFACTS.md).

Expected values for every gate:
[`artifacts/FROZEN_V2.json`](artifacts/FROZEN_V2.json) and
`artifacts/manifests/phase{3,4}_*.json`.

## Attribution

The pipeline in this repository — the benchmark alignment, the metric audit, the
cross-encoder, the bootstrap analysis, the reproduction gates — is my own work.

It grew out of a three-person university course project
(`esci-search-ranking-system`). Components written by teammates — the baseline
and advanced MLP rerankers, the Two-Tower V0 encoder, the current BM25 module,
the ESCI-S integration, the interactive GUI, and the legacy scorer — were
**deliberately not migrated** and remain there. Where a result from that period
is cited here, [`docs/prior_work.md`](docs/prior_work.md) names who wrote the
component that produced it — including two cases easy to misread in my favour,
which are stated flatly. The boundary came from git history, not recollection.

**Two-Tower V2 (`src/retrieval/`) is trained on US-locale data only** — 0 Spanish
and 0 Japanese training pairs, enforced by three separate filters. It is a
monolingual English model sitting in a repository named
*multilingual*-product-ranking, so: any es/jp number for that component is
zero-shot transfer, and a weak Japanese score is the expected consequence of its
training scope rather than a finding about the architecture. The cross-encoder
above is a different model and is trained on all three locales.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

Third-party attributions are in [`NOTICE`](NOTICE), including one restriction
that matters: `jinaai/jina-reranker-v3.5`, used only in the exploratory
zero-shot comparison, is **CC BY-NC 4.0 (non-commercial)**. No result derived
from it may be presented as commercial- or production-ready, and it is confined
to `experiments/exploratory/`. The Apache 2.0 grant covering this repository's
own code does not extend to that model.
