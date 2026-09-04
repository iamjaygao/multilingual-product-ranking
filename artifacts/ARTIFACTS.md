# Out-of-band artifacts

Large binaries are excluded from git. This file is the register of what a fresh
clone does **not** contain, how to obtain each item, and — critically — which
items are **rebuilt from upstream data** rather than transferred.

That distinction is the point of the two-layer split (`MIGRATION_PLAN.md` §4):
the minimum viable loop depends on **zero group-derived artifacts**. Everything
it needs is either fetched from upstream, rebuilt deterministically here, or a
frozen checkpoint used as a regression oracle.

---

## Register

| Artifact | Location | Size | How to obtain | Tracked? |
|---|---|---:|---|---|
| **CE V2 checkpoint** | `artifacts/checkpoints/cross_encoder_v2/` | 1.1 GB | Copy from the frozen source-repo checkpoint (`epoch_2`). sha256 of `model.safetensors`: `29082490a5a275964e11023a342b7e5afb1f2d29773006afd53cf8614cc1e1a1` | no |
| **Official baseline — 3 locale models** | `artifacts/checkpoints/official_baseline/task_1_ranking_model_{us,es,jp}/` | 965 MB | Copy from the frozen source-repo checkpoints. Regenerable in principle via `scripts/reproduce_official_baseline.py`, but **not bit-identically** | no |
| **Oracle predictions** | `artifacts/oracle_predictions/` | 4.6 MB | Produced by the source repo; carried over as the migration **regression oracle** (see below) | no |
| **Layer 1 core pool** | `data/task1/{train,dev,test}_task1_core.parquet` | 101 MB | **REBUILT** — `python -m src.data.build_task1_pool --prereg docs/prereg/pool_construction.md` | no |
| **CE text cache** | `data/task1/ce_text/` | 74 MB dev-only (+425 MB if train is built) | **REBUILT** — `python -m scripts.build_ce_data --splits dev` | no |
| **Raw ESCI parquets** | `$ESCI_DATA_ROOT/shopping_queries_dataset/` | 1.15 GB | Clone `amazon-science/esci-data`; point `ESCI_DATA_ROOT` at it. Never vendored | no |
| **Layer 2 feature matrix** | *not present* | — | Not migrated. Phase 6b only — see the warning below | n/a |

Small manifests and result JSONs under `artifacts/manifests/` and
`artifacts/results/` **are** git-tracked, and are the audit trail for everything
above.

---

## Rebuilt, not transferred — and why it matters

`data/task1/` and `data/task1/ce_text/` are **regenerated from raw ESCI by
migrated code**, not copied from the predecessor repository.

- The Layer 1 pool rebuild is deterministic and verified against a tracked
  oracle: `python -m src.data.verify_split_integrity` compares 52 quantities
  (query and row counts, per-split locale / depth / label distributions, split
  disjointness, and the split-method string) against
  `artifacts/manifests/split_integrity_reference.json`. All 52 match exactly.
- The CE text cache rebuild was verified byte-for-byte against the source repo's
  cache: rendered product text, the query column, their concatenation, the id
  triple and the gain column all agree on sha256.

Determinism depends on **`scikit-learn==1.8.0`**, pinned exactly in
`pyproject.toml`. `train_test_split`'s shuffling may change across major
versions, so the same `random_state=42` can yield a *different* partition on a
different version — silently, because a wrong partition still produces
plausible-looking NDCG. `verify_split_integrity` is what catches that.

## The oracle predictions are not a pipeline input

`artifacts/oracle_predictions/` holds two frozen prediction parquets from the
source repo:

| File | Size |
|---|---:|
| `official_baseline_dev_predictions.parquet` | 2.2 MB |
| `fulldev_epoch2_predictions.parquet` | 2.4 MB |

They are **not consumed by the pipeline**. Their only role is to prove that
migration did not change inference. In Phase 4, predictions were **regenerated**
here from the frozen checkpoints and compared against these files: both score
columns proved bit-identical (sha256 over the float64 bytes; 118,231 / 118,231
rows exactly equal; `max|diff| = 0.0`).

Computing metrics directly off the oracle would have verified only the scorer,
not the inference path — so it was deliberately not done that way.

This is the distinction `MIGRATION_PLAN.md` §1 draws:

```
Public repo reproducibility:
  training config    → approximate retraining      [non-blocking, §10.2b]
  frozen checkpoint  → exact eval regression       [blocking migration test, §10.2a]
```

A third party cloning this repository can retrain from config and land inside a
declared tolerance band without ever receiving a checkpoint. The checkpoints
exist so *this migration* could be proven bit-exact — a one-time validation
need, not an ongoing requirement.

## ⚠️ Layer 2 — `semantic_score` must never be silently regenerated

The Layer 2 feature matrix is **not in this repository**. Its `semantic_score`
column was produced by a **group-originated Two-Tower V0 checkpoint** that did
not migrate (`MIGRATION_PLAN.md` §7.3-2; `docs/prior_work.md` §2.4).

If Phase 6b is ever run:

1. Carry the Layer 2 matrix out-of-band from the source repo.
2. **Verify `semantic_score` is bit-identical** to the source (sha256, recorded
   in `legacy/manifests/semantic_score_sha256.json`) **before** running anything.
3. **Do not regenerate that column with Two-Tower V2.** Doing so changes every
   number in the Task 1 benchmark report — including the externally cited
   LambdaMART 0.8429 — while leaving the file name, the column name and the
   report text identical. Nothing would fail; the numbers would quietly stop
   meaning what they say.
4. If a V2-based column is genuinely wanted, add it under a **different name**
   (`semantic_score_v2`) as a new baseline row, never as a rewrite.

The minimum loop reads none of these columns, so a clone that never runs
Phase 6b cannot hit this failure at all. `tests/test_governance.py::test_smoke_14`
asserts exactly that: with `legacy/` and `data/legacy_features/` hidden, the
§10.2a and §10.4 gates still pass.

## Restoring a working clone

```bash
git clone <this repo> && cd multilingual-product-ranking
pip install -e .                                    # pins scikit-learn==1.8.0

export ESCI_DATA_ROOT=/path/to/esci-data            # amazon-science/esci-data

python -m src.data.build_task1_pool --prereg docs/prereg/pool_construction.md
python -m src.data.verify_split_integrity           # 52/52 must pass (§10.4)
python -m scripts.build_ce_data --splits dev

# copy the checkpoints into artifacts/checkpoints/, then run the gate:
python -m scripts.evaluate_cross_encoder \
    --checkpoint artifacts/checkpoints/cross_encoder_v2 \
    --split dev --output_dir artifacts/results/phase3 --tag ce_v2_dev
```

Expected gate values: `artifacts/FROZEN_V2.json`,
`artifacts/manifests/phase3_ce_reproduction.json`,
`artifacts/manifests/phase4_reproduction.json`.
