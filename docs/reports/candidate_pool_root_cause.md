# Candidate-pool root cause: why does our pool show max = 94?

**Answer: it is not a defect. The official Task 1 data itself has queries with far more
than 40 judged candidates, and our pool reproduces the official pairs bit-for-bit.**

The premise behind the red flag — that the official data caps at ~40 — is an
approximation stated in prose, not a property of the released parquet.

---

## 1. Measured from the raw parquet, not from any processed frame

Source: `esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet`
Filter: `small_version == 1` — nothing else.

```
rows                        1,118,011
unique query_id                48,300
unique product_id             879,141
```

Candidates per query, computed two ways:

| grouping key | groups | mean | p50 | p90 | p95 | p99 | **max** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `query_id` | 48,300 | 23.15 | 16 | 40 | 40 | 48 | **188** |
| `(query_id, product_locale)` | 48,300 | 23.15 | 16 | 40 | 40 | 48 | **188** |

Per split:

| split | rows | queries | mean | **max** |
|---|---:|---:|---:|---:|
| train | 781,638 | 33,804 | 23.12 | **188** |
| test | 336,373 | 14,496 | 23.20 | **95** |

**1,219 of 48,300 official Task 1 queries (2.52%) have more than 40 candidates.**
The size histogram above 40 is not a thin tail — there is a pronounced spike at
exactly 48 (481 queries), plus 41 (269), 42 (97), 43 (63).

Our DEV subset's max of 94 is simply the largest value that happens to occur among
the 5,071 DEV queries drawn from an official TRAIN pool whose true max is 188.

---

## 2. Every hypothesis in the brief, tested and refuted

### A. Does `query_id` repeat across locales, causing `groupby(query_id)` to merge languages?

**No.**

```
query_ids appearing in more than one locale : 0
max locales per query_id                    : 1
```

`groupby(query_id)` and `groupby([query_id, product_locale])` return the identical
48,300 groups with identical statistics. **`query_id` alone is a safe grouping key**
for this dataset. This was the highest-priority hypothesis and it is cleanly refuted.

### B. Duplicate `(query_id, product_id)` or `(query_id, product_locale, product_id)`?

**None.**

```
dup (query_id, product_id)                  : 0
dup (query_id, product_locale, product_id)  : 0
example_id unique                           : True
```

### C. Is `(product_locale, product_id)` unique in the products table? Any many-to-many blow-up?

**Unique.** The products table has no duplicate `(product_id, product_locale)` key, and
our merge is validated `many_to_one`. The official `train.py` merges on exactly the same
two-key pair (`left_on=[col_product_locale, col_product_id]`).

Note the contrast with the *historical* V1 large-version pipeline, which merged on
`product_id` alone and inflated its pool by 6.93% — that defect was found and fixed in an
earlier phase and is **not** present here.

### D. Is our pool a BM25/Two-Tower union, a retrieval pool, or a concatenation artifact?

**No.** Direct set comparison against the raw official pairs:

```
current train + dev            781,638 rows / 33,804 queries
official Task 1 TRAIN          781,638 rows / 33,804 queries

example_id sets identical      True   (|cur\off| = 0, |off\cur| = 0)
(query_id, locale, product_id) identical      True
label mismatches over 781,638 merged rows     0
rows with small_version != 1                  0
rows with split != 'train'                    0
extra pairs added beyond official             0
```

DEV specifically:

```
DEV                                     118,231 rows / 5,071 queries
official pairs for those same query_ids 118,231 rows
example_id sets identical               True
DEV candidates/query   mean 23.32  p90 40  max 94
official, same queries mean 23.32  p90 40  max 94
```

The pool contains **only** official judged pairs. No BM25 candidate, no Two-Tower
candidate, no retrieved negative, no hard negative was ever added.

---

## 3. Where the "~40" figure comes from

The ~40 number is a dataset-description statement about the typical/intended depth. It
holds for 97.5% of queries and matches p90/p95 exactly (both = 40), which suggests 40 was
the collection target. It is not enforced in the released data.

**Any of the following would have corrupted the benchmark and none was done:**

- `drop_duplicates()` — there are no duplicates to drop
- truncating candidate lists to 40 — would silently discard 2.52% of queries' judged
  products and inflate NDCG by shrinking IDCG denominators
- filtering out queries with >40 candidates — would remove 1,219 official queries

---

## 4. Verdict

| item | status |
|---|---|
| max = 94 explained | **YES** — inherent to official data (true max 188 in train, 95 in test) |
| our pool == official pool | **YES** — bit-identical at `example_id` and triple-key level |
| grouping key correct | **YES** — `query_id` never spans locales |
| duplicates | **NONE** in raw or in our pool |
| contamination from retrieval pools | **NONE** |
| any silent repair applied | **NONE** |

`candidate_pool = PASS`.
