"""
4-class Cross-Encoder reranker for the ESCI KDD Task 1 benchmark.

Task
----
Input  : (query, product_text) as a tokenizer sequence pair
Output : 4 logits, in the FIXED order [E, S, C, I]
Loss   : plain CrossEntropyLoss (no class weights, no focal loss, no pairwise
         or listwise objective -- this is a deliberately clean baseline)

Ranking score
-------------
Ranking NEVER uses argmax. The score is the expected ESCI gain under the
predicted class distribution:

    score = P(E)*1.00 + P(S)*0.10 + P(C)*0.01 + P(I)*0.00

which is exactly the gain vector the authoritative Task 1 scorer uses, so the
model is scored on the quantity the metric rewards. The score is a convex
combination of {1.0, 0.1, 0.01, 0.0} and therefore always lies in [0, 1].

This module holds only shared logic. Training, evaluation and data prep live in
scripts/. NDCG is never computed here -- the authoritative scorer is imported
via `load_task1_scorer()`.
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- labels
#: Fixed class order. Index i of the logit vector is LABEL_ORDER[i].
LABEL_ORDER: tuple[str, ...] = ("E", "S", "C", "I")
ID2LABEL: dict[int, str] = {i: l for i, l in enumerate(LABEL_ORDER)}
LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(LABEL_ORDER)}

#: Official competition gains, aligned positionally to LABEL_ORDER.
#: Must stay identical to kdd_task1_ndcg.GAIN_COMPETITION.
GAIN_BY_LABEL: dict[str, float] = {"E": 1.00, "S": 0.10, "C": 0.01, "I": 0.00}
GAIN_VECTOR: np.ndarray = np.array([GAIN_BY_LABEL[l] for l in LABEL_ORDER], dtype=np.float64)

#: Product text fields, in emission order. Values are repo-verified column names
#: from shopping_queries_dataset_products.parquet.
TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("title", "product_title"),
    ("brand", "product_brand"),
    ("color", "product_color"),
    ("bullet", "product_bullet_point"),
    ("description", "product_description"),
)

#: Path to THE authoritative scorer, migrated to `src/metrics/` in Phase 1.
#: Repointed from the source repo's
#: `experiments/ranking_v2/kdd_task1_benchmark/scripts/kdd_task1_ndcg.py`.
TASK1_SCORER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metrics", "kdd_task1_ndcg.py")


# ---------------------------------------------------------------- scorer
def load_task1_scorer():
    """Import THE authoritative Task 1 scorer. No second NDCG implementation
    exists anywhere in this pipeline.

    Migration note (§10.5 smoke tests #4/#5): in the source repo the scorer sat
    outside any package, so it was loaded by file path via
    `importlib.util.spec_from_file_location`. Here it is a real module, so this
    is a plain package import with no `sys.path` insertion. The identity check
    keeps `TASK1_SCORER_PATH` load-bearing: if the module ever resolves to a
    different file, this fails loudly instead of silently scoring against a
    second implementation -- the exact condition §13.2 exists to prevent.
    """
    from src.metrics import kdd_task1_ndcg as mod

    if os.path.abspath(mod.__file__) != os.path.abspath(TASK1_SCORER_PATH):
        raise RuntimeError(
            f"scorer identity mismatch: imported {mod.__file__}, "
            f"expected {TASK1_SCORER_PATH}")
    return mod


def assert_gain_agreement() -> None:
    """Fail loudly if this module's gain vector ever drifts from the scorer's."""
    sc = load_task1_scorer()
    if dict(GAIN_BY_LABEL) != dict(sc.GAIN_COMPETITION):
        raise AssertionError(
            f"gain mismatch: cross_encoder {GAIN_BY_LABEL} vs "
            f"authoritative scorer {sc.GAIN_COMPETITION}")


# ---------------------------------------------------------------- scoring
def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def expected_gain(probs: np.ndarray) -> np.ndarray:
    """Expected ESCI gain: probs @ [1.0, 0.1, 0.01, 0.0].

    `probs` must be (n, 4) with columns ordered [E, S, C, I].
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    if p.shape[1] != len(LABEL_ORDER):
        raise ValueError(f"expected {len(LABEL_ORDER)} columns, got {p.shape[1]}")
    # Explicit weighted sum rather than `p @ GAIN_VECTOR`: numerically identical
    # (verified bit-for-bit) but avoids spurious divide-by-zero/overflow FP flags
    # that this numpy/BLAS build raises from padded SIMD lanes on a 4-wide dot.
    return (p * GAIN_VECTOR).sum(axis=1)


def scores_from_logits(logits: np.ndarray) -> np.ndarray:
    """logits -> softmax -> expected gain. The only sanctioned ranking score."""
    return expected_gain(softmax(logits))


# ---------------------------------------------------------------- text
def _clean(value) -> str:
    """NaN / None / the literal strings 'nan' and 'none' all become ''.

    Guards against the classic `str(np.nan) == 'nan'` leak, which would inject a
    meaningless token into every product with a missing field.
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if s.lower() in {"nan", "none", "<na>", "null"}:
        return ""
    return s


def build_product_text(row, fields: Sequence[tuple[str, str]] = TEXT_FIELDS,
                       max_chars_per_field: int | None = None) -> str:
    """Render a product as labelled lines, in TEXT_FIELDS order.

        title: ...
        brand: ...
        color: ...
        bullet: ...
        description: ...

    Empty fields are omitted entirely rather than emitted as a bare `field:`
    line, so the 256-token budget is spent on content. No tokenizer special
    tokens are added here and the tokenizer vocabulary is never modified.
    """
    parts = []
    for label, col in fields:
        v = _clean(row[col] if not hasattr(row, "get") else row.get(col))
        if not v:
            continue
        v = " ".join(v.split())
        if max_chars_per_field is not None and len(v) > max_chars_per_field:
            v = v[:max_chars_per_field]
        parts.append(f"{label}: {v}")
    return "\n".join(parts)


def build_product_texts(df: pd.DataFrame,
                        fields: Sequence[tuple[str, str]] = TEXT_FIELDS,
                        max_chars_per_field: int | None = None) -> list[str]:
    """Vectorised-ish product text construction over a DataFrame."""
    missing = [c for _, c in fields if c not in df.columns]
    if missing:
        raise KeyError(f"missing product text columns: {missing}")
    cols = {c: df[c].tolist() for _, c in fields}
    n = len(df)
    out = []
    for i in range(n):
        row = {c: cols[c][i] for _, c in fields}
        out.append(build_product_text(row, fields, max_chars_per_field))
    return out


# ---------------------------------------------------------------- dataset
class CrossEncoderDataset:
    """Tokenises (query, product_text) pairs lazily.

    `truncation="only_second"` keeps the query intact and truncates the product
    side. Rows carry their identifying keys so prediction order can never be
    silently lost -- the evaluator re-attaches keys positionally and asserts.
    """

    def __init__(self, queries: Sequence[str], product_texts: Sequence[str],
                 tokenizer, max_length: int = 256,
                 labels: Sequence[int] | None = None):
        if len(queries) != len(product_texts):
            raise ValueError("queries and product_texts must be the same length")
        if labels is not None and len(labels) != len(queries):
            raise ValueError("labels must match the number of rows")
        self.queries = list(queries)
        self.product_texts = list(product_texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict:
        item = {"query": self.queries[idx], "product_text": self.product_texts[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item

    def collate(self, batch: list[dict]) -> dict:
        import torch
        enc = self.tokenizer(
            [b["query"] for b in batch],
            [b["product_text"] for b in batch],
            truncation="only_second",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        if self.labels is not None:
            enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return enc


# ---------------------------------------------------------------- tokens
def token_length_stats(queries: Sequence[str], product_texts: Sequence[str],
                       tokenizer, max_length: int = 256,
                       locales: Sequence[str] | None = None,
                       batch_size: int = 512) -> dict:
    """Token-length and truncation statistics.

    Reports the fraction of rows whose untruncated pair exceeds `max_length`,
    overall and per locale, plus how often the QUERY alone would not fit (which
    is the only way `truncation="only_second"` can lose query tokens).
    """
    n = len(queries)
    q_len = np.zeros(n, dtype=np.int32)
    p_len = np.zeros(n, dtype=np.int32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        q_len[s:e] = [len(x) for x in tokenizer(
            list(queries[s:e]), add_special_tokens=False)["input_ids"]]
        p_len[s:e] = [len(x) for x in tokenizer(
            list(product_texts[s:e]), add_special_tokens=False)["input_ids"]]

    # xlm-roberta pair encoding: <s> A </s></s> B </s>  -> 4 special tokens
    n_special = tokenizer.num_special_tokens_to_add(pair=True)
    total = q_len + p_len + n_special
    truncated = total > max_length
    query_overflow = (q_len + n_special) > max_length

    def blk(mask):
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            return {"rows": 0}
        return {
            "rows": int(m.sum()),
            "query_tokens_mean": round(float(q_len[m].mean()), 3),
            "query_tokens_p95": float(np.percentile(q_len[m], 95)),
            "query_tokens_max": int(q_len[m].max()),
            "product_tokens_mean": round(float(p_len[m].mean()), 3),
            "product_tokens_p50": float(np.percentile(p_len[m], 50)),
            "product_tokens_p95": float(np.percentile(p_len[m], 95)),
            "product_tokens_max": int(p_len[m].max()),
            "pair_tokens_mean": round(float(total[m].mean()), 3),
            "pair_tokens_p95": float(np.percentile(total[m], 95)),
            "pair_tokens_max": int(total[m].max()),
            "truncation_fraction": round(float(truncated[m].mean()), 6),
            "query_overflow_fraction": round(float(query_overflow[m].mean()), 8),
        }

    out = {"max_length": max_length, "num_special_tokens_pair": int(n_special),
           "overall": blk(np.ones(n, dtype=bool)), "by_locale": {}}
    if locales is not None:
        loc = np.asarray(locales)
        for lc in ["us", "es", "jp"]:
            out["by_locale"][lc] = blk(loc == lc)
    return out


# ---------------------------------------------------------------- model
def load_model_and_tokenizer(model_name: str, max_length: int = 256):
    """Any Hugging Face sequence-classification backbone, 4 labels, fixed order."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_ORDER),
        id2label=dict(ID2LABEL),
        label2id=dict(LABEL2ID),
    )
    return model, tokenizer


def sample_query_ids(df: pd.DataFrame, max_queries: int | None,
                     seed: int = 42, query_col: str = "query_id") -> pd.DataFrame:
    """Subsample BY QUERY (never by row), so every sampled query keeps its whole
    candidate list and NDCG stays well defined."""
    if max_queries is None:
        return df
    qs = np.sort(df[query_col].unique())
    if len(qs) <= max_queries:
        return df
    rng = np.random.RandomState(seed)
    keep = set(rng.choice(qs, size=max_queries, replace=False).tolist())
    return df[df[query_col].isin(keep)].copy()
