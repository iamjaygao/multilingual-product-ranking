"""
Tests for the Task 1 Cross-Encoder reranker.

Covers the ten areas required by the Phase 2A brief:
  1  label mapping                     6  scorer adapter on a hand-built query
  2  expected gain                     7  train/dev query overlap assertion
  3  NaN handling in the text builder  +  probability ordering E>S>C>I
  4  query is never truncated          +  score range [0, 1]
  5  prediction row order / keys

Nothing here downloads a model; the two tests that need a tokenizer are skipped
when the backbone is not already cached locally.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from src import paths
from src.ranking.cross_encoder import (
    GAIN_BY_LABEL, GAIN_VECTOR, ID2LABEL, LABEL2ID, LABEL_ORDER, TEXT_FIELDS,
    CrossEncoderDataset, assert_gain_agreement, build_product_text,
    build_product_texts, expected_gain, load_task1_scorer, sample_query_ids,
    scores_from_logits, softmax, token_length_stats,
)

ROOT = str(paths.REPO_ROOT)
BENCH = str(paths.DATA_TASK1)
MODEL_ID = "FacebookAI/xlm-roberta-base"


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{MODEL_ID} not cached locally")


# ------------------------------------------------------------------ 1
class TestLabelMapping:
    def test_fixed_order(self):
        assert LABEL_ORDER == ("E", "S", "C", "I")
        assert ID2LABEL == {0: "E", 1: "S", 2: "C", 3: "I"}
        assert LABEL2ID == {"E": 0, "S": 1, "C": 2, "I": 3}

    def test_round_trip(self):
        for i, l in ID2LABEL.items():
            assert LABEL2ID[l] == i
        assert len(set(LABEL2ID.values())) == 4

    def test_gain_vector_is_positionally_aligned(self):
        assert list(GAIN_VECTOR) == [1.00, 0.10, 0.01, 0.00]
        for i, l in ID2LABEL.items():
            assert GAIN_VECTOR[i] == GAIN_BY_LABEL[l]

    def test_matches_authoritative_scorer(self):
        assert_gain_agreement()
        sc = load_task1_scorer()
        assert dict(GAIN_BY_LABEL) == dict(sc.GAIN_COMPETITION)


# ------------------------------------------------------------------ 2 + ordering + range
class TestExpectedGain:
    def test_one_hot_recovers_the_gain(self):
        eye = np.eye(4)
        got = expected_gain(eye)
        assert np.allclose(got, [1.00, 0.10, 0.01, 0.00])

    def test_hand_computed_mixture(self):
        # P(E)=0.5, P(S)=0.3, P(C)=0.15, P(I)=0.05
        # 0.5*1.0 + 0.3*0.1 + 0.15*0.01 + 0.05*0.0 = 0.5 + 0.03 + 0.0015 = 0.5315
        p = np.array([[0.5, 0.3, 0.15, 0.05]])
        assert expected_gain(p)[0] == pytest.approx(0.5315, abs=1e-12)

    def test_probability_order_E_gt_S_gt_C_gt_I(self):
        """A distribution peaked on E must outscore one peaked on S, and so on."""
        peaked = np.eye(4) * 0.97 + 0.01
        s = expected_gain(peaked)
        assert s[0] > s[1] > s[2] > s[3], s

    def test_score_range(self):
        rng = np.random.RandomState(0)
        logits = rng.randn(2000, 4) * 5
        s = scores_from_logits(logits)
        assert s.min() >= 0.0 and s.max() <= 1.0
        # bounded by the extreme gains
        assert s.max() <= GAIN_BY_LABEL["E"] and s.min() >= GAIN_BY_LABEL["I"]

    def test_softmax_rows_sum_to_one(self):
        rng = np.random.RandomState(1)
        p = softmax(rng.randn(500, 4) * 10)
        assert np.allclose(p.sum(axis=1), 1.0)
        assert (p >= 0).all()

    def test_ranking_is_not_argmax(self):
        """Two rows with the same argmax must still be separable by expected gain."""
        a = np.array([0.60, 0.39, 0.005, 0.005])   # argmax E
        b = np.array([0.60, 0.005, 0.005, 0.39])   # argmax E
        sa, sb = expected_gain(a[None, :])[0], expected_gain(b[None, :])[0]
        assert sa > sb
        assert int(a.argmax()) == int(b.argmax()) == 0

    def test_rejects_wrong_width(self):
        with pytest.raises(ValueError):
            expected_gain(np.ones((3, 3)))


# ------------------------------------------------------------------ 3
class TestTextBuilder:
    def test_all_fields_in_order(self):
        row = {"product_title": "iPad Pro", "product_brand": "Apple",
               "product_color": "Silver", "product_bullet_point": "11 inch",
               "product_description": "A tablet"}
        t = build_product_text(row)
        assert t == ("title: iPad Pro\nbrand: Apple\ncolor: Silver\n"
                     "bullet: 11 inch\ndescription: A tablet")

    @pytest.mark.parametrize("bad", [np.nan, None, float("nan"), pd.NA, "nan", "NaN", "  ", ""])
    def test_missing_values_never_leak_the_string_nan(self, bad):
        row = {"product_title": "T", "product_brand": bad, "product_color": bad,
               "product_bullet_point": bad, "product_description": bad}
        t = build_product_text(row)
        assert t == "title: T"
        assert "nan" not in t.lower()
        assert "brand:" not in t

    def test_all_fields_missing_gives_empty_string(self):
        row = {c: np.nan for _, c in TEXT_FIELDS}
        assert build_product_text(row) == ""

    def test_whitespace_collapsed(self):
        row = {"product_title": "a\n\n  b\tc", "product_brand": "", "product_color": "",
               "product_bullet_point": "", "product_description": ""}
        assert build_product_text(row) == "title: a b c"

    def test_char_cap_applies_per_field(self):
        row = {"product_title": "x" * 50, "product_brand": "", "product_color": "",
               "product_bullet_point": "", "product_description": ""}
        assert build_product_text(row, max_chars_per_field=10) == "title: " + "x" * 10

    def test_dataframe_path_matches_row_path(self):
        df = pd.DataFrame({
            "product_title": ["A", "B"], "product_brand": ["b1", np.nan],
            "product_color": [np.nan, "red"], "product_bullet_point": ["p", np.nan],
            "product_description": [np.nan, "d"]})
        got = build_product_texts(df)
        assert got[0] == "title: A\nbrand: b1\nbullet: p"
        assert got[1] == "title: B\ncolor: red\ndescription: d"

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            build_product_texts(pd.DataFrame({"product_title": ["x"]}))


# ------------------------------------------------------------------ 4
class TestQueryNotTruncated:
    def test_only_second_keeps_the_query_intact(self):
        tok = _tokenizer()
        query = "wireless noise cancelling over ear headphones for travel"
        product = "title: " + ("very long product text " * 400)
        enc = tok([query], [product], truncation="only_second", max_length=64,
                  padding=False)
        ids = enc["input_ids"][0]
        assert len(ids) <= 64
        q_ids = tok(query, add_special_tokens=False)["input_ids"]
        # every query token survives, contiguously, right after the leading <s>
        assert ids[1:1 + len(q_ids)] == q_ids

    def test_query_survives_for_each_locale_script(self):
        tok = _tokenizer()
        for q in ["iphone 15 pro max case",
                  "auriculares inalámbricos con cancelación de ruido",
                  "ワイヤレスイヤホン ノイズキャンセリング"]:
            enc = tok([q], ["title: " + "z " * 500], truncation="only_second",
                      max_length=48, padding=False)
            q_ids = tok(q, add_special_tokens=False)["input_ids"]
            assert enc["input_ids"][0][1:1 + len(q_ids)] == q_ids, q

    def test_token_stats_report_truncation(self):
        tok = _tokenizer()
        queries = ["short query"] * 4
        products = ["title: tiny", "title: " + "w " * 500, "title: tiny",
                    "title: " + "w " * 500]
        st = token_length_stats(queries, products, tok, max_length=64,
                                locales=["us", "us", "jp", "jp"])
        assert st["overall"]["truncation_fraction"] == pytest.approx(0.5)
        assert st["overall"]["query_overflow_fraction"] == 0.0
        assert st["by_locale"]["jp"]["truncation_fraction"] == pytest.approx(0.5)
        assert st["by_locale"]["es"]["rows"] == 0


# ------------------------------------------------------------------ 5
class TestPredictionKeysAndOrder:
    def test_collate_preserves_batch_order(self):
        tok = _tokenizer()
        qs = [f"query {i}" for i in range(8)]
        ps = [f"title: product {i}" for i in range(8)]
        ds = CrossEncoderDataset(qs, ps, tok, max_length=32, labels=list(range(4)) * 2)
        batch = ds.collate([ds[i] for i in range(8)])
        assert batch["input_ids"].shape[0] == 8
        assert batch["labels"].tolist() == list(range(4)) * 2
        # decoding row k must recover query k
        for k in [0, 3, 7]:
            assert f"query {k}" in tok.decode(batch["input_ids"][k], skip_special_tokens=True)

    def test_keys_are_not_lost_when_scores_are_attached(self):
        df = pd.DataFrame({
            "query_id": [1, 1, 2, 2], "product_id": ["a", "b", "c", "d"],
            "product_locale": ["us"] * 4, "esci_label": ["E", "I", "S", "C"]})
        logits = np.array([[3., 0, 0, 0], [0, 0, 0, 3.], [0, 3., 0, 0], [0, 0, 3., 0]])
        out = df.copy()
        out["score"] = scores_from_logits(logits)
        assert len(out) == len(df)
        assert list(out["product_id"]) == ["a", "b", "c", "d"]
        assert list(out["query_id"]) == [1, 1, 2, 2]
        assert out["score"].notna().all()
        assert out.loc[0, "score"] > out.loc[1, "score"]

    def test_length_mismatch_rejected(self):
        tok = _tokenizer()
        with pytest.raises(ValueError):
            CrossEncoderDataset(["a", "b"], ["x"], tok)
        with pytest.raises(ValueError):
            CrossEncoderDataset(["a"], ["x"], tok, labels=[0, 1])


# ------------------------------------------------------------------ 6
class TestScorerAdapter:
    def test_hand_built_query_perfect_ranking(self):
        """One query, E>S>C>I ranked correctly -> NDCG 1.0 on all three cutoffs."""
        sc = load_task1_scorer()
        df = pd.DataFrame({
            "query_id": [1, 1, 1, 1],
            "product_id": ["p1", "p2", "p3", "p4"],
            "product_locale": ["us"] * 4,
            "esci_label": ["E", "S", "C", "I"]})
        df["gain"] = sc.gain_from_labels(df["esci_label"])
        # confident, correctly-ordered predictions
        logits = np.array([[9., 0, 0, 0], [0, 9., 0, 0], [0, 0, 9., 0], [0, 0, 0, 9.]])
        df["score"] = scores_from_logits(logits)
        res = sc.evaluate(df, "score", gain_col="gain", query_col="query_id")
        assert res["ndcg_full"] == pytest.approx(1.0)
        assert res["ndcg_at_10"] == pytest.approx(1.0)
        assert res["ndcg_at_20"] == pytest.approx(1.0)
        assert res["candidate_pool"] == "task1_small_v1"

    def test_inverted_ranking_scores_below_one(self):
        sc = load_task1_scorer()
        df = pd.DataFrame({
            "query_id": [1, 1], "product_id": ["p1", "p2"],
            "product_locale": ["us", "us"], "esci_label": ["E", "S"]})
        df["gain"] = sc.gain_from_labels(df["esci_label"])
        # S predicted more relevant than E -> the known 0.6875501678 case
        df["score"] = scores_from_logits(np.array([[0, 9., 0, 0], [9., 0, 0, 0]]))
        res = sc.evaluate(df, "score", gain_col="gain", query_col="query_id")
        assert res["ndcg_full"] == pytest.approx(0.687550167778977, abs=1e-9)

    def test_expected_gain_beats_argmax_on_a_tie(self):
        """Two candidates share an argmax of E; expected gain still orders the
        truly-better one first, which argmax ranking could not."""
        sc = load_task1_scorer()
        df = pd.DataFrame({
            "query_id": [1, 1], "product_id": ["p_bad", "p_good"],
            "product_locale": ["us", "us"], "esci_label": ["I", "E"]})
        df["gain"] = sc.gain_from_labels(df["esci_label"])
        probs = np.array([[0.55, 0.02, 0.03, 0.40], [0.55, 0.44, 0.005, 0.005]])
        assert probs.argmax(axis=1).tolist() == [0, 0]      # identical argmax
        df["score"] = expected_gain(probs)
        res = sc.evaluate(df, "score", gain_col="gain", query_col="query_id")
        assert res["ndcg_full"] == pytest.approx(1.0)


# ------------------------------------------------------------------ 7
class TestSplitIntegrity:
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(BENCH, "train_task1_core.parquet")),
        reason="Layer 1 core pool not built; run src.data.build_task1_pool")
    def test_train_dev_test_query_disjoint_and_counts_match(self):
        exp = {"train": (663407, 28733), "dev": (118231, 5071), "test": (336373, 14496)}
        qs = {}
        for split, (rows, queries) in exp.items():
            # Layer 1 core pool, rebuilt from raw ESCI (§4.1) -- the source repo
            # read the transferred {split}_task1.parquet here.
            d = pd.read_parquet(os.path.join(BENCH, f"{split}_task1_core.parquet"),
                                columns=["query_id"])
            assert len(d) == rows, f"{split} rows {len(d)} != {rows}"
            assert d["query_id"].nunique() == queries
            qs[split] = set(d["query_id"])
        assert not (qs["train"] & qs["dev"])
        assert not (qs["train"] & qs["test"])
        assert not (qs["dev"] & qs["test"])

    def test_sample_query_ids_is_query_level_and_deterministic(self):
        df = pd.DataFrame({"query_id": np.repeat(np.arange(20), 5),
                           "row": np.arange(100)})
        a = sample_query_ids(df, 5, seed=42)
        b = sample_query_ids(df, 5, seed=42)
        assert a["query_id"].nunique() == 5
        assert len(a) == 25                      # every sampled query keeps all 5 rows
        assert set(a["query_id"]) == set(b["query_id"])
        assert a.groupby("query_id").size().eq(5).all()

    def test_sample_query_ids_noop_when_budget_exceeds_pool(self):
        df = pd.DataFrame({"query_id": [1, 1, 2], "x": [0, 1, 2]})
        assert len(sample_query_ids(df, 99, seed=0)) == 3
        assert len(sample_query_ids(df, None, seed=0)) == 3
