"""Governance rules, enforced as tests.

MIGRATION_PLAN.md §13 requires the two governance rules to ship as executable
checks rather than prose, because a convention that depends on someone
remembering it has already failed once in this project's history.

This file currently covers **§13.1 (TEST discipline)**, which lands in Phase 2.
§13.2 (metric binding) lands in Phase 5 and its assertions will be added here
then.
"""
from __future__ import annotations

import json
import re

import pytest

from src import paths
from src.data import task1_common as tc
# aliased so pytest does not try to collect it as a test class
from src.data.task1_common import TestSplitLocked as _Locked

PREREG = "docs/prereg/pool_construction.md"


# --------------------------------------------------------------------------
# §13.1 -- the lock refuses
# --------------------------------------------------------------------------

def test_load_split_test_raises_without_prereg():
    """The headline rule: TEST is locked."""
    with pytest.raises(_Locked) as e:
        tc.load_split("test")
    assert "TEST is locked" in str(e.value)
    assert "13.1" in str(e.value)


def test_load_split_all_raises_without_prereg():
    """`all` includes test rows, so it is locked on the same terms."""
    with pytest.raises(_Locked):
        tc.load_split("all")


def test_load_examples_inherits_the_lock():
    """load_examples() is routed through load_split(); in the source repo it
    read the parquet directly with no guard at all."""
    with pytest.raises(_Locked):
        tc.load_examples()


# --------------------------------------------------------------------------
# §13.1 -- prereg validation is not a rubber stamp
# --------------------------------------------------------------------------

def test_nonexistent_prereg_path_is_rejected():
    with pytest.raises(_Locked) as e:
        tc.load_split("test", prereg="docs/prereg/does_not_exist.md")
    assert "does not exist" in str(e.value)


def test_boolean_prereg_is_rejected():
    """A bare boolean is too easy to pass -- §13.1 says it must name a file."""
    with pytest.raises(_Locked) as e:
        tc.load_split("test", prereg=True)
    assert "not a pre-registration" in str(e.value).lower() or \
           "must be a path" in str(e.value).lower()


def test_untracked_prereg_is_rejected(tmp_path):
    """A file that exists but is not committed cannot be a pre-registration:
    it could have been written after the results were known."""
    p = tmp_path / "after_the_fact.md"
    p.write_text("# not committed\n")
    with pytest.raises(_Locked) as e:
        tc.load_split("test", prereg=str(p))
    assert "not tracked by git" in str(e.value).lower()


def test_committed_prereg_is_accepted():
    """The construction pre-registration is git-tracked, so it passes."""
    assert (paths.REPO_ROOT / PREREG).exists()
    assert tc._git_tracked(paths.REPO_ROOT / PREREG), \
        f"{PREREG} must be git-tracked for the TEST lock to accept it"


# --------------------------------------------------------------------------
# §13.1 -- metadata_only escape hatch
# --------------------------------------------------------------------------

def test_metadata_only_succeeds_without_prereg():
    """Counts and schema without labels are always permitted."""
    df = tc.load_split("test", metadata_only=True, columns=["query_id"])
    assert len(df) > 0


def test_metadata_only_hides_labels():
    """The escape hatch must not leak ground truth."""
    df = tc.load_split("test", metadata_only=True)
    for col in ("esci_label", "gain", "lgb_label"):
        assert col not in df.columns, f"metadata_only leaked {col}"


def test_task1_test_query_ids_needs_no_prereg():
    """Which queries are held out is not itself ground truth, so the arm-B
    exclusion path can run without spending the TEST budget."""
    qids = tc.task1_test_query_ids()
    assert len(qids) == 14496


# --------------------------------------------------------------------------
# §13.1 -- the audit trail
# --------------------------------------------------------------------------

def test_accepted_test_reads_are_logged():
    """"Spent once" must be auditable, not merely asserted."""
    assert tc.TEST_ACCESS_LOG.exists(), (
        "artifacts/manifests/test_access_log.json should exist after the Phase 2 "
        "pool build, which reads TEST under the construction pre-registration")
    log = json.loads(tc.TEST_ACCESS_LOG.read_text())
    assert log["accesses"], "no logged accesses"
    for e in log["accesses"]:
        assert e["prereg"], "an accepted TEST read was logged without a prereg"
        assert e["timestamp_utc"] and e["caller"]


def test_train_reads_are_not_logged():
    """Only TEST reads consume the budget; train must not pollute the log."""
    before = len(json.loads(tc.TEST_ACCESS_LOG.read_text())["accesses"]) \
        if tc.TEST_ACCESS_LOG.exists() else 0
    tc.load_split("train", columns=["query_id"])
    after = len(json.loads(tc.TEST_ACCESS_LOG.read_text())["accesses"]) \
        if tc.TEST_ACCESS_LOG.exists() else 0
    assert after == before


# --------------------------------------------------------------------------
# Layer separation (§4, §8) -- the minimum loop must not reach into Layer 2
# --------------------------------------------------------------------------

def test_layer1_columns_carry_no_features():
    forbidden = {"bm25_score", "semantic_score", "lgb_label", "word_overlap",
                 "log_price", "stars_clean", "category"}
    assert not (set(tc.LAYER1_COLUMNS) & forbidden), \
        "a Layer 2 feature column leaked into the Layer 1 core schema"


def test_src_does_not_import_legacy():
    """Smoke test #13 (§10.5): nothing on the main path may import Layer 2."""
    hits = []
    for p in list((paths.REPO_ROOT / "src").rglob("*.py")) + \
            list((paths.REPO_ROOT / "scripts").rglob("*.py")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            # Match imports OF the `legacy` package -- not the word appearing
            # in a trailing comment (bm25_pool.py mentions the `legacy` extra).
            code = line.split("#", 1)[0].strip()
            if re.match(r"^(from\s+legacy\b|import\s+legacy\b)", code):
                hits.append(f"{p.relative_to(paths.REPO_ROOT)}:{i}: {code}")
    assert not hits, "main path imports Layer 2:\n" + "\n".join(hits)
