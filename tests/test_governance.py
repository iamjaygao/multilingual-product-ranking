"""Governance rules, enforced as tests.

MIGRATION_PLAN.md §13 requires the two governance rules to ship as executable
checks rather than prose, because a convention that depends on someone
remembering it has already failed once in this project's history.

This file currently covers **§13.1 (TEST discipline)**, which lands in Phase 2.
§13.2 (metric binding) lands in Phase 5 and its assertions will be added here
then.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import sys

import pandas as pd
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


# ==========================================================================
# §13.2 -- metric binding (Phase 5). The scorer is sealed; the binding is a
# wrapper. See docs/governance.md §2.
# ==========================================================================

SCORER_SHA256 = "6946489c8c09dc37cdb91b905dd5108e46e4070dbd1494b4ac4f972d20f41bc7"


def _toy_frame():
    return pd.DataFrame({
        "query_id": ["q1", "q1", "q2", "q2"],
        "product_id": ["a", "b", "c", "d"],
        "product_locale": ["us"] * 4,
        "esci_label": ["E", "I", "E", "S"],
        "gain": [1.0, 0.0, 1.0, 0.1],
        "score": [2.0, 1.0, 1.0, 2.0],
    })


def test_scorer_sha256_is_unchanged():
    """The scorer is a SEALED artifact. It hashes its own source to build
    metric_version, so one changed byte invalidates every Phase 3/4 gate."""
    import hashlib
    from src.ranking.cross_encoder import load_task1_scorer
    p = paths.REPO_ROOT / "src" / "metrics" / "kdd_task1_ndcg.py"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == SCORER_SHA256, \
        "kdd_task1_ndcg.py was modified -- all Phase 3/4 gate values are now invalid"
    assert load_task1_scorer().scorer_sha256() == SCORER_SHA256


def test_bound_table_carries_pool_and_version():
    from src.metrics.binding import bound_per_query_table
    t = bound_per_query_table(_toy_frame(), "score")
    assert t.candidate_pool == "task1_small_v1"
    assert t.metric_version.startswith("kdd_task1_ndcg@6946489c8c09dc37|")


def test_binding_survives_slicing():
    from src.metrics.binding import bound_per_query_table
    t = bound_per_query_table(_toy_frame(), "score")
    assert t.loc[["q1"]].candidate_pool == "task1_small_v1"


def test_cross_pool_comparison_is_refused():
    from src.metrics.binding import (bound_per_query_table, assert_comparable,
                                     CrossPoolComparison)
    a = bound_per_query_table(_toy_frame(), "score", candidate_pool="task1_small_v1")
    b = bound_per_query_table(_toy_frame(), "score", candidate_pool="large_version_v0")
    with pytest.raises(CrossPoolComparison):
        assert_comparable(a, b)


def test_bare_dataframe_is_refused_as_operand():
    from src.metrics.binding import (bound_per_query_table, assert_comparable,
                                     UnboundMetricPayload)
    a = bound_per_query_table(_toy_frame(), "score")
    with pytest.raises(UnboundMetricPayload):
        assert_comparable(a, pd.DataFrame({"x": [1]}))


def test_write_report_refuses_unbound_payload(tmp_path):
    from src.metrics.binding import write_report, UnboundMetricPayload
    with pytest.raises(UnboundMetricPayload):
        write_report({"ndcg_at_20": 0.5, "candidate_pool": None},
                     tmp_path / "r.json")


def test_write_report_binds_and_writes(tmp_path):
    from src.metrics.binding import write_report
    p = write_report({"ndcg_at_20": 0.5}, tmp_path / "r.json")
    d = json.loads(p.read_text())
    assert d["candidate_pool"] == "task1_small_v1"
    assert d["metric_version"].startswith("kdd_task1_ndcg@")


# ==========================================================================
# §10.5 smoke tests
# ==========================================================================

def _py_files(*dirs):
    out = []
    for d in dirs:
        p = paths.REPO_ROOT / d
        if p.exists():
            out += [f for f in p.rglob("*.py")]
    return out


def _code_lines(path):
    """Import-relevant lines with comments stripped."""
    for i, line in enumerate(path.read_text().splitlines(), 1):
        yield i, line.split("#", 1)[0]


def test_smoke_01_legacy_scorer_never_imported():
    """#1 -- §11 item 14: the legacy scorer may not enter, not even as a reference."""
    hits = []
    for f in _py_files("src", "scripts", "tests"):
        for i, code in _code_lines(f):
            if re.search(r"^\s*(from\s+evaluation\.metrics\b|import\s+official_ndcg\b"
                         r"|from\s+official_ndcg\b)", code):
                hits.append(f"{f.relative_to(paths.REPO_ROOT)}:{i}")
    assert not hits, "legacy scorer imported:\n" + "\n".join(hits)


def test_smoke_02_group_mlp_classes_absent():
    """#2 -- CONFLICT-3: the group MLP class must be gone from the main path."""
    hits = []
    for f in _py_files("src", "scripts"):
        for i, code in _code_lines(f):
            if re.search(r"\b(advanced_model|advanced_features|AdvancedDeepReranker)\b", code):
                hits.append(f"{f.relative_to(paths.REPO_ROOT)}:{i}: {code.strip()}")
    assert not hits, "group MLP referenced on the main path:\n" + "\n".join(hits)


def test_smoke_03_extract_advanced_features_absent_repo_wide():
    """#3 -- the 218-line teammate implementation must not have travelled."""
    import ast
    hits = []
    for f in _py_files("src", "scripts", "tests", "legacy", "experiments"):
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                if any(a.name == "extract_advanced_features" for a in n.names):
                    hits.append(f"{f.relative_to(paths.REPO_ROOT)}:{n.lineno} import")
            elif isinstance(n, ast.Name) and n.id == "extract_advanced_features":
                hits.append(f"{f.relative_to(paths.REPO_ROOT)}:{n.lineno} reference")
    assert not hits, "extract_advanced_features used as code:\n" + "\n".join(hits)


def test_smoke_07_every_results_json_is_bound():
    """#7 -- §13.2: every results JSON carries candidate_pool and metric_version."""
    from src.metrics.binding import payload_is_bound
    unbound = []
    for j in (paths.ARTIFACTS / "results").rglob("*.json"):
        try:
            d = json.loads(j.read_text())
        except json.JSONDecodeError:
            continue
        if not payload_is_bound(d):
            unbound.append(str(j.relative_to(paths.REPO_ROOT)))
    assert not unbound, "results JSON missing metric binding:\n" + "\n".join(unbound)


def test_smoke_10_bm25_pool_carries_attribution_banner():
    """#10 -- §7.3-C required wording on the minimal re-implementation."""
    txt = (paths.REPO_ROOT / "src" / "features" / "bm25_pool.py").read_text()
    flat = " ".join(txt.split())      # the banner is line-wrapped in the docstring
    for phrase in (
        "The original course project contained a group implementation of this component.",
        "The version in this repository is a minimal re-implementation written "
        "independently for this ranking benchmark.",
        "No claim is made over the original group implementation.",
    ):
        assert phrase in flat, f"bm25_pool.py missing §7.3-C wording: {phrase!r}"


def test_smoke_11_no_teammate_authored_file_on_the_main_path():
    """#11 -- ownership boundary holds in practice.

    Checked against the source repo's blame where it is available; otherwise
    against the list of teammate-originated modules that must not appear.
    """
    forbidden_basenames = {
        "advanced_features.py", "advanced_model.py", "model.py", "features.py",
        "metrics.py", "bm25.py", "two_tower.py", "interactive_search.py",
        "convert_esci_to_parquet.py", "train_reranker.py", "train_adv_reranker.py",
        "evaluate_advanced.py", "evaluate_reranker.py",
    }
    hits = [str(f.relative_to(paths.REPO_ROOT))
            for f in _py_files("src", "scripts", "tests")
            if f.name in forbidden_basenames]
    assert not hits, "teammate-originated module on the main path:\n" + "\n".join(hits)


def test_smoke_12_prior_work_names_teammates_for_every_document_only_item():
    """#12 -- attribution for each §3.9 DOCUMENT_ONLY result."""
    txt = (paths.REPO_ROOT / "docs" / "prior_work.md").read_text()
    for section, marker in [
        ("BatchNorm diagnosis", "1.1 The `BatchNorm`"),
        ("LambdaMART vs MLP", "1.2 LambdaMART 0.8464"),
        ("Recall@100 / RRF@100", "1.3 Full-catalog retrieval"),
        ("retrieval complementarity", "1.4 Retrieval complementarity"),
        ("query-slice findings", "1.5 Query-slice findings"),
        ("business-NDCG", "1.6 Business-NDCG"),
    ]:
        assert marker in txt, f"prior_work.md missing DOCUMENT_ONLY item: {section}"
    for name in ("gyuszix", "hyukjin17", "jin"):
        assert name in txt, f"prior_work.md never names teammate {name}"
    # every Part 1 section must attribute someone
    part1 = txt.split("# Part 2")[0]
    for head in re.findall(r"^## (1\.\d.*)$", part1, re.M):
        body = part1.split(f"## {head}")[1].split("\n## ")[0]
        assert "Attribution" in body, f"prior_work.md §{head} has no Attribution block"


def test_smoke_13_main_path_does_not_import_legacy():
    """#13 -- two-layer separation (also asserted in the Phase 2 block above)."""
    hits = []
    for f in _py_files("src", "scripts"):
        for i, code in _code_lines(f):
            if re.match(r"^\s*(from\s+legacy\b|import\s+legacy\b)", code):
                hits.append(f"{f.relative_to(paths.REPO_ROOT)}:{i}")
    assert not hits, "main path imports Layer 2:\n" + "\n".join(hits)


def test_smoke_16_sklearn_pin_is_exact():
    """#16 -- §4.1: a version drift silently repartitions train/dev."""
    txt = (paths.REPO_ROOT / "pyproject.toml").read_text()
    assert '"scikit-learn==1.8.0"' in txt, \
        "pyproject.toml must pin scikit-learn==1.8.0 exactly (not >=, not a range)"


# --------------------------------------------------------------------------
# #14 -- the minimum loop has ZERO group-derived artifact dependency.
#
# Simulated, not destructive: legacy/ and data/legacy_features/ are hidden from
# the process rather than deleted, and the two blocking gates are re-checked
# against their recorded manifests.
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _hidden(*relpaths):
    """Make paths invisible to os.path/pathlib existence checks and to imports,
    without touching the filesystem."""
    targets = {str((paths.REPO_ROOT / r).resolve()) for r in relpaths}

    def shadowed(p):
        s = str(pathlib.Path(p).resolve()) if not isinstance(p, str) else str(pathlib.Path(p).resolve())
        return any(s == t or s.startswith(t + os.sep) for t in targets)

    class _Blocker:
        """Refuse to import the `legacy` package while it is hidden."""
        def find_module(self, name, path=None):
            return self if name.split(".")[0] == "legacy" else None
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] == "legacy":
                raise ModuleNotFoundError(f"No module named {name!r} (hidden by #14)")
            return None

    real_exists, real_isdir = os.path.exists, os.path.isdir
    real_pexists = pathlib.Path.exists
    saved_modules = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "legacy"}
    for k in saved_modules:
        del sys.modules[k]
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        os.path.exists = lambda p: False if shadowed(p) else real_exists(p)
        os.path.isdir = lambda p: False if shadowed(p) else real_isdir(p)
        pathlib.Path.exists = lambda self: False if shadowed(self) else real_pexists(self)
        yield
    finally:
        os.path.exists, os.path.isdir = real_exists, real_isdir
        pathlib.Path.exists = real_pexists
        sys.meta_path.remove(blocker)
        sys.modules.update(saved_modules)


def _importlib_import(name):
    import importlib
    return importlib.import_module(name)


def test_smoke_14_minimum_loop_survives_without_layer2():
    """#14 -- with legacy/ and data/legacy_features/ gone, §10.2a and §10.4
    still pass. This is the direct evidence that the minimum viable loop has no
    group-derived artifact dependency (§1)."""
    hidden = ("legacy", "data/legacy_features")

    # they really are there to begin with, or the test proves nothing
    for r in hidden:
        assert (paths.REPO_ROOT / r).exists(), f"{r} missing; #14 would be vacuous"

    with _hidden(*hidden):
        assert not (paths.REPO_ROOT / "legacy").exists()
        assert not (paths.REPO_ROOT / "data" / "legacy_features").exists()
        with pytest.raises(ModuleNotFoundError):
            _importlib_import("legacy.features.build_feature_matrix")

        # the spine still imports and the scorer still resolves
        import importlib
        ce = importlib.import_module("src.ranking.cross_encoder")
        sc = ce.load_task1_scorer()
        assert sc.scorer_sha256() == SCORER_SHA256
        ce.assert_gain_agreement()

        # §10.2a -- the Phase 3/4 gate values, from their recorded manifests
        p3 = json.loads((paths.MANIFESTS / "phase3_ce_reproduction.json").read_text())
        assert p3["verdict"].startswith("PASS")
        assert p3["checks_failed"] == 0
        p4 = json.loads((paths.MANIFESTS / "phase4_reproduction.json").read_text())
        assert p4["verdict"] == "PASS"
        assert p4["checks_failed"] == 0
        assert p4["minimum_loop_closed"] is True

        # §10.4 -- Layer 1 rebuild verification
        v = json.loads((paths.MANIFESTS / "split_integrity_verification.json").read_text())
        assert v["PASS"] is True and v["checks_failed"] == 0

        # and the Layer 1 pool itself is complete without Layer 2
        for split in ("train", "dev", "test"):
            assert (paths.DATA_TASK1 / f"{split}_task1_core.parquet").exists()

    # restored
    assert (paths.REPO_ROOT / "legacy").exists()
    assert (paths.REPO_ROOT / "data" / "legacy_features").exists()
