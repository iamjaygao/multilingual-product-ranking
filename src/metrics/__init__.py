"""Authoritative metric implementations for the Task 1 benchmark.

`kdd_task1_ndcg` is the SOLE scorer of record (MIGRATION_PLAN.md §11 item 14,
§13.2). `official_kdd_ndcg` is the large-version parity reference. The legacy
scorer (`evaluation/metrics.py` in the predecessor repo) is barred from this
repository by hard rule and is not present here in any form.
"""
