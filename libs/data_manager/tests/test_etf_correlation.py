"""etf_correlation 公共逻辑单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_etf_correlation.py -v
"""

from __future__ import annotations

from data_manager.etf_correlation import build_merge_clusters, pick_representative


def test_build_merge_clusters_transitive():
    """A-B、B-C 传递合并为一个簇。"""
    pairs = [
        {"symbol_a": "A", "symbol_b": "B"},
        {"symbol_b": "B", "symbol_a": "C"},
    ]
    clusters = build_merge_clusters(pairs)
    assert len(clusters) == 1
    assert set(clusters[0]) == {"A", "B", "C"}


def test_build_merge_clusters_disjoint():
    """不相交的对分成独立簇。"""
    pairs = [
        {"symbol_a": "A", "symbol_b": "B"},
        {"symbol_a": "C", "symbol_b": "D"},
    ]
    clusters = build_merge_clusters(pairs)
    assert len(clusters) == 2
    assert {frozenset(c) for c in clusters} == {frozenset({"A", "B"}), frozenset({"C", "D"})}


def test_build_merge_clusters_empty():
    assert build_merge_clusters([]) == []


def test_pick_representative_earliest():
    """选 first_date 最早的 symbol。"""
    fd = {"A": "2023-01-01", "B": "2022-01-01", "C": "2024-01-01"}
    assert pick_representative(["A", "B", "C"], fd) == "B"


def test_pick_representative_missing_date():
    """first_date 缺失视为最晚，不参与胜出。"""
    fd = {"A": "2023-01-01", "B": ""}
    assert pick_representative(["A", "B"], fd) == "A"
