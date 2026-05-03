"""Focused regression tests for ETF keyword selection and low-correlation combo search.

Run from project root:
    PYTHONPATH=libs pytest libs/backtesting/tests/test_index_rotation.py -v
"""

from __future__ import annotations

import pandas as pd

from backtesting.index_rotation import (
    build_pairwise_correlation_frame,
    build_representative_etf_frame,
    match_index_keywords,
    search_low_correlation_combos,
    select_keyword_representatives,
)


def test_match_index_keywords_uses_normalized_substrings_and_tokens():
    matched = match_index_keywords("纳指生物科技")
    assert "纳指生物科技" in matched
    assert "纳指" in matched
    assert "生物科技" in matched


def test_select_keyword_representatives_prefers_highest_avg_value_and_filters_short_history():
    profile_frame = pd.DataFrame(
        [
            {
                "symbol": "A1",
                "name": "沪深300ETF甲",
                "type": "指数型-股票",
                "tracked_index": "沪深300",
                "matched_keywords": ("沪深300",),
                "bar_count": 800,
                "avg_value": 100.0,
                "first_date": pd.Timestamp("2020-01-01"),
                "last_date": pd.Timestamp("2024-12-31"),
            },
            {
                "symbol": "A2",
                "name": "沪深300ETF乙",
                "type": "指数型-股票",
                "tracked_index": "沪深300",
                "matched_keywords": ("沪深300",),
                "bar_count": 900,
                "avg_value": 120.0,
                "first_date": pd.Timestamp("2019-01-01"),
                "last_date": pd.Timestamp("2024-12-31"),
            },
            {
                "symbol": "B1",
                "name": "科创50ETF甲",
                "type": "指数型-股票",
                "tracked_index": "科创50",
                "matched_keywords": ("科创50", "科创"),
                "bar_count": 550,
                "avg_value": 500.0,
                "first_date": pd.Timestamp("2022-01-01"),
                "last_date": pd.Timestamp("2024-12-31"),
            },
            {
                "symbol": "B2",
                "name": "科创50ETF乙",
                "type": "指数型-股票",
                "tracked_index": "科创50",
                "matched_keywords": ("科创50", "科创"),
                "bar_count": 780,
                "avg_value": 300.0,
                "first_date": pd.Timestamp("2020-01-01"),
                "last_date": pd.Timestamp("2024-12-31"),
            },
        ]
    )

    selection_frame = select_keyword_representatives(
        profile_frame,
        keywords=["沪深300", "科创50", "科创", "恒生"],
        min_bar_count=600,
    )

    by_keyword = selection_frame.set_index("keyword")
    assert by_keyword.loc["沪深300", "selected_symbol"] == "A2"
    assert by_keyword.loc["沪深300", "candidate_count"] == 2
    assert by_keyword.loc["科创50", "selected_symbol"] == "B2"
    assert by_keyword.loc["科创", "selected_symbol"] == "B2"
    assert by_keyword.loc["恒生", "selected_symbol"] is None


def test_build_representative_etf_frame_deduplicates_symbols_and_keeps_keyword_sources():
    keyword_selection_frame = pd.DataFrame(
        [
            {
                "keyword": "科创50",
                "selected_symbol": "B2",
                "selected_name": "科创50ETF乙",
                "selected_type": "指数型-股票",
                "tracked_index": "科创50",
                "bar_count": 780,
                "avg_value": 300.0,
                "first_date": pd.Timestamp("2020-01-01"),
                "last_date": pd.Timestamp("2024-12-31"),
            },
            {
                "keyword": "科创",
                "selected_symbol": "B2",
                "selected_name": "科创50ETF乙",
                "selected_type": "指数型-股票",
                "tracked_index": "科创50",
                "bar_count": 780,
                "avg_value": 300.0,
                "first_date": pd.Timestamp("2020-01-01"),
                "last_date": pd.Timestamp("2024-12-31"),
            },
            {
                "keyword": "沪深300",
                "selected_symbol": "A2",
                "selected_name": "沪深300ETF乙",
                "selected_type": "指数型-股票",
                "tracked_index": "沪深300",
                "bar_count": 900,
                "avg_value": 120.0,
                "first_date": pd.Timestamp("2019-01-01"),
                "last_date": pd.Timestamp("2024-12-31"),
            },
        ]
    )

    representative_frame = build_representative_etf_frame(
        keyword_selection_frame,
        exclude_symbols=["A2"],
    )

    assert representative_frame["symbol"].tolist() == ["B2"]
    assert representative_frame.iloc[0]["source_keywords"] == ("科创50", "科创")
    assert representative_frame.iloc[0]["source_keywords_text"] == "科创50, 科创"
    assert representative_frame.iloc[0]["source_keyword_count"] == 2


def test_build_pairwise_correlation_frame_marks_threshold_failures():
    corr_matrix = pd.DataFrame(
        [
            [1.0, 0.35, 0.10],
            [0.35, 1.0, 0.25],
            [0.10, 0.25, 1.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    pairwise_frame = build_pairwise_correlation_frame(
        corr_matrix,
        symbols=["A", "B", "C"],
        corr_threshold=0.3,
    )

    by_pair = {
        (row.left_symbol, row.right_symbol): row
        for row in pairwise_frame.itertuples(index=False)
    }
    assert bool(by_pair[("A", "B")].passes_threshold) is False
    assert bool(by_pair[("A", "C")].passes_threshold) is True
    assert bool(by_pair[("B", "C")].passes_threshold) is True


def test_build_pairwise_correlation_frame_honors_lower_bound():
    corr_matrix = pd.DataFrame(
        [
            [1.0, -0.35, -0.10],
            [-0.35, 1.0, 0.20],
            [-0.10, 0.20, 1.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    pairwise_frame = build_pairwise_correlation_frame(
        corr_matrix,
        symbols=["A", "B", "C"],
        corr_threshold=0.3,
        corr_lower_bound=-0.3,
    )

    by_pair = {
        (row.left_symbol, row.right_symbol): row
        for row in pairwise_frame.itertuples(index=False)
    }
    assert bool(by_pair[("A", "B")].passes_threshold) is False
    assert bool(by_pair[("A", "C")].passes_threshold) is True
    assert bool(by_pair[("B", "C")].passes_threshold) is True


def test_search_low_correlation_combos_finds_exact_max_extension_set():
    corr_matrix = pd.DataFrame(
        [
            [1.0, 0.10, 0.10, 0.10, 0.20, 0.20, 0.45],
            [0.10, 1.0, 0.10, 0.10, 0.20, 0.20, 0.20],
            [0.10, 0.10, 1.0, 0.10, 0.20, 0.20, 0.20],
            [0.10, 0.10, 0.10, 1.0, 0.20, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20, 1.0, 0.25, 0.20],
            [0.20, 0.20, 0.20, 0.20, 0.25, 1.0, 0.35],
            [0.45, 0.20, 0.20, 0.20, 0.20, 0.35, 1.0],
        ],
        index=["510300", "518880", "513100", "511010", "E1", "E2", "E3"],
        columns=["510300", "518880", "513100", "511010", "E1", "E2", "E3"],
    )
    extension_metric_frame = pd.DataFrame(
        [
            {"symbol": "E1", "avg_value": 100.0, "source_keywords": ("K1",)},
            {"symbol": "E2", "avg_value": 80.0, "source_keywords": ("K2",)},
            {"symbol": "E3", "avg_value": 90.0, "source_keywords": ("K3",)},
        ]
    )

    combo_frame, combo_lists, base_conflict_frame = search_low_correlation_combos(
        corr_matrix,
        base_symbols=["510300", "518880", "513100", "511010"],
        extension_symbols=["E1", "E2", "E3"],
        corr_threshold=0.3,
        max_results=5,
        extension_metric_frame=extension_metric_frame,
    )

    assert combo_lists == [["510300", "518880", "513100", "511010", "E1", "E2"]]
    assert combo_frame.iloc[0]["extension_symbol_count"] == 2
    assert combo_frame.iloc[0]["max_pairwise_correlation"] < 0.3
    assert bool(base_conflict_frame.set_index("symbol").loc["E3", "compatible_with_base"]) is False


def test_search_low_correlation_combos_returns_no_result_when_base_is_infeasible():
    corr_matrix = pd.DataFrame(
        [
            [1.0, 0.35, 0.10, 0.10, 0.20],
            [0.35, 1.0, 0.10, 0.10, 0.20],
            [0.10, 0.10, 1.0, 0.10, 0.20],
            [0.10, 0.10, 0.10, 1.0, 0.20],
            [0.20, 0.20, 0.20, 0.20, 1.0],
        ],
        index=["510300", "518880", "513100", "511010", "E1"],
        columns=["510300", "518880", "513100", "511010", "E1"],
    )

    combo_frame, combo_lists, _ = search_low_correlation_combos(
        corr_matrix,
        base_symbols=["510300", "518880", "513100", "511010"],
        extension_symbols=["E1"],
        corr_threshold=0.3,
    )

    assert combo_frame.empty
    assert combo_lists == []