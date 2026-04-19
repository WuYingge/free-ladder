from __future__ import annotations

import pandas as pd

from factors.portfolio.low_correlation import (
    LowCorrelationSelectorConfig,
    build_cluster_summary,
    classify_symbol_family,
    select_low_correlation_portfolio,
    validate_and_prepare_correlation_matrix,
)


def _make_corr_matrix() -> pd.DataFrame:
    labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    data = [
        [1.0, 0.92, 0.10, 0.12, 0.05, 0.08],
        [0.92, 1.0, 0.14, 0.11, 0.07, 0.06],
        [0.10, 0.14, 1.0, 0.88, 0.18, 0.16],
        [0.12, 0.11, 0.88, 1.0, 0.17, 0.15],
        [0.05, 0.07, 0.18, 0.17, 1.0, 0.90],
        [0.08, 0.06, 0.16, 0.15, 0.90, 1.0],
    ]
    return pd.DataFrame(data, index=labels, columns=labels)


class TestLowCorrelationSelector:
    def test_validate_and_prepare_accepts_symmetric_matrix(self):
        prepared, report = validate_and_prepare_correlation_matrix(_make_corr_matrix())
        assert prepared.shape == (6, 6)
        assert report.asset_count == 6
        assert report.symmetry_error == 0.0

    def test_build_cluster_summary_contains_expected_columns(self):
        corr = _make_corr_matrix()
        prepared, _ = validate_and_prepare_correlation_matrix(corr)
        cluster_membership = pd.Series([0, 0, 1, 1, 2, 2], index=prepared.index)
        summary = build_cluster_summary(prepared, cluster_membership)
        assert "selection_score" in summary.columns
        assert "cluster_rank" in summary.columns
        assert summary.loc["A1", "cluster_size"] == 2

    def test_selection_respects_target_size_and_cluster_caps(self):
        result = select_low_correlation_portfolio(
            _make_corr_matrix(),
            LowCorrelationSelectorConfig(
                target_size=4,
                cluster_count=3,
                max_per_cluster=1,
                min_cluster_count=3,
            ),
        )
        assert len(result.selected_symbols) == 3
        assert result.selected_frame["cluster_label"].nunique() == 3

    def test_selection_metrics_are_finite(self):
        result = select_low_correlation_portfolio(
            _make_corr_matrix(),
            LowCorrelationSelectorConfig(
                target_size=3,
                cluster_count=3,
                max_per_cluster=1,
            ),
        )
        assert result.metrics["average_pairwise_correlation"] >= 0.0
        assert result.metrics["effective_diversification_number"] > 0.0

    def test_family_classifier_covers_basic_keywords(self):
        assert classify_symbol_family("30年国债ETF鹏扬") == "bond"
        assert classify_symbol_family("黄金ETF华安") == "commodity"
        assert classify_symbol_family("纳指生物科技ETF汇添富") == "healthcare"
        assert classify_symbol_family("金融科技ETF华宝") == "technology"

    def test_selection_respects_family_caps(self):
        result = select_low_correlation_portfolio(
            _make_corr_matrix(),
            LowCorrelationSelectorConfig(
                target_size=6,
                cluster_count=6,
                max_per_cluster=1,
                max_per_family=1,
            ),
        )
        families = result.selected_frame["family_label"].value_counts()
        assert int(families.max()) <= 1