"""测试 forward_returns.py 和 quality.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factor_analysis.forward_returns import compute_forward_returns
from factor_analysis.quality import (
    compute_coverage, compute_distribution_stats, compute_daily_percentiles,
    compute_autocorr, compute_missing_by_volume_group, run_quality_analysis,
)
from factor_analysis.panel import FactorPanel


def _make_test_panel(n_symbols=10, n_dates=100) -> FactorPanel:
    dates = pd.date_range("2020-01-01", periods=n_dates)
    fv = pd.DataFrame(
        np.random.randn(n_dates, n_symbols) * 0.1 + 0.02,
        index=dates,
        columns=[f"S{i}" for i in range(n_symbols)],
    )
    cp = pd.DataFrame(
        100.0 + np.cumsum(np.random.randn(n_dates, n_symbols) * 0.5, axis=0),
        index=dates,
        columns=fv.columns,
    )
    vol = pd.DataFrame(
        np.abs(np.random.randn(n_dates, n_symbols)) * 1e6 + 5e6,
        index=dates,
        columns=fv.columns,
    )
    meta = pd.DataFrame(
        {"bar_count": [n_dates] * n_symbols},
        index=fv.columns,
    )
    return FactorPanel(factor_values=fv, close_prices=cp, volumes=vol, symbol_meta=meta, factor_name="test")


class TestForwardReturns:
    def test_basic(self):
        dates = pd.date_range("2020-01-01", periods=10)
        cp = pd.DataFrame({"A": [100.0 + i for i in range(10)]}, index=dates)
        fwd = compute_forward_returns(cp, periods=(2,))
        assert 2 in fwd
        assert fwd[2].shape == (10, 1)
        # 前 8 行有效 (10 - 2 = 8), 最后 2 行 NaN
        assert fwd[2].iloc[0, 0] == pytest.approx((102.0 / 100.0) - 1.0)

    def test_multiple_periods(self):
        dates = pd.date_range("2020-01-01", periods=20)
        cp = pd.DataFrame({"A": 100.0 + np.arange(20)}, index=dates)
        fwd = compute_forward_returns(cp, periods=(1, 5, 10))
        assert set(fwd.keys()) == {1, 5, 10}


class TestQuality:
    def test_coverage(self):
        panel = _make_test_panel(10, 100)
        cov = compute_coverage(panel)
        assert len(cov) == 100
        assert cov.max() <= 1.0
        assert cov.min() >= 0.0

    def test_distribution_stats(self):
        panel = _make_test_panel(10, 100)
        stats = compute_distribution_stats(panel)
        assert "mean" in stats
        assert "std" in stats
        assert "skewness" in stats
        assert "p50" in stats

    def test_daily_percentiles(self):
        panel = _make_test_panel(10, 100)
        dp = compute_daily_percentiles(panel)
        assert dp.shape == (100, 5)
        assert "P50" in dp.columns

    def test_autocorr(self):
        panel = _make_test_panel(10, 200)
        ac = compute_autocorr(panel, lags=(1, 5))
        assert len(ac) == 2
        assert "mean_autocorr" in ac.columns

    def test_missing_by_volume(self):
        panel = _make_test_panel(10, 100)
        result = compute_missing_by_volume_group(panel, n_groups=3)
        assert len(result) == 3

    def test_run_quality_analysis(self):
        panel = _make_test_panel(10, 100)
        results = run_quality_analysis(panel)
        assert "coverage" in results
        assert "distribution_stats" in results
        assert "daily_percentiles" in results
        assert "autocorr" in results
