"""测试 predictive.py 和 grouping.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factor_analysis.panel import FactorPanel
from factor_analysis.forward_returns import compute_forward_returns
from factor_analysis.predictive import (
    _daily_ic, _ic_summary, compute_rank_ic, compute_pearson_ic,
    compute_ic_decay, compute_rolling_ic, run_predictive_analysis,
)
from factor_analysis.grouping import (
    _assign_quantile_group, compute_quantile_returns, compute_longshort,
    compute_monotonicity, run_grouping_analysis,
)


def _make_test_panel(n_symbols=20, n_dates=200, seed=42) -> FactorPanel:
    """生成本地测试用面板，因子值与前向收益有弱正相关性。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates)
    symbols = [f"S{i}" for i in range(n_symbols)]

    # 因子值：正负随机的截面值
    fv = pd.DataFrame(
        rng.normal(0.02, 0.10, (n_dates, n_symbols)),
        index=dates, columns=symbols,
    )
    cp = pd.DataFrame(
        100.0 + np.cumsum(rng.normal(0, 1.5, (n_dates, n_symbols)), axis=0),
        index=dates, columns=symbols,
    )
    vol = pd.DataFrame(
        rng.uniform(1e6, 1e8, (n_dates, n_symbols)),
        index=dates, columns=symbols,
    )
    meta = pd.DataFrame({"bar_count": [n_dates] * n_symbols}, index=symbols)
    return FactorPanel(factor_values=fv, close_prices=cp, volumes=vol, symbol_meta=meta, factor_name="test")


class TestPredictive:
    def test_daily_ic(self):
        panel = _make_test_panel(20, 200)
        fwd = compute_forward_returns(panel.close_prices, periods=(20,))
        ic = _daily_ic(panel.factor_values, fwd[20])
        assert len(ic) == 200
        assert not ic.isna().all()

    def test_ic_summary(self):
        rng = np.random.default_rng(0)
        ic = pd.Series(rng.normal(0.03, 0.15, 100))
        s = _ic_summary(ic)
        assert abs(s["mean"]) < 1.0
        assert s["n_days"] == 100

    def test_compute_rank_ic(self):
        panel = _make_test_panel(20, 200)
        fwd = compute_forward_returns(panel.close_prices, periods=(20,))
        result = compute_rank_ic(panel, fwd[20])
        assert "ic_series" in result
        assert "summary" in result

    def test_compute_pearson_ic(self):
        panel = _make_test_panel(20, 200)
        fwd = compute_forward_returns(panel.close_prices, periods=(20,))
        result = compute_pearson_ic(panel, fwd[20])
        assert "summary" in result

    def test_ic_decay(self):
        panel = _make_test_panel(20, 200)
        fwd = compute_forward_returns(panel.close_prices, periods=(5, 20, 60))
        decay = compute_ic_decay(panel, fwd)
        assert len(decay) == 3
        assert "ic_mean" in decay.columns

    def test_rolling_ic(self):
        panel = _make_test_panel(20, 200)
        fwd = compute_forward_returns(panel.close_prices, periods=(20,))
        ric = compute_rolling_ic(panel, fwd[20], window=60)
        assert "ic_raw" in ric.columns
        assert "ic_rolling_mean" in ric.columns

    def test_run_predictive_analysis(self):
        panel = _make_test_panel(20, 200)
        fwd = compute_forward_returns(panel.close_prices, periods=(5, 20, 60))
        results = run_predictive_analysis(panel, fwd)
        assert "rank_ic" in results
        assert "pearson_ic" in results
        assert "ic_decay" in results
        # rolling_ic 现在是 {period: DataFrame}
        assert "rolling_ic" in results
        assert isinstance(results["rolling_ic"], dict)
        assert set(results["rolling_ic"].keys()) == {5, 20, 60}
        # rank_ic 现在是 {period: dict}，包含所有 fwd_returns_map 中的持仓期
        assert isinstance(results["rank_ic"], dict)
        assert set(results["rank_ic"].keys()) == {5, 20, 60}
        # 每个 period 的 result 应有 ic_series 和 summary
        for period in (5, 20, 60):
            assert "ic_series" in results["rank_ic"][period]
            assert "summary" in results["rank_ic"][period]
            assert "mean" in results["rank_ic"][period]["summary"]


class TestGrouping:
    def test_assign_quantile_group(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        groups = _assign_quantile_group(series, 3)
        assert set(groups.dropna().unique()) == {1, 2, 3}

    def test_assign_quantile_group_with_nan(self):
        series = pd.Series([1, np.nan, 3, 4, np.nan, 6, 7, 8, 9, 10])
        groups = _assign_quantile_group(series, 3)
        assert groups.isna().sum() == 2

    def test_compute_quantile_returns(self):
        panel = _make_test_panel(30, 200)
        fwd = compute_forward_returns(panel.close_prices, periods=(20,))
        qr = compute_quantile_returns(panel, fwd[20], n_quantiles=5)
        assert qr.shape[1] == 5
        assert all(c.startswith("Q") for c in qr.columns)

    def test_compute_longshort(self):
        dates = pd.date_range("2020-01-01", periods=100)
        qr = pd.DataFrame({
            "Q1": np.random.randn(100) * 0.01,
            "Q2": np.random.randn(100) * 0.01 + 0.001,
            "Q3": np.random.randn(100) * 0.01 + 0.002,
            "Q4": np.random.randn(100) * 0.01 + 0.003,
            "Q5": np.random.randn(100) * 0.01 + 0.005,
        }, index=dates)
        ls = compute_longshort(qr)
        assert "ls_series" in ls
        assert "annualised_return" in ls
        assert "max_drawdown" in ls

    def test_monotonicity(self):
        dates = pd.date_range("2020-01-01", periods=100)
        # 构造严格递减的序列
        qr = pd.DataFrame({
            "Q1": np.ones(100) * 0.005,
            "Q2": np.ones(100) * 0.003,
            "Q3": np.ones(100) * 0.001,
            "Q4": np.ones(100) * -0.001,
            "Q5": np.ones(100) * -0.003,
        }, index=dates)
        mono = compute_monotonicity(qr)
        assert mono["strict_monotonic_ratio"] == 1.0
        assert mono["monotonic_direction"] == "decreasing"

    def test_run_grouping_analysis(self):
        panel = _make_test_panel(30, 200)
        fwd = compute_forward_returns(panel.close_prices, periods=(5, 20, 60))
        results = run_grouping_analysis(panel, fwd, n_quantiles=5)
        # 新结构: {period: {quantile_returns, quantile_summary, longshort, ...}}
        assert isinstance(results, dict)
        assert set(results.keys()) == {5, 20, 60}
        for period in (5, 20, 60):
            gr = results[period]
            assert "quantile_returns" in gr
            assert "quantile_summary" in gr
            assert "longshort" in gr
            assert "quantile_cumret" in gr
            assert "monotonicity" in gr