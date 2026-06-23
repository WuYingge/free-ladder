"""
panel.py 单元测试
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 确保 libs 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factor_analysis.panel import FactorPanel, PanelBuildError, build_factor_panel
from factors.price_return import PriceReturn


class TestFactorPanel:
    """FactorPanel 数据结构测试。"""

    def test_empty_panel_properties(self):
        fv = pd.DataFrame()
        cp = pd.DataFrame()
        vol = pd.DataFrame()
        meta = pd.DataFrame()
        panel = FactorPanel(
            factor_values=fv, close_prices=cp, volumes=vol,
            symbol_meta=meta, factor_name="test",
        )
        assert panel.n_symbols == 0
        assert panel.n_dates == 0

    def test_panel_properties(self):
        dates = pd.date_range("2020-01-01", periods=10)
        fv = pd.DataFrame({"A": np.random.randn(10), "B": np.random.randn(10)}, index=dates)
        cp = pd.DataFrame({"A": 100.0 + np.arange(10), "B": 50.0 + np.arange(10)}, index=dates)
        vol = pd.DataFrame({"A": 1e6 * np.ones(10), "B": 2e6 * np.ones(10)}, index=dates)
        meta = pd.DataFrame({"bar_count": [10, 10]}, index=["A", "B"])
        panel = FactorPanel(
            factor_values=fv, close_prices=cp, volumes=vol,
            symbol_meta=meta, factor_name="test_factor",
        )
        assert panel.n_symbols == 2
        assert panel.n_dates == 10
        assert panel.factor_name == "test_factor"
        assert panel.date_range[0] == pd.Timestamp("2020-01-01")

    def test_summary(self):
        dates = pd.date_range("2020-01-01", periods=5)
        fv = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0, 5.0]}, index=dates)
        cp = pd.DataFrame({"A": [100.0] * 5}, index=dates)
        vol = pd.DataFrame({"A": [1e6] * 5}, index=dates)
        meta = pd.DataFrame({"bar_count": [4]}, index=["A"])
        panel = FactorPanel(
            factor_values=fv, close_prices=cp, volumes=vol,
            symbol_meta=meta, factor_name="test",
        )
        s = panel.summary()
        assert s["n_symbols"] == 1
        assert s["n_dates"] == 5
        assert s["coverage_mean"] == 4.0 / 5.0  # 4 valid out of 5


class TestBuildFactorPanel:
    """面板构建函数测试。"""

    def test_build_with_small_symbol_set(self):
        """用 4 个标的 + 低 min_bars 测试面板构建流程。"""
        factor = PriceReturn(window=60)
        symbols = ["510300", "510500", "159915", "518880"]
        panel = build_factor_panel(
            factor, symbols, min_bars=200, max_workers=2,
        )
        assert panel.n_symbols == 4
        assert panel.n_dates > 1000
        assert panel.factor_name == "PriceReturn_60"
        assert panel.factor_values.shape == (panel.n_dates, 4)
        assert len(panel.errors) == 0

    def test_min_bars_filtering(self):
        """验证 min_bars 过滤逻辑：bar_count 低于阈值的标的被剔除。"""
        factor = PriceReturn(window=60)
        symbols = ["510300", "510500", "159915", "518880"]
        # 用极高 min_bars 期望全部被过滤
        with pytest.raises(RuntimeError, match="过滤后无剩余标的"):
            build_factor_panel(factor, symbols, min_bars=10000, max_workers=2)
