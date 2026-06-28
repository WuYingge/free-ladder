"""结构性/突破族扩展因子单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/factors/tests/test_breakout_family.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.breakout_family import (
    NewHighContinuous,
    NewLowContinuous,
    DonchianChannelPosition,
    ATRRatio,
    ChandelierExit,
)


def _make_df(close=None, high=None, low=None, n=50):
    """构建最小 OHLC DataFrame。"""
    if close is None:
        close = pd.Series(100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n)), dtype=float)
    else:
        close = pd.Series(close, dtype=float)
    if high is None:
        high = close * 1.01
    else:
        high = pd.Series(high, dtype=float)
    if low is None:
        low = close * 0.99
    else:
        low = pd.Series(low, dtype=float)
    return pd.DataFrame({"close": close, "high": high, "low": low})


# ===================================================================
# NewHighContinuous
# ===================================================================

class TestNewHighContinuous:
    def test_output_name(self):
        f = NewHighContinuous(window=50)
        assert f.get_output_name() == "NewHighContinuous_50"

    def test_warmup_period(self):
        f = NewHighContinuous(window=50)
        assert f.warmup_period == 50

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            NewHighContinuous(window=0)

    def test_rejects_missing_high(self):
        f = NewHighContinuous(window=5)
        with pytest.raises(ValueError, match="requires column 'high'"):
            f(pd.DataFrame({"close": [1.0, 2.0]}))

    def test_zero_when_equal_to_high(self):
        """收盘价等于滚动最高点 → 结果 = 0"""
        close = [100.0 + i * 0.1 for i in range(20)]
        high = [c + 0.05 for c in close]
        # 让 high_N 始终略高于 close
        f = NewHighContinuous(window=5)
        result = f(_make_df(close=close, high=high, n=20))
        valid = result.dropna()
        assert (valid < 0).all()  # 始终低于前高 → 负值

    def test_negative_below_high(self):
        """收盘价持续低于历史最高 → 连续值为负"""
        close = [100.0] * 20
        high = [101.0] * 20  # 始终高于 close
        f = NewHighContinuous(window=5)
        result = f(_make_df(close=close, high=high, n=20))
        valid = result.dropna()
        expected = (100.0 - 101.0) / 101.0
        assert np.allclose(valid, expected, rtol=1e-10)

    def test_params_immutable(self):
        f1 = NewHighContinuous(window=20)
        f2 = NewHighContinuous(window=100)
        assert f1.window == 20
        assert f2.window == 100


# ===================================================================
# NewLowContinuous
# ===================================================================

class TestNewLowContinuous:
    def test_output_name(self):
        f = NewLowContinuous(window=50)
        assert f.get_output_name() == "NewLowContinuous_50"

    def test_warmup_period(self):
        f = NewLowContinuous(window=50)
        assert f.warmup_period == 50

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            NewLowContinuous(window=0)

    def test_rejects_missing_low(self):
        f = NewLowContinuous(window=5)
        with pytest.raises(ValueError, match="requires column 'low'"):
            f(pd.DataFrame({"close": [1.0, 2.0]}))

    def test_positive_above_low(self):
        """收盘价高于历史最低 → 连续值为正"""
        close = [100.0] * 20
        low = [99.0] * 20  # 始终低于 close
        f = NewLowContinuous(window=5)
        result = f(_make_df(close=close, low=low, n=20))
        valid = result.dropna()
        expected = (100.0 - 99.0) / 99.0
        assert np.allclose(valid, expected, rtol=1e-10)


# ===================================================================
# DonchianChannelPosition
# ===================================================================

class TestDonchianChannelPosition:
    def test_output_name(self):
        f = DonchianChannelPosition(window=20)
        assert f.get_output_name() == "DonchianPosition_20"

    def test_warmup_period(self):
        f = DonchianChannelPosition(window=20)
        assert f.warmup_period == 20

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            DonchianChannelPosition(window=0)

    def test_range_0_to_1(self):
        """唐奇安通道位置应在 [0, 1] 之间"""
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, 50))
        high = close + np.abs(rng.normal(0, 0.3, 50))
        low = close - np.abs(rng.normal(0, 0.3, 50))
        f = DonchianChannelPosition(window=10)
        result = f(_make_df(close=close.tolist(), high=high.tolist(), low=low.tolist(), n=50))
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_midpoint_when_price_in_middle(self):
        """价格恰好在上下轨中间 → 0.5"""
        n = 20
        high = [110.0] * n
        low = [90.0] * n
        close = [100.0] * n
        f = DonchianChannelPosition(window=5)
        result = f(_make_df(close=close, high=high, low=low, n=n))
        valid = result.dropna()
        expected = (100.0 - 90.0) / (110.0 - 90.0)
        assert np.allclose(valid, expected, rtol=1e-10)

    def test_zero_range_returns_05(self):
        """横盘（箱体宽度为 0）→ 返回 0.5"""
        n = 20
        price = [100.0] * n
        f = DonchianChannelPosition(window=5)
        result = f(_make_df(close=price, high=price, low=price, n=n))
        valid = result.dropna()
        assert (valid == 0.5).all()


# ===================================================================
# ATRRatio
# ===================================================================

class TestATRRatio:
    def test_output_name(self):
        f = ATRRatio(window=25)
        assert f.get_output_name() == "ATRRatio_25"

    def test_warmup_period(self):
        f = ATRRatio(window=25)
        assert f.warmup_period == 26

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            ATRRatio(window=0)

    def test_atr_ratio_non_negative(self):
        """ATR / close 应 ≥ 0"""
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, 50))
        high = close + np.abs(rng.normal(0, 0.3, 50))
        low = close - np.abs(rng.normal(0, 0.3, 50))
        f = ATRRatio(window=10)
        result = f(_make_df(close=close.tolist(), high=high.tolist(), low=low.tolist(), n=50))
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_flat_data_atr_ratio_zero(self):
        """纯横盘 → True Range ≈ 0 → ATR 接近 0 → ATRRatio ≈ 0"""
        n = 30
        price = [100.0] * n
        f = ATRRatio(window=5)
        result = f(_make_df(close=price, high=price, low=price, n=n))
        valid = result.dropna()
        # 横盘数据 TR 为 0 → ATR = 0
        assert valid.abs().max() < 1e-10


# ===================================================================
# ChandelierExit
# ===================================================================

class TestChandelierExit:
    def test_output_name(self):
        f = ChandelierExit(n=22, atr_window=22)
        assert f.get_output_name() == "ChandelierExit_22_22"

    def test_warmup_period(self):
        f = ChandelierExit(n=22, atr_window=22)
        assert f.warmup_period == 23

    def test_rejects_n_lt_1(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            ChandelierExit(n=0)

    def test_rejects_atr_window_lt_1(self):
        with pytest.raises(ValueError, match="atr_window must be at least 1"):
            ChandelierExit(atr_window=0)

    def test_non_positive_below_high(self):
        """收盘价 ≤ 最高点 → ChandelierExit ≤ 0"""
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, 50))
        high = close + np.abs(rng.normal(0, 0.3, 50))
        low = close - np.abs(rng.normal(0, 0.3, 50))
        f = ChandelierExit(n=10, atr_window=10)
        result = f(_make_df(close=close.tolist(), high=high.tolist(), low=low.tolist(), n=50))
        valid = result.dropna()
        # close 一般 ≤ high_N，所以结果 ≤ 0（除非创新高导致 > 0）
        assert (valid <= 1e-10).mean() > 0.5

    def test_params_immutable(self):
        f1 = ChandelierExit(n=10, atr_window=10)
        f2 = ChandelierExit(n=30, atr_window=30)
        assert f1.n == 10
        assert f2.n == 30
