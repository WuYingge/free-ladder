"""超买超卖族因子单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/factors/tests/test_oscillator.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.oscillator import (
    RSI,
    Stochastic,
    CCI,
    WilliamsR,
    MFI,
    UltimateOscillator,
)


def _make_df(close=None, open_=None, high=None, low=None, volume=None, value=None):
    data = {"close": pd.Series(close if close is not None else [100.0] * 30, dtype=float)}
    data["open"] = pd.Series(open_, dtype=float) if open_ else data["close"].copy()
    data["high"] = pd.Series(high, dtype=float) if high else data["close"].copy()
    data["low"] = pd.Series(low, dtype=float) if low else data["close"].copy()
    data["volume"] = pd.Series(volume, dtype=float) if volume else pd.Series([1e6] * len(data["close"]), dtype=float)
    if value:
        data["value"] = pd.Series(value, dtype=float)
    return pd.DataFrame(data)


# ===================================================================
# RSI
# ===================================================================

class TestRSI:
    def test_output_name(self):
        f = RSI(window=14)
        assert f.get_output_name() == "RSI_14"

    def test_warmup_period(self):
        f = RSI(window=14)
        assert f.warmup_period == 15

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            RSI(window=0)

    def test_range_0_to_100(self):
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 0.5, 50))
        f = RSI(window=5)
        result = f(_make_df(close=close.tolist()))
        valid = result.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_only_ups_100(self):
        """只涨不跌 → RSI 接近 100"""
        close = [100.0 + i * 0.1 for i in range(30)]
        f = RSI(window=5)
        result = f(_make_df(close=close))
        valid = result.dropna()
        # 无下跌 → avg_loss=0 → RSI 趋于 100
        assert (valid > 80).all()

    def test_only_downs_0(self):
        """只跌不涨 → RSI 接近 0"""
        close = [100.0 - i * 0.1 for i in range(30)]
        f = RSI(window=5)
        result = f(_make_df(close=close))
        valid = result.dropna()
        assert (valid < 20).all()


# ===================================================================
# Stochastic
# ===================================================================

class TestStochastic:
    def test_output_name(self):
        f = Stochastic(n=14, m=3, output="K")
        assert f.get_output_name() == "Stochastic_14_3_K"

    def test_output_name_D(self):
        f = Stochastic(n=14, m=3, output="D")
        assert f.get_output_name() == "Stochastic_14_3_D"

    def test_warmup_period(self):
        f = Stochastic(n=14, m=3)
        assert f.warmup_period == 16

    def test_rejects_bad_output(self):
        with pytest.raises(ValueError, match="output must be 'K' or 'D'"):
            Stochastic(n=5, m=3, output="X")

    def test_range_0_to_100(self):
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 0.5, 50))
        high = close + np.abs(rng.normal(0, 0.2, 50))
        low = close - np.abs(rng.normal(0, 0.2, 50))
        f = Stochastic(n=5, m=3)
        result = f(_make_df(close=close.tolist(), high=high.tolist(), low=low.tolist()))
        valid = result.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_at_high_100(self):
        """close 在 N 日最高点时 %K = 100"""
        close = [100.0 + i * 0.5 for i in range(20)]
        high = close.copy()
        low = [90.0] * 20
        f = Stochastic(n=5, m=1, output="K")
        result = f(_make_df(close=close, high=high, low=low))
        valid = result.dropna()
        assert (valid > 90).all()

    def test_at_low_0(self):
        """close 在 N 日最低点时 %K = 0"""
        close = [90.0] * 20
        high = [100.0] * 20
        low = close.copy()
        f = Stochastic(n=5, m=1, output="K")
        result = f(_make_df(close=close, high=high, low=low))
        valid = result.dropna()
        assert (valid < 10).all()

    def test_D_is_smoothed_K(self):
        """%D 是 %K 的 MA"""
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 0.5, 50))
        high = close + np.abs(rng.normal(0, 0.2, 50))
        low = close - np.abs(rng.normal(0, 0.2, 50))
        df = _make_df(close=close.tolist(), high=high.tolist(), low=low.tolist())
        fk = Stochastic(n=5, m=3, output="K")
        fd = Stochastic(n=5, m=3, output="D")
        k = fk(df)
        d = fd(df)
        expected_d = k.rolling(3).mean()
        common = k.dropna().index.intersection(d.dropna().index).intersection(expected_d.dropna().index)
        assert np.allclose(d.loc[common], expected_d.loc[common], rtol=1e-10)


# ===================================================================
# CCI
# ===================================================================

class TestCCI:
    def test_output_name(self):
        f = CCI(window=20)
        assert f.get_output_name() == "CCI_20"

    def test_warmup_period(self):
        f = CCI(window=20)
        assert f.warmup_period == 20

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            CCI(window=1)

    def test_pure_trend_high_abs(self):
        """强趋势下 CCI 绝对值较大"""
        close = [100.0 + i * 0.5 for i in range(50)]
        high = [x + 0.2 for x in close]
        low = [x - 0.1 for x in close]
        f = CCI(window=10)
        result = f(_make_df(close=close, high=high, low=low))
        valid = result.dropna()
        # 持续上涨 → CCI 应该偏正
        assert valid.iloc[-10:].mean() > 0


# ===================================================================
# WilliamsR
# ===================================================================

class TestWilliamsR:
    def test_output_name(self):
        f = WilliamsR(window=14)
        assert f.get_output_name() == "WilliamsR_14"

    def test_warmup_period(self):
        f = WilliamsR(window=14)
        assert f.warmup_period == 13

    def test_range_neg100_to_0(self):
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 0.5, 50))
        high = close + np.abs(rng.normal(0, 0.2, 50))
        low = close - np.abs(rng.normal(0, 0.2, 50))
        f = WilliamsR(window=5)
        result = f(_make_df(close=close.tolist(), high=high.tolist(), low=low.tolist()))
        valid = result.dropna()
        assert valid.min() >= -100.0
        assert valid.max() <= 0.0


# ===================================================================
# MFI
# ===================================================================

class TestMFI:
    def test_output_name(self):
        f = MFI(window=14)
        assert f.get_output_name() == "MFI_14"

    def test_warmup_period(self):
        f = MFI(window=14)
        assert f.warmup_period == 15

    def test_requires_volume(self):
        with pytest.raises(ValueError, match="requires column 'volume'"):
            MFI(window=5)(pd.DataFrame({"close": [100.0], "high": [101.0], "low": [99.0]}))

    def test_range_0_to_100(self):
        rng = np.random.default_rng(42)
        n = 50
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        high = close + np.abs(rng.normal(0, 0.2, n))
        low = close - np.abs(rng.normal(0, 0.2, n))
        volume = rng.integers(100000, 1000000, n).astype(float)
        f = MFI(window=5)
        result = f(_make_df(close=close.tolist(), high=high.tolist(), low=low.tolist(), volume=volume.tolist()))
        valid = result.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0


# ===================================================================
# UltimateOscillator
# ===================================================================

class TestUltimateOscillator:
    def test_output_name(self):
        f = UltimateOscillator(short=7, mid=14, long=28)
        assert f.get_output_name() == "UltimateOscillator_7_14_28"

    def test_warmup_period(self):
        f = UltimateOscillator(short=7, mid=14, long=28)
        assert f.warmup_period == 29

    def test_range_0_to_100(self):
        rng = np.random.default_rng(42)
        n = 60
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        high = close + np.abs(rng.normal(0, 0.2, n))
        low = close - np.abs(rng.normal(0, 0.2, n))
        f = UltimateOscillator(short=3, mid=5, long=8)
        result = f(_make_df(close=close.tolist(), high=high.tolist(), low=low.tolist()))
        valid = result.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0
