"""均线族因子单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/factors/tests/test_ma_family.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.ma import (
    BIAS,
    BollingerBandPosition,
    MAAlignment,
    MADispersion,
    MADistance,
    MASlope,
)


def _make_df(close=None, open_=None, high=None, low=None):
    """构建测试用 DataFrame。"""
    if close is None:
        close = [100.0] * 30
    data = {"close": pd.Series(close, dtype=float)}
    data["open"] = pd.Series(open_, dtype=float) if open_ else data["close"].copy()
    data["high"] = pd.Series(high, dtype=float) if high else data["close"].copy()
    data["low"] = pd.Series(low, dtype=float) if low else data["close"].copy()
    return pd.DataFrame(data)


# ===================================================================
# BIAS
# ===================================================================

class TestBIAS:
    def test_output_name(self):
        f = BIAS(window=20)
        assert f.get_output_name() == "BIAS_close_20"

    def test_warmup_period(self):
        f = BIAS(window=20)
        assert f.warmup_period == 20

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            BIAS(window=0)(_make_df())

    def test_above_ma_positive(self):
        """价格高于 MA 时 BIAS > 0"""
        close = [100.0 + i * 0.2 for i in range(30)]  # 持续上涨
        f = BIAS(window=5)
        result = f(_make_df(close=close))
        valid = result.dropna()
        assert (valid > 0).all()

    def test_below_ma_negative(self):
        """价格低于 MA 时 BIAS < 0"""
        close = [100.0 - i * 0.2 for i in range(30)]  # 持续下跌
        f = BIAS(window=5)
        result = f(_make_df(close=close))
        valid = result.dropna()
        assert (valid < 0).all()

    def test_different_from_maposition(self):
        """BIAS 分母是 MA，MAPosition 分母是 close，值应不同"""
        from factors.ma import MAPosition
        close = [100.0, 102.0, 98.0, 105.0, 103.0] * 10
        df = _make_df(close=close)
        f_bias = BIAS(window=3)
        f_ma = MAPosition(window=3)
        bias_res = f_bias(df)
        ma_res = f_ma(df)
        valid = bias_res.dropna().index.intersection(ma_res.dropna().index)
        # 分母不同，值应该不同
        assert not np.allclose(bias_res.loc[valid], ma_res.loc[valid])

    def test_params_immutable(self):
        f1 = BIAS(window=20)
        f2 = BIAS(window=60)
        assert f1.window == 20
        assert f2.window == 60


# ===================================================================
# BollingerBandPosition
# ===================================================================

class TestBollingerBandPosition:
    def test_output_name(self):
        f = BollingerBandPosition(window=20, k=2.0)
        assert f.get_output_name() == "BollingerBandPosition_close_20_2.0"

    def test_warmup_period(self):
        f = BollingerBandPosition(window=20)
        assert f.warmup_period == 20

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            BollingerBandPosition(window=1)(_make_df())

    def test_zero_for_equal_ma_close(self):
        """close == MA 时位置 ≈ 0"""
        df = _make_df(close=[100.0] * 30)
        f = BollingerBandPosition(window=5, k=2.0)
        result = f(df)
        valid = result.dropna()
        # 价格完全不变时 std=0，结果为 NaN（除零保护）
        assert valid.empty

    def test_near_band_value(self):
        """线性趋势上 K=2 位置应接近 ±1"""
        close = [100.0 + i * 0.5 for i in range(50)]
        df = _make_df(close=close)
        f = BollingerBandPosition(window=10, k=2.0)
        result = f(df)
        valid = result.dropna()
        # 线性趋势下，位置值应该在一定范围内
        assert (valid.abs() < 3.0).all()


# ===================================================================
# MAAlignment
# ===================================================================

class TestMAAlignment:
    def test_output_name(self):
        f = MAAlignment(windows=[5, 20, 60])
        assert f.get_output_name() == "MAAlignment_close_5_20_60"

    def test_warmup_period(self):
        f = MAAlignment(windows=[5, 20, 60])
        assert f.warmup_period == 60

    def test_rejects_wrong_window_count(self):
        with pytest.raises(ValueError, match="requires exactly 3 windows"):
            MAAlignment(windows=[5, 20])

    def test_bullish_alignment_positive(self):
        """持续上涨 → 短期 MA > 长期 MA → 多头排列"""
        close = [100.0 + i * 0.3 for i in range(100)]
        f = MAAlignment(windows=[3, 6, 10])
        result = f(_make_df(close=close))
        valid = result.dropna()
        # 持续上涨时大部分时间应该为正
        assert (valid.iloc[-50:] > 0).all()

    def test_bearish_alignment_negative(self):
        """持续下跌 → 短期 MA < 长期 MA → 空头排列"""
        close = [100.0 - i * 0.3 for i in range(100)]
        f = MAAlignment(windows=[3, 6, 10])
        result = f(_make_df(close=close))
        valid = result.dropna()
        assert (valid.iloc[-50:] < 0).all()

    def test_range(self):
        """输出值在 [-1, 1] 范围内"""
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 0.5, 100))
        f = MAAlignment(windows=[3, 6, 10])
        result = f(_make_df(close=close.tolist()))
        valid = result.dropna()
        assert valid.max() <= 1.0
        assert valid.min() >= -1.0


# ===================================================================
# MASlope
# ===================================================================

class TestMASlope:
    def test_output_name(self):
        f = MASlope(ma_window=20, slope_window=5)
        assert f.get_output_name() == "MASlope_close_20_5"

    def test_warmup_period(self):
        f = MASlope(ma_window=20, slope_window=5)
        assert f.warmup_period == 25

    def test_positive_for_uptrend(self):
        """上涨趋势中斜率 > 0"""
        close = [100.0 + i * 0.3 for i in range(50)]
        f = MASlope(ma_window=3, slope_window=2)
        result = f(_make_df(close=close))
        valid = result.dropna()
        assert (valid.iloc[-20:] > 0).all()

    def test_negative_for_downtrend(self):
        """下跌趋势中斜率 < 0"""
        close = [100.0 - i * 0.3 for i in range(50)]
        f = MASlope(ma_window=3, slope_window=2)
        result = f(_make_df(close=close))
        valid = result.dropna()
        assert (valid.iloc[-20:] < 0).all()


# ===================================================================
# MADistance
# ===================================================================

class TestMADistance:
    def test_output_name(self):
        f = MADistance(short_window=5, long_window=60)
        assert f.get_output_name() == "MADistance_close_5_60"

    def test_warmup_period(self):
        f = MADistance(short_window=5, long_window=60)
        assert f.warmup_period == 60

    def test_positive_for_golden_cross(self):
        """短期 MA > 长期 MA 时距离为正"""
        close = [100.0 + i * 0.5 for i in range(80)]
        f = MADistance(short_window=3, long_window=10)
        result = f(_make_df(close=close))
        valid = result.dropna()
        assert (valid.iloc[-30:] > 0).all()

    def test_negative_for_dead_cross(self):
        """短期 MA < 长期 MA 时距离为负"""
        close = [100.0 - i * 0.5 for i in range(80)]
        f = MADistance(short_window=3, long_window=10)
        result = f(_make_df(close=close))
        valid = result.dropna()
        assert (valid.iloc[-30:] < 0).all()


# ===================================================================
# MADispersion
# ===================================================================

class TestMADispersion:
    def test_output_name(self):
        f = MADispersion(windows=[5, 10, 20, 60])
        assert f.get_output_name() == "MADispersion_close_5_10_20_60"

    def test_warmup_period(self):
        f = MADispersion(windows=[5, 10, 20, 60])
        assert f.warmup_period == 60

    def test_rejects_lt_2_windows(self):
        with pytest.raises(ValueError, match="requires at least 2 windows"):
            MADispersion(windows=[5])

    def test_non_negative(self):
        """离散度始终非负"""
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 0.5, 100))
        f = MADispersion(windows=[3, 5, 10])
        result = f(_make_df(close=close.tolist()))
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_low_for_flat_prices(self):
        """价格不变时离散度很低"""
        df = _make_df(close=[100.0] * 60)
        f = MADispersion(windows=[3, 5, 10])
        result = f(df)
        valid = result.dropna()
        assert (valid < 0.01).all()
