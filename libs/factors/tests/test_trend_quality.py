"""趋势质量族因子单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/factors/tests/test_trend_quality.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.trend_quality import (
    HurstExponent,
    KaufmanEfficiencyRatio,
    UpDownRatio,
    ConsecutiveUpDays,
    ConsecutiveDownDays,
    ADX,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ohlc_frame(
    close: list[float] | None = None,
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
) -> pd.DataFrame:
    if close is None:
        close = [100.0] * 5
    close_s = pd.Series(close, dtype=float)
    data: dict[str, pd.Series] = {"close": close_s}
    data["open"] = pd.Series(open_, dtype=float) if open_ is not None else close_s.copy()
    data["high"] = pd.Series(high, dtype=float) if high is not None else close_s.copy()
    data["low"] = pd.Series(low, dtype=float) if low is not None else close_s.copy()
    return pd.DataFrame(data)


# ===================================================================
# HurstExponent
# ===================================================================

class TestHurstExponent:
    def test_output_name(self):
        f = HurstExponent(window=120)
        assert f.get_output_name() == "HurstExponent_120"

    def test_warmup_period(self):
        f = HurstExponent(window=120)
        assert f.warmup_period == 120

    def test_rejects_window_lt_4(self):
        with pytest.raises(ValueError, match="window must be at least 4"):
            HurstExponent(window=3)

    def test_rejects_missing_close(self):
        f = HurstExponent(window=20)
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_warmup_is_nan(self):
        n = 50
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        data = _make_ohlc_frame(close=close.tolist())
        f = HurstExponent(window=10)
        result = f(data)
        # rolling(window=10) 第一个有效值在 index=9
        assert result.iloc[:9].isna().all()

    def test_random_walk_near_0_5(self):
        """随机游走的 Hurst 指数应接近 0.5"""
        rng = np.random.default_rng(42)
        n = 200
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        data = _make_ohlc_frame(close=close.tolist())
        f = HurstExponent(window=100)
        result = f(data)
        valid = result.dropna()
        # 均值应接近 0.5
        assert 0.3 < valid.mean() < 0.7

    def test_trending_series_higher_hurst(self):
        """纯趋势序列的 Hurst 指数应 > 0.5"""
        n = 200
        close = [100.0 + i * 0.1 for i in range(n)]  # 纯直线趋势
        data = _make_ohlc_frame(close=close)
        f = HurstExponent(window=100)
        result = f(data)
        valid = result.dropna()
        assert (valid > 0.5).all()

    def test_params_immutable(self):
        f1 = HurstExponent(window=120)
        f2 = HurstExponent(window=60)
        assert f1.window == 120
        assert f2.window == 60


# ===================================================================
# KaufmanEfficiencyRatio
# ===================================================================

class TestKaufmanEfficiencyRatio:
    def test_output_name(self):
        f = KaufmanEfficiencyRatio(window=20)
        assert f.get_output_name() == "KaufmanER_20"

    def test_warmup_period(self):
        f = KaufmanEfficiencyRatio(window=20)
        assert f.warmup_period == 20

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            KaufmanEfficiencyRatio(window=1)

    def test_rejects_missing_close(self):
        f = KaufmanEfficiencyRatio(window=10)
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_straight_line_is_1(self):
        """纯直线趋势 → 效率 = 1"""
        close = [100.0 + i for i in range(30)]
        data = _make_ohlc_frame(close=close)
        f = KaufmanEfficiencyRatio(window=10)
        result = f(data)
        valid = result.dropna()
        assert (valid == 1.0).all()

    def test_oscillation_low_efficiency(self):
        """来回震荡 → 效率 < 1"""
        close = [100.0, 101.0, 99.0, 101.0, 99.0, 101.0, 99.0, 101.0, 99.0, 101.0, 99.0]
        data = _make_ohlc_frame(close=close)
        f = KaufmanEfficiencyRatio(window=5)
        result = f(data)
        valid = result.dropna()
        assert (valid < 1.0).all()

    def test_output_in_0_1_range(self):
        rng = np.random.default_rng(42)
        n = 100
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        data = _make_ohlc_frame(close=close.tolist())
        f = KaufmanEfficiencyRatio(window=20)
        result = f(data)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_params_immutable(self):
        f1 = KaufmanEfficiencyRatio(window=20)
        f2 = KaufmanEfficiencyRatio(window=60)
        assert f1.window == 20
        assert f2.window == 60


# ===================================================================
# UpDownRatio
# ===================================================================

class TestUpDownRatio:
    def test_output_name(self):
        f = UpDownRatio(window=20)
        assert f.get_output_name() == "UpDownRatio_20"

    def test_warmup_period(self):
        f = UpDownRatio(window=20)
        assert f.warmup_period == 20

    def test_rejects_missing_close(self):
        f = UpDownRatio(window=10)
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_all_up_is_1(self):
        """连续上涨 → 涨跌比 = 1（除首个有效值外）"""
        close = [100.0 + i for i in range(30)]
        data = _make_ohlc_frame(close=close)
        f = UpDownRatio(window=5)
        result = f(data)
        valid = result.dropna()
        # 首个有效值 (index=5)：up[1..5] = [1,1,1,1,1], ratio=5/5=1.0
        assert (valid == 1.0).all()

    def test_all_down_is_0(self):
        """连续下跌 → 涨跌比 = 0"""
        close = [100.0 - i for i in range(30)]
        data = _make_ohlc_frame(close=close)
        f = UpDownRatio(window=5)
        result = f(data)
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_output_in_0_1_range(self):
        rng = np.random.default_rng(42)
        n = 100
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        data = _make_ohlc_frame(close=close.tolist())
        f = UpDownRatio(window=20)
        result = f(data)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_params_immutable(self):
        f1 = UpDownRatio(window=20)
        f2 = UpDownRatio(window=60)
        assert f1.window == 20
        assert f2.window == 60


# ===================================================================
# ConsecutiveUpDays
# ===================================================================

class TestConsecutiveUpDays:
    def test_output_name(self):
        f = ConsecutiveUpDays()
        assert f.get_output_name() == "ConsecutiveUpDays"

    def test_warmup_period(self):
        f = ConsecutiveUpDays()
        assert f.warmup_period == 1

    def test_rejects_missing_close(self):
        f = ConsecutiveUpDays()
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_consecutive_counts(self):
        """连续上涨 3 天，第 4 天下跌 → 连涨天数依次为 0,1,2,0"""
        close = [100.0, 101.0, 102.0, 103.0, 100.0]
        data = _make_ohlc_frame(close=close)
        f = ConsecutiveUpDays()
        result = f(data)
        assert result.iloc[0] == 0
        assert result.iloc[1] == 1
        assert result.iloc[2] == 2
        assert result.iloc[3] == 3
        assert result.iloc[4] == 0

    def test_flat_day_breaks_streak(self):
        """平盘（close == prev_close）应断开连涨"""
        close = [100.0, 101.0, 101.0, 102.0]
        data = _make_ohlc_frame(close=close)
        f = ConsecutiveUpDays()
        result = f(data)
        assert result.iloc[0] == 0
        assert result.iloc[1] == 1
        assert result.iloc[2] == 0  # 平盘断开
        assert result.iloc[3] == 1


# ===================================================================
# ConsecutiveDownDays
# ===================================================================

class TestConsecutiveDownDays:
    def test_output_name(self):
        f = ConsecutiveDownDays()
        assert f.get_output_name() == "ConsecutiveDownDays"

    def test_warmup_period(self):
        f = ConsecutiveDownDays()
        assert f.warmup_period == 1

    def test_rejects_missing_close(self):
        f = ConsecutiveDownDays()
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_consecutive_counts(self):
        """连续下跌 3 天 → 连跌天数依次为 0,1,2"""
        close = [100.0, 99.0, 98.0, 97.0, 101.0]
        data = _make_ohlc_frame(close=close)
        f = ConsecutiveDownDays()
        result = f(data)
        assert result.iloc[0] == 0
        assert result.iloc[1] == 1
        assert result.iloc[2] == 2
        assert result.iloc[3] == 3
        assert result.iloc[4] == 0  # 上涨断开

    def test_flat_day_breaks_streak(self):
        """平盘应断开连跌"""
        close = [100.0, 99.0, 99.0, 98.0]
        data = _make_ohlc_frame(close=close)
        f = ConsecutiveDownDays()
        result = f(data)
        assert result.iloc[0] == 0
        assert result.iloc[1] == 1
        assert result.iloc[2] == 0  # 平盘断开
        assert result.iloc[3] == 1


# ===================================================================
# ADX
# ===================================================================

class TestADX:
    def test_output_name(self):
        f = ADX(window=14, output="adx")
        assert f.get_output_name() == "ADX_14_adx"

    def test_output_name_plus_di(self):
        f = ADX(window=14, output="plus_di")
        assert f.get_output_name() == "ADX_14_plus_di"

    def test_warmup_period(self):
        f = ADX(window=14)
        assert f.warmup_period == 26  # 2 * 14 - 2

    def test_rejects_bad_output(self):
        with pytest.raises(ValueError, match="output must be one of"):
            ADX(window=14, output="bad")

    def test_rejects_missing_columns(self):
        f = ADX(window=14)
        bad = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'high'"):
            f(bad)

    def test_warmup_is_nan(self):
        n = 50
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        high = close + np.abs(rng.normal(0, 0.2, n))
        low = close - np.abs(rng.normal(0, 0.2, n))
        open_ = close + rng.normal(0, 0.1, n)
        data = _make_ohlc_frame(
            close=close.tolist(),
            open_=open_.tolist(),
            high=high.tolist(),
            low=low.tolist(),
        )
        f = ADX(window=14)
        result = f(data)
        # warmup = 2*window-2 = 26，index 0-25 应为 NaN
        assert result.iloc[:26].isna().all()

    def test_adx_positive_for_trending(self):
        """趋势行情中 ADX 应 > 0"""
        n = 100
        rng = np.random.default_rng(42)
        close_arr = 100.0 + np.cumsum(rng.normal(0.1, 0.5, n))  # 轻微向上趋势
        high = close_arr + np.abs(rng.normal(0, 0.3, n))
        low = close_arr - np.abs(rng.normal(0, 0.3, n))
        close_series = pd.Series(close_arr)
        open_ = close_series.shift(1).fillna(100.0).to_numpy() + rng.normal(0, 0.1, n)
        data = _make_ohlc_frame(
            close=close_arr.tolist(),
            open_=open_.tolist(),
            high=high.tolist(),
            low=low.tolist(),
        )
        f = ADX(window=14, output="adx")
        result = f(data)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_all_outputs_work(self):
        """验证所有 output 模式都能正常产出"""
        n = 100
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        high = close + np.abs(rng.normal(0, 0.2, n))
        low = close - np.abs(rng.normal(0, 0.2, n))
        open_ = close + rng.normal(0, 0.1, n)
        data = _make_ohlc_frame(
            close=close.tolist(),
            open_=open_.tolist(),
            high=high.tolist(),
            low=low.tolist(),
        )
        for out in ("adx", "plus_di", "minus_di", "dx"):
            f = ADX(window=14, output=out)
            result = f(data)
            assert isinstance(result, pd.Series)
            assert result.name == f.get_output_name()
            assert len(result) == n
            assert result.dropna().notna().sum() > 0

    def test_dx_equals_diff_over_sum(self):
        """DX = abs(plus_di - minus_di) / (plus_di + minus_di) * 100"""
        n = 100
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        high = close + np.abs(rng.normal(0, 0.2, n))
        low = close - np.abs(rng.normal(0, 0.2, n))
        open_ = close + rng.normal(0, 0.1, n)
        data = _make_ohlc_frame(
            close=close.tolist(),
            open_=open_.tolist(),
            high=high.tolist(),
            low=low.tolist(),
        )
        dx_result = ADX(window=14, output="dx")(data)
        plus_di_result = ADX(window=14, output="plus_di")(data)
        minus_di_result = ADX(window=14, output="minus_di")(data)
        expected_dx = (
            (plus_di_result - minus_di_result).abs()
            / (plus_di_result + minus_di_result).replace(0, np.nan)
            * 100.0
        )
        valid = dx_result.dropna().index.intersection(expected_dx.dropna().index)
        pd.testing.assert_series_equal(
            dx_result.loc[valid], expected_dx.loc[valid], check_names=False
        )

    def test_params_immutable(self):
        f1 = ADX(window=14, output="adx")
        f2 = ADX(window=20, output="dx")
        assert f1.window == 14
        assert f2.window == 20
        assert f1.output == "adx"
        assert f2.output == "dx"
