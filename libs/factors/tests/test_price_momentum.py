"""价格动量族因子单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/factors/tests/test_price_momentum.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.price_momentum import (
    HighPointPosition,
    IntradayMomentum,
    LowPointPosition,
    OvernightReturn,
    RiskAdjustedReturn,
    TimeSeriesMomentum,
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
    """构建一个最小 OHLC DataFrame，未指定的列用 close 填充。"""
    if close is None:
        close = [100.0] * 5
    close_s = pd.Series(close, dtype=float)
    data: dict[str, pd.Series] = {"close": close_s}
    data["open"] = pd.Series(open_, dtype=float) if open_ is not None else close_s.copy()
    data["high"] = pd.Series(high, dtype=float) if high is not None else close_s.copy()
    data["low"] = pd.Series(low, dtype=float) if low is not None else close_s.copy()
    return pd.DataFrame(data)


# ===================================================================
# RiskAdjustedReturn
# ===================================================================

class TestRiskAdjustedReturn:
    def test_output_name(self):
        f = RiskAdjustedReturn(window=20)
        assert f.get_output_name() == "RiskAdjustedReturn_20"

    def test_warmup_period(self):
        f = RiskAdjustedReturn(window=20)
        assert f.warmup_period == 21

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            RiskAdjustedReturn(window=1)

    def test_rejects_missing_close(self):
        f = RiskAdjustedReturn(window=5)
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_returns_nan_when_volatility_is_zero(self):
        """纯横盘数据 → 波动率为 0 → 全部返回 NaN。"""
        data = _make_ohlc_frame(close=[10.0] * 30)
        f = RiskAdjustedReturn(window=5)
        result = f(data)
        assert result.notna().sum() == 0

    def test_computes_expected_values(self):
        """对简单线性序列验证计算值。"""
        close = [100.0 + i for i in range(10)]  # 100 → 109
        data = _make_ohlc_frame(close=close)
        f = RiskAdjustedReturn(window=3)
        result = f(data)

        # index 3: 窗口 [101, 102, 103]
        # N_ret = 103 / 100 - 1 = 0.03
        # daily_ret = [nan, 0.01, 0.00990099, 0.00980392]
        # rolling std @ index 3: std([0.01, 0.00990099, 0.00980392])
        # N_vol = std * sqrt(3)
        expected_ret = 103.0 / 100.0 - 1.0
        rets = pd.Series(close, dtype=float).pct_change()
        expected_vol = rets.iloc[1:4].std() * np.sqrt(3)
        expected = expected_ret / expected_vol

        assert result.iloc[3] == pytest.approx(expected, rel=1e-9)

    def test_params_immutable_across_instances(self):
        f1 = RiskAdjustedReturn(window=20)
        f2 = RiskAdjustedReturn(window=60)
        assert f1.window == 20
        assert f2.window == 60
        assert f1.params["window"] == 20
        assert f2.params["window"] == 60


# ===================================================================
# IntradayMomentum
# ===================================================================

class TestIntradayMomentum:
    def test_output_name(self):
        f = IntradayMomentum()
        assert f.get_output_name() == "IntradayMomentum"

    def test_warmup_period(self):
        f = IntradayMomentum()
        assert f.warmup_period == 0

    def test_rejects_missing_open(self):
        f = IntradayMomentum()
        bad = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'open'"):
            f(bad)

    def test_rejects_missing_close(self):
        f = IntradayMomentum()
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_computes_expected_values(self):
        data = _make_ohlc_frame(
            close=[102.0, 100.0, 103.0],
            open_=[100.0, 101.0, 102.0],
        )
        f = IntradayMomentum()
        result = f(data)

        assert result.iloc[0] == pytest.approx(0.02)  # (102-100)/100
        assert result.iloc[1] == pytest.approx(-0.0099009900990099)  # (100-101)/101
        assert result.iloc[2] == pytest.approx(0.00980392156862745)  # (103-102)/102

    def test_returns_nan_for_zero_open(self):
        data = _make_ohlc_frame(
            close=[100.0, 101.0, 102.0],
            open_=[100.0, 0.0, 102.0],
        )
        f = IntradayMomentum()
        result = f(data)
        assert result.iloc[0] == pytest.approx(0.0)
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(0.0)


# ===================================================================
# OvernightReturn
# ===================================================================

class TestOvernightReturn:
    def test_output_name(self):
        f = OvernightReturn()
        assert f.get_output_name() == "OvernightReturn"

    def test_warmup_period(self):
        f = OvernightReturn()
        assert f.warmup_period == 1

    def test_rejects_missing_open(self):
        f = OvernightReturn()
        bad = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'open'"):
            f(bad)

    def test_rejects_missing_close(self):
        f = OvernightReturn()
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_first_row_is_nan(self):
        data = _make_ohlc_frame(
            close=[100.0, 102.0],
            open_=[101.0, 99.0],
        )
        f = OvernightReturn()
        result = f(data)
        assert np.isnan(result.iloc[0])

    def test_computes_expected_values(self):
        data = _make_ohlc_frame(
            close=[100.0, 102.0, 103.0],
            open_=[101.0, 99.0, 104.0],
        )
        f = OvernightReturn()
        result = f(data)

        # day 1 (index=1): open=99, prev_close=100 → (99-100)/100 = -0.01
        assert result.iloc[1] == pytest.approx(-0.01)
        # day 2 (index=2): open=104, prev_close=102 → (104-102)/102 ≈ 0.019608
        assert result.iloc[2] == pytest.approx(0.0196078431372549)

    def test_returns_nan_for_zero_prev_close(self):
        data = _make_ohlc_frame(
            close=[0.0, 102.0],
            open_=[101.0, 99.0],
        )
        f = OvernightReturn()
        result = f(data)
        # 第一天：NaN（暖启动），第二天：prev_close = 0 → NaN
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])


# ===================================================================
# HighPointPosition
# ===================================================================

class TestHighPointPosition:
    def test_output_name(self):
        f = HighPointPosition(window=20)
        assert f.get_output_name() == "HighPointPosition_20"

    def test_warmup_period(self):
        f = HighPointPosition(window=20)
        assert f.warmup_period == 20

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            HighPointPosition(window=1)

    def test_rejects_missing_high(self):
        f = HighPointPosition(window=5)
        bad = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'high'"):
            f(bad)

    def test_returns_all_nan_when_data_shorter_than_window(self):
        high = [10.0, 12.0, 11.0]
        data = _make_ohlc_frame(high=high)
        f = HighPointPosition(window=5)
        result = f(data)
        assert result.notna().sum() == 0

    def test_monotonically_increasing_high(self):
        """单调递增 → 高点始终在窗口末端 → 归一化 = 1.0。"""
        high = list(range(1, 11))  # 1→10
        data = _make_ohlc_frame(high=high)
        f = HighPointPosition(window=3)
        result = f(data)

        # index 2: 窗口 [1,2,3] → max @ index 2 → 2/(3-1) = 1.0
        # index 3: 窗口 [2,3,4] → max @ index 2 → 1.0
        assert result.iloc[2] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(1.0)
        assert result.iloc[-1] == pytest.approx(1.0)

    def test_monotonically_decreasing_high(self):
        """单调递减 → 高点始终在窗口起始 → 归一化 = 0.0。"""
        high = list(range(10, 0, -1))  # 10→1
        data = _make_ohlc_frame(high=high)
        f = HighPointPosition(window=3)
        result = f(data)

        assert result.iloc[2] == pytest.approx(0.0)
        assert result.iloc[3] == pytest.approx(0.0)
        assert result.iloc[-1] == pytest.approx(0.0)

    def test_mixed_high_positions(self):
        high = [10.0, 12.0, 11.0, 13.0, 9.0]
        data = _make_ohlc_frame(high=high)
        f = HighPointPosition(window=3)
        result = f(data)

        # index 2: 窗口 [10, 12, 11] → max=12 @ idx 1 → 1/2 = 0.5
        assert result.iloc[2] == pytest.approx(0.5)
        # index 3: 窗口 [12, 11, 13] → max=13 @ idx 2 → 2/2 = 1.0
        assert result.iloc[3] == pytest.approx(1.0)
        # index 4: 窗口 [11, 13, 9] → max=13 @ idx 1 → 1/2 = 0.5
        assert result.iloc[4] == pytest.approx(0.5)

    def test_all_same_high(self):
        """所有 high 相同 → argmax 返回 0 → 归一化 = 0.0。"""
        high = [10.0] * 10
        data = _make_ohlc_frame(high=high)
        f = HighPointPosition(window=5)
        result = f(data)
        assert result.dropna().iloc[0] == pytest.approx(0.0)
        # 所有有效行都应返回 0
        assert (result.dropna() == 0.0).all()

    def test_params_immutable_across_instances(self):
        f1 = HighPointPosition(window=10)
        f2 = HighPointPosition(window=30)
        assert f1.window == 10
        assert f2.window == 30
        assert f1.params["window"] == 10


# ===================================================================
# LowPointPosition
# ===================================================================

class TestLowPointPosition:
    def test_output_name(self):
        f = LowPointPosition(window=20)
        assert f.get_output_name() == "LowPointPosition_20"

    def test_warmup_period(self):
        f = LowPointPosition(window=20)
        assert f.warmup_period == 20

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            LowPointPosition(window=1)

    def test_rejects_missing_low(self):
        f = LowPointPosition(window=5)
        bad = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'low'"):
            f(bad)

    def test_returns_all_nan_when_data_shorter_than_window(self):
        low = [10.0, 12.0, 11.0]
        data = _make_ohlc_frame(low=low)
        f = LowPointPosition(window=5)
        result = f(data)
        assert result.notna().sum() == 0

    def test_monotonically_increasing_low(self):
        """单调递增 → 低点始终在窗口起始 → 归一化 = 0.0。"""
        low = list(range(1, 11))  # 1→10
        data = _make_ohlc_frame(low=low)
        f = LowPointPosition(window=3)
        result = f(data)

        assert result.iloc[2] == pytest.approx(0.0)
        assert result.iloc[3] == pytest.approx(0.0)
        assert result.iloc[-1] == pytest.approx(0.0)

    def test_monotonically_decreasing_low(self):
        """单调递减 → 低点始终在窗口末端 → 归一化 = 1.0。"""
        low = list(range(10, 0, -1))  # 10→1
        data = _make_ohlc_frame(low=low)
        f = LowPointPosition(window=3)
        result = f(data)

        assert result.iloc[2] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(1.0)
        assert result.iloc[-1] == pytest.approx(1.0)

    def test_mixed_low_positions(self):
        low = [10.0, 8.0, 9.0, 7.0, 12.0]
        data = _make_ohlc_frame(low=low)
        f = LowPointPosition(window=3)
        result = f(data)

        # index 2: 窗口 [10, 8, 9] → min=8 @ idx 1 → 1/2 = 0.5
        assert result.iloc[2] == pytest.approx(0.5)
        # index 3: 窗口 [8, 9, 7] → min=7 @ idx 2 → 2/2 = 1.0
        assert result.iloc[3] == pytest.approx(1.0)
        # index 4: 窗口 [9, 7, 12] → min=7 @ idx 1 → 1/2 = 0.5
        assert result.iloc[4] == pytest.approx(0.5)

    def test_all_same_low(self):
        """所有 low 相同 → argmin 返回 0 → 归一化 = 0.0。"""
        low = [10.0] * 10
        data = _make_ohlc_frame(low=low)
        f = LowPointPosition(window=5)
        result = f(data)
        assert result.dropna().iloc[0] == pytest.approx(0.0)
        assert (result.dropna() == 0.0).all()


# ===================================================================
# TimeSeriesMomentum
# ===================================================================

class TestTimeSeriesMomentum:
    def test_output_name(self):
        f = TimeSeriesMomentum(window=252)
        assert f.get_output_name() == "TimeSeriesMomentum_252"

    def test_warmup_period(self):
        f = TimeSeriesMomentum(window=252)
        assert f.warmup_period == 253

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            TimeSeriesMomentum(window=0)

    def test_rejects_missing_close(self):
        with pytest.raises(ValueError, match="requires column 'close'"):
            TimeSeriesMomentum(window=5)(pd.DataFrame({"open": [1.0, 2.0]}))

    def test_output_is_0_or_1(self):
        """输出应为 0 或 1"""
        close = [100.0 + i for i in range(20)]
        data = _make_ohlc_frame(close=close)
        f = TimeSeriesMomentum(window=5)
        result = f(data)
        valid = result.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_uptrend_returns_1_after_warmup(self):
        """持续上涨 → warmup 后时序动量 = 1"""
        close = list(range(100, 130))
        data = _make_ohlc_frame(close=close)
        f = TimeSeriesMomentum(window=5)
        result = f(data)
        valid = result.dropna()
        assert (valid == 1).all()
        assert len(valid) > 0

    def test_downtrend_returns_0(self):
        """持续下跌 → 时序动量 = 0"""
        close = list(range(130, 100, -1))
        data = _make_ohlc_frame(close=close)
        f = TimeSeriesMomentum(window=5)
        result = f(data)
        valid = result.dropna()
        assert (valid == 0).all()

    def test_first_pct_is_nan(self):
        """shift(N) 导致前 N 行为 NaN"""
        close = [100.0 + i for i in range(20)]
        data = _make_ohlc_frame(close=close)
        f = TimeSeriesMomentum(window=5)
        result = f(data)
        assert result.iloc[:5].isna().all()
        assert result.iloc[5:].notna().all()

    def test_params_immutable(self):
        f1 = TimeSeriesMomentum(window=60)
        f2 = TimeSeriesMomentum(window=252)
        assert f1.window == 60
        assert f2.window == 252
        assert f1.params["window"] == 60
        assert f2.params["window"] == 252


# ===================================================================
# Integration: all factors run on standard OHLC data without error
# ===================================================================

class TestIntegration:
    def test_all_factors_run_on_standard_ohlc(self):
        dates = pd.date_range("2025-01-01", periods=30, freq="B")
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(30) * 0.5)
        data = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(30) * 0.002),
                "high": close * (1 + np.abs(np.random.randn(30) * 0.01)),
                "low": close * (1 - np.abs(np.random.randn(30) * 0.01)),
                "close": close,
            },
            index=dates,
        )

        factors = [
            RiskAdjustedReturn(window=20),
            IntradayMomentum(),
            OvernightReturn(),
            HighPointPosition(window=20),
            LowPointPosition(window=20),
            TimeSeriesMomentum(window=10),
        ]

        for f in factors:
            result = f(data)
            assert isinstance(result, pd.Series)
            assert result.name == f.get_output_name()
            assert len(result) == len(data)
