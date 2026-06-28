"""波动率族因子单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/factors/tests/test_volatility_family.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.volatility_family import (
    DownsideVolatility,
    ParkinsonVolatility,
    GarmanKlassVolatility,
    VolOfVol,
    MaxDrawdown,
    AvgDrawdown,
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
# DownsideVolatility
# ===================================================================

class TestDownsideVolatility:
    def test_output_name(self):
        f = DownsideVolatility(window=20)
        assert f.get_output_name() == "DownsideVolatility_20"

    def test_warmup_period(self):
        f = DownsideVolatility(window=20)
        assert f.warmup_period == 21

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            DownsideVolatility(window=1)

    def test_rejects_missing_close(self):
        f = DownsideVolatility(window=5)
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_only_downside_counted(self):
        """只上涨的序列 → 下行波动率为 0"""
        close = [100.0 + i * 0.1 for i in range(30)]  # 只涨不跌
        data = _make_ohlc_frame(close=close)
        f = DownsideVolatility(window=5)
        result = f(data)
        # 没有下行收益的情况，波动率为 0
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_downside_vs_total_volatility(self):
        """下行波动率应 ≤ 总波动率"""
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.02, 50)
        close = 100.0 * np.exp(np.cumsum(rets))
        data = _make_ohlc_frame(close=close.tolist())
        f_total = pd.Series(rets).rolling(20).std()
        f_down = DownsideVolatility(window=20)
        result = f_down(data)
        valid = result.dropna()
        total_valid = f_total.dropna()
        # 对齐后比较
        common_idx = valid.index.intersection(total_valid.index)
        assert (valid.loc[common_idx] <= total_valid.loc[common_idx] * 1.01).all()

    def test_params_immutable(self):
        f1 = DownsideVolatility(window=20)
        f2 = DownsideVolatility(window=60)
        assert f1.window == 20
        assert f2.window == 60


# ===================================================================
# ParkinsonVolatility
# ===================================================================

class TestParkinsonVolatility:
    def test_output_name(self):
        f = ParkinsonVolatility(window=20)
        assert f.get_output_name() == "ParkinsonVolatility_20"

    def test_output_name_annualized(self):
        f = ParkinsonVolatility(window=20, annualize=True)
        assert f.get_output_name() == "ParkinsonVolatility_20_ann"

    def test_warmup_period(self):
        f = ParkinsonVolatility(window=20)
        assert f.warmup_period == 20

    def test_rejects_missing_columns(self):
        f = ParkinsonVolatility(window=10)
        bad = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'high'"):
            f(bad)

    def test_zero_vol_for_flat_prices(self):
        """价格不变时波动率应为 0"""
        data = _make_ohlc_frame(
            close=[100.0] * 30,
            high=[100.0] * 30,
            low=[100.0] * 30,
        )
        f = ParkinsonVolatility(window=5)
        result = f(data)
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_positive_vol_for_moving_prices(self):
        """有波动时波动率 > 0"""
        rng = np.random.default_rng(42)
        n = 50
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        high = close + np.abs(rng.normal(0, 0.3, n))
        low = close - np.abs(rng.normal(0, 0.3, n))
        data = _make_ohlc_frame(close=close.tolist(), high=high.tolist(), low=low.tolist())
        f = ParkinsonVolatility(window=10)
        result = f(data)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_annualize_larger_than_raw(self):
        """年化值应 > 原始值"""
        rng = np.random.default_rng(42)
        n = 50
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        high = close + np.abs(rng.normal(0, 0.3, n))
        low = close - np.abs(rng.normal(0, 0.3, n))
        data = _make_ohlc_frame(close=close.tolist(), high=high.tolist(), low=low.tolist())
        f_raw = ParkinsonVolatility(window=10, annualize=False)
        f_ann = ParkinsonVolatility(window=10, annualize=True)
        result_raw = f_raw(data)
        result_ann = f_ann(data)
        common = result_raw.dropna().index.intersection(result_ann.dropna().index)
        assert (result_ann.loc[common] > result_raw.loc[common]).all()


# ===================================================================
# GarmanKlassVolatility
# ===================================================================

class TestGarmanKlassVolatility:
    def test_output_name(self):
        f = GarmanKlassVolatility(window=20)
        assert f.get_output_name() == "GarmanKlassVolatility_20"

    def test_warmup_period(self):
        f = GarmanKlassVolatility(window=20)
        assert f.warmup_period == 20

    def test_rejects_missing_open(self):
        f = GarmanKlassVolatility(window=10)
        # 不用 _make_ohlc_frame 因为它会自动填充 open 列
        bad = pd.DataFrame({"close": [100.0] * 30, "high": [101.0] * 30, "low": [99.0] * 30})
        with pytest.raises(ValueError, match="requires column 'open'"):
            f(bad)

    def test_no_negative_variance(self):
        """即使 log(close/open)² 很大，方差也不应变成负数"""
        # 构造一个 close/open 极端偏离的序列
        rng = np.random.default_rng(42)
        n = 50
        open_ = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        close = open_ + rng.normal(0, 2.0, n)
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
        data = _make_ohlc_frame(
            close=close.tolist(),
            open_=open_.tolist(),
            high=high.tolist(),
            low=low.tolist(),
        )
        f = GarmanKlassVolatility(window=10)
        result = f(data)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_params_immutable(self):
        f1 = GarmanKlassVolatility(window=20)
        f2 = GarmanKlassVolatility(window=60, annualize=True)
        assert f1.window == 20
        assert f2.window == 60
        assert f1.annualize is False
        assert f2.annualize is True


# ===================================================================
# VolOfVol
# ===================================================================

class TestVolOfVol:
    def test_output_name(self):
        f = VolOfVol(vol_window=20, std_window=60)
        assert f.get_output_name() == "VolOfVol_20_60"

    def test_warmup_period(self):
        f = VolOfVol(vol_window=20, std_window=60)
        assert f.warmup_period == 80

    def test_rejects_missing_close(self):
        f = VolOfVol(vol_window=10, std_window=20)
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_returns_nan_in_warmup(self):
        n = 100
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        data = _make_ohlc_frame(close=close.tolist())
        f = VolOfVol(vol_window=20, std_window=60)
        result = f(data)
        assert result.iloc[:79].isna().all()

    def test_params_immutable(self):
        f1 = VolOfVol(vol_window=20, std_window=60)
        f2 = VolOfVol(vol_window=10, std_window=30)
        assert f1.vol_window == 20
        assert f2.vol_window == 10


# ===================================================================
# MaxDrawdown
# ===================================================================

class TestMaxDrawdown:
    def test_output_name(self):
        f = MaxDrawdown(window=60)
        assert f.get_output_name() == "MaxDrawdown_60"

    def test_warmup_period(self):
        f = MaxDrawdown(window=60)
        assert f.warmup_period == 60

    def test_at_peak_is_zero(self):
        """不断创新高时回撤应为 0"""
        close = [100.0 + i for i in range(30)]  # 100 → 129
        data = _make_ohlc_frame(close=close)
        f = MaxDrawdown(window=10)
        result = f(data)
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_calculates_drawdown(self):
        """从 100 涨到 120 回到 108 → 回撤 = 108/120 - 1 = -0.1"""
        close = [100.0, 110.0, 120.0, 115.0, 108.0]
        data = _make_ohlc_frame(close=close)
        f = MaxDrawdown(window=5)
        result = f(data)
        assert result.iloc[4] == pytest.approx(108.0 / 120.0 - 1.0)

    def test_range_within_negative_one_to_zero(self):
        """取值应在 (−1, 0]"""
        rng = np.random.default_rng(42)
        n = 200
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        data = _make_ohlc_frame(close=close.tolist())
        f = MaxDrawdown(window=60)
        result = f(data)
        valid = result.dropna()
        assert (valid <= 0.0).all()
        assert (valid > -1.0).all()

    def test_params_immutable(self):
        f1 = MaxDrawdown(window=60)
        f2 = MaxDrawdown(window=120)
        assert f1.window == 60
        assert f2.window == 120


# ===================================================================
# AvgDrawdown
# ===================================================================

class TestAvgDrawdown:
    def test_output_name(self):
        f = AvgDrawdown(window=60)
        assert f.get_output_name() == "AvgDrawdown_60"

    def test_warmup_period(self):
        f = AvgDrawdown(window=60)
        assert f.warmup_period == 60

    def test_average_is_smoother(self):
        """AvgDrawdown 的极值应比 MaxDrawdown 更窄"""
        rng = np.random.default_rng(42)
        n = 200
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        data = _make_ohlc_frame(close=close.tolist())
        f_max = MaxDrawdown(window=60)
        f_avg = AvgDrawdown(window=60)
        r_max = f_max(data).dropna()
        r_avg = f_avg(data).dropna()
        common = r_max.index.intersection(r_avg.index)
        assert r_max.loc[common].min() <= r_avg.loc[common].min()

    def test_params_immutable(self):
        f1 = AvgDrawdown(window=60)
        f2 = AvgDrawdown(window=120)
        assert f1.window == 60
        assert f2.window == 120
