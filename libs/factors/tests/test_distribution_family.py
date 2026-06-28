"""分布形态族因子单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/factors/tests/test_distribution_family.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.distribution_family import (
    ReturnSkew,
    ReturnKurtosis,
    HistoricalVaR,
    CVaR,
    MaxFavorableExcursion,
    MaxAdverseExcursion,
    InformationDiscreteness,
)


def _make_close(close=None, n=50):
    """构建最小 DataFrame，含 close 列。"""
    if close is None:
        close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    return pd.DataFrame({"close": pd.Series(close, dtype=float)})


# ===================================================================
# ReturnSkew
# ===================================================================

class TestReturnSkew:
    def test_output_name(self):
        f = ReturnSkew(window=60)
        assert f.get_output_name() == "ReturnSkew_60"

    def test_warmup_period(self):
        f = ReturnSkew(window=60)
        assert f.warmup_period == 61

    def test_rejects_window_lt_3(self):
        with pytest.raises(ValueError, match="window must be at least 3"):
            ReturnSkew(window=2)

    def test_rejects_missing_close(self):
        with pytest.raises(ValueError, match="requires column 'close'"):
            ReturnSkew(window=5)(pd.DataFrame({"open": [1.0, 2.0]}))

    def test_symmetric_returns_near_zero(self):
        """对称随机游走 → 偏度 ≈ 0"""
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, 80))
        f = ReturnSkew(window=60)
        result = f(_make_close(close=close.tolist(), n=80))
        valid = result.dropna()
        assert valid.abs().mean() < 2.0  # 大致对称

    def test_params_immutable_across_instances(self):
        f1 = ReturnSkew(window=30)
        f2 = ReturnSkew(window=90)
        assert f1.window == 30
        assert f2.window == 90


# ===================================================================
# ReturnKurtosis
# ===================================================================

class TestReturnKurtosis:
    def test_output_name(self):
        f = ReturnKurtosis(window=60)
        assert f.get_output_name() == "ReturnKurtosis_60"

    def test_warmup_period(self):
        f = ReturnKurtosis(window=60)
        assert f.warmup_period == 61

    def test_rejects_window_lt_4(self):
        with pytest.raises(ValueError, match="window must be at least 4"):
            ReturnKurtosis(window=3)

    def test_rejects_missing_close(self):
        with pytest.raises(ValueError, match="requires column 'close'"):
            ReturnKurtosis(window=5)(pd.DataFrame({"open": [1.0, 2.0]}))

    def test_flat_returns_yields_finite(self):
        """恒定价格 → 收益全零 → 分布退化 → 峰度为 -3（pandas 行为）"""
        close = [100.0] * 80
        f = ReturnKurtosis(window=60)
        result = f(_make_close(close=close, n=80))
        valid = result.dropna()
        assert np.isfinite(valid).all()

    def test_params_immutable(self):
        f1 = ReturnKurtosis(window=30)
        f2 = ReturnKurtosis(window=90)
        assert f1.window == 30
        assert f2.window == 90


# ===================================================================
# HistoricalVaR
# ===================================================================

class TestHistoricalVaR:
    def test_output_name(self):
        f = HistoricalVaR(window=252, q=0.05)
        assert f.get_output_name() == "HistoricalVaR_252_q05"

    def test_warmup_period(self):
        f = HistoricalVaR(window=252)
        assert f.warmup_period == 253

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            HistoricalVaR(window=1)

    def test_rejects_q_out_of_range(self):
        with pytest.raises(ValueError, match="q must be in"):
            HistoricalVaR(window=20, q=0.0)
        with pytest.raises(ValueError, match="q must be in"):
            HistoricalVaR(window=20, q=1.0)

    def test_vaR_always_negative_or_zero(self):
        """VaR(5%) 应 ≤ 0（除非极端牛市）"""
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, 100))
        f = HistoricalVaR(window=50, q=0.05)
        result = f(_make_close(close=close.tolist(), n=100))
        valid = result.dropna()
        # 绝大多数情况下 VaR 应为负值
        assert (valid <= 0).mean() > 0.8

    def test_small_window_vaR_is_finite(self):
        """小窗口 VaR 应该是有限值"""
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, 20))
        f = HistoricalVaR(window=10, q=0.25)
        result = f(_make_close(close=close.tolist(), n=20))
        valid = result.dropna()
        assert np.isfinite(valid).all()


# ===================================================================
# CVaR
# ===================================================================

class TestCVaR:
    def test_output_name(self):
        f = CVaR(window=252, q=0.05)
        assert f.get_output_name() == "CVaR_252_q05"

    def test_warmup_period(self):
        f = CVaR(window=252)
        assert f.warmup_period == 253

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            CVaR(window=1)

    def test_cvar_leq_var(self):
        """CVaR 应 ≤ VaR（同分位数下，尾部均值 ≤ 分位数值）"""
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, 100))
        f_var = HistoricalVaR(window=50, q=0.25)
        f_cvar = CVaR(window=50, q=0.25)
        data = _make_close(close=close.tolist(), n=100)
        var = f_var(data)
        cvar = f_cvar(data)
        valid = var.notna() & cvar.notna()
        assert (cvar[valid] <= var[valid]).all()


# ===================================================================
# MaxFavorableExcursion
# ===================================================================

class TestMaxFavorableExcursion:
    def test_output_name(self):
        f = MaxFavorableExcursion(window=20)
        assert f.get_output_name() == "MFE_20"

    def test_warmup_period(self):
        f = MaxFavorableExcursion(window=20)
        assert f.warmup_period == 21

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            MaxFavorableExcursion(window=0)

    def test_mfe_range(self):
        """MFE 在持续上涨时应为正，持续下跌时可为负"""
        # 上涨窗口：最高点 > 起点 → MFE > 0
        up = list(range(100, 130))
        f = MaxFavorableExcursion(window=5)
        valid = f(_make_close(close=up, n=30)).dropna()
        assert (valid > 0).all()

        # 下跌窗口：最高点 < 起点 → MFE < 0
        down = list(range(130, 100, -1))
        valid = f(_make_close(close=down, n=30)).dropna()
        assert (valid < 0).all()

    def test_uptrend_mfe_positive(self):
        """持续上涨 → MFE > 0"""
        close = [100.0 + i for i in range(30)]
        f = MaxFavorableExcursion(window=5)
        result = f(_make_close(close=close, n=30))
        valid = result.dropna()
        assert (valid > 0).all()


# ===================================================================
# MaxAdverseExcursion
# ===================================================================

class TestMaxAdverseExcursion:
    def test_output_name(self):
        f = MaxAdverseExcursion(window=20)
        assert f.get_output_name() == "MAE_20"

    def test_warmup_period(self):
        f = MaxAdverseExcursion(window=20)
        assert f.warmup_period == 21

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            MaxAdverseExcursion(window=0)

    def test_mae_range(self):
        """MAE 在持续下跌时为负，持续上涨时可为正"""
        # 下跌窗口：最低点 < 起点 → MAE < 0
        down = list(range(130, 100, -1))
        f = MaxAdverseExcursion(window=5)
        valid = f(_make_close(close=down, n=30)).dropna()
        assert (valid < 0).all()

        # 上涨窗口：最低点 > 起点 → MAE > 0
        up = list(range(100, 130))
        valid = f(_make_close(close=up, n=30)).dropna()
        assert (valid > 0).all()

    def test_downtrend_mae_negative(self):
        """持续下跌 → MAE < 0"""
        close = [100.0 - i for i in range(30)]
        f = MaxAdverseExcursion(window=5)
        result = f(_make_close(close=close, n=30))
        valid = result.dropna()
        assert (valid < 0).all()


# ===================================================================
# InformationDiscreteness
# ===================================================================

class TestInformationDiscreteness:
    def test_output_name(self):
        f = InformationDiscreteness(window=20)
        assert f.get_output_name() == "ID_20"

    def test_warmup_period(self):
        f = InformationDiscreteness(window=20)
        assert f.warmup_period == 21

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            InformationDiscreteness(window=1)

    def test_id_range_0_to_1(self):
        """ID 应在 [0, 1] 之间"""
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, 50))
        f = InformationDiscreteness(window=10)
        result = f(_make_close(close=close.tolist(), n=50))
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_id_zero_for_monotonic(self):
        """单调上涨 → 方向从不切换 → ID=0"""
        close = [100.0 + i for i in range(30)]
        f = InformationDiscreteness(window=10)
        result = f(_make_close(close=close, n=30))
        valid = result.dropna()
        assert (valid == 0).all()

    def test_id_one_for_alternating(self):
        """严格交替涨跌 → 每天切换 → ID=1"""
        close = []
        price = 100.0
        for i in range(30):
            close.append(price)
            price += 1.0 if i % 2 == 0 else -1.0
        f = InformationDiscreteness(window=10)
        result = f(_make_close(close=close, n=30))
        valid = result.dropna()
        assert (valid == 1).all()
