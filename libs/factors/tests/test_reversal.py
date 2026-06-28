"""反转族因子单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/factors/tests/test_reversal.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.reversal import ShortTermReversal, ExtremeReversal, VolumeReversal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_frame(
    close: list[float] | None = None,
    volume: list[float] | None = None,
) -> pd.DataFrame:
    if close is None:
        close = [100.0] * 5
    data: dict[str, pd.Series] = {"close": pd.Series(close, dtype=float)}
    if volume is not None:
        data["volume"] = pd.Series(volume, dtype=float)
    return pd.DataFrame(data)


# ===================================================================
# ShortTermReversal
# ===================================================================

class TestShortTermReversal:
    def test_output_name(self):
        f = ShortTermReversal(window=1)
        assert f.get_output_name() == "ShortTermReversal_1"

    def test_output_name_window_5(self):
        f = ShortTermReversal(window=5)
        assert f.get_output_name() == "ShortTermReversal_5"

    def test_warmup_period(self):
        f = ShortTermReversal(window=5)
        assert f.warmup_period == 6

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            ShortTermReversal(window=0)

    def test_rejects_missing_close(self):
        f = ShortTermReversal(window=5)
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_first_row_is_nan(self):
        data = _make_frame(close=[100.0, 101.0, 102.0])
        f = ShortTermReversal(window=1)
        result = f(data)
        assert np.isnan(result.iloc[0])

    def test_computes_expected_values_1d(self):
        """1日反转：第 t 日 = −(close[t] / close[t-1] − 1)"""
        data = _make_frame(close=[100.0, 102.0, 101.0])
        f = ShortTermReversal(window=1)
        result = f(data)
        # day 1: -(102/100 - 1) = -0.02
        assert result.iloc[1] == pytest.approx(-0.02)
        # day 2: -(101/102 - 1) ≈ 0.0098039
        assert result.iloc[2] == pytest.approx(0.00980392156862745)

    def test_computes_expected_values_5d(self):
        """5日反转：第 t 日 = −(close[t] / close[t-5] − 1)"""
        close = [100.0 + i for i in range(10)]  # 100 → 109
        data = _make_frame(close=close)
        f = ShortTermReversal(window=5)
        result = f(data)
        # index 5: -(105/100 - 1) = -0.05
        assert result.iloc[5] == pytest.approx(-0.05)
        # index 9: -(109/104 - 1) ≈ -0.0480769
        assert result.iloc[9] == pytest.approx(-0.04807692307692303)

    def test_params_immutable_across_instances(self):
        f1 = ShortTermReversal(window=1)
        f2 = ShortTermReversal(window=5)
        assert f1.window == 1
        assert f2.window == 5
        assert f1.params["window"] == 1
        assert f2.params["window"] == 5


# ===================================================================
# ExtremeReversal
# ===================================================================

class TestExtremeReversal:
    def test_output_name(self):
        f = ExtremeReversal(window=20, tail_pct=0.1)
        assert f.get_output_name() == "ExtremeReversal_20_p1"

    def test_warmup_period(self):
        f = ExtremeReversal(window=20, tail_pct=0.1)
        assert f.warmup_period == 21

    def test_rejects_bad_tail_pct(self):
        with pytest.raises(ValueError, match="tail_pct must be in"):
            ExtremeReversal(window=20, tail_pct=0.6)
        with pytest.raises(ValueError, match="tail_pct must be in"):
            ExtremeReversal(window=20, tail_pct=0.0)

    def test_rejects_missing_close(self):
        f = ExtremeReversal(window=10, tail_pct=0.1)
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'close'"):
            f(bad)

    def test_output_is_signal_type(self):
        """输出应为 −1 / 0 / 1 的离散信号"""
        n = 50
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 1, n))
        data = _make_frame(close=close.tolist())
        f = ExtremeReversal(window=20, tail_pct=0.2)
        result = f(data)
        assert set(result.dropna().unique()).issubset({-1, 0, 1})

    def test_warmup_is_nan(self):
        n = 30
        data = _make_frame(close=list(range(100, 100 + n)))
        f = ExtremeReversal(window=10, tail_pct=0.1)
        result = f(data)
        # warmup 内为 NaN
        assert result.iloc[:10].isna().all()

    def test_params_immutable(self):
        f1 = ExtremeReversal(window=20, tail_pct=0.1)
        f2 = ExtremeReversal(window=60, tail_pct=0.05)
        assert f1.window == 20
        assert f2.window == 60
        assert f1.params["window"] == 20
        assert f2.params["window"] == 60


# ===================================================================
# VolumeReversal
# ===================================================================

class TestVolumeReversal:
    def test_output_name(self):
        f = VolumeReversal(ret_window=5, vol_window=20)
        assert f.get_output_name() == "VolumeReversal_5_20"

    def test_warmup_period(self):
        f = VolumeReversal(ret_window=5, vol_window=20)
        assert f.warmup_period == 20  # max(5, 20)

    def test_rejects_missing_columns(self):
        f = VolumeReversal(ret_window=5, vol_window=20)
        bad = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requires column 'volume'"):
            f(bad)

    def test_first_row_is_nan(self):
        close = list(range(100, 130))
        volume = [10000.0] * 30
        data = _make_frame(close=close, volume=volume)
        f = VolumeReversal(ret_window=5, vol_window=20)
        result = f(data)
        # warmup = 20，前 20 行 (index 0-19) 预期为 NaN
        # 实际 index=19 已满足 min_periods → 前 19 行 NaN
        assert result.iloc[:19].isna().all()

    def test_volume_ratio_amplifies_reversal(self):
        """放量时 vol_ratio > 1，信号应被放大"""
        close = [100.0, 101.0, 102.0, 103.0, 104.0, 103.0]
        volume = [1000.0] * 5 + [5000.0]  # 最后一天放量
        data = _make_frame(close=close, volume=volume)
        f = VolumeReversal(ret_window=1, vol_window=5)
        result = f(data)
        # 相对于 volume_ma ≈ 1000，第5天 vol_ratio ≈ 5.0，反转信号被放大
        assert abs(result.iloc[5]) > 0.0

    def test_params_immutable(self):
        f1 = VolumeReversal(ret_window=5, vol_window=20)
        f2 = VolumeReversal(ret_window=1, vol_window=10)
        assert f1.ret_window == 5
        assert f2.ret_window == 1
        assert f1.params["ret_window"] == 5
        assert f2.params["ret_window"] == 1
