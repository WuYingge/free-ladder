"""MinDailyReturn 因子单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/factors/tests/test_min_daily_return.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.min_daily_return import MinDailyReturn


def _make_frame(close: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": pd.Series(close, dtype=float)})


class TestMinDailyReturn:
    def test_output_name(self):
        f = MinDailyReturn(window=3)
        assert f.get_output_name() == "MinDailyReturn_3"

    def test_output_name_window_5(self):
        f = MinDailyReturn(window=5)
        assert f.get_output_name() == "MinDailyReturn_5"

    def test_warmup_period(self):
        f = MinDailyReturn(window=3)
        assert f.warmup_period == 4

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            MinDailyReturn(window=0)

    def test_rejects_bad_price_column(self):
        with pytest.raises(ValueError, match="price_column"):
            MinDailyReturn(window=3, price_column="volume")

    def test_steady_uptrend(self):
        # 100 → 103 → 106 → 109: 单日涨幅均为正
        f = MinDailyReturn(window=3)
        out = f(_make_frame([100.0, 103.0, 106.0, 109.0]))
        assert np.isnan(out.iloc[0])          # 首日无收益率
        assert np.isnan(out.iloc[1])          # pct_change 首值 NaN, 窗口未满
        assert np.isnan(out.iloc[2])          # 窗口含 NaN
        # index 3: min(3%, 2.913%, 2.830%) = 2.830%
        assert np.isclose(out.iloc[3], (109.0 - 106.0) / 106.0, atol=1e-9)

    def test_daily_drop_detected(self):
        # 100 → 103 → 100 → 102: 中间一天 -2.91%（103→100）
        f = MinDailyReturn(window=3)
        out = f(_make_frame([100.0, 103.0, 100.0, 102.0]))
        assert np.isclose(out.iloc[3], (100.0 - 103.0) / 103.0, atol=1e-9)

    def test_drop_exceeding_3pct(self):
        # 100 → 103 → 99 → 101: 103→99 = -3.88% < -3%
        f = MinDailyReturn(window=3)
        out = f(_make_frame([100.0, 103.0, 99.0, 101.0]))
        assert out.iloc[3] < -0.03

    def test_window_bounds(self):
        # window=2 只看最近 2 个收益率
        f = MinDailyReturn(window=2)
        out = f(_make_frame([100.0, 110.0, 100.0, 108.0]))
        # 最近 2 个收益率: 110→100 (-9.09%), 100→108 (+8%)
        assert np.isclose(out.iloc[3], (100.0 - 110.0) / 110.0, atol=1e-9)
