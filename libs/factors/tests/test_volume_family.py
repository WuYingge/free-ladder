"""成交量/流动性族因子单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/factors/tests/test_volume_family.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.volume_family import (
    VolumeRatio,
    VolumePriceCorrelation,
    OBV,
    VPT,
    AmihudIlliquidity,
    VolumeStd,
    VolumeSkew,
)


def _make_df(close=None, volume=None, value=None):
    data = {"close": pd.Series(close if close is not None else [100.0] * 30, dtype=float)}
    data["volume"] = pd.Series(volume, dtype=float) if volume else pd.Series([1e6] * len(data["close"]), dtype=float)
    if value is not None:
        data["value"] = pd.Series(value, dtype=float)
    else:
        data["value"] = data["close"] * data["volume"]
    return pd.DataFrame(data)


# ===================================================================
# VolumeRatio
# ===================================================================

class TestVolumeRatio:
    def test_output_name(self):
        f = VolumeRatio(window=5)
        assert f.get_output_name() == "VolumeRatio_5"

    def test_warmup_period(self):
        f = VolumeRatio(window=5)
        assert f.warmup_period == 5

    def test_rejects_window_lt_1(self):
        with pytest.raises(ValueError, match="window must be at least 1"):
            VolumeRatio(window=0)

    def test_ratio_1_for_flat_volume(self):
        vol = [1e6] * 30
        f = VolumeRatio(window=5)
        result = f(_make_df(volume=vol))
        valid = result.dropna()
        assert np.allclose(valid, 1.0, rtol=1e-10)

    def test_above_1_for_increasing(self):
        vol = [1e6 + i * 1e4 for i in range(30)]
        f = VolumeRatio(window=5)
        result = f(_make_df(volume=vol))
        valid = result.dropna()
        assert (valid.iloc[-10:] > 1.0).all()

    def test_below_1_for_decreasing(self):
        vol = [2e6 - i * 1e4 for i in range(30)]
        f = VolumeRatio(window=5)
        result = f(_make_df(volume=vol))
        valid = result.dropna()
        assert (valid.iloc[-10:] < 1.0).all()


# ===================================================================
# VolumePriceCorrelation
# ===================================================================

class TestVolumePriceCorrelation:
    def test_output_name(self):
        f = VolumePriceCorrelation(window=20)
        assert f.get_output_name() == "VolumePriceCorrelation_20"

    def test_warmup_period(self):
        f = VolumePriceCorrelation(window=20)
        assert f.warmup_period == 20

    def test_range_neg1_to_1(self):
        rng = np.random.default_rng(42)
        n = 50
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        vol = rng.integers(100000, 1000000, n).astype(float)
        f = VolumePriceCorrelation(window=5)
        result = f(_make_df(close=close.tolist(), volume=vol.tolist()))
        valid = result.dropna()
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_perfect_positive(self):
        """完全单调同向序列 → 相关 ≈ 1"""
        n = 30
        close = [100.0 + i for i in range(n)]
        vol = [1e5 + i * 1000 for i in range(n)]
        f = VolumePriceCorrelation(window=5)
        result = f(_make_df(close=close, volume=vol))
        valid = result.dropna()
        # Spearman 完全单调 → 接近 1.0
        assert (valid > 0.95).all()

    def test_perfect_negative(self):
        """完全单调反向序列 → 相关 ≈ -1"""
        n = 30
        close = [100.0 + i for i in range(n)]
        vol = [1e6 - i * 1000 for i in range(n)]
        f = VolumePriceCorrelation(window=5)
        result = f(_make_df(close=close, volume=vol))
        valid = result.dropna()
        assert (valid < -0.95).all()


# ===================================================================
# OBV
# ===================================================================

class TestOBV:
    def test_output_name(self):
        f = OBV()
        assert f.get_output_name() == "OBV"

    def test_warmup_period(self):
        f = OBV()
        assert f.warmup_period == 1

    def test_cumulative(self):
        """OBV 应该随涨跌累积"""
        close = [100.0, 101.0, 99.0, 102.0, 101.0]
        vol = [1000.0, 2000.0, 1500.0, 3000.0, 1000.0]
        f = OBV()
        result = f(_make_df(close=close, volume=vol))
        # day0: 0 (no prev close, direction=0)
        # day1: +2000
        # day2: -1500 → cum=500
        # day3: +3000 → cum=3500
        # day4: -1000 → cum=2500
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == pytest.approx(2000.0)
        assert result.iloc[2] == pytest.approx(500.0)
        assert result.iloc[3] == pytest.approx(3500.0)
        assert result.iloc[4] == pytest.approx(2500.0)


# ===================================================================
# VPT
# ===================================================================

class TestVPT:
    def test_output_name(self):
        f = VPT()
        assert f.get_output_name() == "VPT"

    def test_warmup_period(self):
        f = VPT()
        assert f.warmup_period == 1

    def test_cumulative(self):
        """VPT 应累积 volume × pct_change"""
        close = [100.0, 102.0, 99.0]
        vol = [2000.0, 3000.0, 1000.0]
        f = VPT()
        result = f(_make_df(close=close, volume=vol))
        # day0: 0
        # day1: vol[1] * (102-100)/100 = 3000 * 0.02 = 60
        # day2: vol[2] * (99-102)/102 = 1000 * -0.02941... ≈ -29.41
        # cum: 0 + 60 -29.41 ≈ 30.59
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == pytest.approx(60.0, rel=1e-6)
        assert result.iloc[2] == pytest.approx(30.588235294, rel=1e-6)


# ===================================================================
# AmihudIlliquidity
# ===================================================================

class TestAmihudIlliquidity:
    def test_output_name(self):
        f = AmihudIlliquidity(window=20)
        assert f.get_output_name() == "AmihudIlliquidity_20"

    def test_warmup_period(self):
        f = AmihudIlliquidity(window=20)
        assert f.warmup_period == 21

    def test_requires_value(self):
        with pytest.raises(ValueError, match="requires column 'value'"):
            AmihudIlliquidity(window=5)(pd.DataFrame({"close": [100.0]}))

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        n = 50
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        vol = rng.integers(100000, 1000000, n).astype(float)
        value = (close * vol).tolist()
        f = AmihudIlliquidity(window=5)
        result = f(_make_df(close=close.tolist(), value=value))
        valid = result.dropna()
        assert (valid >= 0).all()


# ===================================================================
# VolumeStd
# ===================================================================

class TestVolumeStd:
    def test_output_name(self):
        f = VolumeStd(window=20)
        assert f.get_output_name() == "VolumeStd_20"

    def test_warmup_period(self):
        f = VolumeStd(window=20)
        assert f.warmup_period == 20

    def test_zero_for_flat_volume(self):
        vol = [1e6] * 30
        f = VolumeStd(window=5)
        result = f(_make_df(volume=vol))
        valid = result.dropna()
        assert np.allclose(valid, 0.0)

    def test_positive_for_varying(self):
        rng = np.random.default_rng(42)
        vol = rng.integers(50000, 200000, 50).astype(float)
        f = VolumeStd(window=5)
        result = f(_make_df(volume=vol.tolist()))
        valid = result.dropna()
        assert (valid > 0).all()


# ===================================================================
# VolumeSkew
# ===================================================================

class TestVolumeSkew:
    def test_output_name(self):
        f = VolumeSkew(window=20)
        assert f.get_output_name() == "VolumeSkew_20"

    def test_warmup_period(self):
        f = VolumeSkew(window=20)
        assert f.warmup_period == 20

    def test_rejects_window_lt_3(self):
        with pytest.raises(ValueError, match="window must be at least 3"):
            VolumeSkew(window=2)

    def test_no_crash(self):
        rng = np.random.default_rng(42)
        vol = rng.integers(50000, 200000, 50).astype(float)
        f = VolumeSkew(window=5)
        result = f(_make_df(volume=vol.tolist()))
        valid = result.dropna()
        assert len(valid) > 0
