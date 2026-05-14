from __future__ import annotations

import pandas as pd
import pytest

from factors.rolling_ols import RollingOLS


def _make_frame(values: list[float]) -> pd.DataFrame:
    series = pd.Series(values, dtype=float)
    return pd.DataFrame(
        {
            "open": series,
            "high": series,
            "low": series,
            "close": series,
        }
    )


def test_rolling_ols_returns_expected_slope_for_linear_series():
    data = _make_frame([10.0, 12.0, 14.0, 16.0, 18.0])

    factor = RollingOLS(window=3, output="slope")
    result = factor(data)

    assert result.name == "RollingOLS_close_3_slope"
    assert result.iloc[0] != result.iloc[0]
    assert result.iloc[1] != result.iloc[1]
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(2.0)


def test_rolling_ols_returns_expected_r2_for_linear_series():
    data = _make_frame([10.0, 12.0, 14.0, 16.0, 18.0])

    factor = RollingOLS(window=3, output="r2")
    result = factor(data)

    assert result.name == "RollingOLS_close_3_r2"
    assert result.iloc[2] == pytest.approx(1.0)
    assert result.iloc[3] == pytest.approx(1.0)
    assert result.iloc[4] == pytest.approx(1.0)


def test_rolling_ols_supports_custom_value_column():
    data = pd.DataFrame(
        {
            "close": pd.Series([1.0, 1.0, 1.0, 1.0], dtype=float),
            "high": pd.Series([3.0, 5.0, 7.0, 9.0], dtype=float),
        }
    )

    factor = RollingOLS(window=2, value_column="high", output="slope")
    result = factor(data)

    assert result.name == "RollingOLS_high_2_slope"
    assert result.iloc[1] == pytest.approx(2.0)
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(2.0)


def test_rolling_ols_rejects_missing_column():
    factor = RollingOLS(window=3, value_column="value")

    with pytest.raises(ValueError, match="requires column 'value'"):
        factor(_make_frame([1.0, 2.0, 3.0]))