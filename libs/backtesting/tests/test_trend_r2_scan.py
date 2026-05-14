from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.trend_r2 import compute_trend_r2
from scripts.run_trend_r2_scan import (
    accumulate_cross_sectional_stats,
    compute_direction_hit_rate,
    compute_forward_return,
    compute_top_group_return,
    finalize_cross_sectional_ic,
    summarize_symbol_rows,
)


def test_compute_trend_r2_uses_cumulative_return_regression():
    close = pd.Series([100.0, 110.0, 120.0], dtype=float)

    result = compute_trend_r2(close=close, window=3)

    assert result.slope.iloc[0] != result.slope.iloc[0]
    assert result.slope.iloc[1] != result.slope.iloc[1]
    assert result.slope.iloc[2] == pytest.approx(0.1)
    assert result.r2.iloc[2] == pytest.approx(1.0)


def test_compute_trend_r2_returns_nan_r2_for_flat_window():
    close = pd.Series([100.0, 100.0, 100.0], dtype=float)

    result = compute_trend_r2(close=close, window=3)

    assert result.slope.iloc[2] == pytest.approx(0.0)
    assert pd.isna(result.r2.iloc[2])


def test_compute_trend_r2_rejects_invalid_window():
    close = pd.Series([100.0, 110.0], dtype=float)

    with pytest.raises(ValueError, match="window must be at least 2"):
        compute_trend_r2(close=close, window=1)


def test_compute_forward_return_aligns_with_future_window():
    close = pd.Series([100.0, 110.0, 121.0], dtype=float)

    result = compute_forward_return(close=close, holding_window=1)

    assert result.iloc[0] == pytest.approx(10.0)
    assert result.iloc[1] == pytest.approx(10.0)
    assert pd.isna(result.iloc[2])


def test_compute_top_group_return_uses_top_ceil_20_percent():
    r2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    fwd_ret = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=float)

    result = compute_top_group_return(r2=r2, fwd_ret=fwd_ret, top_fraction=0.2)

    assert result == pytest.approx(55.0)


def test_compute_direction_hit_rate_uses_median_split():
    r2 = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=float)
    fwd_ret = pd.Series([-1.0, -2.0, 2.0, 3.0], dtype=float)

    result = compute_direction_hit_rate(r2=r2, fwd_ret=fwd_ret)

    assert result == pytest.approx(1.0)


def test_summarize_symbol_rows_aggregates_combo_metrics():
    symbol_rows = pd.DataFrame(
        [
            {
                "symbol": "A",
                "formation_window": 60,
                "holding_window": 5,
                "valid_pair_count": 300,
                "eligible": True,
                "symbol_corr": 0.2,
                "top_buy_return_pct": 1.0,
                "hit_rate": 0.5,
            },
            {
                "symbol": "B",
                "formation_window": 60,
                "holding_window": 5,
                "valid_pair_count": 280,
                "eligible": True,
                "symbol_corr": -0.1,
                "top_buy_return_pct": 2.0,
                "hit_rate": 0.75,
            },
            {
                "symbol": "C",
                "formation_window": 60,
                "holding_window": 5,
                "valid_pair_count": 120,
                "eligible": False,
                "symbol_corr": float("nan"),
                "top_buy_return_pct": float("nan"),
                "hit_rate": float("nan"),
            },
        ]
    )

    result = summarize_symbol_rows(
        symbol_rows=symbol_rows,
        formation_windows=(60,),
        holding_windows=(5,),
    )

    row = result.iloc[0]
    assert row["candidate_etf_count"] == 3
    assert row["eligible_etf_count"] == 2
    assert row["skipped_etf_count"] == 1
    assert row["valid_pair_count"] == 580
    assert row["mean_symbol_corr"] == pytest.approx(0.05)
    assert row["positive_symbol_share"] == pytest.approx(0.5)
    assert row["mean_top_buy_return_pct"] == pytest.approx(1.5)
    assert row["mean_hit_rate"] == pytest.approx(0.625)


def test_finalize_cross_sectional_ic_aggregates_daily_values():
    accumulator: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    first_date = pd.Timestamp("2024-01-01").value
    second_date = pd.Timestamp("2024-01-02").value

    accumulate_cross_sectional_stats(
        accumulator=accumulator,
        combo_key=(60, 5),
        date_ns=np.array([first_date, second_date], dtype=np.int64),
        r2_values=np.array([1.0, 1.0], dtype=float),
        fwd_values=np.array([1.0, 3.0], dtype=float),
    )
    accumulate_cross_sectional_stats(
        accumulator=accumulator,
        combo_key=(60, 5),
        date_ns=np.array([first_date, second_date], dtype=np.int64),
        r2_values=np.array([2.0, 2.0], dtype=float),
        fwd_values=np.array([2.0, 2.0], dtype=float),
    )
    accumulate_cross_sectional_stats(
        accumulator=accumulator,
        combo_key=(60, 5),
        date_ns=np.array([first_date, second_date], dtype=np.int64),
        r2_values=np.array([3.0, 3.0], dtype=float),
        fwd_values=np.array([3.0, 1.0], dtype=float),
    )

    result = finalize_cross_sectional_ic(
        accumulator=accumulator,
        formation_windows=(60,),
        holding_windows=(5,),
        min_symbols=3,
    )

    row = result.iloc[0]
    assert row["mean_daily_xs_ic"] == pytest.approx(0.0)
    assert row["positive_daily_xs_ic_share"] == pytest.approx(0.5)
    assert row["valid_xs_ic_date_count"] == 2
    assert row["xs_eligible_date_count"] == 2