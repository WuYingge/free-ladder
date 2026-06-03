from __future__ import annotations

import math
from typing import Mapping

import pandas as pd
import pytest

from backtesting.wide_momentum_baseline import (
    PreparedWideMomentumUniverse,
    SymbolBaselineData,
    WideMomentumBaselineConfig,
    prepare_wide_momentum_universe_from_etf_data_map,
    prepare_wide_momentum_universe_from_frames,
    run_wide_momentum_baseline_from_prepared,
    save_wide_momentum_baseline_result,
)
from core.models.etf_daily_data import EtfData
from factors.base_factor import BaseFactor


def _make_frame(
    dates: pd.DatetimeIndex,
    open_values: list[float],
    close_values: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_values,
            "close": close_values,
        }
    )


def _make_etf(
    symbol: str,
    dates: pd.DatetimeIndex,
    open_values: list[float],
    close_values: list[float],
) -> EtfData:
    open_series = pd.Series(open_values, index=dates, dtype=float)
    close_series = pd.Series(close_values, index=dates, dtype=float)
    frame = pd.DataFrame(
        {
            "open": open_series,
            "high": pd.Series(
                [max(open_value, close_value) for open_value, close_value in zip(open_values, close_values)],
                index=dates,
                dtype=float,
            ),
            "low": pd.Series(
                [min(open_value, close_value) for open_value, close_value in zip(open_values, close_values)],
                index=dates,
                dtype=float,
            ),
            "close": close_series,
            "volume": pd.Series([1_000] * len(dates), index=dates, dtype="int64"),
            "value": pd.Series([100_000] * len(dates), index=dates, dtype="int64"),
            "turnOver": pd.Series([0.01] * len(dates), index=dates, dtype=float),
            "gain": pd.Series([0.0] * len(dates), index=dates, dtype=float),
            "change": pd.Series([0.0] * len(dates), index=dates, dtype=float),
        }
    )
    return EtfData(frame, symbol=symbol)


class CloseLevelFactor(BaseFactor):
    name = "CloseLevel"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(data["close"], index=data.index, dtype=float)
        result.name = self.get_output_name()
        return result


class CloseMinusOpenFactor(BaseFactor):
    name = "CloseMinusOpen"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(data["close"] - data["open"], index=data.index, dtype=float)
        result.name = self.get_output_name()
        return result


def test_prepare_universe_uses_shifted_momentum_and_listing_proxy() -> None:
    dates = pd.bdate_range("2023-01-02", periods=8)
    symbol_frame_map = {
        "AAA": _make_frame(
            dates=dates,
            open_values=[10, 11, 12, 13, 14, 15, 16, 17],
            close_values=[10, 11, 12, 13, 14, 15, 16, 17],
        )
    }
    config = WideMomentumBaselineConfig(
        top_n_values=(1,),
        min_listing_days=2,
        momentum_window=2,
        momentum_skip_recent=1,
        rebalance_interval=2,
        start_date=None,
        end_date=None,
    )

    prepared = prepare_wide_momentum_universe_from_frames(
        symbol_frame_map=symbol_frame_map,
        config=config,
        calendar=dates,
        cluster_lookup={"AAA": 7},
    )

    symbol_data = prepared.symbol_data_map["AAA"]
    assert symbol_data.listing_proxy_date == dates[0]
    assert symbol_data.cluster_label == 7
    assert prepared.start_date == dates[3]

    momentum_value = symbol_data.frame.loc[dates[3], "momentum"]
    expected = (12.0 - 10.0) / 10.0
    assert momentum_value == expected
    assert not bool(symbol_data.frame.loc[dates[2], "eligible_signal"])
    assert bool(symbol_data.frame.loc[dates[3], "eligible_signal"])


def test_monthly_pool_diagnostics_marks_stable_start() -> None:
    dates = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        )
    )
    symbol_frame_map = {
        "AAA": _make_frame(dates, [10, 11, 12], [10, 11, 12]),
        "BBB": _make_frame(dates, [20, 21, 22], [20, 21, 22]),
    }
    config = WideMomentumBaselineConfig(
        top_n_values=(1,),
        min_listing_days=0,
        momentum_window=1,
        momentum_skip_recent=1,
        rebalance_interval=1,
        stable_pool_size=2,
        start_date=None,
        end_date=None,
    )

    prepared = prepare_wide_momentum_universe_from_frames(
        symbol_frame_map=symbol_frame_map,
        config=config,
        calendar=dates,
        cluster_lookup={"AAA": 1, "BBB": -1},
    )

    diagnostics = prepared.monthly_pool_diagnostics
    stable_row = diagnostics[diagnostics["is_stable_start"]].iloc[0]
    assert stable_row["month"] == "2023-03"
    assert int(stable_row["eligible_symbol_count"]) == 2
    assert int(stable_row["covered_cluster_count"]) == 1
    assert int(stable_row["unassigned_cluster_symbol_count"]) == 1


def test_run_baseline_from_prepared_tracks_turnover_and_period_win_rate() -> None:
    dates = pd.bdate_range("2023-01-02", periods=6)
    monthly_pool_diagnostics = pd.DataFrame(
        [
            {
                "month": "2023-01",
                "month_end_date": dates[-1],
                "eligible_symbol_count": 2,
                "covered_cluster_count": 2,
                "unassigned_cluster_symbol_count": 0,
                "is_pool_stable": True,
                "is_stable_start": True,
            }
        ]
    )
    symbol_data_map = {
        "AAA": SymbolBaselineData(
            symbol="AAA",
            listing_proxy_date=dates[0],
            cluster_label=1,
            frame=pd.DataFrame(
                {
                    "open": [10.0, 10.0, 11.0, 12.0, 12.0, 11.0],
                    "close": [10.0, 10.5, 11.0, 12.0, 11.5, 11.0],
                    "momentum": [2.0, 2.0, 0.1, 0.1, -0.5, -0.5],
                    "eligible_signal": [True, True, True, True, True, True],
                },
                index=dates,
            ),
        ),
        "BBB": SymbolBaselineData(
            symbol="BBB",
            listing_proxy_date=dates[0],
            cluster_label=2,
            frame=pd.DataFrame(
                {
                    "open": [20.0, 20.0, 20.0, 20.0, 22.0, 24.0],
                    "close": [20.0, 20.0, 20.0, 21.0, 24.0, 24.0],
                    "momentum": [1.0, 1.0, 3.0, 3.0, 2.0, 2.0],
                    "eligible_signal": [True, True, True, True, True, True],
                },
                index=dates,
            ),
        ),
    }
    prepared = PreparedWideMomentumUniverse(
        symbol_data_map=symbol_data_map,
        calendar=dates,
        start_date=dates[0],
        end_date=dates[-1],
        recent_complete_date=dates[-1],
        monthly_pool_diagnostics=monthly_pool_diagnostics,
        source_symbol_count=2,
    )
    config = WideMomentumBaselineConfig(
        top_n_values=(1,),
        rebalance_interval=2,
        cash=100.0,
        commission=0.0,
        risk_free_rate=0.0,
        start_date=None,
        end_date=None,
    )

    result = run_wide_momentum_baseline_from_prepared(prepared=prepared, config=config)
    variant = result.variant_results[1]

    assert variant.summary["rebalance_count"] == 3
    assert variant.summary["completed_period_count"] == 2
    assert variant.summary["rebalance_win_rate_pct"] == 100.0
    assert variant.summary["experiment_name"] == "wide_momentum_baseline"
    assert "min_momentum_value" not in variant.summary
    assert "active_candidate_filter_count" not in variant.summary
    assert variant.rebalance_log.iloc[1]["turnover"] == 1.0
    assert variant.rebalance_log.iloc[0]["candidate_count"] == 2
    assert variant.rebalance_log.iloc[0]["filtered_candidate_count"] == 2
    assert variant.rebalance_log.iloc[0]["active_candidate_filters"] == []
    assert variant.rebalance_log.iloc[0]["selected_symbols"] == ["AAA"]
    assert variant.rebalance_log.iloc[1]["selected_symbols"] == ["BBB"]
    assert variant.rebalance_log.iloc[0]["period_return"] == pytest.approx(0.2)
    assert variant.rebalance_log.iloc[1]["period_return"] == pytest.approx(0.2)
    assert variant.annual_returns.iloc[0]["year"] == 2023
    assert variant.summary["cumulative_return_pct"] is not None
    assert math.isfinite(float(variant.summary["cumulative_return_pct"]))


def test_run_baseline_from_prepared_applies_optional_momentum_filter_before_ranking() -> None:
    dates = pd.bdate_range("2023-01-02", periods=6)
    monthly_pool_diagnostics = pd.DataFrame(
        [
            {
                "month": "2023-01",
                "month_end_date": dates[-1],
                "eligible_symbol_count": 2,
                "covered_cluster_count": 2,
                "unassigned_cluster_symbol_count": 0,
                "is_pool_stable": True,
                "is_stable_start": True,
            }
        ]
    )
    symbol_data_map = {
        "AAA": SymbolBaselineData(
            symbol="AAA",
            listing_proxy_date=dates[0],
            cluster_label=1,
            frame=pd.DataFrame(
                {
                    "open": [10.0, 10.0, 11.0, 12.0, 12.0, 11.0],
                    "close": [10.0, 10.5, 11.0, 12.0, 11.5, 11.0],
                    "momentum": [2.0, 2.0, -0.1, -0.1, -0.5, -0.5],
                    "eligible_signal": [True, True, True, True, True, True],
                },
                index=dates,
            ),
        ),
        "BBB": SymbolBaselineData(
            symbol="BBB",
            listing_proxy_date=dates[0],
            cluster_label=2,
            frame=pd.DataFrame(
                {
                    "open": [20.0, 20.0, 20.0, 20.0, 22.0, 24.0],
                    "close": [20.0, 20.0, 20.0, 21.0, 24.0, 24.0],
                    "momentum": [1.0, 1.0, 3.0, 3.0, 2.0, 2.0],
                    "eligible_signal": [True, True, True, True, True, True],
                },
                index=dates,
            ),
        ),
    }
    prepared = PreparedWideMomentumUniverse(
        symbol_data_map=symbol_data_map,
        calendar=dates,
        start_date=dates[0],
        end_date=dates[-1],
        recent_complete_date=dates[-1],
        monthly_pool_diagnostics=monthly_pool_diagnostics,
        source_symbol_count=2,
    )
    config = WideMomentumBaselineConfig(
        top_n_values=(2,),
        min_momentum_value=0.0,
        rebalance_interval=2,
        cash=100.0,
        commission=0.0,
        risk_free_rate=0.0,
        start_date=None,
        end_date=None,
    )

    result = run_wide_momentum_baseline_from_prepared(prepared=prepared, config=config)
    variant = result.variant_results[2]

    assert variant.summary["experiment_name"] == "wide_momentum_baseline__min_momentum_value_ge_0"
    assert "min_momentum_value" not in variant.summary
    assert "active_candidate_filter_count" not in variant.summary
    assert variant.rebalance_log.iloc[0]["candidate_count"] == 2
    assert variant.rebalance_log.iloc[0]["filtered_candidate_count"] == 2
    assert variant.rebalance_log.iloc[1]["candidate_count"] == 2
    assert variant.rebalance_log.iloc[1]["filtered_candidate_count"] == 1
    assert variant.rebalance_log.iloc[1]["active_candidate_filters"] == ["momentum>=0.0"]
    assert variant.rebalance_log.iloc[0]["selected_symbols"] == ["AAA", "BBB"]
    assert variant.rebalance_log.iloc[1]["selected_symbols"] == ["BBB"]


def test_prepare_universe_supports_extensible_factors_and_callable_filters() -> None:
    dates = pd.bdate_range("2023-01-02", periods=4)
    ranking_factor = CloseLevelFactor()
    spread_factor = CloseMinusOpenFactor()

    def minimum_spread(candidate_etf_data: EtfData) -> bool:
        return float(candidate_etf_data.factor_results[spread_factor].iloc[-1]) >= 3.0

    def stronger_than_benchmark(
        candidate_etf_data: EtfData,
        signal_view_map: Mapping[str, EtfData],
    ) -> bool:
        candidate_spread = float(candidate_etf_data.factor_results[spread_factor].iloc[-1])
        benchmark_spread = float(signal_view_map["BENCH"].factor_results[spread_factor].iloc[-1])
        return candidate_spread > benchmark_spread

    config = WideMomentumBaselineConfig(
        top_n_values=(2,),
        ranking_factor=ranking_factor,
        factor_pipeline=(spread_factor,),
        candidate_filters=(minimum_spread, stronger_than_benchmark),
        min_listing_days=0,
        rebalance_interval=1,
        cash=100.0,
        commission=0.0,
        risk_free_rate=0.0,
        start_date=None,
        end_date=None,
    )
    prepared = prepare_wide_momentum_universe_from_etf_data_map(
        etf_data_map={
            "AAA": _make_etf("AAA", dates, [10.0, 10.0, 10.0, 10.0], [13.0, 13.0, 13.0, 13.0]),
            "BBB": _make_etf("BBB", dates, [10.0, 10.0, 10.0, 10.0], [11.0, 11.0, 11.0, 11.0]),
            "BENCH": _make_etf("BENCH", dates, [10.0, 10.0, 10.0, 10.0], [12.0, 12.0, 12.0, 12.0]),
        },
        config=config,
        calendar=dates,
        cluster_lookup={"AAA": 1, "BBB": 2, "BENCH": 3},
    )
    result = run_wide_momentum_baseline_from_prepared(prepared=prepared, config=config)
    variant = result.variant_results[2]

    assert prepared.symbol_data_map["AAA"].ranking_output_name == ranking_factor.get_output_name()
    assert prepared.symbol_data_map["AAA"].etf_data is not None
    assert spread_factor in prepared.symbol_data_map["AAA"].etf_data.factor_results
    assert ranking_factor.get_output_name() in prepared.symbol_data_map["AAA"].frame.columns
    assert variant.rebalance_log.iloc[0]["candidate_count"] == 3
    assert variant.rebalance_log.iloc[0]["filtered_candidate_count"] == 1
    assert variant.rebalance_log.iloc[0]["active_candidate_filters"] == [
        "minimum_spread",
        "stronger_than_benchmark",
    ]
    assert variant.rebalance_log.iloc[0]["selected_symbols"] == ["AAA"]


def test_save_result_writes_expected_artifacts(tmp_path) -> None:
    dates = pd.bdate_range("2023-01-02", periods=4)
    monthly_pool_diagnostics = pd.DataFrame(
        [
            {
                "month": "2023-01",
                "month_end_date": dates[-1],
                "eligible_symbol_count": 1,
                "covered_cluster_count": 1,
                "unassigned_cluster_symbol_count": 0,
                "is_pool_stable": True,
                "is_stable_start": True,
            }
        ]
    )
    prepared = PreparedWideMomentumUniverse(
        symbol_data_map={
            "AAA": SymbolBaselineData(
                symbol="AAA",
                listing_proxy_date=dates[0],
                cluster_label=1,
                frame=pd.DataFrame(
                    {
                        "open": [10.0, 10.0, 11.0, 11.0],
                        "close": [10.0, 10.5, 11.0, 11.5],
                        "momentum": [1.0, 1.0, 1.0, 1.0],
                        "eligible_signal": [True, True, True, True],
                    },
                    index=dates,
                ),
            )
        },
        calendar=dates,
        start_date=dates[0],
        end_date=dates[-1],
        recent_complete_date=dates[-1],
        monthly_pool_diagnostics=monthly_pool_diagnostics,
        source_symbol_count=1,
    )
    config = WideMomentumBaselineConfig(
        top_n_values=(1,),
        rebalance_interval=2,
        cash=100.0,
        commission=0.0,
        risk_free_rate=0.0,
        start_date=None,
        end_date=None,
    )
    result = run_wide_momentum_baseline_from_prepared(prepared=prepared, config=config)

    output_dir = save_wide_momentum_baseline_result(result=result, output_dir=tmp_path / "baseline")
    summary_df = pd.read_csv(output_dir / "summary.csv")

    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "monthly_pool_diagnostics.csv").exists()
    assert (output_dir / "run_metadata.json").exists()
    assert (output_dir / "top_1" / "equity_curve.csv").exists()
    assert (output_dir / "top_1" / "annual_returns.csv").exists()
    assert (output_dir / "top_1" / "rebalance_log.csv").exists()
    assert (output_dir / "top_1" / "summary.json").exists()
    assert (output_dir / "top_1" / "equity_curve.png").exists()
    assert "experiment_name" in summary_df.columns
    assert "min_momentum_value" not in summary_df.columns
    assert "active_candidate_filter_count" not in summary_df.columns