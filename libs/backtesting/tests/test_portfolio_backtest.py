"""Minimal regression tests for the multi-symbol portfolio backtest mode.

Run from project root:
    PYTHONPATH=libs pytest libs/backtesting/tests/test_portfolio_backtest.py -v
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from backtesting import (
    PortfolioBatchConfig,
    example_equal_weight_momentum_signal,
    run_portfolio_backtest_batch,
)
from backtesting.engine import PortfolioBacktestConfig, run_portfolio_backtest_from_feeds


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

def _make_feed_df(
    n: int = 120,
    start: str = "2023-01-01",
    seed: int = 0,
    factor_value: float = 1.0,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with a constant momentum factor column."""
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    close = 10.0 * (1 + rng.normal(0, 0.01, n)).cumprod()
    df = pd.DataFrame(
        {
            "open": close * (1 - rng.uniform(0, 0.005, n)),
            "high": close * (1 + rng.uniform(0, 0.01, n)),
            "low": close * (1 - rng.uniform(0, 0.01, n)),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
            "momentum": factor_value,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


SYMBOLS = ["A", "B", "C"]


@pytest.fixture()
def symbol_feed_map() -> dict[str, pd.DataFrame]:
    return {
        "A": _make_feed_df(seed=1, factor_value=3.0),
        "B": _make_feed_df(seed=2, factor_value=1.0),
        "C": _make_feed_df(seed=3, factor_value=-1.0),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_signal(
    snapshot: pd.DataFrame,
    context: dict[str, Any],
) -> dict[str, float]:
    """Hold equal weight in all symbols every bar."""
    n = len(snapshot)
    if n == 0:
        return {}
    w = 0.9 / n
    return {sym: w for sym in snapshot.index}


# ---------------------------------------------------------------------------
# engine-level: run_portfolio_backtest_from_feeds
# ---------------------------------------------------------------------------

class TestRunPortfolioBacktestFromFeeds:
    def test_returns_expected_result_type(self, symbol_feed_map):
        from backtesting.engine import PortfolioBacktestResult

        result = run_portfolio_backtest_from_feeds(
            PortfolioBacktestConfig(
                symbol_feed_map=symbol_feed_map,
                strategy_callable=_noop_signal,
                cash=100_000,
                commission=0.001,
                slippage_perc=0.0002,
            )
        )
        assert isinstance(result, PortfolioBacktestResult)
        assert result.symbol == "PORTFOLIO"
        assert result.start_value == pytest.approx(100_000.0)
        assert result.end_value > 0
        assert isinstance(result.trades_total, int)

    def test_time_return_is_non_empty(self, symbol_feed_map):
        result = run_portfolio_backtest_from_feeds(
            PortfolioBacktestConfig(
                symbol_feed_map=symbol_feed_map,
                strategy_callable=_noop_signal,
                cash=100_000,
            )
        )
        time_return = result.analyzer_raw.get("time_return", {})
        assert len(time_return) > 0

    def test_empty_symbol_map_raises(self):
        with pytest.raises(ValueError, match="symbol_feed_map"):
            run_portfolio_backtest_from_feeds(
                PortfolioBacktestConfig(
                    symbol_feed_map={},
                    strategy_callable=_noop_signal,
                )
            )

    def test_invalid_slippage_raises(self, symbol_feed_map):
        with pytest.raises(ValueError, match="slippage"):
            run_portfolio_backtest_from_feeds(
                PortfolioBacktestConfig(
                    symbol_feed_map=symbol_feed_map,
                    strategy_callable=_noop_signal,
                    slippage_perc=-0.001,
                )
            )


# ---------------------------------------------------------------------------
# batch-level: run_portfolio_backtest_batch
# ---------------------------------------------------------------------------

class TestRunPortfolioBatchConfig:
    def test_summary_has_portfolio_row(self, symbol_feed_map):
        config = PortfolioBatchConfig(
            cash=100_000,
            commission=0.001,
        )
        summary_df, errors, equity_curves = run_portfolio_backtest_batch(
            symbol_feed_map=symbol_feed_map,
            strategy_callable=_noop_signal,
            config=config,
        )
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 1
        assert summary_df.iloc[0]["symbol"] == "PORTFOLIO"

    def test_equity_curves_contains_portfolio_key(self, symbol_feed_map):
        config = PortfolioBatchConfig(cash=100_000)
        _, _, equity_curves = run_portfolio_backtest_batch(
            symbol_feed_map=symbol_feed_map,
            strategy_callable=_noop_signal,
            config=config,
        )
        assert "PORTFOLIO" in equity_curves
        ec = equity_curves["PORTFOLIO"]
        assert isinstance(ec, pd.DataFrame)
        assert "strategy" in ec.columns
        assert "benchmark" in ec.columns
        assert len(ec) > 0

    def test_no_errors_on_valid_input(self, symbol_feed_map):
        config = PortfolioBatchConfig(cash=100_000)
        _, errors, _ = run_portfolio_backtest_batch(
            symbol_feed_map=symbol_feed_map,
            strategy_callable=_noop_signal,
            config=config,
        )
        assert errors == []

    def test_summary_metrics_are_finite(self, symbol_feed_map):
        config = PortfolioBatchConfig(cash=100_000)
        summary_df, _, _ = run_portfolio_backtest_batch(
            symbol_feed_map=symbol_feed_map,
            strategy_callable=_noop_signal,
            config=config,
        )
        row = summary_df.iloc[0]
        assert row.get("error") is None
        for col in ("strategy_cumulative_return", "max_drawdown_pct"):
            val = row.get(col)
            if val is not None:
                assert math.isfinite(float(val)), f"{col} is not finite: {val}"

    def test_empty_symbol_map_raises(self):
        config = PortfolioBatchConfig(cash=100_000)
        with pytest.raises(ValueError, match="symbol_feed_map"):
            run_portfolio_backtest_batch(
                symbol_feed_map={},
                strategy_callable=_noop_signal,
                config=config,
            )


# ---------------------------------------------------------------------------
# example signal: example_equal_weight_momentum_signal
# ---------------------------------------------------------------------------

class TestExampleMomentumSignal:
    def test_selects_top_by_momentum(self, symbol_feed_map):
        snapshot = pd.DataFrame({"momentum": [3.0, 1.0, -1.0]}, index=["A", "B", "C"])
        weights = example_equal_weight_momentum_signal(
            snapshot,
            {},
            momentum_column="momentum",
            min_score=0.0,
            max_positions=2,
        )
        # A and B selected; C excluded (score < 0 filtered by backtrader engine on min_score=0)
        assert "A" in weights
        assert "B" in weights
        assert "C" not in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_empty_snapshot_returns_empty(self):
        weights = example_equal_weight_momentum_signal(
            pd.DataFrame(),
            {},
            momentum_column="momentum",
        )
        assert weights == {}

    def test_missing_column_returns_empty(self, symbol_feed_map):
        snapshot = pd.DataFrame({"other": [1.0]}, index=["A"])
        weights = example_equal_weight_momentum_signal(
            snapshot,
            {},
            momentum_column="momentum",
        )
        assert weights == {}

    def test_weights_sum_to_one(self, symbol_feed_map):
        snapshot = pd.DataFrame({"momentum": [2.0, 1.5, 0.5]}, index=["A", "B", "C"])
        weights = example_equal_weight_momentum_signal(
            snapshot, {}, momentum_column="momentum", max_positions=3
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# weight normalization edge cases
# ---------------------------------------------------------------------------

class TestFunctionalPortfolioWeightNormalization:
    """Exercise weight-normalization logic through a custom signal + batch."""

    def _make_signal(self, raw_weights: dict[str, float]):
        def signal(snapshot, context, **_):
            return raw_weights
        return signal

    def _run(self, symbol_feed_map, raw_weights):
        config = PortfolioBatchConfig(cash=100_000, max_gross_exposure=0.95)
        return run_portfolio_backtest_batch(
            symbol_feed_map=symbol_feed_map,
            strategy_callable=self._make_signal(raw_weights),
            config=config,
        )

    def test_zero_weight_signal_does_not_crash(self, symbol_feed_map):
        summary_df, errors, _ = self._run(symbol_feed_map, {"A": 0.0, "B": 0.0, "C": 0.0})
        assert errors == []
        assert len(summary_df) == 1

    def test_overweight_signal_gets_scaled(self, symbol_feed_map):
        # sum of weights > max_gross_exposure; engine should scale down
        summary_df, errors, _ = self._run(symbol_feed_map, {"A": 0.6, "B": 0.6, "C": 0.6})
        assert errors == []
        assert len(summary_df) == 1

    def test_unknown_symbol_in_signal_is_ignored(self, symbol_feed_map):
        summary_df, errors, _ = self._run(
            symbol_feed_map, {"A": 0.3, "UNKNOWN": 0.5, "B": 0.3}
        )
        assert errors == []
        assert len(summary_df) == 1
