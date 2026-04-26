from __future__ import annotations

from typing import Optional

import pandas as pd

from backtesting.performance import (
    _apply_analysis_start_date,
    _compute_bnh_returns,
    _to_series,
    build_equity_curves,
    compute_performance_metrics,
    equity_curve,
)


def _normalize_static_weights(symbols: list[str], weights: Optional[dict[str, float]]) -> dict[str, float]:
    if not symbols:
        return {}
    if not weights:
        equal = 1.0 / float(len(symbols))
        return {symbol: equal for symbol in symbols}

    normalized: dict[str, float] = {symbol: 0.0 for symbol in symbols}
    for symbol, raw_weight in weights.items():
        if symbol not in normalized:
            continue
        try:
            normalized[symbol] = max(0.0, float(raw_weight))
        except (TypeError, ValueError):
            normalized[symbol] = 0.0

    total = sum(normalized.values())
    if total <= 0:
        equal = 1.0 / float(len(symbols))
        return {symbol: equal for symbol in symbols}
    return {symbol: weight / total for symbol, weight in normalized.items()}


def build_portfolio_benchmark_returns(
    price_df_by_symbol: dict[str, pd.DataFrame],
    *,
    benchmark_weights: Optional[dict[str, float]] = None,
    analysis_start_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    symbols = sorted(price_df_by_symbol.keys())
    if not symbols:
        return pd.Series(dtype=float)

    weights = _normalize_static_weights(symbols, benchmark_weights)
    weighted_returns: list[pd.Series] = []

    for symbol in symbols:
        price_df = price_df_by_symbol[symbol]
        returns = _compute_bnh_returns(price_df)
        returns = _apply_analysis_start_date(returns, analysis_start_date)
        weighted_returns.append(returns * weights.get(symbol, 0.0))

    if not weighted_returns:
        return pd.Series(dtype=float)

    benchmark = pd.concat(weighted_returns, axis=1).fillna(0.0).sum(axis=1)
    benchmark.name = "benchmark_return"
    benchmark.index = pd.to_datetime(benchmark.index)
    return benchmark.sort_index().astype(float)


def compute_portfolio_performance_metrics(
    strategy_time_return: dict[str, float],
    price_df_by_symbol: dict[str, pd.DataFrame],
    *,
    name: str = "Portfolio",
    trades_total: int = 0,
    trades_won: int = 0,
    trades_lost: int = 0,
    max_drawdown_pct_from_engine: Optional[float] = None,
    benchmark_weights: Optional[dict[str, float]] = None,
    analysis_start_date: Optional[pd.Timestamp] = None,
) -> dict:
    strategy_returns = _to_series(strategy_time_return)
    strategy_returns = _apply_analysis_start_date(strategy_returns, analysis_start_date)

    benchmark_returns = build_portfolio_benchmark_returns(
        price_df_by_symbol,
        benchmark_weights=benchmark_weights,
        analysis_start_date=analysis_start_date,
    )

    if len(strategy_returns) >= 2 and len(benchmark_returns) > 0:
        start_date = strategy_returns.index[0]
        end_date = strategy_returns.index[-1]
        benchmark_returns = benchmark_returns.loc[
            (benchmark_returns.index >= start_date) & (benchmark_returns.index <= end_date)
        ]

    synthetic_close = equity_curve(benchmark_returns, start=1.0).rename("close")
    benchmark_price_df = pd.DataFrame({"close": synthetic_close})

    return compute_performance_metrics(
        strategy_time_return={str(k): float(v) for k, v in strategy_returns.items()},
        price_df=benchmark_price_df,
        symbol="PORTFOLIO",
        name=name,
        trades_total=trades_total,
        trades_won=trades_won,
        trades_lost=trades_lost,
        max_drawdown_pct_from_engine=max_drawdown_pct_from_engine,
        analysis_start_date=analysis_start_date,
    )


def build_portfolio_equity_curves(
    strategy_time_return: dict[str, float],
    price_df_by_symbol: dict[str, pd.DataFrame],
    *,
    benchmark_weights: Optional[dict[str, float]] = None,
    analysis_start_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    benchmark_returns = build_portfolio_benchmark_returns(
        price_df_by_symbol,
        benchmark_weights=benchmark_weights,
        analysis_start_date=analysis_start_date,
    )
    synthetic_close = equity_curve(benchmark_returns, start=1.0).rename("close")
    benchmark_price_df = pd.DataFrame({"close": synthetic_close})
    return build_equity_curves(
        strategy_time_return=strategy_time_return,
        price_df=benchmark_price_df,
        analysis_start_date=analysis_start_date,
    )
