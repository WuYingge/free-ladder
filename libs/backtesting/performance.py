"""
backtesting.performance
=======================
Compute portfolio performance metrics from daily return series.

All public functions operate on ``pandas.Series`` indexed by ``datetime.date``
(or anything that can serve as a date-like index).  Inputs come from the
``analyzer_raw["time_return"]`` dict produced by the enhanced engine.

Baseline convention
-------------------
The *benchmark* for every symbol is **buy-and-hold** on that same instrument:
    daily_return_t = close_t / close_{t-1} - 1

The strategy return series is taken from Backtrader's ``TimeReturn`` analyser
which records the fractional change in total portfolio value each day
(cash + open position mark-to-market).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252


def _to_series(raw: dict[str, float]) -> pd.Series:
    """Convert the string-keyed time_return dict from analyzer_raw to a Series."""
    if not raw:
        return pd.Series(dtype=float)
    s = pd.Series(raw)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().astype(float)


def _compute_bnh_returns(price_df: pd.DataFrame) -> pd.Series:
    """Daily buy-and-hold returns for a normalised OHLCV DataFrame.

    ``price_df`` must have a datetime index and a ``close`` column.
    Returns a Series of fractional daily returns (first day → NaN dropped).
    """
    close = pd.to_numeric(price_df["close"], errors="coerce")
    close = close[close > 0].dropna()
    returns = close.pct_change().dropna()
    returns.index = pd.to_datetime(returns.index)
    return returns.astype(float)


# ---------------------------------------------------------------------------
# Core metrics  (operate on a single return Series, 252 trading days / year)
# ---------------------------------------------------------------------------

def annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return."""
    n = len(returns)
    if n == 0:
        return float("nan")
    cumulative = (1.0 + returns).prod()
    return float(cumulative ** (TRADING_DAYS_PER_YEAR / n) - 1.0)


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown as a *positive* fraction (e.g. 0.15 means −15%)."""
    if len(returns) == 0:
        return float("nan")
    cum = (1.0 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    mdd = float(dd.min())
    return abs(mdd)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> Optional[float]:
    """Annualised Sharpe ratio (daily returns, assumes 252 trading days)."""
    ann_vol = annualised_volatility(returns)
    if not ann_vol or math.isnan(ann_vol) or ann_vol == 0.0:
        return None
    excess = annualised_return(returns) - risk_free_rate
    return float(excess / ann_vol)


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """Annualised Information Ratio.

    IR = annualised(excess_returns) / tracking_error
    where excess_returns = strategy - benchmark (aligned by date).
    Returns None when tracking error is zero or there are fewer than 2 aligned days.
    """
    strat, bench = strategy_returns.align(benchmark_returns, join="inner")
    excess = strat - bench
    if len(excess) < 2:
        return None
    te = float(excess.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    if te == 0.0 or math.isnan(te):
        return None
    ann_excess = annualised_return(excess)
    return float(ann_excess / te)


def cumulative_return(returns: pd.Series) -> float:
    """Total cumulative return over all periods."""
    if len(returns) == 0:
        return float("nan")
    return float((1.0 + returns).prod() - 1.0)


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """Cumulative equity curve starting at *start*."""
    return (start * (1.0 + returns).cumprod()).rename("equity")


# ---------------------------------------------------------------------------
# Top-level aggregation
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    strategy_time_return: dict[str, float],
    price_df: pd.DataFrame,
    *,
    symbol: str = "",
    name: str = "",
    trades_total: int = 0,
    trades_won: int = 0,
    trades_lost: int = 0,
    max_drawdown_pct_from_engine: Optional[float] = None,
) -> dict:
    """Return a flat dict of all performance metrics for one symbol.

    Parameters
    ----------
    strategy_time_return:
        ``analyzer_raw["time_return"]`` from
        ``SingleFactorSingleTargetBacktestResult``.  Keys are ISO date strings.
    price_df:
        Normalised OHLCV DataFrame (datetime index, ``close`` column required).
    symbol / name:
        Passed through into the result dict for identification.
    trades_total / trades_won / trades_lost:
        From the engine result, forwarded as-is.
    max_drawdown_pct_from_engine:
        Engine-level max drawdown (from Backtrader ``DrawDown`` analyser) used as
        a cross-check fallback; if None we compute it from the return series.
    """
    strat_ret = _to_series(strategy_time_return)
    bench_ret = _compute_bnh_returns(price_df)

    # Align to common date window (strategy determines the active period).
    if len(strat_ret) >= 2:
        start_date = strat_ret.index[0]
        end_date = strat_ret.index[-1]
        bench_ret = bench_ret.loc[
            (bench_ret.index >= start_date) & (bench_ret.index <= end_date)
        ]

    strat_cum = cumulative_return(strat_ret)
    bench_cum = cumulative_return(bench_ret)
    excess_cum = (
        (strat_cum + 1.0) / (bench_cum + 1.0) - 1.0
        if not (math.isnan(strat_cum) or math.isnan(bench_cum))
        else float("nan")
    )

    strat_ann = annualised_return(strat_ret)
    bench_ann = annualised_return(bench_ret)
    strat_vol = annualised_volatility(strat_ret)
    strat_sharpe = sharpe_ratio(strat_ret)
    strat_mdd = max_drawdown(strat_ret)
    ir = information_ratio(strat_ret, bench_ret)
    win_rate = (
        float(trades_won / trades_total) if trades_total > 0 else float("nan")
    )

    date_range_start = str(strat_ret.index[0].date()) if len(strat_ret) > 0 else ""
    date_range_end = str(strat_ret.index[-1].date()) if len(strat_ret) > 0 else ""

    return {
        "symbol": symbol,
        "name": name,
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        # Returns
        "strategy_cumulative_return": round(strat_cum, 6) if not math.isnan(strat_cum) else None,
        "benchmark_cumulative_return": round(bench_cum, 6) if not math.isnan(bench_cum) else None,
        "excess_cumulative_return": round(excess_cum, 6) if not math.isnan(excess_cum) else None,
        "strategy_annualised_return": round(strat_ann, 6) if not math.isnan(strat_ann) else None,
        "benchmark_annualised_return": round(bench_ann, 6) if not math.isnan(bench_ann) else None,
        "strategy_annualised_volatility": round(strat_vol, 6) if not math.isnan(strat_vol) else None,
        # Risk-adjusted
        "sharpe": round(strat_sharpe, 4) if strat_sharpe is not None else None,
        "information_ratio": round(ir, 4) if ir is not None else None,
        "max_drawdown_pct": round(strat_mdd * 100, 4) if not math.isnan(strat_mdd) else max_drawdown_pct_from_engine,
        # Trades
        "trades_total": trades_total,
        "trades_won": trades_won,
        "trades_lost": trades_lost,
        "win_rate": round(win_rate, 4) if not math.isnan(win_rate) else None,
    }


def build_equity_curves(
    strategy_time_return: dict[str, float],
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a DataFrame with ``strategy`` and ``benchmark`` equity curves.

    Both series start at 1.0 and are aligned over the strategy's active period.
    Suitable for plotting.
    """
    strat_ret = _to_series(strategy_time_return)
    bench_ret = _compute_bnh_returns(price_df)

    if len(strat_ret) == 0:
        return pd.DataFrame(columns=["strategy", "benchmark"])

    start_date = strat_ret.index[0]
    end_date = strat_ret.index[-1]
    bench_ret = bench_ret.loc[
        (bench_ret.index >= start_date) & (bench_ret.index <= end_date)
    ]

    strat_curve = equity_curve(strat_ret, start=1.0).rename("strategy")
    bench_curve = equity_curve(bench_ret, start=1.0).rename("benchmark")
    df = pd.concat([strat_curve, bench_curve], axis=1).sort_index()
    df.index.name = "date"
    return df
