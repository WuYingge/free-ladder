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
from typing import Any, Optional

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


def _apply_analysis_start_date(
    returns: pd.Series,
    analysis_start_date: Optional[pd.Timestamp],
) -> pd.Series:
    """Trim return series to dates on/after ``analysis_start_date`` when provided."""
    if analysis_start_date is None or len(returns) == 0:
        return returns
    start = pd.to_datetime(analysis_start_date)
    return returns.loc[returns.index >= start]


# ---------------------------------------------------------------------------
# Core metrics  (operate on a single return Series, 252 trading days / year)
# ---------------------------------------------------------------------------

def annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return."""
    n = len(returns)
    if n == 0:
        return float("nan")
    cumulative_raw: Any = (1.0 + returns).prod()
    cumulative = float(cumulative_raw)
    return cumulative ** (TRADING_DAYS_PER_YEAR / n) - 1.0


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
    cumulative_raw: Any = (1.0 + returns).prod()
    cumulative = float(cumulative_raw)
    return cumulative - 1.0


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
    analysis_start_date: Optional[pd.Timestamp] = None,
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
    analysis_start_date:
        Optional backtest analysis start date. When provided, strategy and
        benchmark return series are both trimmed to this date and later.
    """
    strat_ret = _to_series(strategy_time_return)
    bench_ret = _compute_bnh_returns(price_df)

    strat_ret = _apply_analysis_start_date(strat_ret, analysis_start_date)
    bench_ret = _apply_analysis_start_date(bench_ret, analysis_start_date)

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
    analysis_start_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Build a DataFrame with ``strategy`` and ``benchmark`` equity curves.

    Both series start at 1.0 and are aligned over the strategy's active period.
    Suitable for plotting.
    """
    strat_ret = _to_series(strategy_time_return)
    bench_ret = _compute_bnh_returns(price_df)

    strat_ret = _apply_analysis_start_date(strat_ret, analysis_start_date)
    bench_ret = _apply_analysis_start_date(bench_ret, analysis_start_date)

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


# ---------------------------------------------------------------------------
# Periodic (分段) performance metrics
# ---------------------------------------------------------------------------

def compute_periodic_metrics(
    returns: pd.Series,
    freq: str = "YE",
    risk_free_rate: float = 0.0,
    periods: Optional[list[tuple[str, str]]] = None,
) -> pd.DataFrame:
    """按时间段分组计算绩效指标，返回每段一行。

    支持两种分段方式（``periods`` 优先于 ``freq``）：

    * **固定频率** — 通过 ``freq`` 指定 pandas 时间频率：
      ``'YE'``(年)、``'QE'``(季)、``'ME'``(月)、``'W'``(周)。
    * **自定义时间段** — 通过 ``periods`` 指定起止日列表：
      ``[("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")]``。

    Parameters
    ----------
    returns:
        日收益率序列，索引为 ``datetime`` 类型。
    freq:
        pandas 时间频率标签（``periods`` 为 None 时生效）。
    risk_free_rate:
        年化无风险利率，用于夏普比率计算。
    periods:
        自定义时间段列表，每个元素为 ``(start_date_str, end_date_str)``。
        传入后 ``freq`` 被忽略。

    Returns
    -------
    pd.DataFrame，列包括：
        - ``period``: 时间段标签
        - ``start_date`` / ``end_date``: 时间段起止日
        - ``trading_days``: 该段交易日数
        - ``cumulative_return_pct``: 累积收益(%)
        - ``annualised_return_pct``: 年化收益(%)
        - ``annualised_volatility_pct``: 年化波动率(%)
        - ``sharpe``: 夏普比率
        - ``max_drawdown_pct``: 最大回撤(%)
        - ``calmar``: Calmar 比率

    若 ``returns`` 为空或所有段均无足够数据，返回空 DataFrame。
    """
    _empty_columns = [
        "period", "start_date", "end_date", "trading_days",
        "cumulative_return_pct", "annualised_return_pct",
        "annualised_volatility_pct", "sharpe", "max_drawdown_pct",
        "calmar",
    ]
    if len(returns) < 2:
        return pd.DataFrame(columns=_empty_columns)

    rows: list[dict[str, Any]] = []

    if periods is not None:
        # ---- 自定义时间段模式 ----
        for start_str, end_str in periods:
            start = pd.Timestamp(start_str)
            end = pd.Timestamp(end_str)
            group = returns.loc[(returns.index >= start) & (returns.index <= end)].dropna()
            if len(group) < 2:
                continue
            row = _compute_single_period_row(
                group, period_label=f"{start_str} → {end_str}",
                risk_free_rate=risk_free_rate,
            )
            rows.append(row)
        return pd.DataFrame(rows)

    # ---- 固定频率模式 ----
    groups = returns.groupby(pd.Grouper(freq=freq))
    _freq_label = _freq_to_label_format(freq)

    for period_start, group in groups:
        group = group.dropna()
        if len(group) < 2:
            continue
        period_end = group.index[-1]
        label = _format_period_label(
            pd.Timestamp(period_start), pd.Timestamp(period_end), _freq_label,
        )
        row = _compute_single_period_row(
            group, period_label=label,
            risk_free_rate=risk_free_rate,
        )
        # 覆盖 start_date / end_date 为分组实际边界
        row["start_date"] = str(pd.Timestamp(period_start).date())
        row["end_date"] = str(period_end.date())
        rows.append(row)

    return pd.DataFrame(rows)


def _compute_single_period_row(
    group: pd.Series,
    period_label: str,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """对单个时间段的收益率序列计算绩效指标，返回一行 dict。"""
    cum = cumulative_return(group)
    ann = annualised_return(group)
    vol = annualised_volatility(group)
    sh = sharpe_ratio(group, risk_free_rate=risk_free_rate)
    mdd = max_drawdown(group)
    cal = None
    if not math.isnan(ann) and not math.isnan(mdd) and mdd > 0:
        cal = ann / mdd

    return {
        "period": period_label,
        "start_date": str(group.index[0].date()),
        "end_date": str(group.index[-1].date()),
        "trading_days": len(group),
        "cumulative_return_pct": (
            round(float(cum) * 100.0, 4) if not math.isnan(cum) else None
        ),
        "annualised_return_pct": (
            round(float(ann) * 100.0, 4) if not math.isnan(ann) else None
        ),
        "annualised_volatility_pct": (
            round(float(vol) * 100.0, 4) if not math.isnan(vol) else None
        ),
        "sharpe": round(float(sh), 4) if sh is not None else None,
        "max_drawdown_pct": (
            round(float(mdd) * 100.0, 4) if not math.isnan(mdd) else None
        ),
        "calmar": round(float(cal), 4) if cal is not None else None,
    }


def _freq_to_label_format(freq: str) -> str:
    """将 pandas 频率转换为时段标签格式键。"""
    _freq_label_map = {
        "YE": "year",
        "Y": "year",
        "QE": "quarter",
        "Q": "quarter",
        "ME": "month",
        "M": "month",
        "W": "week",
    }
    return _freq_label_map.get(freq.upper(), freq)


def _format_period_label(period_start: pd.Timestamp, period_end: pd.Timestamp, fmt: str) -> str:
    """根据频率类型生成可读的时段标签。"""
    if fmt == "year":
        return str(period_start.year)
    elif fmt == "quarter":
        return f"{period_start.year}Q{period_start.quarter}"
    elif fmt == "month":
        return period_start.strftime("%Y-%m")
    elif fmt == "week":
        return f"{period_start.strftime('%Y-%m-%d')}~{period_end.strftime('%Y-%m-%d')}"
    else:
        return f"{period_start.date()} → {period_end.date()}"
