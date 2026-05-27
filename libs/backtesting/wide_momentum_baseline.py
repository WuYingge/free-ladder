from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd

from config import DataPath
from data_manager.providers.cluster_provider import ClusterInfo
from data_manager.providers.etf_list_provider import ETF_LIST
from factors.price_return import PriceReturn

from .data import load_etf_dataframe
from .performance import (
    annualised_return,
    annualised_volatility,
    cumulative_return,
    sharpe_ratio,
)


@dataclass(slots=True)
class WideMomentumBaselineConfig:
    top_n_values: tuple[int, ...] = (5, 10)
    min_listing_days: int = 1200
    momentum_window: int = 20
    momentum_skip_recent: int = 1
    rebalance_interval: int = 20
    cash: float = 100_000.0
    commission: float = 0.0005
    risk_free_rate: float = 0.02
    stable_pool_size: int = 100
    start_date: Optional[str | pd.Timestamp] = None
    end_date: Optional[str | pd.Timestamp] = None
    symbols: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not self.top_n_values:
            raise ValueError("top_n_values cannot be empty")
        if any(int(top_n) <= 0 for top_n in self.top_n_values):
            raise ValueError("top_n_values must all be >= 1")
        if int(self.momentum_window) <= 0:
            raise ValueError("momentum_window must be >= 1")
        if int(self.momentum_skip_recent) < 0:
            raise ValueError("momentum_skip_recent must be >= 0")
        if int(self.rebalance_interval) <= 0:
            raise ValueError("rebalance_interval must be >= 1")
        if float(self.cash) <= 0:
            raise ValueError("cash must be > 0")
        if float(self.commission) < 0:
            raise ValueError("commission must be >= 0")
        if float(self.risk_free_rate) < 0:
            raise ValueError("risk_free_rate must be >= 0")
        if int(self.stable_pool_size) <= 0:
            raise ValueError("stable_pool_size must be >= 1")
        if int(self.min_listing_days) < 0:
            raise ValueError("min_listing_days must be >= 0")


@dataclass(slots=True)
class SymbolBaselineData:
    symbol: str
    listing_proxy_date: pd.Timestamp
    cluster_label: int
    frame: pd.DataFrame


@dataclass(slots=True)
class PreparedWideMomentumUniverse:
    symbol_data_map: dict[str, SymbolBaselineData]
    calendar: pd.DatetimeIndex
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    recent_complete_date: pd.Timestamp
    monthly_pool_diagnostics: pd.DataFrame
    source_symbol_count: int
    load_errors: list[dict[str, str]] = field(default_factory=list)
    excluded_symbols: dict[str, str] = field(default_factory=dict)

    @property
    def stable_start_month(self) -> Optional[pd.Timestamp]:
        if self.monthly_pool_diagnostics.empty:
            return None
        stable_rows = self.monthly_pool_diagnostics[
            self.monthly_pool_diagnostics["is_stable_start"]
        ]
        if stable_rows.empty:
            return None
        return pd.to_datetime(stable_rows.iloc[0]["month_end_date"])


@dataclass(slots=True)
class WideMomentumVariantResult:
    top_n: int
    summary: dict[str, Any]
    equity_curve: pd.DataFrame
    annual_returns: pd.DataFrame
    rebalance_log: pd.DataFrame


@dataclass(slots=True)
class WideMomentumBaselineResult:
    config: WideMomentumBaselineConfig
    prepared_universe: PreparedWideMomentumUniverse
    variant_results: dict[int, WideMomentumVariantResult]


def _normalize_price_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    local_df = frame.copy()
    if "date" in local_df.columns:
        local_df["date"] = pd.to_datetime(local_df["date"], errors="coerce")
        local_df = local_df.set_index("date")
    else:
        local_df.index = pd.to_datetime(local_df.index, errors="coerce")

    local_df = local_df[~local_df.index.isna()].sort_index()
    if local_df.empty:
        raise ValueError(f"{symbol} has no valid rows")

    required_columns = {"open", "close"}
    missing_columns = required_columns.difference(local_df.columns)
    if missing_columns:
        raise ValueError(f"{symbol} missing required columns: {sorted(missing_columns)}")

    local_df["open"] = pd.to_numeric(local_df["open"], errors="coerce")
    local_df["close"] = pd.to_numeric(local_df["close"], errors="coerce")
    local_df = local_df.dropna(subset=["open", "close"])
    local_df = local_df[local_df["open"] > 0]
    local_df = local_df[local_df["close"] > 0]
    if local_df.empty:
        raise ValueError(f"{symbol} has no valid open/close rows after normalization")

    return local_df[["open", "close"]].copy()


def _resolve_recent_complete_date(
    last_dates: list[pd.Timestamp],
    explicit_end_date: Optional[str | pd.Timestamp],
) -> pd.Timestamp:
    if explicit_end_date is not None:
        return pd.to_datetime(explicit_end_date)

    counts = pd.Series(pd.to_datetime(last_dates)).value_counts()
    if counts.empty:
        raise ValueError("No last_dates available to resolve end date")

    top_count = int(counts.iloc[0])
    candidates = sorted(pd.to_datetime(counts[counts == top_count].index))
    return pd.Timestamp(candidates[-1])


def _load_trading_calendar(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    calendar_df = pd.read_csv(
        DataPath.CALANDAR_DF,
        parse_dates=["trade_date"],
    )
    trade_dates = pd.to_datetime(calendar_df["trade_date"], errors="coerce")
    trade_dates = trade_dates.dropna().sort_values().drop_duplicates()
    mask = (trade_dates >= start_date) & (trade_dates <= end_date)
    calendar = pd.DatetimeIndex(trade_dates.loc[mask])
    if calendar.empty:
        raise ValueError(
            f"No trading days found between {start_date.date()} and {end_date.date()}"
        )
    return calendar


def _resolve_cluster_label(
    symbol: str,
    cluster_lookup: Optional[Mapping[str, int] | Callable[[str], int]],
) -> int:
    if cluster_lookup is None:
        return int(ClusterInfo.get_cluster(symbol))
    if callable(cluster_lookup):
        return int(cluster_lookup(symbol))
    return int(cluster_lookup.get(symbol, -1))


def _month_end_dates(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if len(calendar) == 0:
        return []
    month_end_series = pd.Series(calendar, index=calendar)
    return [pd.Timestamp(value) for value in month_end_series.groupby(calendar.to_period("M")).max()]


def _build_monthly_pool_diagnostics(
    symbol_data_map: Mapping[str, SymbolBaselineData],
    calendar: pd.DatetimeIndex,
    stable_pool_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stable_marked = False

    for month_end_date in _month_end_dates(calendar):
        eligible_symbols: list[str] = []
        covered_clusters: set[int] = set()
        unassigned_cluster_symbols = 0

        for symbol, symbol_data in symbol_data_map.items():
            eligible = bool(
                symbol_data.frame["eligible_signal"].reindex([month_end_date], fill_value=False).iloc[0]
            )
            if not eligible:
                continue
            eligible_symbols.append(symbol)
            if symbol_data.cluster_label >= 0:
                covered_clusters.add(int(symbol_data.cluster_label))
            else:
                unassigned_cluster_symbols += 1

        pool_size = len(eligible_symbols)
        is_pool_stable = pool_size >= stable_pool_size
        is_stable_start = bool(is_pool_stable and not stable_marked)
        if is_stable_start:
            stable_marked = True

        rows.append(
            {
                "month": month_end_date.strftime("%Y-%m"),
                "month_end_date": month_end_date,
                "eligible_symbol_count": pool_size,
                "covered_cluster_count": len(covered_clusters),
                "unassigned_cluster_symbol_count": int(unassigned_cluster_symbols),
                "is_pool_stable": is_pool_stable,
                "is_stable_start": is_stable_start,
            }
        )

    return pd.DataFrame(rows)


def prepare_wide_momentum_universe_from_frames(
    symbol_frame_map: Mapping[str, pd.DataFrame],
    config: WideMomentumBaselineConfig,
    *,
    calendar: Optional[pd.DatetimeIndex] = None,
    cluster_lookup: Optional[Mapping[str, int] | Callable[[str], int]] = None,
) -> PreparedWideMomentumUniverse:
    normalized_frames: dict[str, pd.DataFrame] = {}
    load_errors: list[dict[str, str]] = []

    for symbol, frame in symbol_frame_map.items():
        try:
            normalized_frames[symbol] = _normalize_price_frame(frame=frame, symbol=symbol)
        except Exception as exc:
            load_errors.append({"symbol": symbol, "error": str(exc)})

    if not normalized_frames:
        raise ValueError("No valid ETF frames available for baseline prep")

    recent_complete_date = _resolve_recent_complete_date(
        last_dates=[frame.index[-1] for frame in normalized_frames.values()],
        explicit_end_date=config.end_date,
    )

    factor = PriceReturn(
        window=int(config.momentum_window),
        skip_recent=int(config.momentum_skip_recent),
    )
    symbol_data_map: dict[str, SymbolBaselineData] = {}
    excluded_symbols: dict[str, str] = {}

    for symbol, frame in normalized_frames.items():
        if frame.index[-1] < recent_complete_date:
            excluded_symbols[symbol] = "stale_before_recent_complete_date"
            continue

        local_frame = frame.loc[frame.index <= recent_complete_date].copy()
        listing_proxy_date = pd.Timestamp(local_frame.index[0])
        momentum = factor(local_frame)
        eligible_signal = (local_frame.index > listing_proxy_date + pd.Timedelta(days=int(config.min_listing_days))) & momentum.notna()
        local_frame["momentum"] = momentum.astype(float)
        local_frame["eligible_signal"] = eligible_signal.astype(bool)

        if not bool(local_frame["eligible_signal"].any()):
            excluded_symbols[symbol] = "never_eligible_in_window"
            continue

        symbol_data_map[symbol] = SymbolBaselineData(
            symbol=symbol,
            listing_proxy_date=listing_proxy_date,
            cluster_label=_resolve_cluster_label(symbol, cluster_lookup),
            frame=local_frame,
        )

    if not symbol_data_map:
        raise ValueError("No symbols remain after eligibility filtering")

    computed_start_date = min(
        pd.Timestamp(symbol_data.frame.index[symbol_data.frame["eligible_signal"]][0])
        for symbol_data in symbol_data_map.values()
    )
    start_date = computed_start_date
    if config.start_date is not None:
        start_date = max(start_date, pd.to_datetime(config.start_date))

    resolved_calendar = pd.DatetimeIndex(pd.to_datetime(calendar)) if calendar is not None else _load_trading_calendar(
        start_date=start_date,
        end_date=recent_complete_date,
    )
    resolved_calendar = resolved_calendar[(resolved_calendar >= start_date) & (resolved_calendar <= recent_complete_date)]
    resolved_calendar = resolved_calendar.sort_values().drop_duplicates()
    if resolved_calendar.empty:
        raise ValueError("Resolved trading calendar is empty")

    aligned_start_date = pd.Timestamp(resolved_calendar[0])
    monthly_pool_diagnostics = _build_monthly_pool_diagnostics(
        symbol_data_map=symbol_data_map,
        calendar=resolved_calendar,
        stable_pool_size=int(config.stable_pool_size),
    )

    return PreparedWideMomentumUniverse(
        symbol_data_map=symbol_data_map,
        calendar=resolved_calendar,
        start_date=aligned_start_date,
        end_date=pd.Timestamp(resolved_calendar[-1]),
        recent_complete_date=recent_complete_date,
        monthly_pool_diagnostics=monthly_pool_diagnostics,
        source_symbol_count=len(symbol_frame_map),
        load_errors=load_errors,
        excluded_symbols=excluded_symbols,
    )


def prepare_wide_momentum_universe(
    config: WideMomentumBaselineConfig,
    *,
    symbols: Optional[list[str]] = None,
    cluster_lookup: Optional[Mapping[str, int] | Callable[[str], int]] = None,
) -> PreparedWideMomentumUniverse:
    target_symbols = symbols or list(config.symbols or tuple(ETF_LIST.get_all_symbol()))
    symbol_frame_map: dict[str, pd.DataFrame] = {}

    for symbol in target_symbols:
        try:
            symbol_frame_map[symbol] = load_etf_dataframe(symbol)
        except Exception:
            continue

    return prepare_wide_momentum_universe_from_frames(
        symbol_frame_map=symbol_frame_map,
        config=config,
        cluster_lookup=cluster_lookup,
    )


def _portfolio_value(
    positions: Mapping[str, float],
    cash: float,
    date: pd.Timestamp,
    symbol_data_map: Mapping[str, SymbolBaselineData],
    price_column: str,
) -> float:
    total_value = float(cash)
    for symbol, shares in positions.items():
        if abs(float(shares)) <= 1e-12:
            continue
        price = symbol_data_map[symbol].frame[price_column].reindex([date]).iloc[0]
        if pd.isna(price) or float(price) <= 0:
            raise ValueError(f"Missing {price_column} price for {symbol} on {date.date()}")
        total_value += float(shares) * float(price)
    return float(total_value)


def _portfolio_weights(
    positions: Mapping[str, float],
    cash: float,
    date: pd.Timestamp,
    symbol_data_map: Mapping[str, SymbolBaselineData],
    price_column: str,
) -> dict[str, float]:
    portfolio_value = _portfolio_value(
        positions=positions,
        cash=cash,
        date=date,
        symbol_data_map=symbol_data_map,
        price_column=price_column,
    )
    if portfolio_value <= 0:
        return {}

    weights: dict[str, float] = {}
    for symbol, shares in positions.items():
        if abs(float(shares)) <= 1e-12:
            continue
        price = float(symbol_data_map[symbol].frame[price_column].reindex([date]).iloc[0])
        weights[symbol] = float(shares) * price / float(portfolio_value)
    return weights


def _select_target_weights(
    symbol_data_map: Mapping[str, SymbolBaselineData],
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    top_n: int,
) -> tuple[dict[str, float], int]:
    candidates: list[tuple[str, float]] = []

    for symbol, symbol_data in symbol_data_map.items():
        frame = symbol_data.frame
        signal_row = frame.reindex([signal_date])
        execution_row = frame.reindex([execution_date])
        if signal_row.empty or execution_row.empty:
            continue

        eligible = bool(signal_row["eligible_signal"].iloc[0])
        momentum = signal_row["momentum"].iloc[0]
        next_open = execution_row["open"].iloc[0]
        if not eligible or pd.isna(momentum) or pd.isna(next_open):
            continue
        if float(next_open) <= 0:
            continue
        candidates.append((symbol, float(momentum)))

    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected = candidates[: int(top_n)]
    if not selected:
        return {}, len(candidates)

    weight = 1.0 / float(len(selected))
    return {symbol: weight for symbol, _ in selected}, len(candidates)


def _solve_target_exposures(
    portfolio_value: float,
    current_exposures: Mapping[str, float],
    target_weights: Mapping[str, float],
    commission: float,
) -> tuple[dict[str, float], float, float, float]:
    if portfolio_value <= 0:
        return {}, 0.0, 0.0, 0.0

    if not target_weights:
        trade_notional = float(sum(abs(float(value)) for value in current_exposures.values()))
        commission_paid = trade_notional * float(commission)
        cash_after = float(portfolio_value) - commission_paid
        return {}, commission_paid, trade_notional, cash_after

    normalized_weights = {
        symbol: float(weight) / float(sum(target_weights.values()))
        for symbol, weight in target_weights.items()
        if float(weight) > 0.0
    }

    symbols = set(current_exposures).union(normalized_weights)
    low = 0.0
    high = 1.0

    def _cash_after(scale: float) -> float:
        desired_total = float(portfolio_value) * float(scale)
        trade_notional = 0.0
        for symbol in symbols:
            desired = desired_total * normalized_weights.get(symbol, 0.0)
            current = float(current_exposures.get(symbol, 0.0))
            trade_notional += abs(desired - current)
        return float(portfolio_value) - desired_total - float(commission) * trade_notional

    for _ in range(60):
        mid = (low + high) / 2.0
        if _cash_after(mid) >= 0.0:
            low = mid
        else:
            high = mid

    desired_total = float(portfolio_value) * low
    desired_exposures = {
        symbol: desired_total * weight
        for symbol, weight in normalized_weights.items()
        if desired_total * weight > 1e-12
    }
    trade_notional = 0.0
    for symbol in symbols:
        trade_notional += abs(
            desired_exposures.get(symbol, 0.0) - float(current_exposures.get(symbol, 0.0))
        )
    commission_paid = float(commission) * trade_notional
    cash_after = float(portfolio_value) - sum(desired_exposures.values()) - commission_paid
    if abs(cash_after) < 1e-10:
        cash_after = 0.0
    return desired_exposures, commission_paid, trade_notional, cash_after


def _execute_rebalance(
    *,
    positions: Mapping[str, float],
    cash: float,
    execution_date: pd.Timestamp,
    target_weights: Mapping[str, float],
    symbol_data_map: Mapping[str, SymbolBaselineData],
    commission: float,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    open_prices: dict[str, float] = {}
    involved_symbols = set(positions).union(target_weights)

    for symbol in involved_symbols:
        open_price = symbol_data_map[symbol].frame["open"].reindex([execution_date]).iloc[0]
        if pd.isna(open_price) or float(open_price) <= 0:
            raise ValueError(f"Missing execution open price for {symbol} on {execution_date.date()}")
        open_prices[symbol] = float(open_price)

    current_exposures = {
        symbol: float(shares) * open_prices[symbol]
        for symbol, shares in positions.items()
        if abs(float(shares)) > 1e-12
    }
    portfolio_value_before_trade = float(cash) + sum(current_exposures.values())
    desired_exposures, commission_paid, trade_notional, cash_after = _solve_target_exposures(
        portfolio_value=portfolio_value_before_trade,
        current_exposures=current_exposures,
        target_weights=target_weights,
        commission=commission,
    )

    new_positions: dict[str, float] = {}
    for symbol, desired_exposure in desired_exposures.items():
        shares = desired_exposure / open_prices[symbol]
        if abs(shares) > 1e-12:
            new_positions[symbol] = float(shares)

    portfolio_value_after_trade = float(cash_after) + sum(desired_exposures.values())
    executed_weights_open = {
        symbol: desired_exposure / portfolio_value_after_trade
        for symbol, desired_exposure in desired_exposures.items()
        if portfolio_value_after_trade > 0
    }

    execution_info = {
        "execution_date": execution_date,
        "portfolio_value_before_trade": float(portfolio_value_before_trade),
        "portfolio_value_after_trade": float(portfolio_value_after_trade),
        "commission_paid": float(commission_paid),
        "trade_notional": float(trade_notional),
        "cash_after_trade": float(cash_after),
        "executed_weights_open": executed_weights_open,
    }
    return new_positions, float(cash_after), execution_info


def _compute_weight_turnover(
    previous_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
) -> float:
    symbols = set(previous_weights).union(current_weights)
    return 0.5 * float(
        sum(
            abs(float(current_weights.get(symbol, 0.0)) - float(previous_weights.get(symbol, 0.0)))
            for symbol in symbols
        )
    )


def _finalize_rebalance_log(
    rebalance_entries: list[dict[str, Any]],
) -> pd.DataFrame:
    previous_target_weights: Optional[dict[str, float]] = None
    for entry in rebalance_entries:
        current_target_weights = dict(entry.get("target_weights", {}))
        if previous_target_weights is None:
            entry["turnover"] = None
        else:
            entry["turnover"] = _compute_weight_turnover(
                previous_weights=previous_target_weights,
                current_weights=current_target_weights,
            )
        previous_target_weights = current_target_weights

    for idx, entry in enumerate(rebalance_entries[:-1]):
        next_entry = rebalance_entries[idx + 1]
        after_trade_value = entry.get("portfolio_value_after_trade")
        next_before_trade = next_entry.get("portfolio_value_before_trade")
        if after_trade_value is None or next_before_trade is None:
            entry["period_return"] = None
            continue
        entry["period_return"] = float(next_before_trade) / float(after_trade_value) - 1.0

    if rebalance_entries:
        rebalance_entries[-1]["period_return"] = None

    rebalance_df = pd.DataFrame(rebalance_entries)
    if rebalance_df.empty:
        return rebalance_df
    for column in ("signal_date", "execution_date"):
        if column in rebalance_df.columns:
            rebalance_df[column] = pd.to_datetime(rebalance_df[column])
    return rebalance_df


def _compute_drawdown_details(equity: pd.Series) -> dict[str, Any]:
    if equity.empty:
        return {
            "max_drawdown": float("nan"),
            "peak_date": None,
            "trough_date": None,
        }

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    trough_date = pd.Timestamp(drawdown.idxmin())
    peak_date = pd.Timestamp(equity.loc[:trough_date].idxmax())
    max_drawdown = abs(float(drawdown.min()))
    return {
        "max_drawdown": max_drawdown,
        "peak_date": peak_date,
        "trough_date": trough_date,
    }


def _build_annual_returns(equity_curve_df: pd.DataFrame) -> pd.DataFrame:
    if equity_curve_df.empty:
        return pd.DataFrame(columns=["year", "annual_return_pct"])

    returns = equity_curve_df["equity"].pct_change().dropna()
    if returns.empty:
        return pd.DataFrame(columns=["year", "annual_return_pct"])

    annual = returns.groupby(returns.index.year).apply(lambda values: (1.0 + values).prod() - 1.0)
    annual_df = annual.rename("annual_return_pct").mul(100.0).reset_index()
    annual_df = annual_df.rename(columns={"date": "year", "index": "year"})
    annual_df["annual_return_pct"] = annual_df["annual_return_pct"].round(4)
    return annual_df


def _build_variant_summary(
    *,
    top_n: int,
    equity_curve_df: pd.DataFrame,
    rebalance_df: pd.DataFrame,
    config: WideMomentumBaselineConfig,
    prepared: PreparedWideMomentumUniverse,
) -> dict[str, Any]:
    equity = equity_curve_df["equity"] if not equity_curve_df.empty else pd.Series(dtype=float)
    returns = equity.pct_change().dropna()

    cumulative = cumulative_return(returns) if not returns.empty else float("nan")
    annualised = annualised_return(returns) if not returns.empty else float("nan")
    volatility = annualised_volatility(returns) if len(returns) >= 2 else float("nan")
    sharpe = sharpe_ratio(returns, risk_free_rate=float(config.risk_free_rate)) if len(returns) >= 2 else None
    drawdown_details = _compute_drawdown_details(equity)
    max_drawdown = drawdown_details["max_drawdown"]
    calmar = None
    if not math.isnan(annualised) and max_drawdown not in (0.0, float("nan")):
        if max_drawdown > 0:
            calmar = annualised / max_drawdown

    period_returns = rebalance_df["period_return"].dropna() if "period_return" in rebalance_df.columns else pd.Series(dtype=float)
    turnover_series = rebalance_df["turnover"].dropna() if "turnover" in rebalance_df.columns else pd.Series(dtype=float)
    stable_start = prepared.stable_start_month

    return {
        "top_n": int(top_n),
        "date_range_start": str(prepared.start_date.date()),
        "date_range_end": str(prepared.end_date.date()),
        "recent_complete_date": str(prepared.recent_complete_date.date()),
        "eligible_symbol_count": int(len(prepared.symbol_data_map)),
        "pool_stable_start_month": str(stable_start.date()) if stable_start is not None else None,
        "risk_free_rate_pct": round(float(config.risk_free_rate) * 100.0, 4),
        "commission_pct": round(float(config.commission) * 100.0, 4),
        "rebalance_interval_days": int(config.rebalance_interval),
        "cumulative_return_pct": round(float(cumulative) * 100.0, 4) if not math.isnan(cumulative) else None,
        "annualised_return_pct": round(float(annualised) * 100.0, 4) if not math.isnan(annualised) else None,
        "annualised_volatility_pct": round(float(volatility) * 100.0, 4) if not math.isnan(volatility) else None,
        "sharpe": round(float(sharpe), 4) if sharpe is not None else None,
        "max_drawdown_pct": round(float(max_drawdown) * 100.0, 4) if not math.isnan(max_drawdown) else None,
        "max_drawdown_peak_date": (
            str(drawdown_details["peak_date"].date())
            if drawdown_details["peak_date"] is not None
            else None
        ),
        "max_drawdown_trough_date": (
            str(drawdown_details["trough_date"].date())
            if drawdown_details["trough_date"] is not None
            else None
        ),
        "calmar": round(float(calmar), 4) if calmar is not None else None,
        "rebalance_win_rate_pct": (
            round(float((period_returns > 0.0).mean()) * 100.0, 4)
            if not period_returns.empty
            else None
        ),
        "monthly_turnover_pct": (
            round(float(turnover_series.mean()) * 100.0, 4)
            if not turnover_series.empty
            else None
        ),
        "rebalance_count": int(len(rebalance_df)),
        "completed_period_count": int(len(period_returns)),
    }


def _run_top_n_variant(
    *,
    prepared: PreparedWideMomentumUniverse,
    config: WideMomentumBaselineConfig,
    top_n: int,
) -> WideMomentumVariantResult:
    positions: dict[str, float] = {}
    cash = float(config.cash)
    rebalance_entries: list[dict[str, Any]] = []
    pending_rebalance: Optional[dict[str, Any]] = None
    equity_rows: list[dict[str, Any]] = []

    for bar_index, current_date in enumerate(prepared.calendar):
        current_date = pd.Timestamp(current_date)

        if pending_rebalance is not None and current_date == pending_rebalance["execution_date"]:
            positions, cash, execution_info = _execute_rebalance(
                positions=positions,
                cash=cash,
                execution_date=current_date,
                target_weights=pending_rebalance["target_weights"],
                symbol_data_map=prepared.symbol_data_map,
                commission=float(config.commission),
            )
            rebalance_entries[pending_rebalance["entry_index"]].update(execution_info)
            pending_rebalance = None

        close_value = _portfolio_value(
            positions=positions,
            cash=cash,
            date=current_date,
            symbol_data_map=prepared.symbol_data_map,
            price_column="close",
        )
        equity_rows.append(
            {
                "date": current_date,
                "equity": float(close_value),
                "cash": float(cash),
            }
        )

        is_signal_bar = (bar_index % int(config.rebalance_interval) == 0) and (bar_index + 1 < len(prepared.calendar))
        if not is_signal_bar:
            continue

        execution_date = pd.Timestamp(prepared.calendar[bar_index + 1])
        target_weights, candidate_count = _select_target_weights(
            symbol_data_map=prepared.symbol_data_map,
            signal_date=current_date,
            execution_date=execution_date,
            top_n=top_n,
        )
        current_weights = _portfolio_weights(
            positions=positions,
            cash=cash,
            date=current_date,
            symbol_data_map=prepared.symbol_data_map,
            price_column="close",
        )
        rebalance_entries.append(
            {
                "signal_date": current_date,
                "execution_date": execution_date,
                "bar_index": int(bar_index),
                "candidate_count": int(candidate_count),
                "selected_symbols": sorted(target_weights.keys()),
                "current_weights": current_weights,
                "target_weights": target_weights,
                "signal_equity": float(close_value),
            }
        )
        pending_rebalance = {
            "entry_index": len(rebalance_entries) - 1,
            "execution_date": execution_date,
            "target_weights": target_weights,
        }

    equity_curve_df = pd.DataFrame(equity_rows).set_index("date")
    equity_curve_df.index = pd.to_datetime(equity_curve_df.index)
    equity_curve_df["daily_return"] = equity_curve_df["equity"].pct_change()
    equity_curve_df["cumulative_return_pct"] = (
        equity_curve_df["equity"] / float(config.cash) - 1.0
    ) * 100.0

    rebalance_df = _finalize_rebalance_log(rebalance_entries)
    annual_returns = _build_annual_returns(equity_curve_df)
    summary = _build_variant_summary(
        top_n=top_n,
        equity_curve_df=equity_curve_df,
        rebalance_df=rebalance_df,
        config=config,
        prepared=prepared,
    )
    return WideMomentumVariantResult(
        top_n=top_n,
        summary=summary,
        equity_curve=equity_curve_df,
        annual_returns=annual_returns,
        rebalance_log=rebalance_df,
    )


def run_wide_momentum_baseline_from_prepared(
    prepared: PreparedWideMomentumUniverse,
    config: WideMomentumBaselineConfig,
) -> WideMomentumBaselineResult:
    variant_results = {
        int(top_n): _run_top_n_variant(prepared=prepared, config=config, top_n=int(top_n))
        for top_n in config.top_n_values
    }
    return WideMomentumBaselineResult(
        config=config,
        prepared_universe=prepared,
        variant_results=variant_results,
    )


def run_wide_momentum_baseline(
    config: WideMomentumBaselineConfig,
    *,
    symbols: Optional[list[str]] = None,
    cluster_lookup: Optional[Mapping[str, int] | Callable[[str], int]] = None,
) -> WideMomentumBaselineResult:
    prepared = prepare_wide_momentum_universe(
        config=config,
        symbols=symbols,
        cluster_lookup=cluster_lookup,
    )
    return run_wide_momentum_baseline_from_prepared(prepared=prepared, config=config)


def _prepare_rebalance_log_for_csv(rebalance_df: pd.DataFrame) -> pd.DataFrame:
    if rebalance_df.empty:
        return rebalance_df.copy()

    csv_df = rebalance_df.copy()
    for column in ("selected_symbols", "current_weights", "target_weights", "executed_weights_open"):
        if column in csv_df.columns:
            csv_df[column] = csv_df[column].apply(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                if isinstance(value, (dict, list))
                else value
            )
    return csv_df


def _plot_equity_curve(equity_curve_df: pd.DataFrame, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = equity_curve_df.reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["date"], plot_df["cumulative_return_pct"], linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_wide_momentum_baseline_result(
    result: WideMomentumBaselineResult,
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        variant_result.summary
        for _, variant_result in sorted(result.variant_results.items(), key=lambda item: item[0])
    ]
    pd.DataFrame(summary_rows).to_csv(
        output_path / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result.prepared_universe.monthly_pool_diagnostics.to_csv(
        output_path / "monthly_pool_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "config": {
            "top_n_values": [int(value) for value in result.config.top_n_values],
            "min_listing_days": int(result.config.min_listing_days),
            "momentum_window": int(result.config.momentum_window),
            "momentum_skip_recent": int(result.config.momentum_skip_recent),
            "rebalance_interval": int(result.config.rebalance_interval),
            "cash": float(result.config.cash),
            "commission": float(result.config.commission),
            "risk_free_rate": float(result.config.risk_free_rate),
            "stable_pool_size": int(result.config.stable_pool_size),
            "start_date": str(result.config.start_date) if result.config.start_date is not None else None,
            "end_date": str(result.config.end_date) if result.config.end_date is not None else None,
        },
        "prepared_universe": {
            "source_symbol_count": int(result.prepared_universe.source_symbol_count),
            "eligible_symbol_count": int(len(result.prepared_universe.symbol_data_map)),
            "load_error_count": int(len(result.prepared_universe.load_errors)),
            "excluded_symbol_count": int(len(result.prepared_universe.excluded_symbols)),
            "start_date": str(result.prepared_universe.start_date.date()),
            "end_date": str(result.prepared_universe.end_date.date()),
            "recent_complete_date": str(result.prepared_universe.recent_complete_date.date()),
            "stable_start_month": (
                str(result.prepared_universe.stable_start_month.date())
                if result.prepared_universe.stable_start_month is not None
                else None
            ),
        },
        "load_errors": result.prepared_universe.load_errors,
        "excluded_symbols": result.prepared_universe.excluded_symbols,
    }
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    for top_n, variant_result in sorted(result.variant_results.items(), key=lambda item: item[0]):
        variant_dir = output_path / f"top_{top_n}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_result.equity_curve.to_csv(
            variant_dir / "equity_curve.csv",
            index=True,
            encoding="utf-8-sig",
        )
        variant_result.annual_returns.to_csv(
            variant_dir / "annual_returns.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _prepare_rebalance_log_for_csv(variant_result.rebalance_log).to_csv(
            variant_dir / "rebalance_log.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (variant_dir / "summary.json").write_text(
            json.dumps(variant_result.summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        _plot_equity_curve(
            equity_curve_df=variant_result.equity_curve,
            output_path=variant_dir / "equity_curve.png",
            title=f"Wide Momentum Baseline Top {top_n}",
        )

    return output_path


__all__ = [
    "PreparedWideMomentumUniverse",
    "SymbolBaselineData",
    "WideMomentumBaselineConfig",
    "WideMomentumBaselineResult",
    "WideMomentumVariantResult",
    "prepare_wide_momentum_universe",
    "prepare_wide_momentum_universe_from_frames",
    "run_wide_momentum_baseline",
    "run_wide_momentum_baseline_from_prepared",
    "save_wide_momentum_baseline_result",
]