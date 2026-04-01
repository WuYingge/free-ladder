from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, cast

import backtrader as bt

from .custom_strategy_example import ExampleCustomTimingStrategy
from .data import build_bt_feed_dataframe


@dataclass(slots=True)
class SingleFactorSingleTargetBacktestConfig:
    """Single-instrument backtest configuration.

    The engine only loads data and runs strategy execution.
    Any factor combination and signal logic should be implemented in
    ``strategy_cls.next()`` based on feed lines.
    """

    symbol: str
    cash: float = 100000.0
    commission: float = 0.0005
    stake: int = 100
    data_dir: Optional[str | Path] = None
    strategy_cls: type[bt.Strategy] = ExampleCustomTimingStrategy
    # If None, engine creates a PandasData subclass exposing all extra columns.
    data_feed_cls: Optional[type[bt.feeds.PandasData]] = None
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SingleFactorSingleTargetBacktestResult:
    symbol: str
    # 初始值
    start_value: float
    # 结束值
    end_value: float
    # 盈亏金额
    pnl: float
    # 盈亏百分比
    pnl_pct: float
    # 夏普比率，越高越好，通常大于1.0被认为是不错的策略
    sharpe: Optional[float]
    # 最大回撤百分比，越低越好，通常小于20%被认为是不错的策略
    max_drawdown_pct: Optional[float]
    trades_total: int
    trades_won: int
    trades_lost: int
    analyzer_raw: dict[str, Any]

def _safe_extract_sharpe(analysis: dict[str, Any]) -> Optional[float]:
    value = analysis.get("sharperatio") if analysis else None
    if value is None:
        return None
    return float(value)


def _extract_strategy_param_names(strategy_cls: type[bt.Strategy]) -> set[str]:
    params = getattr(strategy_cls, "params", None)
    if params is None:
        return set()

    get_keys = getattr(params, "_getkeys", None)
    if callable(get_keys):
        try:
            keys = cast(Iterable[Any], get_keys())
            return {str(key) for key in keys}
        except TypeError:
            return set()

    try:
        return {str(name) for name, _ in params}
    except TypeError:
        return set()


def _extract_datafeed_param_names(data_feed_cls: type[bt.feeds.PandasData]) -> set[str]:
    params = getattr(data_feed_cls, "params", None)
    if params is None:
        return set()

    get_keys = getattr(params, "_getkeys", None)
    if callable(get_keys):
        try:
            keys = cast(Iterable[Any], get_keys())
            return {str(key) for key in keys}
        except TypeError:
            return set()

    try:
        return {str(name) for name, _ in params}
    except TypeError:
        return set()


def _build_auto_data_feed_cls(feed_df_columns: Iterable[str]) -> type[bt.feeds.PandasData]:
    reserved_columns = {"open", "high", "low", "close", "volume", "openinterest"}
    extra_columns = tuple(column for column in feed_df_columns if column not in reserved_columns)
    params = tuple((column, column) for column in extra_columns)

    return cast(
        type[bt.feeds.PandasData],
        type(
            "AutoFactorPandasDataFeed",
            (bt.feeds.PandasData,),
            {
                "lines": extra_columns,
                "params": params,
            },
        ),
    )


def _build_strategy_kwargs(
    config: SingleFactorSingleTargetBacktestConfig,
) -> dict[str, Any]:
    strategy_kwargs = dict(config.strategy_kwargs)
    accepted_params = _extract_strategy_param_names(config.strategy_cls)
    default_strategy_kwargs = {
        "stake": config.stake,
    }

    for key, value in default_strategy_kwargs.items():
        if key in accepted_params and key not in strategy_kwargs:
            strategy_kwargs[key] = value

    return strategy_kwargs


def run_single_factor_single_target_backtest(
    config: SingleFactorSingleTargetBacktestConfig,
) -> SingleFactorSingleTargetBacktestResult:
    # 1) Load CSV into a Backtrader-compatible DataFrame.
    feed_df = build_bt_feed_dataframe(
        symbol=config.symbol,
        data_dir=config.data_dir,
    )

    cerebro = bt.Cerebro()
    data_feed_cls = config.data_feed_cls or _build_auto_data_feed_cls(feed_df.columns)
    data_feed_param_names = _extract_datafeed_param_names(data_feed_cls)
    data_feed_kwargs: dict[str, Any] = {}
    # Keep signal mapping compatibility when feed exposes a signal param.
    if "signal" in data_feed_param_names and "signal" in feed_df.columns:
        data_feed_kwargs["signal"] = "signal"

    data_feed = data_feed_cls(dataname=feed_df, **data_feed_kwargs)  # type: ignore[call-arg]
    cerebro.adddata(data_feed, name=config.symbol)
    # 2) Strategy consumes feed lines and converts them into orders.
    cerebro.addstrategy(config.strategy_cls, **_build_strategy_kwargs(config))

    cerebro.broker.setcash(config.cash)
    cerebro.broker.setcommission(commission=config.commission)

    # 3) Keep analyzer set small and stable for daily tuning workflow.
    timeframe = getattr(bt.TimeFrame, "Days", None)
    if timeframe is not None:
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=timeframe)
    else:
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    # TimeReturn provides daily portfolio value changes needed for IR / excess return.
    if timeframe is not None:
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return", timeframe=timeframe)
    else:
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

    start_value = float(cerebro.broker.getvalue())
    result = cerebro.run()[0]
    end_value = float(cerebro.broker.getvalue())
    pnl = end_value - start_value
    pnl_pct = pnl / start_value if start_value else 0.0

    sharpe_analysis = result.analyzers.sharpe.get_analysis()
    drawdown_analysis = result.analyzers.drawdown.get_analysis()
    trade_analysis = result.analyzers.trades.get_analysis()
    time_return_analysis = result.analyzers.time_return.get_analysis()

    max_dd = drawdown_analysis.get("max", {}).get("drawdown")
    trades_total = int(trade_analysis.get("total", {}).get("total", 0) or 0)
    trades_won = int(trade_analysis.get("won", {}).get("total", 0) or 0)
    trades_lost = int(trade_analysis.get("lost", {}).get("total", 0) or 0)

    # 4) Return both flat metrics and raw analyzer payload for later drill-down.
    return SingleFactorSingleTargetBacktestResult(
        symbol=config.symbol,
        start_value=start_value,
        end_value=end_value,
        pnl=pnl,
        pnl_pct=pnl_pct,
        sharpe=_safe_extract_sharpe(sharpe_analysis),
        max_drawdown_pct=float(max_dd) if max_dd is not None else None,
        trades_total=trades_total,
        trades_won=trades_won,
        trades_lost=trades_lost,
        analyzer_raw={
            "sharpe": sharpe_analysis,
            "drawdown": drawdown_analysis,
            "trades": trade_analysis,
            # keyed by datetime.date → daily return fraction (e.g. 0.01 = +1%)
            "time_return": {str(k): v for k, v in time_return_analysis.items()},
        },
    )
