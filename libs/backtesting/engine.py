from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import backtrader as bt

from factors.base_factor import BaseFactor
from .data import build_bt_feed_dataframe
from .single_factor_single_target_strategy import (
    SingleFactorSingleTargetDataFeed,
    SingleFactorSingleTargetStrategy,
)


@dataclass(slots=True)
class SingleFactorSingleTargetBacktestConfig:
    """Single-instrument backtest configuration.

    `factor` is the strategy signal source and should return a Series aligned
    with the ETF price index, typically with buy/sell values such as 1/-1.
    """

    symbol: str
    factor: BaseFactor
    cash: float = 100000.0
    commission: float = 0.0005
    stake: int = 100
    data_dir: Optional[str | Path] = None
    buy_signal: float = 1.0
    sell_signal: float = -1.0


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


@dataclass(slots=True)
class SingleFactorSingleTargetBatchBacktestResult:
    total_symbols: int
    success_count: int
    failure_count: int
    results: list[SingleFactorSingleTargetBacktestResult]
    errors: list[dict[str, str]]


def _safe_extract_sharpe(analysis: dict[str, Any]) -> Optional[float]:
    value = analysis.get("sharperatio") if analysis else None
    if value is None:
        return None
    return float(value)


def run_single_factor_single_target_backtest(
    config: SingleFactorSingleTargetBacktestConfig,
) -> SingleFactorSingleTargetBacktestResult:
    # 1) Load CSV and compute factor signal into a Backtrader-compatible DataFrame.
    feed_df = build_bt_feed_dataframe(
        symbol=config.symbol,
        factor=config.factor,
        data_dir=config.data_dir,
    )

    cerebro = bt.Cerebro()
    data_feed = SingleFactorSingleTargetDataFeed(dataname=feed_df)  # type: ignore[call-arg]
    cerebro.adddata(data_feed, name=config.symbol)
    # 2) Strategy only consumes `signal` and converts it to orders.
    cerebro.addstrategy(
        SingleFactorSingleTargetStrategy,
        stake=config.stake,
        buy_signal=config.buy_signal,
        sell_signal=config.sell_signal,
    )

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

    start_value = float(cerebro.broker.getvalue())
    result = cerebro.run()[0]
    end_value = float(cerebro.broker.getvalue())
    pnl = end_value - start_value
    pnl_pct = pnl / start_value if start_value else 0.0

    sharpe_analysis = result.analyzers.sharpe.get_analysis()
    drawdown_analysis = result.analyzers.drawdown.get_analysis()
    trade_analysis = result.analyzers.trades.get_analysis()

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
        },
    )


def _discover_etf_symbols(data_dir: Optional[str | Path]) -> list[str]:
    if data_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / "data" / "etf_data"
    else:
        candidate = Path(data_dir)

    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"ETF data directory not found: {candidate}")

    return sorted(path.stem for path in candidate.glob("*.csv"))


def run_single_factor_single_target_backtest_all_etfs(
    factor: BaseFactor,
    *,
    symbols: Optional[list[str]] = None,
    cash: float = 100000.0,
    commission: float = 0.0005,
    stake: int = 100,
    data_dir: Optional[str | Path] = None,
    buy_signal: float = 1.0,
    sell_signal: float = -1.0,
) -> SingleFactorSingleTargetBatchBacktestResult:
    """Run a single-factor strategy backtest for all ETF symbols.

    By default, symbols are auto-discovered from `data/etf_data/*.csv`.
    You can pass `symbols` to limit scope and/or `data_dir` to override source.
    """

    symbol_list = symbols if symbols is not None else _discover_etf_symbols(data_dir)
    results: list[SingleFactorSingleTargetBacktestResult] = []
    errors: list[dict[str, str]] = []

    for symbol in symbol_list:
        try:
            result = run_single_factor_single_target_backtest(
                SingleFactorSingleTargetBacktestConfig(
                    symbol=symbol,
                    factor=factor,
                    cash=cash,
                    commission=commission,
                    stake=stake,
                    data_dir=data_dir,
                    buy_signal=buy_signal,
                    sell_signal=sell_signal,
                )
            )
            results.append(result)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})

    return SingleFactorSingleTargetBatchBacktestResult(
        total_symbols=len(symbol_list),
        success_count=len(results),
        failure_count=len(errors),
        results=results,
        errors=errors,
    )


# Backward-compatible aliases for existing imports.
BacktestConfig = SingleFactorSingleTargetBacktestConfig
BacktestResult = SingleFactorSingleTargetBacktestResult
BatchBacktestResult = SingleFactorSingleTargetBatchBacktestResult
run_backtest = run_single_factor_single_target_backtest
run_backtest_all_etfs = run_single_factor_single_target_backtest_all_etfs
