"""
backtesting.timing_batch
========================
Batch orchestration for single-symbol timing strategy backtests.

Design principles
-----------------
* Filter logic runs in the **main process** (notebook-defined functions are not
  picklable and cannot be sent to worker processes).
* Only fully-serialisable config and strategy objects are sent to worker
    processes.
* Each worker task is fully independent: one symbol → one Cerebro run.

Typical notebook usage
----------------------
::

    from backtesting.timing_batch import run_timing_backtest_batch, TimingBatchConfig
    from backtesting.strategies import ExampleCustomTimingStrategy

    batch_cfg = TimingBatchConfig(
        cash=100_000,
        commission=0.0005,
        max_workers=8,
        strategy_cls=ExampleCustomTimingStrategy,
        output_dir="data/backtest_results/timing_run_01",
    )

    results_df, errors, equity_curves = run_timing_backtest_batch(
        filtered_symbols=["159915", "159001"],
        etf_data_map=etf_data_map,         # dict[symbol, EtfData]
        config=batch_cfg,
    )
"""
from __future__ import annotations

import json
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from backtesting.engine import (
    SingleFactorSingleTargetBacktestConfig,
    SingleFactorSingleTargetBacktestResult,
    run_single_factor_single_target_backtest,
)
from backtesting.performance import (
    build_equity_curves,
    compute_performance_metrics,
)
from backtesting.strategies import ExampleCustomTimingStrategy
from backtesting.data import build_bt_feed_dataframe_from_etf_data, load_etf_dataframe
from core.models.etf_daily_data import EtfData


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TimingBatchConfig:
    """Configuration shared across all single-symbol timing backtests in a batch."""

    cash: float = 100_000.0
    commission: float = 0.0005
    stake: int = 100
    # Parallel worker count.  None → os.cpu_count()
    max_workers: int = 4
    # Where to write summary.csv, details.json, run_metadata.json.
    # If None, results are returned in-memory only.
    output_dir: Optional[str | Path] = None
    # Optional path override for ETF CSV data (forwarded to engine).
    data_dir: Optional[str | Path] = None
    strategy_cls: type = ExampleCustomTimingStrategy
    data_feed_cls: Optional[type] = None
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)
    # Prefer providing factors from notebook/preprocessing pipeline so strategy
    # logic remains stable while factor params change frequently.
    analysis_factors: tuple[Any, ...] = field(default_factory=tuple)
    # Explicit warm-up bars override. If None, infer from analysis_factors,
    # then fall back to strategy_cls.get_max_warmup_bars().
    analysis_warmup_bars: Optional[int] = None


# ---------------------------------------------------------------------------
# Per-symbol result container (enriched vs engine's raw result)
# ---------------------------------------------------------------------------

@dataclass
class TimingSymbolResult:
    """Full result for one symbol: engine output + computed performance metrics."""

    symbol: str
    name: str
    engine_result: Optional[SingleFactorSingleTargetBacktestResult]
    metrics: dict
    equity_curve: pd.DataFrame
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Top-level worker function (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _backtest_worker(
    symbol: str,
    cash: float,
    commission: float,
    stake: int,
    data_dir: Optional[str | Path],
    feed_df: Optional[pd.DataFrame],
    strategy_cls: type,
    data_feed_cls: Optional[type],
    strategy_kwargs: dict[str, Any],
) -> tuple[str, Optional[SingleFactorSingleTargetBacktestResult], Optional[str]]:
    """Run one single-symbol backtest; returns (symbol, result, error_str)."""
    try:
        result = run_single_factor_single_target_backtest(
            SingleFactorSingleTargetBacktestConfig(
                symbol=symbol,
                cash=cash,
                commission=commission,
                stake=stake,
                data_dir=data_dir,
                feed_df=feed_df,
                strategy_cls=strategy_cls,
                data_feed_cls=data_feed_cls,
                strategy_kwargs=strategy_kwargs,
            )
        )
        return symbol, result, None
    except Exception:
        return symbol, None, traceback.format_exc()


def _validate_parallel_class(class_obj: type, label: str, workers: int) -> None:
    module_name = getattr(class_obj, "__module__", "")
    qualname = getattr(class_obj, "__qualname__", getattr(class_obj, "__name__", repr(class_obj)))

    if module_name == "__main__":
        raise ValueError(
            f"{label} {qualname} is defined in __main__. Parallel batch backtests "
            f"require importable classes from libs modules. Move it into libs and retry "
            f"with max_workers={workers}."
        )

    try:
        pickle.dumps(class_obj)
    except Exception as exc:
        raise ValueError(
            f"{label} {module_name}.{qualname} cannot be pickled for parallel batch backtests "
            f"(max_workers={workers}). Move it into an importable libs module. Original error: {exc}"
        ) from exc


def _validate_parallel_payload(payload: Any, label: str, workers: int) -> None:
    try:
        pickle.dumps(payload)
    except Exception as exc:
        raise ValueError(
            f"{label} must be picklable for parallel batch backtests (max_workers={workers}). "
            f"Original error: {exc}"
        ) from exc


def _validate_parallel_config(config: TimingBatchConfig, workers: int) -> None:
    _validate_parallel_class(config.strategy_cls, "strategy_cls", workers)
    if config.data_feed_cls is not None:
        _validate_parallel_class(config.data_feed_cls, "data_feed_cls", workers)
    _validate_parallel_payload(config.strategy_kwargs, "strategy_kwargs", workers)


def _max_warmup_from_factors(factors: tuple[Any, ...]) -> Optional[int]:
    """Return max warm-up bars from factor instances exposing get_max_warmup_period()."""
    if not factors:
        return None

    max_warmup = 0
    seen_valid_factor = False
    for factor in factors:
        getter = getattr(factor, "get_max_warmup_period", None)
        if not callable(getter):
            continue
        seen_valid_factor = True
        warmup_raw: Any = getter()
        warmup = int(warmup_raw)
        if warmup < 0:
            raise ValueError(f"Invalid factor warm-up bars: {warmup}. Expected >= 0.")
        if warmup > max_warmup:
            max_warmup = warmup

    if not seen_valid_factor:
        return None
    return max_warmup


def _resolve_analysis_warmup_bars(config: TimingBatchConfig) -> int:
    """Resolve warm-up bars with priority: explicit > factors > strategy fallback."""
    if config.analysis_warmup_bars is not None:
        warmup = int(config.analysis_warmup_bars)
        if warmup < 0:
            raise ValueError("analysis_warmup_bars must be a non-negative integer.")
        return warmup

    warmup_from_factors = _max_warmup_from_factors(config.analysis_factors)
    if warmup_from_factors is not None:
        return warmup_from_factors

    get_max_warmup_bars = getattr(config.strategy_cls, "get_max_warmup_bars", None)
    if callable(get_max_warmup_bars):
        warmup_raw: Any = get_max_warmup_bars()
        warmup = int(warmup_raw)
        if warmup < 0:
            raise ValueError("strategy warm-up bars must be a non-negative integer.")
        return warmup

    return 0


def _analysis_start_date_from_warmup_bars(
    warmup_bars: int,
    price_df: pd.DataFrame,
) -> Optional[pd.Timestamp]:
    """Convert resolved warm-up bars into analysis start date for one symbol."""
    if len(price_df) == 0 or warmup_bars <= 0:
        return None
    if len(price_df.index) <= warmup_bars:
        return None
    return pd.to_datetime(price_df.index[warmup_bars])


# ---------------------------------------------------------------------------
# Public orchestration entry point
# ---------------------------------------------------------------------------

def run_timing_backtest_batch(
    filtered_symbols: list[str],
    etf_data_map: dict[str, EtfData],
    config: TimingBatchConfig,
    *,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[pd.DataFrame, list[dict], dict[str, pd.DataFrame]]:
    """Run single-symbol timing backtests for all *filtered_symbols* in parallel.

    Parameters
    ----------
    filtered_symbols:
        Pre-filtered list of ETF symbols to backtest.  Filtering must happen
        in the calling (notebook) process before invoking this function.
    etf_data_map:
        Mapping from symbol → EtfData, used to compute benchmark returns and
        equity curves.  Must contain entries for every symbol in
        *filtered_symbols*.
    config:
        Shared ``TimingBatchConfig`` for this batch run.
    progress_callback:
        Optional ``(done_count, total_count, symbol)`` callable called after
        each symbol completes.  Useful to feed a tqdm bar in notebooks.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per symbol with all performance metrics.  Pre-sorted by
        ``sharpe`` descending (NaN last).
    errors : list[dict]
        ``[{"symbol": ..., "error": ...}, ...]`` for failed symbols.
    equity_curves : dict[str, pd.DataFrame]
        Mapping from symbol → equity curve DataFrame (``strategy``, ``benchmark``
        columns, datetime index).
    """
    total = len(filtered_symbols)
    raw_results: dict[str, Optional[SingleFactorSingleTargetBacktestResult]] = {}
    error_list: list[dict] = []

    # --- Parallel execution (workers receive only serialisable primitives) ---
    workers = min(config.max_workers, total) if total > 0 else 1
    if workers > 1:
        _validate_parallel_config(config, workers)

    analysis_warmup_bars = _resolve_analysis_warmup_bars(config)

    feed_df_by_symbol: dict[str, Optional[pd.DataFrame]] = {}
    for sym in filtered_symbols:
        etf = etf_data_map.get(sym)
        if etf is None:
            feed_df_by_symbol[sym] = None
            continue
        try:
            feed_df_by_symbol[sym] = build_bt_feed_dataframe_from_etf_data(etf)
        except Exception:
            # Fall back to worker-side disk loading for malformed in-memory payload.
            feed_df_by_symbol[sym] = None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_symbol = {}
        for sym in filtered_symbols:
            future = executor.submit(
                _backtest_worker,
                sym,
                config.cash,
                config.commission,
                config.stake,
                config.data_dir,
                feed_df_by_symbol.get(sym),
                config.strategy_cls,
                config.data_feed_cls,
                dict(config.strategy_kwargs),
            )
            future_to_symbol[future] = sym
        done = 0
        for future in as_completed(future_to_symbol):
            sym, result, err = future.result()
            done += 1
            if err:
                raw_results[sym] = None
                error_list.append({"symbol": sym, "error": err})
            else:
                raw_results[sym] = result
            if progress_callback:
                progress_callback(done, total, sym)

    # --- Post-process: compute metrics + equity curves in main process ---
    rows: list[dict] = []
    equity_curves: dict[str, pd.DataFrame] = {}

    for sym in filtered_symbols:
        etf = etf_data_map.get(sym)
        engine_res = raw_results.get(sym)
        etf_name = etf.name if etf is not None else ""

        if engine_res is None:
            # Engine failed for this symbol; emit a null-metrics row.
            err_msg = next(
                (e["error"] for e in error_list if e["symbol"] == sym), "unknown error"
            )
            rows.append({"symbol": sym, "name": etf_name, "error": err_msg[:200]})
            equity_curves[sym] = pd.DataFrame(columns=["strategy", "benchmark"])
            continue

        # Get normalised OHLCV DataFrame for benchmark computation.
        # Prefer in-memory etf_data_map; fall back to disk via data.load_etf_dataframe.
        try:
            if etf is not None and etf.data is not None and len(etf.data) > 0:
                price_df = etf.data.copy()
                if "date" in price_df.columns:
                    price_df["date"] = pd.to_datetime(price_df["date"])
                    price_df = price_df.set_index("date").sort_index()
            else:
                price_df = load_etf_dataframe(sym, data_dir=config.data_dir)
        except Exception as exc:
            error_list.append({"symbol": sym, "error": f"price_df load failed: {exc}"})
            rows.append({"symbol": sym, "name": etf_name, "error": str(exc)[:200]})
            equity_curves[sym] = pd.DataFrame(columns=["strategy", "benchmark"])
            continue

        analysis_start_date = _analysis_start_date_from_warmup_bars(
            warmup_bars=analysis_warmup_bars,
            price_df=price_df,
        )

        metrics = compute_performance_metrics(
            strategy_time_return=engine_res.analyzer_raw.get("time_return", {}),
            price_df=price_df,
            symbol=sym,
            name=etf_name,
            trades_total=engine_res.trades_total,
            trades_won=engine_res.trades_won,
            trades_lost=engine_res.trades_lost,
            max_drawdown_pct_from_engine=engine_res.max_drawdown_pct,
            analysis_start_date=analysis_start_date,
        )
        metrics["error"] = None
        rows.append(metrics)
        equity_curves[sym] = build_equity_curves(
            strategy_time_return=engine_res.analyzer_raw.get("time_return", {}),
            price_df=price_df,
            analysis_start_date=analysis_start_date,
        )

    summary_df = pd.DataFrame(rows)
    # Sort by sharpe descending (NaN last).
    if "sharpe" in summary_df.columns:
        summary_df = summary_df.sort_values("sharpe", ascending=False, na_position="last")
    summary_df = summary_df.reset_index(drop=True)

    # --- Persist results if output_dir is provided ---
    if config.output_dir is not None:
        _save_results(
            output_dir=Path(config.output_dir),
            summary_df=summary_df,
            error_list=error_list,
            equity_curves=equity_curves,
            config=config,
            total_symbols=total,
        )

    return summary_df, error_list, equity_curves


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_results(
    output_dir: Path,
    summary_df: pd.DataFrame,
    error_list: list[dict],
    equity_curves: dict[str, pd.DataFrame],
    config: TimingBatchConfig,
    total_symbols: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary table
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    # 2. Per-symbol details (metrics dict + abbreviated equity curve)
    details: list[dict] = []
    for _, row in summary_df.iterrows():
        sym = row.get("symbol", "")
        detail = row.to_dict()
        # Attach equity curve as list-of-records (trimmed to reduce size).
        ec = equity_curves.get(sym)
        if ec is not None and not ec.empty:
            detail["equity_curve"] = ec.reset_index().rename(
                columns={"index": "date"}
            ).to_dict(orient="records")
        else:
            detail["equity_curve"] = []
        details.append(detail)

    with open(output_dir / "details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2, default=str)

    # 3. Errors
    with open(output_dir / "errors.json", "w", encoding="utf-8") as f:
        json.dump(error_list, f, ensure_ascii=False, indent=2)

    # 4. Run metadata (reproducibility)
    import datetime

    metadata = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "total_symbols": total_symbols,
        "success_count": int(summary_df["error"].isna().sum()) if "error" in summary_df.columns else total_symbols,
        "failure_count": len(error_list),
        "config": {
            "cash": config.cash,
            "commission": config.commission,
            "stake": config.stake,
            "max_workers": config.max_workers,
            "data_dir": str(config.data_dir) if config.data_dir else None,
            "strategy_cls": f"{config.strategy_cls.__module__}.{config.strategy_cls.__qualname__}",
            "data_feed_cls": (
                f"{config.data_feed_cls.__module__}.{config.data_feed_cls.__qualname__}"
                if config.data_feed_cls is not None
                else None
            ),
            "strategy_kwargs": config.strategy_kwargs,
            "analysis_warmup_bars": _resolve_analysis_warmup_bars(config),
            "analysis_factors": [
                f"{factor.__class__.__module__}.{factor.__class__.__qualname__}"
                for factor in config.analysis_factors
            ],
        },
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
