from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import traceback

import pandas as pd

from core.models.etf_daily_data import EtfData
from data_manager.etf_data_manager import get_etf_data_by_symbol
from factors.base_factor import BaseFactor


@dataclass
class PreprocessError:
    symbol: str
    stage: str
    error: str
    retried_serial: bool = False


@dataclass
class LoadFilterResult:
    etf_data_map: dict[str, EtfData]
    filtered_symbols: list[str]
    errors: list[PreprocessError]


@dataclass
class FactorCalcResult:
    etf_data_map: dict[str, EtfData]
    errors: list[PreprocessError]


@dataclass
class DateRangeSliceResult:
    etf_data_map: dict[str, EtfData]
    filtered_symbols: list[str]
    dropped_symbols: list[str]


def _load_symbol_worker(symbol: str) -> tuple[str, Optional[EtfData], Optional[str]]:
    try:
        etf_data = get_etf_data_by_symbol(symbol)
        return symbol, etf_data, None
    except Exception:
        return symbol, None, traceback.format_exc(limit=8)


def _calc_factors_worker(
    symbol: str,
    factor_pipeline: list[BaseFactor],
) -> tuple[str, Optional[dict[str, pd.Series]], Optional[str]]:
    try:
        etf_data = get_etf_data_by_symbol(symbol)
        etf_data.factors.clear()
        etf_data.factor_results.clear()

        for factor in factor_pipeline:
            etf_data.add_factors(factor)

        etf_data.calc_factors()

        # Keep transport payload serializable and detached from factor object identity.
        by_name = {
            factor.get_output_name(): series.copy()
            for factor, series in etf_data.factor_results.items()
        }
        return symbol, by_name, None
    except Exception:
        return symbol, None, traceback.format_exc(limit=8)


def _collect_factor_objects(factor_pipeline: list[BaseFactor]) -> dict[str, BaseFactor]:
    seen_factors: set[BaseFactor] = set()
    factor_by_output_name: dict[str, BaseFactor] = {}

    def visit(f: BaseFactor) -> None:
        if f in seen_factors:
            return
        seen_factors.add(f)
        for dep in f.dependencies:
            visit(dep)
        output_name = f.get_output_name()
        existing = factor_by_output_name.get(output_name)
        if existing is not None and existing != f:
            raise ValueError(
                f"Duplicate factor output name {output_name!r} for {existing!r} and {f!r}"
            )
        factor_by_output_name[output_name] = f

    for f in factor_pipeline:
        visit(f)

    return factor_by_output_name


def parallel_load_filter_etf_data(
    candidate_symbols: list[str],
    filter_fn: Callable[[EtfData], bool],
    max_workers: int,
    retry_serial_on_fail: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> LoadFilterResult:
    etf_data_map: dict[str, EtfData] = {}
    filtered_symbols: list[str] = []
    errors: list[PreprocessError] = []

    total = len(candidate_symbols)
    done = 0
    load_failed_symbols: list[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_load_symbol_worker, symbol): symbol
            for symbol in candidate_symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            done += 1
            try:
                sym, etf_data, err = future.result()
                if err or etf_data is None:
                    load_failed_symbols.append(sym)
                    errors.append(
                        PreprocessError(
                            symbol=sym,
                            stage="load",
                            error=err or "unknown load error",
                            retried_serial=False,
                        )
                    )
                else:
                    etf_data_map[sym] = etf_data
                    try:
                        if filter_fn(etf_data):
                            filtered_symbols.append(sym)
                    except Exception as filter_err:
                        errors.append(
                            PreprocessError(
                                symbol=sym,
                                stage="filter",
                                error=str(filter_err),
                                retried_serial=False,
                            )
                        )
            except Exception as exec_err:
                load_failed_symbols.append(symbol)
                errors.append(
                    PreprocessError(
                        symbol=symbol,
                        stage="load",
                        error=str(exec_err),
                        retried_serial=False,
                    )
                )

            if progress_callback is not None:
                progress_callback(done, total, symbol)

    if retry_serial_on_fail and load_failed_symbols:
        unresolved = set(load_failed_symbols)
        for symbol in load_failed_symbols:
            try:
                etf_data = get_etf_data_by_symbol(symbol)
                etf_data_map[symbol] = etf_data
                if filter_fn(etf_data):
                    filtered_symbols.append(symbol)
                unresolved.discard(symbol)
            except Exception:
                pass

        if unresolved:
            for e in errors:
                if e.symbol in unresolved and e.stage == "load":
                    e.retried_serial = True
        else:
            errors = [
                e
                for e in errors
                if not (e.symbol in set(load_failed_symbols) and e.stage == "load")
            ]

    return LoadFilterResult(
        etf_data_map=etf_data_map,
        filtered_symbols=filtered_symbols,
        errors=errors,
    )


def parallel_calc_factors_for_map(
    etf_data_map: dict[str, EtfData],
    factor_pipeline: list[BaseFactor],
    symbols: Optional[list[str]] = None,
    max_workers: int = 4,
    retry_serial_on_fail: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> FactorCalcResult:
    target_symbols = symbols or list(etf_data_map.keys())
    factor_by_output_name = _collect_factor_objects(factor_pipeline)
    errors: list[PreprocessError] = []

    total = len(target_symbols)
    done = 0
    failed_symbols: list[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_calc_factors_worker, symbol, factor_pipeline): symbol
            for symbol in target_symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            done += 1
            try:
                sym, by_name, err = future.result()
                if err or by_name is None:
                    failed_symbols.append(sym)
                    errors.append(
                        PreprocessError(
                            symbol=sym,
                            stage="factor_calc",
                            error=err or "unknown factor calc error",
                            retried_serial=False,
                        )
                    )
                else:
                    etf_data = etf_data_map.get(sym)
                    if etf_data is None:
                        errors.append(
                            PreprocessError(
                                symbol=sym,
                                stage="factor_apply",
                                error="symbol not found in etf_data_map",
                                retried_serial=False,
                            )
                        )
                    else:
                        etf_data.factors.clear()
                        etf_data.factor_results.clear()
                        for f in factor_pipeline:
                            etf_data.add_factors(f)
                        missing_names = [
                            name for name in factor_by_output_name.keys() if name not in by_name
                        ]
                        unexpected_names = [
                            name for name in by_name.keys() if name not in factor_by_output_name
                        ]
                        if missing_names or unexpected_names:
                            details: list[str] = []
                            if missing_names:
                                details.append(f"missing outputs: {missing_names}")
                            if unexpected_names:
                                details.append(f"unexpected outputs: {unexpected_names}")
                            errors.append(
                                PreprocessError(
                                    symbol=sym,
                                    stage="factor_apply",
                                    error="; ".join(details),
                                    retried_serial=False,
                                )
                            )
                        else:
                            for name, factor_obj in factor_by_output_name.items():
                                etf_data.factor_results[factor_obj] = by_name[name]
            except Exception as exec_err:
                failed_symbols.append(symbol)
                errors.append(
                    PreprocessError(
                        symbol=symbol,
                        stage="factor_calc",
                        error=str(exec_err),
                        retried_serial=False,
                    )
                )

            if progress_callback is not None:
                progress_callback(done, total, symbol)

    if retry_serial_on_fail and failed_symbols:
        unresolved = set(failed_symbols)
        for symbol in failed_symbols:
            try:
                etf_data = etf_data_map[symbol]
                etf_data.factors.clear()
                etf_data.factor_results.clear()
                for factor in factor_pipeline:
                    etf_data.add_factors(factor)
                etf_data.calc_factors()
                unresolved.discard(symbol)
            except Exception:
                pass

        if unresolved:
            for e in errors:
                if e.symbol in unresolved and e.stage == "factor_calc":
                    e.retried_serial = True
        else:
            errors = [
                e
                for e in errors
                if not (e.symbol in set(failed_symbols) and e.stage == "factor_calc")
            ]

    return FactorCalcResult(etf_data_map=etf_data_map, errors=errors)


def slice_etf_data_map_by_date_range(
    etf_data_map: dict[str, EtfData],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbols: Optional[list[str]] = None,
    drop_empty: bool = True,
) -> DateRangeSliceResult:
    target_symbols = symbols or list(etf_data_map.keys())

    if start_date and end_date:
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        if start_ts > end_ts:
            raise ValueError("start_date must be earlier than or equal to end_date")

    sliced_map: dict[str, EtfData] = {}
    filtered_symbols: list[str] = []
    dropped_symbols: list[str] = []

    for symbol in target_symbols:
        etf_data = etf_data_map.get(symbol)
        if etf_data is None:
            dropped_symbols.append(symbol)
            continue

        sliced_etf_data = etf_data.slice_date_range(
            start_date=start_date,
            end_date=end_date,
        )
        has_rows = sliced_etf_data.data is not None and len(sliced_etf_data.data) > 0
        if drop_empty and not has_rows:
            dropped_symbols.append(symbol)
            continue

        sliced_map[symbol] = sliced_etf_data
        filtered_symbols.append(symbol)

    return DateRangeSliceResult(
        etf_data_map=sliced_map,
        filtered_symbols=filtered_symbols,
        dropped_symbols=dropped_symbols,
    )


def ensure_output_compatibility(
    etf_data_map: dict[str, EtfData],
    filtered_symbols: list[str],
) -> None:
    missing = [sym for sym in filtered_symbols if sym not in etf_data_map]
    if missing:
        raise ValueError(
            f"Found {len(missing)} symbols missing from etf_data_map: {missing[:10]}"
        )


__all__ = [
    "PreprocessError",
    "LoadFilterResult",
    "FactorCalcResult",
    "parallel_load_filter_etf_data",
    "parallel_calc_factors_for_map",
    "ensure_output_compatibility",
]
