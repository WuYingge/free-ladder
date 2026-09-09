"""
Alpha101 原始输入面板与 FactorPanel 构建 (Alpha101 Panel Builder)

面板化计算的入口：
  * build_alpha101_inputs  -> 组装 date x symbol 的原始输入（OHLCV + 派生 returns/vwap）
  * build_alpha101_panel   -> 调用某个 alpha 公式，产出供 factor_analysis 复用的 FactorPanel

与 `libs/factor_analysis/panel.py::build_factor_panel` 的区别：
  后者逐标的调用 BaseFactor（仅支持时序算子）；alpha101 的横截面算子需要整块面板，
  因此在这里一次性加载全池、在面板上计算，再复用 `FactorPanel` 交给下游 IC/分组分析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from factor_analysis.panel import FactorPanel
from factors.alpha101.universe import Alpha101Universe


@dataclass
class Alpha101Inputs:
    """Alpha101 公式的原始输入集合（每项均为 date x symbol 的 float DataFrame）。

    open/high/low/close/value/volume 来自原始数据；returns/vwap 为派生输入。
    adv{d} 由公式侧通过 operator.adv(inputs.value, d) 按需计算。
    cap（流通市值）来自 data/daily_basic（东财口径）；daily_basic 无数据时为
    None，依赖 cap 的 alpha 由扫描侧按可用性启用/跳过。
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    value: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame
    vwap: pd.DataFrame
    cap: pd.DataFrame | None = None

    def align(self, frame: pd.DataFrame) -> pd.DataFrame:
        """把任意 date x symbol 矩阵对齐到本面板的日期/标的网格。"""
        return frame.reindex(index=self.close.index, columns=self.close.columns)


def build_alpha101_inputs(
    universe: Alpha101Universe,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    max_workers: int | None = None,
) -> Alpha101Inputs:
    """并行加载全池标的，对齐为 date x symbol 的原始输入。"""
    symbols = universe.list_symbols()
    if not symbols:
        raise ValueError("Alpha101 universe 未提供任何标的。")

    loaded = _load_symbols_parallel(universe, symbols, max_workers=max_workers)
    if not loaded:
        raise RuntimeError("Alpha101 全池加载失败，无任何可用标的。")

    # 所有标的的日期并集作为公共索引。
    all_dates = set().union(*(df.index for df in loaded.values()))
    common_index = pd.DatetimeIndex(sorted(all_dates)).sort_values()

    # 构建各 input 矩阵。
    open_cols: dict[str, pd.Series] = {}
    high_cols: dict[str, pd.Series] = {}
    low_cols: dict[str, pd.Series] = {}
    close_cols: dict[str, pd.Series] = {}
    value_cols: dict[str, pd.Series] = {}
    volume_cols: dict[str, pd.Series] = {}

    for symbol, df in loaded.items():
        open_cols[symbol] = df["open"].reindex(common_index).astype(float)
        high_cols[symbol] = df["high"].reindex(common_index).astype(float)
        low_cols[symbol] = df["low"].reindex(common_index).astype(float)
        close_cols[symbol] = df["close"].reindex(common_index).astype(float)
        value_cols[symbol] = df["value"].reindex(common_index).astype(float)
        volume_cols[symbol] = df["volume"].reindex(common_index).astype(float)

    open_m = pd.DataFrame(open_cols, index=common_index).sort_index()
    high_m = pd.DataFrame(high_cols, index=common_index).sort_index()
    low_m = pd.DataFrame(low_cols, index=common_index).sort_index()
    close_m = pd.DataFrame(close_cols, index=common_index).sort_index()
    value_m = pd.DataFrame(value_cols, index=common_index).sort_index()
    volume_m = pd.DataFrame(volume_cols, index=common_index).sort_index()

    returns_m = close_m.pct_change()
    vwap_m = value_m.div(volume_m)  # 成交额 / 成交量 = 当日均价（VWAP 近似）
    cap_m = _load_cap_matrix(symbols, common_index, max_workers=max_workers)

    inputs = Alpha101Inputs(
        open=open_m,
        high=high_m,
        low=low_m,
        close=close_m,
        value=value_m,
        volume=volume_m,
        returns=returns_m,
        vwap=vwap_m,
        cap=cap_m,
    )

    if start_date is not None or end_date is not None:
        inputs = _slice_inputs(inputs, start_date, end_date)

    return inputs


def _slice_inputs(
    inputs: Alpha101Inputs,
    start_date: str | None,
    end_date: str | None,
) -> Alpha101Inputs:
    """按日期范围切分所有 input 矩阵。"""
    mask = pd.Series(True, index=inputs.close.index)
    if start_date is not None:
        mask &= inputs.close.index >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= inputs.close.index <= pd.Timestamp(end_date)
    return Alpha101Inputs(
        open=inputs.open.loc[mask],
        high=inputs.high.loc[mask],
        low=inputs.low.loc[mask],
        close=inputs.close.loc[mask],
        value=inputs.value.loc[mask],
        volume=inputs.volume.loc[mask],
        returns=inputs.returns.loc[mask],
        vwap=inputs.vwap.loc[mask],
        cap=inputs.cap.loc[mask] if inputs.cap is not None else None,
    )


def build_alpha101_panel(
    alpha_func: Callable[[Alpha101Inputs], pd.DataFrame],
    universe: Alpha101Universe,
    *,
    factor_name: str,
    min_bars: int = 252,
    start_date: str | None = None,
    end_date: str | None = None,
    max_workers: int | None = None,
) -> FactorPanel:
    """计算一个 alpha 公式并构造 FactorPanel（供下游 IC/分组/质量分析复用）。

    Parameters
    ----------
    alpha_func: 接收 Alpha101Inputs、返回 date x symbol 因子矩阵的函数。
    universe: 数据源。
    factor_name: 因子输出名（如 "Alpha101_001"）。
    min_bars: 单标的至少需要的有效值天数，低于此值的被剔除。
    start_date / end_date: 日期过滤。
    max_workers: 数据加载并行度。
    """
    inputs = build_alpha101_inputs(
        universe, start_date=start_date, end_date=end_date, max_workers=max_workers,
    )
    factor_matrix = alpha_func(inputs).reindex(
        index=inputs.close.index, columns=inputs.close.columns
    ).astype(float)

    close_matrix = inputs.close
    value_matrix = inputs.value

    # 过滤 bar_count 不足的标的（沿用因子分析框架约定）。
    bar_counts = factor_matrix.notna().sum(axis=0)
    keep_symbols = bar_counts[bar_counts >= min_bars].index.tolist()
    filtered_symbols = {
        str(s): int(c) for s, c in bar_counts[bar_counts < min_bars].items()
    }

    if not keep_symbols:
        raise RuntimeError(
            f"Alpha101[{factor_name}] 面板构建失败：min_bars={min_bars} 过滤后无剩余标的。"
        )

    factor_matrix = factor_matrix[keep_symbols]
    close_matrix = close_matrix[keep_symbols]
    value_matrix = value_matrix[keep_symbols]

    meta_rows = {}
    for symbol in keep_symbols:
        meta_rows[symbol] = {
            "bar_count": int(bar_counts[symbol]),
            "first_valid_date": factor_matrix[symbol].first_valid_index(),
        }
    symbol_meta = pd.DataFrame.from_dict(meta_rows, orient="index")
    symbol_meta.index.name = "symbol"

    return FactorPanel(
        factor_values=factor_matrix,
        close_prices=close_matrix,
        volumes=value_matrix,
        symbol_meta=symbol_meta,
        factor_name=factor_name,
        errors=[],
        filtered_symbols=filtered_symbols,
    )


# ── 并行加载 ──────────────────────────────────────────────────────────────────


def _load_one(universe: Alpha101Universe, symbol: str) -> tuple[str, pd.DataFrame | None, str | None]:
    """加载单个标的，返回 (symbol, df, error)。"""
    try:
        return symbol, universe.load(symbol), None
    except Exception as exc:  # noqa: BLE001
        return symbol, None, f"{type(exc).__name__}: {exc}"


def _load_symbols_parallel(
    universe: Alpha101Universe,
    symbols: list[str],
    *,
    max_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    """并行加载全池标的。单个标的失败仅记录并跳过，不影响整体。"""
    if max_workers is None or max_workers <= 1 or len(symbols) <= 1:
        loaded: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            sym, df, err = _load_one(universe, symbol)
            if df is not None:
                loaded[sym] = df
        return loaded

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {executor.submit(_load_one, universe, symbol): symbol for symbol in symbols}
        loaded = {}
        for future in as_completed(futures):
            sym, df, err = future.result()
            if df is not None:
                loaded[sym] = df
    return loaded


# ── cap (市值) 面板 ───────────────────────────────────────────────────────────


def _load_cap_one(symbol: str) -> tuple[str, pd.Series | None, str | None]:
    """加载单个标的的流通市值序列，返回 (symbol, series, error)。"""
    try:
        from data_manager.daily_basic_manager import load_daily_basic

        df = load_daily_basic(symbol)
        if df is None or df.empty:
            return symbol, None, "no daily_basic"
        return symbol, df["circ_mv"], None
    except Exception as exc:  # noqa: BLE001
        return symbol, None, f"{type(exc).__name__}: {exc}"


def _load_cap_symbols_parallel(
    symbols: list[str],
    *,
    max_workers: int | None = None,
) -> dict[str, pd.Series]:
    """并行加载市值序列。单个标的失败仅记录并跳过，不影响整体。"""
    if max_workers is None or max_workers <= 1 or len(symbols) <= 1:
        loaded: dict[str, pd.Series] = {}
        for symbol in symbols:
            sym, series, err = _load_cap_one(symbol)
            if series is not None:
                loaded[sym] = series
        return loaded

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {executor.submit(_load_cap_one, symbol): symbol for symbol in symbols}
        loaded = {}
        for future in as_completed(futures):
            sym, series, err = future.result()
            if series is not None:
                loaded[sym] = series
    return loaded


def _load_cap_matrix(
    symbols: list[str],
    common_index: pd.DatetimeIndex,
    *,
    max_workers: int | None = None,
) -> pd.DataFrame | None:
    """构建 date x symbol 的流通市值矩阵；全池均无数据时返回 None。"""
    series_map = _load_cap_symbols_parallel(symbols, max_workers=max_workers)
    if not series_map:
        return None
    cap_cols = {
        symbol: series.reindex(common_index).astype(float)
        for symbol, series in series_map.items()
    }
    matrix = pd.DataFrame(cap_cols, index=common_index).sort_index()
    matrix.index.name = "date"
    return matrix
