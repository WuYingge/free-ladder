"""
截面因子面板构建 (Cross-Sectional Factor Panel Builder)

将多个单标的因子计算结果组装为统一的 date × symbol 矩阵，
是整个因子分析框架的数据地基。

核心输出: FactorPanel(factor_values, close_prices, volumes, symbol_meta, factor_name)

计算流程:
  1. (多进程) 对每个 symbol 加载数据、计算因子值
  2. 将所有 symbol 对齐到统一的日期索引
  3. 构建 3 个 date × symbol 矩阵（因子值 / 收盘价 / 成交额）
  4. 过滤 bar_count < min_bars 的标的
  5. 按 start_date / end_date 切片
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.models.etf_daily_data import EtfData
from data_manager.etf_data_manager import get_etf_data_by_symbol
from factors.base_factor import BaseFactor


# ── 数据结构 ────────────────────────────────────────────────────────────────


@dataclass
class PanelBuildError:
    """单个标的在面板构建中出错时的记录。"""
    symbol: str
    stage: str  # "load" | "compute"
    error: str


@dataclass
class FactorPanel:
    """截面因子面板 —— 整个分析框架的核心数据结构。

    Attributes
    ----------
    factor_values:
        date (Index) × symbol (columns) 的因子值矩阵。
        每个单元格是该日该标的的因子值；NaN 表示因子无法计算（如 warmup 期内）。

    close_prices:
        date × symbol 的收盘价矩阵。用于计算前向收益率和分组收益。

    volumes:
        date × symbol 的成交额（value，单位元）矩阵。用于缺失分析中的成交额分档。
        注意：东方财富返回的 "成交额" 列在本地存储中命名为 "value"。

    symbol_meta:
        symbol (Index) × 属性列 的 DataFrame。
        列：name（ETF 名称）、bar_count（有效交易日数）、listing_date（首日日期）。

    factor_name:
        因子输出列名，如 "PriceReturn_60"。

    errors:
        构建过程中出错的 symbol 列表（load 失败 / 计算失败）。

    filtered_symbols:
        被 min_bars 过滤掉的 symbol 及各自的 bar_count 列表。
    """

    factor_values: pd.DataFrame
    close_prices: pd.DataFrame
    volumes: pd.DataFrame
    symbol_meta: pd.DataFrame
    factor_name: str
    errors: list[PanelBuildError] = field(default_factory=list)
    filtered_symbols: dict[str, int] = field(default_factory=dict)

    @property
    def n_symbols(self) -> int:
        """有效标的数（列数）。"""
        return self.factor_values.shape[1]

    @property
    def n_dates(self) -> int:
        """有效日期数（行数）。"""
        return self.factor_values.shape[0]

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """面板的起止日期。"""
        return self.factor_values.index[0], self.factor_values.index[-1]

    def summary(self) -> dict[str, Any]:
        """返回面板概况（供报告使用）。"""
        return {
            "n_symbols": self.n_symbols,
            "n_dates": self.n_dates,
            "start_date": str(self.date_range[0].date()),
            "end_date": str(self.date_range[1].date()),
            "factor_name": self.factor_name,
            "n_errors": len(self.errors),
            "n_filtered": len(self.filtered_symbols),
            "coverage_mean": float(self.factor_values.notna().sum(axis=1).mean() / self.n_symbols),
        }


# ── 多进程 worker ───────────────────────────────────────────────────────────


def _calc_factor_worker(
    symbol: str,
    factor: BaseFactor,
) -> tuple[str, dict[str, pd.Series] | None, str | None]:
    """多进程 worker：加载单个标的的数据并计算因子值。

    返回 (symbol, series_dict, error_msg)。
    series_dict 包含: {factor_output_name: factor_series, "close": close_series, "volume": volume_series}
    """
    try:
        etf_data: EtfData = get_etf_data_by_symbol(symbol)
        raw_df = etf_data.data

        # 确保索引是 DatetimeIndex。
        # EtfData.from_csv 读入时 date 列是字符串列，需转换为 datetime 索引。
        if "date" in raw_df.columns:
            date_series = pd.to_datetime(raw_df["date"], errors="coerce")
            raw_df = raw_df.set_index(date_series)
        elif not isinstance(raw_df.index, pd.DatetimeIndex):
            raw_df.index = pd.to_datetime(raw_df.index, errors="coerce")

        # 提取 close / volume，确保索引一致
        close = raw_df.get("close", pd.Series(dtype=float))
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        close = pd.to_numeric(close, errors="coerce")

        volume = raw_df.get("value", pd.Series(dtype=float))
        if isinstance(volume, pd.DataFrame):
            volume = volume.squeeze()
        volume = pd.to_numeric(volume, errors="coerce")
        # value 列记录了当日成交额（单位：元）
        # 清洗：0 或负值 → NaN（无效交易日的成交额无分析意义）
        volume = volume.where(volume > 0, other=float("nan"))

        # 计算因子值（传入含 DatetimeIndex 的 DataFrame）
        factor_series = factor(raw_df)
        if not isinstance(factor_series, pd.Series):
            factor_series = pd.Series(factor_series, index=raw_df.index)
        factor_series = pd.to_numeric(factor_series, errors="coerce")

        output_name = factor.get_output_name()
        return symbol, {
            output_name: factor_series,
            "close": close,
            "volume": volume,
        }, None
    except Exception:
        return symbol, None, traceback.format_exc(limit=8)


# ── 面板构建主函数 ──────────────────────────────────────────────────────────


def build_factor_panel(
    factor: BaseFactor,
    symbols: list[str],
    *,
    min_bars: int = 252,
    start_date: str | None = None,
    end_date: str | None = None,
    max_workers: int | None = None,
) -> FactorPanel:
    """构建截面因子面板。

    Parameters
    ----------
    factor:
        因子实例（已设好参数）。对每个 symbol 调用 factor(data) 计算。
    symbols:
        待分析的 ETF 标的列表。
    min_bars:
        单个标的至少需要的有效交易日数。bar_count 低于此值的标的不参与后续分析。
    start_date / end_date:
        日期范围过滤。"YYYY-MM-DD" 格式，None = 不限制。
    max_workers:
        多进程 worker 数量。None = os.cpu_count()。

    Returns
    -------
    FactorPanel
        包含 factor_values / close_prices / volumes / symbol_meta 及构建错误信息。
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(symbols))

    factor_name = factor.get_output_name()
    errors: list[PanelBuildError] = []
    series_by_symbol: dict[str, dict[str, pd.Series]] = {}

    # ── 1. 多进程并行计算 ──────────────────────────────────────────────
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_calc_factor_worker, symbol, factor): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol, result, error = future.result()
            if error is not None:
                errors.append(PanelBuildError(symbol=symbol, stage="compute", error=error))
                continue
            if result is None:
                errors.append(PanelBuildError(symbol=symbol, stage="compute", error="empty result"))
                continue
            series_by_symbol[symbol] = result

    if not series_by_symbol:
        raise RuntimeError(
            f"面板构建失败：{len(symbols)} 个标的全部计算失败，"
            f"首条错误: {errors[0].error if errors else 'unknown'}"
        )

    # ── 2. 对齐索引：找出所有 symbol 的日期并集 ─────────────────────────
    all_dates = sorted(set().union(*(s.index for s_dict in series_by_symbol.values() for s in s_dict.values())))
    common_index = pd.DatetimeIndex(all_dates).sort_values()

    # ── 3. 构建 3 个 date × symbol 矩阵 ─────────────────────────────────
    # 先收集所有列到 dict，再用 pd.concat 一次性构建，避免逐列插入的性能警告
    factor_cols: dict[str, pd.Series] = {}
    close_cols: dict[str, pd.Series] = {}
    volume_cols: dict[str, pd.Series] = {}

    for symbol, s_dict in series_by_symbol.items():
        factor_series = s_dict.get(factor_name)
        close_series = s_dict.get("close")
        volume_series = s_dict.get("volume")

        if factor_series is not None:
            factor_cols[symbol] = factor_series.reindex(common_index).astype(float)
        if close_series is not None:
            close_cols[symbol] = close_series.reindex(common_index).astype(float)
        if volume_series is not None:
            volume_cols[symbol] = volume_series.reindex(common_index).astype(float)

    factor_matrix = pd.DataFrame(factor_cols, index=common_index)
    close_matrix = pd.DataFrame(close_cols, index=common_index)
    volume_matrix = pd.DataFrame(volume_cols, index=common_index)

    # ── 4. 过滤 bar_count 不足的标的 ────────────────────────────────────
    bar_counts = factor_matrix.notna().sum(axis=0)
    low_bar_symbols = bar_counts[bar_counts < min_bars]
    filtered = {str(sym): int(cnt) for sym, cnt in low_bar_symbols.items()}

    keep_symbols = bar_counts[bar_counts >= min_bars].index.tolist()
    factor_matrix = factor_matrix[keep_symbols]
    close_matrix = close_matrix[keep_symbols]
    volume_matrix = volume_matrix[keep_symbols]

    if factor_matrix.empty:
        raise RuntimeError(
            f"面板构建失败：min_bars={min_bars} 过滤后无剩余标的"
        )

    # ── 5. 日期范围切片 ────────────────────────────────────────────────
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        factor_matrix = factor_matrix.loc[factor_matrix.index >= start_ts]
        close_matrix = close_matrix.loc[close_matrix.index >= start_ts]
        volume_matrix = volume_matrix.loc[volume_matrix.index >= start_ts]

    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        factor_matrix = factor_matrix.loc[factor_matrix.index <= end_ts]
        close_matrix = close_matrix.loc[close_matrix.index <= end_ts]
        volume_matrix = volume_matrix.loc[volume_matrix.index <= end_ts]

    # 按 index 自然递增排序（确保后续 shift/rolling 运算正确）
    factor_matrix = factor_matrix.sort_index()
    close_matrix = close_matrix.sort_index()
    volume_matrix = volume_matrix.sort_index()

    # ── 6. 构建 symbol_meta ─────────────────────────────────────────────
    meta_rows = {}
    for symbol in keep_symbols:
        first_valid_date = factor_matrix[symbol].first_valid_index()
        meta_rows[symbol] = {
            "bar_count": int(bar_counts[symbol]),
            "first_valid_date": first_valid_date,
        }

    symbol_meta = pd.DataFrame.from_dict(meta_rows, orient="index")
    symbol_meta.index.name = "symbol"

    return FactorPanel(
        factor_values=factor_matrix,
        close_prices=close_matrix,
        volumes=volume_matrix,
        symbol_meta=symbol_meta,
        factor_name=factor_name,
        errors=errors,
        filtered_symbols=filtered,
    )
