"""
因子走势服务

根据因子名 + symbol 构造 FactorPanel，提取单标的因子值、收盘价、成交额时序数据。
使用 LRU 缓存避免重复计算。
"""
from __future__ import annotations

import functools
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP
from factor_analysis.panel import build_factor_panel, FactorPanel
from services.factor_registry import get_factor_instance, get_available_factors


# ── LRU 缓存 ──────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=16)
def _cached_build_panel(factor_name: str, tuple_symbols: tuple[str, ...]) -> FactorPanel:
    """缓存 FactorPanel 构建结果。

    Parameters
    ----------
    factor_name: 因子名
    tuple_symbols: 所有 symbols 的元组（哈希友好）
    """
    factor = get_factor_instance(factor_name)
    if factor is None:
        raise ValueError(f"无法解析因子: {factor_name}，该因子可能为复杂衍生因子，暂不支持走势查询。")

    symbols = list(tuple_symbols)
    return build_factor_panel(
        factor=factor,
        symbols=symbols,
        min_bars=252,
    )


def _get_all_symbols() -> list[str]:
    """获取所有可用 ETF symbols。"""
    try:
        return ETF_INDEX_MAP.get_all_symbols()
    except Exception:
        # fallback：从 _index.json 获取
        return []


# ── 公共 API ──────────────────────────────────────────────────────────────────

def get_trend_data(
    factor_name: str,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """获取因子在某 symbol 上的走势时序数据。

    Parameters
    ----------
    factor_name: 因子名
    symbol: ETF 代码
    start_date: 起始日期 YYYY-MM-DD，默认近一年
    end_date: 截止日期 YYYY-MM-DD，默认最新

    Returns
    -------
    dict: {factor_name, symbol, symbol_name, start_date, end_date, data: [...], stats: {...}}
    """
    # 获取所有 symbols
    all_symbols = _get_all_symbols()
    if not all_symbols:
        raise ValueError("无法获取 ETF 标的列表")

    if symbol not in all_symbols:
        # 尝试部分匹配
        matched = [s for s in all_symbols if symbol in s]
        if not matched:
            raise ValueError(f"未找到 symbol: {symbol}")
        symbol = matched[0]

    # 获取 symbol 名称
    symbol_name = _get_symbol_name(symbol)

    # 构建面板（从缓存获取）
    try:
        panel = _cached_build_panel(factor_name, tuple(all_symbols))
    except ValueError as e:
        raise
    except Exception as e:
        raise RuntimeError(f"构建因子面板失败: {e}")

    # 从面板中提取该 symbol 的数据
    if symbol not in panel.factor_values.columns:
        raise ValueError(f"因子 {factor_name} 的面板中不含 symbol: {symbol}")

    factor_series = panel.factor_values[symbol].dropna()
    close_series = panel.close_prices[symbol].dropna()
    volume_series = panel.volumes[symbol].dropna()

    # 确定日期范围
    if start_date is None:
        # 默认近一年
        latest_date = factor_series.index.max()
        start_dt = latest_date - pd.Timedelta(days=365)
        start_date = start_dt.strftime("%Y-%m-%d")
    if end_date is None:
        end_date = factor_series.index.max().strftime("%Y-%m-%d")

    # 按日期范围筛选
    mask = (factor_series.index >= start_date) & (factor_series.index <= end_date)
    factor_series = factor_series[mask]
    close_series = close_series[close_series.index >= start_date]
    close_series = close_series[close_series.index <= end_date]
    volume_series = volume_series[volume_series.index >= start_date]
    volume_series = volume_series[volume_series.index <= end_date]

    # 构建 data list
    data_points: list[dict] = []
    for dt in factor_series.index:
        dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
        fv = factor_series.get(dt)
        close = close_series.get(dt)
        vol = volume_series.get(dt)

        data_points.append({
            "date": dt_str,
            "factor_value": float(fv) if pd.notna(fv) else None,
            "close": float(close) if pd.notna(close) else None,
            "volume": float(vol) if pd.notna(vol) else None,
        })

    # 统计
    fv_np = factor_series.values
    stats = {
        "factor_mean": float(np.mean(fv_np)) if len(fv_np) > 0 else None,
        "factor_std": float(np.std(fv_np)) if len(fv_np) > 0 else None,
        "factor_min": float(np.min(fv_np)) if len(fv_np) > 0 else None,
        "factor_max": float(np.max(fv_np)) if len(fv_np) > 0 else None,
        "factor_latest": float(fv_np[-1]) if len(fv_np) > 0 else None,
        "close_latest": float(close_series.iloc[-1]) if len(close_series) > 0 else None,
        "data_points": len(data_points),
    }

    return {
        "factor_name": factor_name,
        "symbol": symbol,
        "symbol_name": symbol_name,
        "start_date": start_date,
        "end_date": end_date,
        "data": data_points,
        "stats": stats,
    }


def _get_symbol_name(symbol: str) -> str:
    """获取 symbol 对应的 ETF 名称。"""
    try:
        info = ETF_INDEX_MAP.get_info(symbol)
        return info.get("name", symbol) if info else symbol
    except Exception:
        return symbol


def get_available_symbols() -> list[dict[str, str]]:
    """获取可用 ETF 标的列表（用于前端输入提示）。"""
    all_syms = _get_all_symbols()
    result = []
    for s in all_syms:
        name = _get_symbol_name(s)
        result.append({"symbol": s, "name": name})
    return result


def get_trendable_factors() -> list[str]:
    """返回可以做走势查询的因子列表（所有有 report.json 的因子）。

    不做实例化验证（太慢），验证延迟到实际请求时。
    """
    return get_available_factors()
