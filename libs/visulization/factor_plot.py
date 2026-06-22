"""因子可视化绘图工具。

提供 `plot_factors` 函数，用于在 notebook 中快速将多个因子的时间序列
与标的收盘价一起绘制成折线图，方便直观观察因子值变化。
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.models.etf_daily_data import EtfData
from data_manager.etf_data_manager import get_etf_data_by_symbol
from factors.base_factor import BaseFactor

# ── 离散信号因子的典型输出值集合 ─────────────────────────────────────────
# 这些因子的输出是有限个离散值（如买入/卖出信号），用 step 线绘制更直观。
_DISCRETE_SIGNAL_VALUES: set[float | int] = {-1, 0, 1, 2}


def _is_discrete_signal(series: pd.Series) -> bool:
    """判断一个因子序列是否为离散信号（如 1/0/-1 三类信号）。"""
    unique_vals = set(series.dropna().unique())
    if len(unique_vals) <= 5 and unique_vals.issubset(_DISCRETE_SIGNAL_VALUES):
        return True
    return False


def _detect_date_index(df: pd.DataFrame) -> tuple[pd.Index, str]:
    """检测日期索引并返回日期轴标签。

    优先使用 index（DatetimeIndex），其次使用 'date' 列。
    返回 (date_index, x_label)。
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index, "日期"
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        return dates, "日期"
    return df.index, "Index"


def plot_factors(
    symbol_or_data: str | EtfData,
    factors: Sequence[BaseFactor],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    show_price: bool = True,
    figsize: tuple[float, float] = (16, 8),
    title: str | None = None,
    price_color: str = "#333333",
    price_alpha: float = 0.8,
    factor_alpha: float = 0.85,
    grid: bool = True,
    legend_loc: str = "best",
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """绘制指定标的的收盘价与多个因子的时间序列折线图。

    Parameters
    ----------
    symbol_or_data : str | EtfData
        ETF 代码（如 "510300"）或已加载的 EtfData 对象。
    factors : Sequence[BaseFactor]
        要绘制的因子实例列表。每个因子将通过 ``factor(data)`` 计算。
    start_date : str | None
        起始日期，如 "2024-01-01"。默认不限制。
    end_date : str | None
        结束日期，如 "2025-12-31"。默认不限制。
    show_price : bool
        是否在面板上方绘制收盘价折线。默认 True。
    figsize : tuple[float, float]
        图形尺寸 (宽, 高)，默认 (16, 8)。
    title : str | None
        图表标题。默认自动生成。
    price_color : str
        收盘价折线颜色，默认深灰。
    price_alpha : float
        收盘价折线透明度。
    factor_alpha : float
        因子折线透明度。
    grid : bool
        是否显示网格线。
    legend_loc : str
        图例位置，默认 "best"。

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, ...]]
        (figure, (price_ax, factor_ax)) 或 (figure, (factor_ax,))。
        可在调用后继续自定义图表。

    Examples
    --------
    >>> from factors.rsrs import RsrsFactor
    >>> from factors.price_return import PriceReturn
    >>> from factors.trend_r2 import TrendR2Factor
    >>> from visulization.factor_plot import plot_factors
    >>>
    >>> factors = [
    ...     RsrsFactor(output="zscore"),
    ...     PriceReturn(window=60),
    ...     TrendR2Factor(window=120, output="r2"),
    ... ]
    >>> fig, axes = plot_factors("510300", factors,
    ...     start_date="2024-01-01", end_date="2025-06-01")
    """
    # ── 1. 加载数据 ──────────────────────────────────────────────────
    if isinstance(symbol_or_data, EtfData):
        etf_data = symbol_or_data
        symbol = etf_data.symbol or "unknown"
    else:
        symbol = str(symbol_or_data)
        etf_data = get_etf_data_by_symbol(symbol)

    # ── 2. 日期过滤 ──────────────────────────────────────────────────
    if start_date is not None or end_date is not None:
        etf_data = etf_data.slice_date_range(
            start_date=start_date, end_date=end_date
        )

    # ── 3. 计算因子 ─────────────────────────────────────────────────
    for factor in factors:
        etf_data.add_factors(factor)
    factor_df = etf_data.calc_factors()

    close = factor_df["close"].astype(float)
    date_index, x_label = _detect_date_index(factor_df)

    # ── 4. 确定因子输出名称列表 ──────────────────────────────────────
    factor_names = [factor.get_output_name() for factor in factors]

    # 确保所有因子列都存在
    missing = [n for n in factor_names if n not in factor_df.columns]
    if missing:
        ohlcv_cols = {
            "open", "high", "low", "close", "volume",
            "value", "turnOver", "gain", "change",
        }
        available = [c for c in factor_df.columns if c not in ohlcv_cols]
        raise KeyError(
            f"因子列 {missing} 不在计算结果中。"
            f"可用的因子列: {available}"
        )

    # ── 5. 构建图表 ─────────────────────────────────────────────────
    n_panels = 2 if show_price else 1
    height_ratios = [3, 5] if show_price else [1]

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = (axes,)

    price_ax = axes[0] if show_price else None
    factor_ax = axes[-1]

    # ── 5a. 收盘价面板 ──────────────────────────────────────────────
    if show_price and price_ax is not None:
        price_ax.plot(
            date_index, close.values,
            color=price_color, linewidth=1.5, alpha=price_alpha,
            label=f"{symbol} 收盘价",
        )
        price_ax.set_ylabel("收盘价")
        if grid:
            price_ax.grid(True, alpha=0.25)
        price_ax.legend(loc=legend_loc)

    # ── 5b. 因子面板 ────────────────────────────────────────────────
    palette = sns.color_palette("husl", n_colors=len(factor_names))

    for i, name in enumerate(factor_names):
        series = factor_df[name].astype(float)
        color = palette[i]

        if _is_discrete_signal(series):
            factor_ax.step(
                date_index, series.values,
                where="post", color=color, linewidth=1.2,
                alpha=factor_alpha, label=name,
            )
        else:
            factor_ax.plot(
                date_index, series.values,
                color=color, linewidth=1.2,
                alpha=factor_alpha, label=name,
            )

    factor_ax.set_ylabel("因子值")
    factor_ax.set_xlabel(x_label)
    if grid:
        factor_ax.grid(True, alpha=0.25)

    # 离散信号因子：设置合适的 y 轴刻度
    all_discrete = all(
        _is_discrete_signal(factor_df[n].astype(float))
        for n in factor_names
    )
    if all_discrete:
        unique_vals = sorted(set().union(*(
            factor_df[n].dropna().unique() for n in factor_names
        )))
        factor_ax.set_yticks(unique_vals)

    # 零线参考（因子值跨越正负时）
    factor_values = factor_df[factor_names].values
    if bool(np.nanmax(factor_values) > 0) and bool(np.nanmin(factor_values) < 0):
        factor_ax.axhline(
            y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5
        )

    factor_ax.legend(loc=legend_loc)

    # ── 6. 标题与布局 ────────────────────────────────────────────────
    if title is None:
        name = etf_data.name or symbol
        factor_desc = ", ".join(factor_names)
        title = f"{name} ({symbol}) — 因子走势\n{factor_desc}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    return fig, axes
