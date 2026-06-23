"""
前向收益计算 (Forward Return Computation)

计算每个标的在每个交易日后的未来 N 日收益率矩阵，
供 Layer 2 (IC 计算) 和 Layer 3 (分组检验) 使用。

核心公式:
    fwd_return_{t, period} = close_{t+period} / close_t - 1

注意:
    - 未来收益使用收盘价计算（close-to-close）。
    - t+period 超出数据范围的尾部日期会自动填充 NaN（IC 计算时自动跳过）。
    - period=0 无意义（同日收益），不在默认参数中。
    - 不同 period 之间完全独立，各自从 close_prices 计算，互不影响。
"""

from __future__ import annotations

import pandas as pd


def compute_forward_returns(
    close_prices: pd.DataFrame,
    periods: tuple[int, ...],
) -> dict[int, pd.DataFrame]:
    """计算多个持仓期的前向收益率矩阵。

    Parameters
    ----------
    close_prices:
        date (Index) × symbol (columns) 的收盘价矩阵。
        必须按时间升序排列（shift 前向取未来值依赖此顺序）。
    periods:
        持仓期列表（单位：交易日）。如 (5, 10, 20, 60)。
        5  ≈ 1 周（检验短期 alpha 衰减速度）
        10 ≈ 2 周（中短期观察点）
        20 ≈ 1 个月（月频调仓参考）
        60 ≈ 1 个季度（长期趋势持续性）

    Returns
    -------
    dict[int, pd.DataFrame]
        {period: date × symbol 的前向收益率矩阵}。
        收益率以小数形式存储（如 0.05 表示 +5%）。

    Examples
    --------
    >>> close = pd.DataFrame(...)  # date × symbol
    >>> fwd = compute_forward_returns(close, periods=(5, 20))
    >>> fwd_5d = fwd[5]   # 5 日前向收益矩阵
    >>> fwd_20d = fwd[20]  # 20 日前向收益矩阵
    """
    if close_prices.empty:
        return {}

    # 确保按时间升序排列（shift 依赖此顺序）
    close = close_prices.sort_index()

    result: dict[int, pd.DataFrame] = {}
    for period in periods:
        if period <= 0:
            raise ValueError(f"period 必须为正整数，收到 {period}")
        # 未来 N 日收益 = close_{t+N} / close_t - 1
        # shift(-period) 把 N 天后的价格拉到当前行
        future_close = close.shift(-period)
        fwd_ret = future_close.divide(close) - 1.0
        # 尾部 period 行的 future_close 为 NaN，fwd_ret 自然为 NaN
        result[period] = fwd_ret

    return result
