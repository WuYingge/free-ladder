"""
分组检验 (Grouping Analysis)

Layer 3：从"因子能否排序"深化到"按因子分组能赚多少钱"。
将 IC 的统计显著性转化为可交易的经济显著性。

包含四项分析:
    3.1 分位数组合 — 每日截面按因子值分组，等权持有，各组的平均收益
    3.2 Long-Short 组合 — 多 Top 组 + 空 Bottom 组的多空收益序列
    3.3 分组累计收益曲线 — 各分组的累计收益叠加图
    3.4 单调性检验 — 检查分组收益是否严格单调 (Q1 > Q2 > ... > Qn)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from factor_analysis.panel import FactorPanel
from factor_analysis.forward_returns import compute_forward_returns


# ── 分位数分组辅助 ───────────────────────────────────────────────────────────


def _assign_quantile_group(
    factor_row: pd.Series,
    n_quantiles: int,
) -> pd.Series:
    """将单日截面的因子值（含 NaN）分为 n_quantiles 组。

    使用 pd.qcut（等频分位），duplicates='drop' 处理边界重复值。
    NaN 的标的分配 group=NaN（不参与任何组）。

    Returns
    -------
    pd.Series
        index=symbol, values=group_label (1 ~ n_quantiles，1=最低组，n=最高组)
    """
    valid = factor_row.dropna()
    if len(valid) < n_quantiles:
        return pd.Series(np.nan, index=factor_row.index)

    try:
        groups = pd.qcut(valid, q=n_quantiles, labels=False, duplicates="drop")
        # pd.qcut 的 labels=False 返回 0-based group index
        result = pd.Series(np.nan, index=factor_row.index)
        result[groups.index] = groups.values + 1  # 转为 1-based
        return result
    except ValueError:
        return pd.Series(np.nan, index=factor_row.index)


# ── 3.1 分位数组合 ──────────────────────────────────────────────────────────


def compute_quantile_returns(
    panel: FactorPanel,
    fwd_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """计算各分位数组的每日等权平均收益。

    每日截面：
    1. 按因子值将标的分为 n_quantiles 组
    2. 每组等权取该组标的未来收益的均值
    3. 得到每个分位组的时间序列

    Parameters
    ----------
    panel: 截面因子面板。
    fwd_returns: 未来收益率矩阵（date × symbol）。
    n_quantiles: 分组数。5 = 五分位（Q1~Q5）。

    Returns
    -------
    pd.DataFrame
        index=日期, columns=[Q1, Q2, ..., Qn]。
        Q1 = 因子值最低组（Bottom），Qn = 因子值最高组（Top）。
    """
    fv = panel.factor_values
    fwd = fwd_returns

    common_dates = fv.index.intersection(fwd.index)
    common_symbols = fv.columns.intersection(fwd.columns)

    if len(common_dates) == 0 or len(common_symbols) == 0:
        return pd.DataFrame()

    fv_aligned = fv.loc[common_dates, common_symbols]
    fwd_aligned = fwd.loc[common_dates, common_symbols]

    # 预分配：每日期 × 每组
    group_returns: dict[str, list[float]] = {
        f"Q{q}": [] for q in range(1, n_quantiles + 1)
    }
    date_index: list[pd.Timestamp] = []

    for date in common_dates:
        factor_row = fv_aligned.loc[date]
        fwd_row = fwd_aligned.loc[date]

        group_labels = _assign_quantile_group(factor_row, n_quantiles)

        date_index.append(date)
        for q in range(1, n_quantiles + 1):
            syms_in_group = group_labels[group_labels == q].index
            if len(syms_in_group) == 0:
                group_returns[f"Q{q}"].append(float("nan"))
            else:
                # 等权平均：该组所有标的的未来收益均值
                rets = fwd_row[syms_in_group].dropna()
                if len(rets) == 0:
                    group_returns[f"Q{q}"].append(float("nan"))
                else:
                    group_returns[f"Q{q}"].append(float(rets.mean()))

    result = pd.DataFrame(group_returns, index=pd.DatetimeIndex(date_index))
    result.index.name = "date"
    return result


def compute_quantile_summary(
    group_returns: pd.DataFrame,
) -> pd.DataFrame:
    """计算各分位组的平均收益和统计量。

    Returns
    -------
    pd.DataFrame
        index=Q1..Qn, columns=[mean_return, std_return, sharpe, win_rate]
    """
    if group_returns.empty:
        return pd.DataFrame()

    records: list[dict[str, float]] = []
    for col in group_returns.columns:
        valid = group_returns[col].dropna()
        if len(valid) < 2:
            records.append({
                "group": col,
                "mean_return": float("nan"),
                "std_return": float("nan"),
                "win_rate": float("nan"),
            })
            continue
        mean_ret = float(valid.mean())
        std_ret = float(valid.std(ddof=1))
        win_rate = float((valid > 0).sum() / len(valid))
        records.append({
            "group": col,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "win_rate": win_rate,
        })

    return pd.DataFrame(records).set_index("group")


# ── 3.2 Long-Short 组合 ────────────────────────────────────────────────────


def compute_longshort(
    group_returns: pd.DataFrame,
) -> dict[str, Any]:
    """计算多空组合（Long Top - Short Bottom）的收益序列和性能指标。

    做多因子值最高组 (Top = Qn) + 做空因子值最低组 (Bottom = Q1)。
    多空收益 = Top 组收益 - Bottom 组收益。
    这是因子纯 alpha 的最直接度量——剥离了市场 beta。

    Parameters
    ----------
    group_returns: 各组收益矩阵，columns=[Q1, Q2, ..., Qn]。

    Returns
    -------
    dict
        {
            "ls_series": pd.Series,     # 多空每日收益序列
            "annualised_return": float,
            "annualised_volatility": float,
            "sharpe": float | None,
            "max_drawdown": float,
            "cumulative_return": float,
        }
    """
    if group_returns.empty or len(group_returns.columns) < 2:
        return {}

    cols = group_returns.columns.tolist()
    bottom_col = cols[0]   # Q1 = 因子值最低 = 做空
    top_col = cols[-1]     # Qn = 因子值最高 = 做多

    ls = group_returns[top_col] - group_returns[bottom_col]
    ls = ls.dropna()
    ls.name = "long_short"

    if len(ls) < 2:
        return {"ls_series": ls}

    # 复用 performance.py 的指标计算（避免重复造轮子）
    try:
        from backtesting.performance import (
            annualised_return,
            annualised_volatility,
            cumulative_return,
            max_drawdown,
            sharpe_ratio,
        )
    except ImportError:
        # 降级为本地计算
        return _compute_longshort_local(ls)

    return {
        "ls_series": ls,
        "annualised_return": annualised_return(ls),
        "annualised_volatility": annualised_volatility(ls),
        "sharpe": sharpe_ratio(ls),
        "max_drawdown": max_drawdown(ls),
        "cumulative_return": cumulative_return(ls),
    }


def _compute_longshort_local(ls: pd.Series) -> dict[str, Any]:
    """本地计算 Long-Short 指标（不依赖 performance.py 时使用）。"""
    trading_days_per_year = 242.0  # A 股年均交易日
    n = len(ls)

    cum_ret = float((1.0 + ls).prod() - 1.0)
    ann_ret = float((1.0 + cum_ret) ** (trading_days_per_year / n) - 1.0) if n > 0 else float("nan")

    ann_vol = float(ls.std(ddof=1) * (trading_days_per_year ** 0.5)) if n > 1 else float("nan")

    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else None

    cum = (1.0 + ls).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    mdd = abs(float(dd.min())) if len(dd) > 0 else float("nan")

    return {
        "ls_series": ls,
        "annualised_return": ann_ret,
        "annualised_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "cumulative_return": cum_ret,
    }


# ── 3.3 分组累计收益 ────────────────────────────────────────────────────────


def compute_quantile_cumret(
    group_returns: pd.DataFrame,
) -> pd.DataFrame:
    """计算各分位组的累计收益曲线。

    Returns
    -------
    pd.DataFrame
        index=日期, columns=[Q1, Q2, ..., Qn], values=累计净值（起始=1.0）。
    """
    if group_returns.empty:
        return pd.DataFrame()

    return (1.0 + group_returns).cumprod()


# ── 3.4 单调性检验 ──────────────────────────────────────────────────────────


def compute_monotonicity(
    group_returns: pd.DataFrame,
) -> dict[str, Any]:
    """检验分组收益是否严格单调递减/递增。

    对每日截面检查：Q1 > Q2 > ... > Qn 是否成立（完全单调成立的比例）。
    同时使用 Jonckheere-Terpstra 趋势检验评估单调性。

    Parameters
    ----------
    group_returns: columns=[Q1, Q2, ..., Qn] 的每日分组收益。

    Returns
    -------
    dict
        {
            "strict_monotonic_ratio": float,   # 严格单调成立的天数占比
            "loose_monotonic_ratio": float,     # 宽松单调（Q1 >= Q2 >= ...）的占比
            "monotonic_direction": str,         # "increasing" | "decreasing" | "none"
            "jt_statistic": float | None,       # Jonckheere-Terpstra 统计量
            "jt_pvalue": float | None,          # JT 检验 p 值
        }
    """
    if group_returns.empty or len(group_returns.columns) < 2:
        return {}

    cols = group_returns.columns.tolist()
    n_days = len(group_returns)

    # 检查每日的排列关系
    strict_true = 0
    loose_true = 0
    increasing_days = 0
    decreasing_days = 0

    for _, row in group_returns.iterrows():
        values = row.dropna().values
        if len(values) < 2:
            continue

        # 严格单调：每个值都大于下一个
        if all(values[i] > values[i + 1] for i in range(len(values) - 1)):
            strict_true += 1
        # 宽松单调：每个值 >= 下一个
        if all(values[i] >= values[i + 1] for i in range(len(values) - 1)):
            loose_true += 1

        # 方向判断：比较首尾
        if values[0] > values[-1]:
            decreasing_days += 1
        elif values[0] < values[-1]:
            increasing_days += 1

    strict_ratio = strict_true / n_days if n_days > 0 else float("nan")
    loose_ratio = loose_true / n_days if n_days > 0 else float("nan")

    if increasing_days > decreasing_days:
        direction = "increasing"
    elif decreasing_days > increasing_days:
        direction = "decreasing"
    else:
        direction = "none"

    # Jonckheere-Terpstra 趋势检验（需 scipy >= 1.7）
    jt_stat = None
    jt_pvalue = None
    try:
        # 将数据组织为 [sample1, sample2, ...] 格式
        samples = [group_returns[col].dropna().values for col in cols]
        samples = [s for s in samples if len(s) > 0]
        if len(samples) >= 2:
            jt_result = scipy_stats.jonckheere(*samples)
            jt_stat = float(jt_result.statistic) if hasattr(jt_result, "statistic") else None
            jt_pvalue = float(jt_result.pvalue) if hasattr(jt_result, "pvalue") else None
    except Exception:
        pass

    return {
        "strict_monotonic_ratio": strict_ratio,
        "loose_monotonic_ratio": loose_ratio,
        "monotonic_direction": direction,
        "jt_statistic": jt_stat,
        "jt_pvalue": jt_pvalue,
    }


# ── 汇总 ─────────────────────────────────────────────────────────────────────


def run_grouping_analysis(
    panel: FactorPanel,
    fwd_returns_map: dict[int, pd.DataFrame],
    *,
    n_quantiles: int = 5,
    holding_period: int = 20,  # 用哪个持仓期做分组检验（默认 20 日月频）
) -> dict[str, Any]:
    """一次性跑完 Layer 3 全部分组检验。

    Parameters
    ----------
    panel: 截面因子面板。
    fwd_returns_map: {period: fwd_returns_df}。
    n_quantiles: 分组数。
    holding_period: 用哪个持仓期做分组检验。

    Returns
    -------
    dict
        {
            "quantile_returns": pd.DataFrame,
            "quantile_summary": pd.DataFrame,
            "longshort": dict,
            "quantile_cumret": pd.DataFrame,
            "monotonicity": dict,
        }
    """
    fwd = fwd_returns_map.get(holding_period)
    if fwd is None:
        fwd = next(iter(fwd_returns_map.values())) if fwd_returns_map else pd.DataFrame()

    q_returns = compute_quantile_returns(panel, fwd, n_quantiles=n_quantiles)

    return {
        "quantile_returns": q_returns,
        "quantile_summary": compute_quantile_summary(q_returns),
        "longshort": compute_longshort(q_returns),
        "quantile_cumret": compute_quantile_cumret(q_returns),
        "monotonicity": compute_monotonicity(q_returns),
    }
