"""
因子质量分析 (Factor Quality Analysis)

Layer 1：回答"这个因子本身是否健康可用"。
所有分析均不涉及未来收益，仅评估因子自身的数据质量特征。

包含四项检查:
    1. 覆盖率 — 每日有效因子值的标的比例
    2. 分布特征 — 全样本分布统计量 + 分位数带时序
    3. 缺失模式 — 按成交额/上市时长分组的缺失率
    4. 自相关 — 因子值 rank autocorrelation 衰减曲线

注意:
    - 所有分析基于 FactorPanel.factor_values (date × symbol) 进行。
    - 自相关对截面因子（如 PriceReturn）意义有限，对时序因子
      （如 RSRS zscore、TrendR2 r2）更有参考价值。
      —— 设计文档 1.4 节对此有明确说明。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from factor_analysis.panel import FactorPanel


# ── 1.1 覆盖率 ──────────────────────────────────────────────────────────────


def compute_coverage(panel: FactorPanel) -> pd.Series:
    """计算每日因子值覆盖率。

    覆盖率 = 当日有有效因子值的标的数 / 总标的数。

    Parameters
    ----------
    panel: 截面因子面板。

    Returns
    -------
    pd.Series
        index=日期, values=覆盖率比例 (0.0 ~ 1.0)。
    """
    n_symbols = panel.n_symbols
    if n_symbols == 0:
        return pd.Series(dtype=float)
    # 每日有效标的数 / 总标的数
    coverage = panel.factor_values.notna().sum(axis=1) / n_symbols
    coverage.name = "coverage"
    return coverage


# ── 1.2 分布特征 ────────────────────────────────────────────────────────────


def compute_distribution_stats(panel: FactorPanel) -> dict[str, float]:
    """计算全样本因子值的分布统计量。

    将整个 factor_values 矩阵的所有非 NaN 值视为一个总体，
    计算其均值、标准差、偏度、峰度及各分位数。

    Parameters
    ----------
    panel: 截面因子面板。

    Returns
    -------
    dict
        {mean, std, skewness, kurtosis, min, p01, p05, p25, p50, p75, p95, p99, max}
    """
    # 全样本向量化展平
    values = panel.factor_values.values.ravel()
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return {}

    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)),
        "skewness": float(scipy_stats.skew(valid)),
        "kurtosis": float(scipy_stats.kurtosis(valid, fisher=True)),
        "min": float(np.min(valid)),
        "p01": float(np.percentile(valid, 1)),
        "p05": float(np.percentile(valid, 5)),
        "p25": float(np.percentile(valid, 25)),
        "p50": float(np.percentile(valid, 50)),
        "p75": float(np.percentile(valid, 75)),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
        "max": float(np.max(valid)),
    }


def compute_daily_percentiles(
    panel: FactorPanel,
    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
) -> pd.DataFrame:
    """计算每日截面的因子值分位数时序。

    Parameters
    ----------
    panel: 截面因子面板。
    percentiles: 分位点列表，如 (5, 25, 50, 75, 95)。

    Returns
    -------
    pd.DataFrame
        index=日期, columns=P5/P25/P50/P75/P95, 值为当日截面的分位数。
    """
    fv = panel.factor_values
    if fv.empty:
        return pd.DataFrame()

    # 使用 numpy percentile 逐行计算（避免每行调一次 describe）
    arr = fv.values  # (n_dates, n_symbols)
    result_cols: dict[str, list[float]] = {}

    for p in percentiles:
        col_name = f"P{int(p)}"
        # nanpercentile 沿 axis=1 计算每行的分位数
        pct_values = np.nanpercentile(arr, p, axis=1).tolist()
        result_cols[col_name] = pct_values

    df = pd.DataFrame(result_cols, index=fv.index)
    df.index.name = "date"
    return df


# ── 1.3 缺失模式 ────────────────────────────────────────────────────────────


def compute_missing_by_volume_group(
    panel: FactorPanel,
    n_groups: int = 3,
) -> dict[str, float]:
    """按日均成交额分档统计因子缺失率。

    将标的按全历史日均成交额从低到高分为 n_groups 档（等频分档），
    统计每档内因子值的缺失比例。

    Parameters
    ----------
    panel: 截面因子面板（需要 volumes 矩阵）。
    n_groups: 分档数。3 = 低成交额 / 中成交额 / 高成交额。

    Returns
    -------
    dict
        {分组标签: 缺失率}。缺失率 = 该组内 NaN 数 / 该组总单元格数。
    """
    vol = panel.volumes
    fv = panel.factor_values

    if vol.empty or fv.empty:
        return {}

    # 每个标的的日均成交额（剔除 NaN）
    avg_volume = vol.mean(axis=0)
    valid_mask = avg_volume.notna() & (avg_volume > 0)

    if valid_mask.sum() < n_groups:
        return {}

    vol_valid = avg_volume[valid_mask]
    # 按日均成交额等频分档（qcut），标签用中文可读
    try:
        labels = ["低成交额", "中成交额", "高成交额"][:n_groups]
        groups = pd.qcut(vol_valid, q=n_groups, labels=labels)
    except ValueError:
        # 有重复值导致 qcut 失败时降级为 cut
        groups = pd.cut(vol_valid, bins=n_groups, labels=labels)

    result: dict[str, float] = {}
    for label in labels:
        syms_in_group = groups[groups == label].index
        if len(syms_in_group) == 0:
            result[label] = float("nan")
            continue
        group_fv = fv[syms_in_group.intersection(fv.columns)]
        missing_rate = float(group_fv.isna().sum().sum() / group_fv.size)
        result[label] = missing_rate

    return result


def compute_missing_by_bar_count_group(
    panel: FactorPanel,
    n_groups: int = 3,
) -> dict[str, float]:
    """按 bar_count（有效交易日数）分档统计因子缺失率。

    bar_count 可作为"上市时长"的代理变量。
    上市时间短的 ETF 数据不完整，因子缺失率可能更高。

    Parameters
    ----------
    panel: 截面因子面板（需要 symbol_meta 中的 bar_count）。
    n_groups: 分档数。

    Returns
    -------
    dict
        {分组标签: 缺失率}。
    """
    meta = panel.symbol_meta
    fv = panel.factor_values

    if meta.empty or fv.empty or "bar_count" not in meta.columns:
        return {}

    bar_counts = meta["bar_count"]
    valid_mask = bar_counts.notna() & (bar_counts > 0)

    if valid_mask.sum() < n_groups:
        return {}

    bc_valid = bar_counts[valid_mask]
    labels = ["短上市", "中上市", "长上市"][:n_groups]
    try:
        groups = pd.qcut(bc_valid, q=n_groups, labels=labels)
    except ValueError:
        groups = pd.cut(bc_valid, bins=n_groups, labels=labels)

    result: dict[str, float] = {}
    for label in labels:
        syms_in_group = groups[groups == label].index
        if len(syms_in_group) == 0:
            result[label] = float("nan")
            continue
        group_fv = fv[syms_in_group.intersection(fv.columns)]
        missing_rate = float(group_fv.isna().sum().sum() / group_fv.size)
        result[label] = missing_rate

    return result


# ── 1.4 自相关 ──────────────────────────────────────────────────────────────


def compute_autocorr(
    panel: FactorPanel,
    lags: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """计算因子值的 rank autocorrelation 衰减曲线。

    对每个 symbol 单独计算 rank autocorr(lag)，取所有 symbol 的截面均值。

    Rank autocorr = Spearman 秩相关系数(factor_t, factor_{t-lag})。
    值域 [0, 1]，接近 1 = 高自相关 = 因子值变化慢 = 低换手。

    Parameters
    ----------
    panel: 截面因子面板。
    lags: 滞后天数。1 = 日频、5 = 周频、10 = 双周、20 = 月频。

    Returns
    -------
    pd.DataFrame
        columns = [lag, mean_autocorr, median_autocorr, std_autocorr]
    """
    fv = panel.factor_values
    if fv.empty:
        return pd.DataFrame(columns=["lag", "mean_autocorr", "median_autocorr", "std_autocorr"])

    records: list[dict[str, float]] = []

    for lag in lags:
        # 每个 symbol 计算 rank autocorr
        ac_values: list[float] = []
        for sym in fv.columns:
            series = fv[sym].dropna()
            if len(series) <= lag + 10:
                continue
            # Spearman 秩相关（对异常值稳健）
            corr = series.rolling(window=2).apply(
                lambda _: np.nan, raw=False
            )
            # 计算 factor_t 与 factor_{t-lag} 的 Spearman 秩相关系数
            aligned_current = series.iloc[lag:]
            aligned_lagged = series.shift(lag).iloc[lag:]
            mask = aligned_current.notna() & aligned_lagged.notna()
            if mask.sum() < 20:  # 至少需要 20 对有效观测
                continue
            from scipy.stats import spearmanr
            r, _ = spearmanr(aligned_current[mask], aligned_lagged[mask])
            if not np.isnan(r):
                ac_values.append(r)

        if ac_values:
            mean_ac = float(np.mean(ac_values))
            median_ac = float(np.median(ac_values))
            std_ac = float(np.std(ac_values, ddof=1))
        else:
            mean_ac = float("nan")
            median_ac = float("nan")
            std_ac = float("nan")

        records.append({
            "lag": lag,
            "mean_autocorr": mean_ac,
            "median_autocorr": median_ac,
            "std_autocorr": std_ac,
        })

    return pd.DataFrame(records)


# ── 汇总 ─────────────────────────────────────────────────────────────────────


def run_quality_analysis(panel: FactorPanel) -> dict[str, Any]:
    """一次性跑完 Layer 1 全部检查，返回结构化的结果字典。

    Parameters
    ----------
    panel: 截面因子面板。

    Returns
    -------
    dict
        {
            "coverage": pd.Series,
            "distribution_stats": dict,
            "daily_percentiles": pd.DataFrame,
            "missing_by_volume": dict,
            "missing_by_bar_count": dict,
            "autocorr": pd.DataFrame,
        }
    """
    return {
        "coverage": compute_coverage(panel),
        "distribution_stats": compute_distribution_stats(panel),
        "daily_percentiles": compute_daily_percentiles(panel),
        "missing_by_volume": compute_missing_by_volume_group(panel),
        "missing_by_bar_count": compute_missing_by_bar_count_group(panel),
        "autocorr": compute_autocorr(panel),
    }
