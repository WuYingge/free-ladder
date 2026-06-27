"""
因子预测力分析 (Factor Predictive Power Analysis)

Layer 2：回答"这个因子能否预测未来收益"。
这是单因子评估中最核心的部分。

包含五项分析:
    2.1 Rank IC — 每日截面 Spearman 秩相关系数（因子值 × 未来收益）
    2.2 Pearson IC — 每日截面 Pearson 线性相关系数
    2.3 IC 衰减曲线 — 不同持仓期 (5/10/20/60日) 的 IC 变化
    2.4 滚动 IC 稳定性 — 固定窗口滚动 IC 均值时序
    2.5 参数敏感度网格 — 遍历参数 × 持仓期，生成 IC 热力图

IC 尺度感参考（设计文档 2.1 节）:
    > 0.10  → 顶级
    0.05~0.10 → 靠谱
    0.02~0.05 → 可用
    < 0.02 → 边际
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from factor_analysis.panel import FactorPanel
from factor_analysis.forward_returns import compute_forward_returns


# ── 工具函数 ─────────────────────────────────────────────────────────────────


def _daily_ic(
    factor_values: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """计算每日截面的 IC（信息系数）序列。

    Parameters
    ----------
    factor_values: date × symbol 因子值矩阵。
    fwd_returns: date × symbol 未来收益率矩阵。
    method: "spearman" → Rank IC, "pearson" → Pearson IC。

    Returns
    -------
    pd.Series
        index=日期, values=当日 IC 值。当日截面有效样本 < 5 时返回 NaN。
    """
    # 只取两矩阵共有的日期和标的
    common_dates = factor_values.index.intersection(fwd_returns.index)
    common_symbols = factor_values.columns.intersection(fwd_returns.columns)

    if len(common_dates) == 0 or len(common_symbols) == 0:
        return pd.Series(dtype=float)

    fv = factor_values.loc[common_dates, common_symbols]
    fwd = fwd_returns.loc[common_dates, common_symbols]

    ic_values: list[float] = []
    for date in common_dates:
        fv_row = fv.loc[date].astype(float)
        fwd_row = fwd.loc[date].astype(float)

        # 取两行均非 NaN 的标的
        mask = fv_row.notna() & fwd_row.notna()
        n_valid = mask.sum()

        if n_valid < 5:
            ic_values.append(float("nan"))
            continue

        try:
            if method == "spearman":
                r, _ = spearmanr(fv_row[mask], fwd_row[mask])
            else:
                r, _ = pearsonr(fv_row[mask], fwd_row[mask])
            ic_values.append(float(r) if not np.isnan(r) else float("nan"))
        except Exception:
            ic_values.append(float("nan"))

    return pd.Series(ic_values, index=common_dates, name=f"IC_{method}")


def _ic_summary(ic_series: pd.Series) -> dict[str, float]:
    """从 IC 序列计算汇总统计量。

    Returns
    -------
    dict
        {mean, std, ir (=mean/std), t_stat, pos_ratio (>0的比例), n_days}
    """
    valid = ic_series.dropna()
    n = len(valid)

    if n < 2:
        return {
            "mean": float("nan"), "std": float("nan"),
            "ir": float("nan"), "t_stat": float("nan"),
            "pos_ratio": float("nan"), "n_days": n,
        }

    mean_ic = float(valid.mean())
    std_ic = float(valid.std(ddof=1))

    # IC Information Ratio = mean / std（IC 的信噪比，相当于 t-stat 的单调变换）
    ir = mean_ic / std_ic if std_ic > 0 else float("nan")

    # t statistic = mean / (std / sqrt(n))
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else float("nan")

    # IC > 0 的比例：即因子方向正确的天数占比
    pos_ratio = float((valid > 0).sum() / n)

    return {
        "mean": mean_ic,
        "std": std_ic,
        "ir": ir,
        "t_stat": t_stat,
        "pos_ratio": pos_ratio,
        "n_days": n,
    }


# ── 2.1 Rank IC ─────────────────────────────────────────────────────────────


def compute_rank_ic(panel: FactorPanel, fwd_returns: pd.DataFrame) -> dict[str, Any]:
    """计算 Rank IC（截面 Spearman 秩相关系数）。

    对每日截面，计算因子值与未来 N 日收益的 Spearman rank correlation。

    Rank IC 对异常值稳健，不需要正态假设，是截面因子评估的核心指标。
    """
    ic_series = _daily_ic(panel.factor_values, fwd_returns, method="spearman")
    summary = _ic_summary(ic_series)
    return {"ic_series": ic_series, "summary": summary}


# ── 2.2 Pearson IC ──────────────────────────────────────────────────────────


def compute_pearson_ic(panel: FactorPanel, fwd_returns: pd.DataFrame) -> dict[str, Any]:
    """计算 Pearson IC（截面线性相关系数）。

    Pearson IC 对线性关系敏感。与 Rank IC 对比:
    - Rank IC >> Pearson IC → 因子收益由少数极端值驱动（肥尾效应）
    - 两者接近 → 因子收益分布较均匀，无极端值主导
    """
    ic_series = _daily_ic(panel.factor_values, fwd_returns, method="pearson")
    summary = _ic_summary(ic_series)
    return {"ic_series": ic_series, "summary": summary}


# ── 2.3 IC 衰减曲线 ────────────────────────────────────────────────────────


def compute_ic_decay(
    panel: FactorPanel,
    fwd_returns_map: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """计算 IC 在不同持仓期上的衰减曲线。

    同一批因子值，分别对 T+5/10/20/60 日收益计算 Rank IC，
    观察 IC 随持仓期延长如何变化。

    Parameters
    ----------
    panel: 截面因子面板。
    fwd_returns_map: {period: fwd_returns_df}，来自 compute_forward_returns。

    Returns
    -------
    pd.DataFrame
        columns = [period, ic_mean, ic_std, ic_ir]
    """
    records: list[dict[str, float]] = []

    for period, fwd in sorted(fwd_returns_map.items()):
        ic_series = _daily_ic(panel.factor_values, fwd, method="spearman")
        summary = _ic_summary(ic_series)
        records.append({
            "period": period,
            "ic_mean": summary["mean"],
            "ic_std": summary["std"],
            "ic_ir": summary["ir"],
        })

    return pd.DataFrame(records)


# ── 2.4 滚动 IC 稳定性 ─────────────────────────────────────────────────────


def compute_rolling_ic(
    panel: FactorPanel,
    fwd_returns: pd.DataFrame,
    window: int = 120,
) -> pd.DataFrame:
    """计算滚动窗口 IC 均值时序。

    用固定窗口（如 120 日 ≈ 半年）计算滚动 IC 均值，
    画成时序图可以观察 IC 在什么时间段有效。
    "IC 在什么时间段有效"比"IC 均值多少"更重要。

    Parameters
    ----------
    panel: 截面因子面板。
    fwd_returns: 未来收益率矩阵（通常用 20 日持仓期）。
    window: 滚动窗口（交易日）。120 ≈ 半年，60 ≈ 一个季度。

    Returns
    -------
    pd.DataFrame
        index=日期, columns=[ic_raw, ic_rolling_mean]。
        ic_raw = 每日原始 IC 值，ic_rolling_mean = 滚动窗口均值。
    """
    ic_series = _daily_ic(panel.factor_values, fwd_returns, method="spearman")
    rolling_mean = ic_series.rolling(window=window, min_periods=max(20, window // 2)).mean()

    result = pd.DataFrame({
        "ic_raw": ic_series,
        "ic_rolling_mean": rolling_mean,
    }, index=ic_series.index)
    return result


# ── 2.5 参数敏感度网格 ─────────────────────────────────────────────────────


def compute_param_grid(
    factor_cls: type,
    param_grid: dict[str, list],
    symbols: list[str],
    forward_periods: tuple[int, ...],
    *,
    min_bars: int = 252,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """遍历因子参数空间，计算每个参数组合 × 每个持仓期的 IC。

    这是防范过拟合的关键分析：必须在全参数空间里看 IC 分布的稳健性，
    而非只看最优参数。如果只在某个窄参数区间有效，因子可能不可靠。

    Parameters
    ----------
    factor_cls:
        因子类（如 PriceReturn），不是实例。框架会遍历参数组合实例化。
    param_grid:
        参数扫描范围。{"window": [20, 60, 120], "skip_recent": [0, 20]}
        笛卡尔积遍历所有组合。
    symbols:
        标的列表。
    forward_periods:
        持仓期列表。
    min_bars:
        每个参数组合的面板最少 bar 数。
    max_workers:
        多进程 worker 数。注意：内层还有因子计算的多进程，
        此处建议单进程（串行遍历参数组合，每组合内用多进程计算面板）。

    Returns
    -------
    dict
        {
            "records": list[dict],  # 每条记录 = {参数名: 值, ..., period: N, ic_mean: ...}
            "matrix": pd.DataFrame,  # 透视表：行=参数组合, 列=持仓期, 值=ic_mean
            "best_params": dict,     # 最优参数组合及对应 IC
        }
    """
    import itertools
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import traceback

    from factor_analysis.panel import build_factor_panel

    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    records: list[dict[str, Any]] = []

    # 串行遍历参数组合（每组合内部 panel 构建用多进程，避免嵌套多进程冲突）
    for combo in combinations:
        params = dict(zip(param_names, combo))
        try:
            factor = factor_cls(**params)
            panel = build_factor_panel(
                factor, symbols, min_bars=min_bars, max_workers=max_workers,
            )
            fwd_map = compute_forward_returns(panel.close_prices, forward_periods)

            for period in forward_periods:
                fwd = fwd_map.get(period)
                if fwd is None:
                    continue
                ic_summary = _ic_summary(
                    _daily_ic(panel.factor_values, fwd, method="spearman")
                )
                record = {**params, "period": period, "ic_mean": ic_summary["mean"]}
                records.append(record)
        except Exception:
            # 记录失败的参数组合
            records.append({**params, "period": -1, "ic_mean": float("nan"), "error": traceback.format_exc(limit=2)})

    records_df = pd.DataFrame(records)

    # 透视表：行=参数标签，列=period，值=ic_mean
    if not records_df.empty and "ic_mean" in records_df.columns:
        # 构造参数组合标签（如 "window=60,skip=20"）
        def _make_label(row):
            parts = [f"{k}={v}" for k, v in row.items() if k in param_names]
            return ", ".join(parts)

        records_df["param_label"] = records_df.apply(_make_label, axis=1)
        try:
            matrix = records_df.pivot_table(
                index="param_label", columns="period", values="ic_mean",
            )
        except Exception:
            matrix = pd.DataFrame()
    else:
        matrix = pd.DataFrame()

    # 找最优组合
    best = {}
    if not records_df.empty and "ic_mean" in records_df.columns:
        valid = records_df.dropna(subset=["ic_mean"])
        if not valid.empty:
            best_idx = valid["ic_mean"].idxmax()
            best_row = valid.loc[best_idx]
            best = {k: best_row[k] for k in param_names + ("period", "ic_mean")}

    return {
        "records": records,
        "matrix": matrix,
        "best_params": best,
    }


# ── 汇总 ─────────────────────────────────────────────────────────────────────


def run_predictive_analysis(
    panel: FactorPanel,
    fwd_returns_map: dict[int, pd.DataFrame],
    *,
    rolling_ic_window: int = 120,
    rolling_ic_period: int = 20,
    param_grid: dict[str, list] | None = None,
    factor_cls: type | None = None,
    symbols: list[str] | None = None,
    forward_periods: tuple[int, ...] = (5, 10, 20, 60),
    min_bars: int = 252,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """一次性跑完 Layer 2 全部分析。

    与旧版不同：Rank IC 和 Pearson IC 现在遍历 fwd_returns_map 中所有持仓期
    （如 5/10/20/60 日），而非仅取默认 20 日。IC 衰减曲线直接从 rank_ic
    结果中提取，避免重复计算。

    Parameters
    ----------
    panel: 截面因子面板。
    fwd_returns_map: {period: fwd_returns_df}。
    rolling_ic_window: 滚动 IC 窗口（交易日）。
    rolling_ic_period: 已废弃。滚动 IC 现在遍历所有持仓期。
    param_grid: 可选，参数网格扫描范围。
    factor_cls: 因子类（param_grid 非空时必填）。
    symbols: 标的列表（param_grid 非空时必填）。
    forward_periods: 持仓期列表（param_grid 非空时需要）。

    Returns
    -------
    dict
        {
            "rank_ic": {period: {"ic_series": pd.Series, "summary": dict}, ...},
            "pearson_ic": {period: {"ic_series": pd.Series, "summary": dict}, ...},
            "ic_decay": pd.DataFrame,   # 从 rank_ic 提取，列=[period, ic_mean, ic_std, ic_ir]
            "rolling_ic": {period: pd.DataFrame, ...},
            "param_grid": dict | None,
        }
    """
    # ── Rank IC / Pearson IC：遍历所有持仓期 ──────────────────────────────
    rank_ic: dict[int, dict[str, Any]] = {}
    pearson_ic: dict[int, dict[str, Any]] = {}
    for period in sorted(fwd_returns_map.keys()):
        fwd = fwd_returns_map[period]
        rank_ic[period] = compute_rank_ic(panel, fwd)
        pearson_ic[period] = compute_pearson_ic(panel, fwd)

    # ── IC 衰减曲线：直接从 rank_ic 提取，避免重复计算 ───────────────────
    decay_records: list[dict[str, float]] = []
    for period in sorted(rank_ic.keys()):
        s = rank_ic[period]["summary"]
        decay_records.append({
            "period": period,
            "ic_mean": s["mean"],
            "ic_std": s["std"],
            "ic_ir": s["ir"],
        })
    ic_decay = pd.DataFrame(decay_records)

    # ── 滚动 IC：每个持仓期各算一份 ────────────────────────────────────
    rolling_ic: dict[int, pd.DataFrame] = {}
    for period in sorted(fwd_returns_map.keys()):
        fwd = fwd_returns_map[period]
        rolling_ic[period] = compute_rolling_ic(panel, fwd, window=rolling_ic_window)

    # ── 参数网格（可选） ─────────────────────────────────────────────────
    pg_result: dict | None = None
    if param_grid is not None and factor_cls is not None and symbols is not None:
        pg_result = compute_param_grid(
            factor_cls=factor_cls,
            param_grid=param_grid,
            symbols=symbols,
            forward_periods=forward_periods,
            min_bars=min_bars,
            max_workers=max_workers,
        )

    return {
        "rank_ic": rank_ic,
        "pearson_ic": pearson_ic,
        "ic_decay": ic_decay,
        "rolling_ic": rolling_ic,
        "param_grid": pg_result,
    }
