"""Utilities for single-factor single-target timing ``summary_df`` comparison.

The public API is intentionally notebook-friendly:

1) ``compare_timing_summary_dfs`` returns structured comparison outputs.
2) ``build_timing_comparison_report`` turns those outputs into a readable report.

Typical usage
-------------
::

    from backtesting.summary_compare import compare_timing_summary_dfs, build_timing_comparison_report

    comparison = compare_timing_summary_dfs(
        {
            "rsrs": rsrs_summary_df,
            "new_high": new_high_summary_df,
        }
    )
    print(build_timing_comparison_report(comparison))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ComparisonMetricSpec:
    """Metric config used in cross-strategy scoring.

    ``higher_is_better`` controls whether z-score is used directly (True)
    or inverted (False), for example for drawdown.
    """

    name: str
    weight: float
    higher_is_better: bool = True


TIMING_SINGLE_FACTOR_DEFAULT_METRICS: tuple[ComparisonMetricSpec, ...] = (
    ComparisonMetricSpec("information_ratio", 0.50, True),
    ComparisonMetricSpec("excess_cumulative_return", 0.35, True),
    ComparisonMetricSpec("max_drawdown_pct", 0.10, False),
    ComparisonMetricSpec("win_rate", 0.05, True),
)

# Backward-compatible alias for existing imports.
DEFAULT_METRICS = TIMING_SINGLE_FACTOR_DEFAULT_METRICS


def _safe_numeric_mean(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _safe_numeric_median(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _build_strategy_aggregate_row(strategy_name: str, summary_df: pd.DataFrame) -> dict:
    valid_df = summary_df.copy()
    has_error_col = "error" in valid_df.columns
    if has_error_col:
        valid_df = valid_df[valid_df["error"].isna()]

    total_rows = int(len(summary_df))
    valid_rows = int(len(valid_df))
    coverage = (valid_rows / total_rows) if total_rows > 0 else 0.0

    row = {
        "strategy": strategy_name,
        "symbols_total": total_rows,
        "symbols_valid": valid_rows,
        "coverage": coverage,
    }

    metric_names = {
        "information_ratio",
        "benchmark_annualised_return",
        "excess_cumulative_return",
        "max_drawdown_pct",
        "win_rate",
    }
    for metric in metric_names:
        if metric not in valid_df.columns:
            row[f"{metric}_mean"] = None
            row[f"{metric}_median"] = None
            continue
        row[f"{metric}_mean"] = _safe_numeric_mean(valid_df[metric])
        row[f"{metric}_median"] = _safe_numeric_median(valid_df[metric])

    return row


def _normalise_weights(metrics: tuple[ComparisonMetricSpec, ...]) -> tuple[ComparisonMetricSpec, ...]:
    total_weight = sum(max(0.0, float(m.weight)) for m in metrics)
    if total_weight <= 0.0:
        raise ValueError("At least one metric weight must be > 0.")

    return tuple(
        ComparisonMetricSpec(
            name=m.name,
            weight=max(0.0, float(m.weight)) / total_weight,
            higher_is_better=m.higher_is_better,
        )
        for m in metrics
    )


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    std = float(valid.std(ddof=0))
    if std == 0.0:
        return pd.Series(0.0, index=series.index)
    mean = float(valid.mean())
    return (s - mean) / std


def compare_timing_summary_dfs(
    strategy_summaries: dict[str, pd.DataFrame],
    metrics: tuple[ComparisonMetricSpec, ...] = DEFAULT_METRICS,
) -> dict[str, pd.DataFrame | dict | str]:
    """Compare single-factor single-target timing summary tables and rank them.

    Parameters
    ----------
    strategy_summaries:
        Mapping ``strategy_name -> summary_df``.
    metrics:
        Metric definitions for weighted composite score.

    Returns
    -------
    dict with keys:
    - ``strategy_stats``: per-strategy aggregate metrics table.
    - ``metric_score_table``: z-score based sub-scores by metric.
    - ``ranking``: final ranking sorted by composite score.
    - ``winner``: top strategy name (empty string when no data).
    """
    if not strategy_summaries:
        raise ValueError("strategy_summaries cannot be empty.")

    normalised_metrics = _normalise_weights(metrics)

    aggregate_rows: list[dict] = []
    for strategy_name, summary_df in strategy_summaries.items():
        if not isinstance(summary_df, pd.DataFrame):
            raise TypeError(f"summary for strategy '{strategy_name}' is not a pandas DataFrame.")
        aggregate_rows.append(_build_strategy_aggregate_row(strategy_name, summary_df))

    strategy_stats = pd.DataFrame(aggregate_rows)
    if strategy_stats.empty:
        return {
            "strategy_stats": strategy_stats,
            "metric_score_table": pd.DataFrame(),
            "ranking": pd.DataFrame(),
            "winner": "",
        }

    metric_score_table = strategy_stats[["strategy"]].copy()
    metric_score_table["coverage_bonus"] = _zscore(strategy_stats["coverage"]).fillna(0.0)

    composite = np.zeros(len(strategy_stats), dtype=float)

    for metric in normalised_metrics:
        source_col = f"{metric.name}_mean"
        if source_col not in strategy_stats.columns:
            z = pd.Series(0.0, index=strategy_stats.index)
        else:
            z = _zscore(strategy_stats[source_col]).fillna(0.0)
        if not metric.higher_is_better:
            z = -z

        metric_col = f"score_{metric.name}"
        metric_score_table[metric_col] = z
        composite += z.to_numpy() * metric.weight

    # Encourage robust runs with higher valid coverage; bonus kept small.
    composite += metric_score_table["coverage_bonus"].to_numpy() * 0.05

    ranking = strategy_stats[["strategy", "symbols_total", "symbols_valid", "coverage"]].copy()
    ranking["composite_score"] = composite
    ranking = ranking.sort_values(
        ["composite_score", "coverage", "symbols_valid"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1

    ranking = ranking.merge(
        strategy_stats,
        on=["strategy", "symbols_total", "symbols_valid", "coverage"],
        how="left",
    )

    winner = str(ranking.iloc[0]["strategy"]) if len(ranking) > 0 else ""

    return {
        "strategy_stats": strategy_stats,
        "metric_score_table": metric_score_table,
        "ranking": ranking,
        "winner": winner,
    }


def build_timing_comparison_report(comparison_result: dict[str, pd.DataFrame | dict | str]) -> str:
    """Render a concise report for single-factor single-target timing comparison."""
    ranking = comparison_result.get("ranking")
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        return "没有可比较的数据。"

    winner = str(comparison_result.get("winner", ""))
    top = ranking.iloc[0]

    lines = [
        "单因子单标的择时对比结论",
        f"- 最优策略: {winner}",
        f"- 综合得分: {float(top['composite_score']):.4f}",
        f"- 有效覆盖: {int(top['symbols_valid'])}/{int(top['symbols_total'])} ({float(top['coverage']) * 100:.1f}%)",
    ]

    lines.append("")
    lines.append("排名明细")
    for _, row in ranking.iterrows():
        strategy = str(row.get("strategy", ""))
        score = float(row.get("composite_score", 0.0))
        ir = row.get("information_ratio_mean")
        excess = row.get("excess_cumulative_return_mean")
        mdd = row.get("max_drawdown_pct_mean")
        lines.append(
            "- "
            f"#{int(row.get('rank', 0))} {strategy}: score={score:.4f}, "
            f"信息比率={_fmt(ir)}, 超额收益={_fmt_pct(excess)}, 回撤={_fmt_pct(mdd, pct_is_100_scale=True)}"
        )

    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(f):
        return "NA"
    return f"{f:.4f}"


def _fmt_pct(value: Optional[float], *, pct_is_100_scale: bool = False) -> str:
    if value is None:
        return "NA"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(f):
        return "NA"
    if pct_is_100_scale:
        return f"{f:.2f}%"
    return f"{f * 100:.2f}%"


__all__ = [
    "ComparisonMetricSpec",
    "TIMING_SINGLE_FACTOR_DEFAULT_METRICS",
    "DEFAULT_METRICS",
    "compare_timing_summary_dfs",
    "build_timing_comparison_report",
    "compare_summary_dfs",
    "build_comparison_text_report",
]


def compare_summary_dfs(
    strategy_summaries: dict[str, pd.DataFrame],
    metrics: tuple[ComparisonMetricSpec, ...] = DEFAULT_METRICS,
) -> dict[str, pd.DataFrame | dict | str]:
    """Backward-compatible wrapper. Prefer ``compare_timing_summary_dfs``."""
    return compare_timing_summary_dfs(strategy_summaries=strategy_summaries, metrics=metrics)


def build_comparison_text_report(comparison_result: dict[str, pd.DataFrame | dict | str]) -> str:
    """Backward-compatible wrapper. Prefer ``build_timing_comparison_report``."""
    return build_timing_comparison_report(comparison_result)
