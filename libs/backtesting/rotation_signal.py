"""ETF 轮动信号工厂模块。

使用方式：
    from backtesting.rotation_signal import make_rotation_signal

    signal_fn = make_rotation_signal(
        score_cols=[("PriceReturn_60", 1.0), ("PriceReturn_20", 0.5)],
        top_n=8,
        trend_filter_col="NewHigh",        # 值 > 0 时才入场，None 表示不过滤
        trend_filter_threshold=0.0,
        weighting="equal",
        min_score=None,
    )

    summary_df, errors, equity_curves = run_portfolio_backtest_batch(
        symbol_feed_map=symbol_feed_map,
        strategy_callable=signal_fn,
        config=PortfolioBatchConfig(...),
    )
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import pandas as pd

# WeightSignalFunction: (snapshot_df, context) -> dict[symbol, weight]
WeightSignalFunction = Callable[[pd.DataFrame, dict[str, Any]], dict[str, float]]


def make_rotation_signal(
    score_cols: list[tuple[str, float]],
    top_n: int = 10,
    trend_filter_col: Optional[str] = None,
    trend_filter_threshold: float = 0.0,
    weighting: Literal["equal", "score"] = "equal",
    min_score: Optional[float] = None,
) -> WeightSignalFunction:
    """返回符合 WeightSignalFunction 签名的轮动信号函数。

    Args:
        score_cols: 因子列名与对应权重的列表，例如
            [("PriceReturn_60", 1.0), ("PriceReturn_20", 0.5)]。
            每列先做 rank 百分位归一化（pct=True），再按权重加权求和得到
            composite_score，避免不同因子量纲差异干扰排名。
        top_n: 每次调仓最多持有的 ETF 数量。
        trend_filter_col: 趋势过滤列名。如果指定，则只有该列值
            > trend_filter_threshold 的标的才能入场。典型用法：
            传入 "NewHigh" 列并设置 threshold=0 以过滤下跌标的。
            设为 None 表示不做趋势过滤。
        trend_filter_threshold: 趋势过滤阈值，默认 0.0。
        weighting: 权重分配方式。
            "equal"  - 选中标的等权分配（推荐，稳健）；
            "score"  - 按 composite_score 比例分配（动量强的给更多仓位）。
        min_score: 合成得分的最低阈值。低于该值的标的即便排名靠前也不入场。
            None 表示不设阈值。

    Returns:
        signal_fn(snapshot, context) -> dict[str, float]
            snapshot: pd.DataFrame，index 为标的代码，columns 包含因子列。
            context: dict，包含 datetime / bar_index / current_weights 等。
    """

    def _signal_fn(snapshot: pd.DataFrame, context: dict[str, Any]) -> dict[str, float]:
        if snapshot.empty:
            return {}

        # ------------------------------------------------------------------
        # 1. 提取需要的得分列，缺失的直接跳过（不强制报错，允许因子未计算）
        # ------------------------------------------------------------------
        available_cols = [col for col, _ in score_cols if col in snapshot.columns]
        if not available_cols:
            return {}

        # ------------------------------------------------------------------
        # 2. 对每列做横截面 rank 百分位归一化 (0~1)，再加权合成
        # ------------------------------------------------------------------
        n = len(snapshot)
        composite = pd.Series(0.0, index=snapshot.index)
        total_weight = 0.0
        for col, w in score_cols:
            if col not in snapshot.columns:
                continue
            col_series = snapshot[col]
            valid_mask = col_series.notna()
            if valid_mask.sum() < 2:
                # 有效数据不足，跳过该列
                continue
            ranked = col_series.rank(pct=True, na_option="bottom")
            composite += ranked * w
            total_weight += w

        if total_weight == 0.0:
            return {}

        composite /= total_weight

        # ------------------------------------------------------------------
        # 3. 趋势过滤
        # ------------------------------------------------------------------
        candidate_mask = pd.Series(True, index=snapshot.index)
        if trend_filter_col is not None and trend_filter_col in snapshot.columns:
            candidate_mask = snapshot[trend_filter_col] > trend_filter_threshold

        # ------------------------------------------------------------------
        # 4. 最低得分过滤
        # ------------------------------------------------------------------
        if min_score is not None:
            candidate_mask = candidate_mask & (composite >= min_score)

        candidates = composite[candidate_mask].dropna()
        if candidates.empty:
            return {}

        # ------------------------------------------------------------------
        # 5. 按得分排序，取 top_n
        # ------------------------------------------------------------------
        selected = candidates.nlargest(top_n)

        # ------------------------------------------------------------------
        # 6. 分配权重
        # ------------------------------------------------------------------
        if weighting == "score":
            score_sum = selected.sum()
            if score_sum <= 0.0:
                weights = {sym: 1.0 / len(selected) for sym in selected.index}
            else:
                weights = {sym: float(score) / score_sum for sym, score in selected.items()}
        else:
            w_each = 1.0 / len(selected)
            weights = {sym: w_each for sym in selected.index}

        return weights

    # 方便调试时打印
    _signal_fn.__name__ = (
        f"rotation_signal(top_n={top_n}, weighting={weighting}, "
        f"filter={trend_filter_col})"
    )
    return _signal_fn
