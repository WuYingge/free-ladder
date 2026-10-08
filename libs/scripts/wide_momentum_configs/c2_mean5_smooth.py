"""C2.mean(5) vs C2 基线 对比回测配置（489 池，同一口径）。

背景
----
狗子要求做一个 C2.mean(5)：对 C2 因子输出做 5 日简单移动平均（rolling_mean(C2, 5)），
属"平滑输出"杠杆（类比 2026-08-31 平滑 A/B 中的 C2_out20，只是窗口从 20 缩到 5），
检验更短平滑窗口是否在保持收益的同时降低因子震荡。池子仍为 489 池。

组（2 × 2 = 4 组对比）
----------------------
  C2           —— 基线 -(MAE_40 zscore 120)，无过滤（对照组）
  C2_rf_b03_a0 —— 基线 + RankFilter(vol120, b0.3, a0)（实盘基准）
  C2_mean5     —— C2.mean(5)，无过滤
  C2_mean5_rf  —— C2.mean(5) + RankFilter(vol120, b0.3, a0)

口径基准：C2_clean_pool.py（top 1/2/5，rebal 5/10，invvol，withbond，
2020-01-01 → 2026-07-17，全量 489 池）。
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    RankFilter,
    StopRuleSpec,
    factor_threshold_stop,
    equal_weight_allocator,
    score_proportional_allocator,
    make_factor_weighted_allocator,
)
from factors.volatility import Volatility
from factors.distribution_family import MaxAdverseExcursion
from factors.meta_factor import CombineFactor, NegateFactor, TransformFactor


# ====================================================================
# 1. 因子定义
# ====================================================================
mae_40 = MaxAdverseExcursion(window=40)

# ── C2 基线：-(MAE_40 z120) ──
c2 = NegateFactor(
    TransformFactor(dependency=mae_40, transform="zscore", window=120)
)

# ── C2.mean(5)：rolling_mean(C2, 5) 平滑输出 ──
c2_mean5 = TransformFactor(dependency=c2, transform="rolling_mean", window=5)


# ====================================================================
# 2. 共享管道（过滤/权重/rank filter 用到的因子）
# ====================================================================
vol20 = Volatility(window=20)
vol120 = Volatility(window=120, annualize=False)
SHARED_PIPELINE: tuple = (
    vol20,
    vol120,
)


# ====================================================================
# 3. 过滤器 / RankFilter
# ====================================================================
NO_FILTERS: tuple = ()

RF_TARGET: tuple[RankFilter, ...] = (
    RankFilter(vol120, exclude_below_pct=0.3, exclude_above_pct=0.0,
               name="rf_vol120_b0.3_a0.0"),
)


# ====================================================================
# 4. 止损规则（不启用，与 C2_clean_pool 基准一致）
# ====================================================================
SHARED_STOP_RULES: tuple[StopRuleSpec, ...] = ()


# ====================================================================
# 5. 组定义（2 变体 × {plain, rankfilter}）
# ====================================================================
GROUPS: list[tuple] = [
    ("C2",             c2,       NO_FILTERS, ()),
    ("C2_rf_b03_a0",   c2,       NO_FILTERS, RF_TARGET),
    ("C2_mean5",       c2_mean5, NO_FILTERS, ()),
    ("C2_mean5_rf_b03", c2_mean5,NO_FILTERS, RF_TARGET),
]


# ====================================================================
# 6. Grid Search 参数
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1, 2, 5)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (False,)
GRID_HOLD_OVERLAP: tuple[bool, ...] = (False,)


# ====================================================================
# 7. 权重分配器
# ====================================================================
alloc_equal = equal_weight_allocator
alloc_momentum = score_proportional_allocator
alloc_momentum.__name__ = "momentum"

alloc_inv_vol = make_factor_weighted_allocator(vol20.get_output_name(), inverse=True)
alloc_inv_vol.__name__ = "invvol"

WEIGHT_ALLOCATORS: tuple = (
    alloc_inv_vol,
)


# ====================================================================
# 8. 执行参数
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
BASENAME_TAG: str = "c2_mean5_smooth_compare_489pool"
TITLE: str = "宽动量基线回测 — C2.mean(5) 对比 (489pool)"
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-17"
MAX_WORKERS: int | None = 2
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
CROSS_GROUP_PARALLEL: bool = True


# ====================================================================
# 9. 标的池：全量 ETF_INDEX_MAP（489 只）
# ====================================================================
SYMBOLS: tuple[str, ...] | None = None
