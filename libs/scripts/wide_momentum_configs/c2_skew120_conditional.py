"""C2 因子 120 日滚动偏度条件测试 — 3 组回测配置。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.c2_skew120_conditional

因子清单:
    C2      `MaxAdverseExcursion_40__zscore_120__neg`   最大不利偏移 zscore（负向取反）
    C2_skew `...__neg__rolling_skew_120`                C2 因子的 120 日滚动偏度
    C2_skew>0 条件因子  C2 仅在 C2_skew > 0 时生效
    C2_skew<0 条件因子  C2 仅在 C2_skew < 0 时生效

组定义:
    C2 原始因子、C2_skew>0 条件、C2_skew<0 条件，各一组。
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    RankFilter,
    ThresholdFilter,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    make_rank_filters,
    score_proportional_allocator,
)
from factors.volatility import Volatility
from factors.distribution_family import MaxAdverseExcursion
from factors.meta_factor import (
    ConditionalFactor,
    NegateFactor,
    TransformFactor,
)


# ====================================================================
# 因子定义
# ====================================================================

# ── C2: MAE_40__zscore_120（负向，取反）──
mae_40 = MaxAdverseExcursion(window=40)
mae_zscore = TransformFactor(dependency=mae_40, transform="zscore", window=120)
c2_factor = NegateFactor(mae_zscore)

# ── C2_skew: C2 的 120 日滚动偏度（三阶矩，窗口 120）──
c2_skew = TransformFactor(dependency=c2_factor, transform="rolling_skew", window=120)

# ── 条件因子: C2_skew > 0 时 C2 生效（否则 NaN，标的不参与排名）──
c2_skew_gt0 = ConditionalFactor(
    signal=c2_factor,
    condition=c2_skew,
    op="gt",
    threshold=0.0,
    false_value="nan",
)

# ── 条件因子: C2_skew < 0 时 C2 生效（否则 NaN，标的不参与排名）──
c2_skew_lt0 = ConditionalFactor(
    signal=c2_factor,
    condition=c2_skew,
    op="lt",
    threshold=0.0,
    false_value="nan",
)


# ====================================================================
# 共享管道因子（排名因子及条件依赖由引擎自动预计算，无需列出）
# ====================================================================
vol20 = Volatility(window=20)
vol120 = Volatility(window=120)
SHARED_PIPELINE: tuple = (vol20, vol120,)


# ====================================================================
# 过滤器定义（暂不使用，保留以备后续扩展）
# ====================================================================
NO_FILTERS = ()
RANK_FILTERS_120: tuple[RankFilter, ...] = make_rank_filters(vol120, (0.0, 0.1, 0.2, 0.3))

# ====================================================================
# 组定义: 3 组（C2 原始 + 两个偏度条件因子，无过滤器）
# ====================================================================
# (label, ranking_factor, builtin_filters)
GROUPS: list[tuple] = [
    (f"C2_{rf}",        c2_factor,  NO_FILTERS, (rf,)) for rf in RANK_FILTERS_120] + [
    (f"C2_skew120_gt0_{rf}",   c2_skew_gt0, NO_FILTERS, (rf,)) for rf in RANK_FILTERS_120] + [
    (f"C2_skew120_lt0_{rf}",   c2_skew_lt0, NO_FILTERS, (rf,)) for rf in RANK_FILTERS_120
]


# ====================================================================
# Grid Search 参数（按模板默认）
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1, 2, 5, 10,)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 20)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (False,)
GRID_HOLD_OVERLAP: tuple[bool, ...] = (False,)


# ====================================================================
# 权重分配器（按模板默认）
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
# 执行参数（按模板默认）
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
TITLE: str              = "宽动量基线回测 — C2 因子 120日滚动偏度条件测试"
BASENAME_TAG: str       = "c2_skew120_conditional"
START_DATE: str         = "2020-01-01"
END_DATE: str           = "2026-07-17"
START_DATES: tuple[str, ...] | None = None
END_DATES: tuple[str, ...] | None = None
SYMBOLS: tuple[str, ...] | None = None
MAX_WORKERS: int | None = None
CROSS_GROUP_PARALLEL: bool = True
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
