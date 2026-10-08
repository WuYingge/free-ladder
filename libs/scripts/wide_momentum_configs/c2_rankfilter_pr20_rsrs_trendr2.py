"""C2 × Volatility RankFilter 与 PR20 × (RSRS + TrendR2) 条件过滤回测配置。

组 1: C2 rank filter
    - 排名因子: C2 = MAE_40__zscore_120__neg（MaxAdverseExcursion(40) zscore(120) 取负）
    - 横截面过滤器: RankFilter(Volatility(120), exclude_below_pct=0.3, exclude_above_pct=0.1)

组 2: PR20 with RSRS + TrendR2 filter
    - 排名因子: PR20 仅在 RSRS(14,600) zscore > 0 且 TrendR2(120) r2 > 0.5 时有效
      （MultiConditionalFactor, AND 逻辑；条件不满足时输出 NaN，标的不参与排名）

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.c2_rankfilter_pr20_rsrs_trendr2
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    RankFilter,
    ThresholdFilter,
    StopRuleSpec,
    factor_threshold_stop,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)
from factors.rsrs import RsrsFactor
from factors.trend_r2 import TrendR2Factor
from factors.distribution_family import MaxAdverseExcursion
from factors.meta_factor import (
    ConditionSpec,
    MultiConditionalFactor,
    NegateFactor,
    TransformFactor,
)
from factors.volatility import Volatility
from factors.price_return import PriceReturn


# ====================================================================
# 1. 因子定义
# ====================================================================

# ── C2: MAE_40__zscore_120__neg ──
mae_40 = MaxAdverseExcursion(window=40)
mae_zscore = TransformFactor(dependency=mae_40, transform="zscore", window=120)
c2_factor = NegateFactor(mae_zscore)

# ── PR20 with RSRS + TrendR2 filter ──
pr_20 = PriceReturn(window=20)
rsrs_zscore = RsrsFactor(regression_window=14, zscore_window=600, output="zscore")
trend_r2_120 = TrendR2Factor(window=120, output="r2")

pr20_rsrs_trendr2 = MultiConditionalFactor(
    signal=pr_20,
    conditions=[
        ConditionSpec(condition=rsrs_zscore, op="gt", threshold=0.0),
        ConditionSpec(condition=trend_r2_120, op="gt", threshold=0.5),
    ],
    logic="and",
)

# ── 权重与 RankFilter 所需因子 ──
vol20 = Volatility(window=20)
vol120 = Volatility(window=120, annualize=False)


# ====================================================================
# 2. 共享管道
# ====================================================================
# 排名因子由引擎自动纳入预计算；这里只放权重与横截面过滤所需的因子。
SHARED_PIPELINE: tuple = (
    vol20,
    vol120,
)


# ====================================================================
# 3. 过滤器定义
# ====================================================================
# 本配置不使用 ThresholdFilter 硬性过滤器。
FULL_FILTERS = ()
NO_FILTERS = ()

# C2 的横截面波动率 RankFilter：淘汰波动率最低 30%，再额外淘汰最高 10%。
C2_RANK_FILTER = RankFilter(
    factor=vol120,
    exclude_below_pct=0.3,
    exclude_above_pct=0.1,
    name="rank_vol120_b0.3_a0.1",
)


# ====================================================================
# 4. 止损/止盈规则（与模板一致: 不启用）
# ====================================================================
SHARED_STOP_RULES: tuple[StopRuleSpec, ...] = ()


# ====================================================================
# 5. ICIR 动态因子选择（与模板一致: 不启用）
# ====================================================================
RANKING_FACTOR_CANDIDATES: tuple = ()
IC_WINDOW: int = 120
IC_SELECTION_MODE: str = "icir"


# ====================================================================
# 6. 组定义
# ====================================================================
# 每组格式: (label, ranking_factor, builtin_filters, cross_sectional_filters)
GROUPS: list[tuple] = [
    ("C2_rank_filter_vol120_b0.3_a0.1", c2_factor, NO_FILTERS, (C2_RANK_FILTER,)),
    ("PR20_rsrs_trendr2", pr20_rsrs_trendr2, NO_FILTERS, ()),
]


# ====================================================================
# 7. Grid Search 参数（与模板一致）
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (
    1,
    5,
    10,
    20,
)

GRID_MIN_MOMENTUM: tuple = (
    None,
)

GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (
    0,
)

GRID_REBALANCE_INTERVAL: tuple[int, ...] = (
    5,
    10,
    20,
)

GRID_EXCLUDE_BONDS: tuple[bool, ...] = (
    True,
    False,
)

GRID_HOLD_OVERLAP: tuple[bool, ...] = (
    False,
)


# ====================================================================
# 8. 权重分配器（与模板一致: 反波动率加权）
# ====================================================================
alloc_equal = equal_weight_allocator

alloc_momentum = score_proportional_allocator
alloc_momentum.__name__ = "momentum"

alloc_inv_vol = make_factor_weighted_allocator(
    vol20.get_output_name(), inverse=True
)
alloc_inv_vol.__name__ = "invvol"

WEIGHT_ALLOCATORS: tuple = (
    alloc_inv_vol,
)


# ====================================================================
# 9. 执行参数
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
TITLE: str = "宽动量基线回测 — C2_rank_filter + PR20_rsrs_trendr2"
BASENAME_TAG: str = "c2_rankfilter_pr20_rsrs_trendr2"

START_DATE: str = "2020-01-01"
END_DATE: str = "2026-08-24"

START_DATES: tuple[str, ...] | None = None
END_DATES: tuple[str, ...] | None = None

SYMBOLS: tuple[str, ...] | None = None

MAX_WORKERS: int | None = None
CROSS_GROUP_PARALLEL: bool = True

PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
