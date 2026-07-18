"""TrendR2_120_r2_rm5 × C2_MAE40_z120_neg 乘积因子回测。

排名因子: TrendR2_120_r2__rolling_mean_5__product_MAE_40__zscore_120__neg
         = F3 (TrendR2 r2 rolling_mean 5) × C2 (MAE zscore 120 neg)

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.trendr2_rm5_prod_c2_mae40_z120_neg
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)
from factors.trend_r2 import TrendR2Factor
from factors.distribution_family import MaxAdverseExcursion, MaxFavorableExcursion
from factors.meta_factor import (
    CombineFactor,
    CompositeRankFactor,
    ConditionSpec,
    ConditionalFactor,
    NegateFactor,
    TransformFactor,
    MultiConditionalFactor
)
from factors.volatility import Volatility
from factors.volatility_family import AvgDrawdown
from factors.volume_family import VolumePriceCorrelation, VolumeStd
from factors.price_return import PriceReturn
from factors.price_momentum import HighPointPosition, LowPointPosition, TimeSeriesMomentum
from factors.ma import MAPosition, MADispersion, MADistance, MASlope, BIAS, LogBIAS
from factors.reversal import ShortTermReversal, ExtremeReversal, VolumeReversal
from factors.daily_rebound import DailyRebound
from factors.average_amount import AverageAmount


# ====================================================================
# 因子定义: TrendR2_120_r2_rm5 × C2_MAE40_z120_neg
# ====================================================================

# ── F3: TrendR2_120_r2__rolling_mean_5 ──
trend_r2_120 = TrendR2Factor(window=120, output="r2")
f3_factor = TransformFactor(dependency=trend_r2_120, transform="rolling_mean", window=5)


# ── C2: MAE_40__zscore_120__neg ──
mae_20 = MaxAdverseExcursion(window=20)
mae_40 = MaxAdverseExcursion(window=40)
mae_zscore = TransformFactor(dependency=mae_40, transform="zscore", window=120)
c2_factor = NegateFactor(mae_zscore)

logBias = LogBIAS(window=20)

logBias_zscore = TransformFactor(dependency=logBias, transform="zscore", window=252)

condition_c2_logBias = ConditionalFactor(
    signal=c2_factor,
    condition=logBias,
    op = "gt",
    threshold=0.05,
)

condition_logBias_07 = ConditionalFactor(
    signal=logBias,
    condition=logBias,
    op = "lt",
    threshold=0.7
)

pr20_condition_logBias = MultiConditionalFactor(
    signal=PriceReturn(window=20),
    conditions=[
        ConditionSpec(logBias, "lt", 0.15),
        ConditionSpec(logBias, "gt", 0.05),
    ]
)

pr20_condition_logBias2 = MultiConditionalFactor(
    signal=PriceReturn(window=20),
    conditions=[
        ConditionSpec(logBias, "lt", 0.10),
        ConditionSpec(logBias, "gt", 0.05),
    ]
)

trendR2_condition_logBias2 = MultiConditionalFactor(
    signal=TrendR2Factor(window=120, output="r2"),
    conditions=[
        ConditionSpec(logBias, "lt", 0.10),
        ConditionSpec(logBias, "gt", 0.05),
    ]
)

# ── 反波动率加权所用的波动率因子 ──
vol20 = Volatility(window=20)

# ====================================================================
# 共享管道（所有因子必须在此，才能被预计算）
# ====================================================================
SHARED_PIPELINE: tuple = (
    # f3_factor, 
    # c2_factor, 
    # # product_factor, 
    # product_pr_20, 
    vol20,
    )

# ====================================================================
# 组定义
# ====================================================================
GROUPS: list[tuple[str, object, tuple[ThresholdFilter, ...]]] = [
    # ── HTML 选出的 50 个因子 ──
    # ("MAE_40__zscore_120__neg", c2_factor, ()),
    # ("LogBIAS_20__zscore_252", logBias_zscore, ()),
    # ("Conditional_C2_MAE40_zscore_120_neg_gt_LogBIAS_20_0.05", condition_c2_logBias, ()),
    # ("Conditional_LogBIAS_20_lt_0.7", condition_logBias_07, ()),
    # ("MultiConditional_PR20_LogBIAS_20_lt_0.10_gt_0.05", pr20_condition_logBias2, ()),
    ("MultiConditional_TrendR2_LogBIAS_20_lt_0.10_gt_0.05", trendR2_condition_logBias2, ()),
]


# ====================================================================
# Grid Search 参数
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1, 5, 10)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 20)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (True, False)
GRID_HOLD_OVERLAP: tuple[bool, ...] = (False,)


# ====================================================================
# 权重分配器
# ====================================================================
alloc_equal = equal_weight_allocator

alloc_momentum = score_proportional_allocator
alloc_momentum.__name__ = "momentum"

alloc_inv_vol = make_factor_weighted_allocator(vol20.get_output_name(), inverse=True)
alloc_inv_vol.__name__ = "invvol"

WEIGHT_ALLOCATORS: tuple = (
    alloc_inv_vol,
    # alloc_momentum,
    # alloc_equal,
)


# ====================================================================
# 执行参数
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
BASENAME_TAG: str = "logBias"
TITLE: str = "宽动量基线回测 — 广发刘晨明主线偏离率"
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-07"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
