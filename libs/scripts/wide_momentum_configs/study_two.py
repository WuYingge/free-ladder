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
from factors.rsrs import RsrsFactor
from factors.trend_r2 import TrendR2Factor
from factors.distribution_family import MaxAdverseExcursion, MaxFavorableExcursion
from factors.meta_factor import (
    CombineFactor,
    CompositeRankFactor,
    ConditionSpec,
    ConditionalFactor,
    NegateFactor,
    TransformFactor,
    MultiConditionalFactor,
)
from factors.volatility import Volatility
from factors.volatility_family import AvgDrawdown
from factors.volume_family import VolumePriceCorrelation, VolumeStd
from factors.price_return import PriceReturn
from factors.price_momentum import HighPointPosition, LowPointPosition, TimeSeriesMomentum
from factors.ma import MAPosition, MADispersion, MADistance, MASlope, BIAS
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

pr_20 = PriceReturn(window=20)

vol252 = Volatility(window=252, annualize=True)

# ── 波动率环境过滤器（用于 builtin_filters，天然 point-in-time）──
FILT_VOL_BOND  = ThresholdFilter(
    field=vol252.get_output_name(), operator="<",  value=0.06,
    name="vol<6%"
)
FILT_VOL_LOW_A = ThresholdFilter(
    field=vol252.get_output_name(), operator=">=", value=0.06,
    name="vol>=6%"
)
FILT_VOL_LOW_B = ThresholdFilter(
    field=vol252.get_output_name(), operator="<",  value=0.20,
    name="vol<20%"
)
FILT_VOL_MID_A = ThresholdFilter(
    field=vol252.get_output_name(), operator=">=", value=0.20,
    name="vol>=20%"
)
FILT_VOL_MID_B = ThresholdFilter(
    field=vol252.get_output_name(), operator="<",  value=0.36,
    name="vol<36%"
)
FILT_VOL_HIGH  = ThresholdFilter(
    field=vol252.get_output_name(), operator=">=", value=0.36,
    name="vol>=36%"
)

FILT_PR_20 = ThresholdFilter(
    field=pr_20.get_output_name(), operator=">", value=0.0,
    name="pr_20>0"
)



pr_20_3_filter = MultiConditionalFactor(
    signal=pr_20,
    conditions=[
        ConditionSpec(
            condition=RsrsFactor(regression_window=14, zscore_window=600, output="zscore"),
            op="gt", threshold=0.0,
        ),
        ConditionSpec(
            condition=MAPosition(window=200, price_column="close"),
            op="gt", threshold=0.0,
        ),
        ConditionSpec(
            condition=TrendR2Factor(window=120, output="r2"),
            op="gt", threshold=0.5,
        ),
    ],
)




# ── 反波动率加权所用的波动率因子 ──
vol20 = Volatility(window=20)

# ====================================================================
# 共享管道（所有因子必须在此，才能被预计算）
# ====================================================================
SHARED_PIPELINE: tuple = (
    vol20,
    vol252,
    pr_20,
)

# ====================================================================
# 组定义
# ====================================================================
GROUPS: list[tuple[str, object, tuple[ThresholdFilter, ...]]] = [
    # ── 基线（无波动率过滤）──
    # ("MAE_40__zscore_120__neg", c2_factor, ()),
    # ("TrendR2_120_r2__rolling_mean_5", f3_factor, ()),
    # ("PR_20_3_filter", pr_20_3_filter, ()),

    # # ── C2 因子 × 波动率环境 ──
    # ("C2_bond", c2_factor, (FILT_VOL_BOND,)),
    # ("C2_low",  c2_factor, (FILT_VOL_LOW_A, FILT_VOL_LOW_B)),
    # ("C2_mid",  c2_factor, (FILT_VOL_MID_A, FILT_VOL_MID_B)),
    # ("C2_high", c2_factor, (FILT_VOL_HIGH,)),

    # # ── PR20_3filter × 波动率环境 ──
    # ("PR20_3f_bond", pr_20_3_filter, (FILT_VOL_BOND,)),
    # ("PR20_3f_low",  pr_20_3_filter, (FILT_VOL_LOW_A, FILT_VOL_LOW_B)),
    # ("PR20_3f_mid",  pr_20_3_filter, (FILT_VOL_MID_A, FILT_VOL_MID_B)),
    # ("PR20_3f_high", pr_20_3_filter, (FILT_VOL_HIGH,)),
    
    # ── PR20_3filter × PR20>0 ──
    ("PR20_3f_pr20_gt_0", pr_20_3_filter, (FILT_PR_20,)),
]


# ====================================================================
# Grid Search 参数
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1, 5, 10)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 20)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (False,)
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
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest/c2_pool_backtest"
BASENAME_TAG: str = "study2"
TITLE: str = "宽动量基线回测 — C2_MAE40_z120_neg 分组变换"
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-07"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None