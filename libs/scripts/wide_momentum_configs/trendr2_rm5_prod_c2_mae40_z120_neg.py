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
from factors.distribution_family import MaxAdverseExcursion
from factors.meta_factor import (
    CombineFactor,
    CompositeRankFactor,
    ConditionalFactor,
    NegateFactor,
    TransformFactor,
)
from factors.volatility import Volatility
from factors.price_return import PriceReturn
from factors.ma import MAPosition, MADispersion


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
c2_20_factor = NegateFactor(TransformFactor(dependency=mae_20, transform="zscore", window=120))
c2_factor_z_252 = NegateFactor(TransformFactor(dependency=mae_40, transform="zscore", window=252))
c2_factor_z_60 = NegateFactor(TransformFactor(dependency=mae_40, transform="zscore", window=60))

# ── 乘积: TrendR2_120_r2_rm5 × C2_MAE40_z120_neg ──
# normalize=False：product 不做 z-score 标准化，避免不必要的 +251 天 warmup
product_factor = CombineFactor(
    factor_a=f3_factor,
    factor_b=c2_factor,
    method="product",
    normalize=False,
)

product_pr_20 = CombineFactor(
    factor_a=c2_factor,
    factor_b=PriceReturn(window=20),
    method="product",
    normalize=False,
)
condition_pr20_gt_0 = ConditionalFactor(
    signal=c2_factor,
    condition=PriceReturn(window=20),
    op = "gt",
    threshold = 0.0,
)
condition_ma200_gt_0 = ConditionalFactor(
    signal=c2_factor,
    condition=MAPosition(window=200, price_column="close"),
    op = "gt",
    threshold = 0.0,
)
condition_tr2r2_gt_0_5 = ConditionalFactor(
    signal=c2_factor,
    condition=f3_factor,
    op = "gt",
    threshold = 0.5,
)
pr20_condition = ConditionalFactor(
    signal=PriceReturn(window=20),
    condition=c2_factor,
    op = "gt",
    threshold = 0.0,
)

MADispersion_close_5_10_20_60_delta_5 = TransformFactor(
    dependency=MADispersion(),
    transform="delta",
    window=5,
)

product_madis_delta_5 = CombineFactor(
    factor_a=c2_factor,
    factor_b=MADispersion_close_5_10_20_60_delta_5,
    method="product",
    normalize=False,
)

# ── CompositeRank: 0.5 * rank(MADispersion_delta_5) + 0.5 * rank(C2_MAE40_z120_neg) ──
composite_madis_c2 = CompositeRankFactor(
    factors=[
        (MADispersion_close_5_10_20_60_delta_5, 0.5),
        (c2_factor, 0.5),
    ]
)

composite_pr20_c2 = CompositeRankFactor(
    factors=[
        (PriceReturn(window=20), 0.5),
        (c2_factor, 0.5),
    ]
)

mae40_vol252 = CombineFactor(
    factor_a=mae_40,
    factor_b=Volatility(window=252),
    method="ratio",
    normalize=False
)

c2_vol252 = NegateFactor(TransformFactor(dependency=mae40_vol252, transform="zscore", window=120))

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
# 组定义: 单组 — 乘积因子纯排名
# ====================================================================
GROUPS: list[tuple[str, object, tuple[ThresholdFilter, ...]]] = [
    # ("TR2R2_rm5_prod_C2_MAE40_z120_neg", product_factor, ()),
    # ("PR20_prod_C2_MAE40_z120_neg", product_pr_20, ()),
    # ("C2_MAE40_z120_neg_cond_MA200_gt_0", condition_ma200_gt_0, ()),
    # ("C2_MAE40_z120_neg_cond_TR2R2_gt_0_5", condition_tr2r2_gt_0_5, ()),
    # ("PR20_cond_C2_MAE40_z120_neg", pr20_condition, ()),
    # ("MADispersion_delta_5_prod_C2", product_madis_delta_5, ()),
    # ("MADispersion_delta_5_C2_composite_rank", composite_madis_c2, ()),
    # ("PR20_C2_composite_rank", composite_pr20_c2, ()),
    ("C2_MAE40_z120_neg", c2_factor, ()),
    ("C2_MAE40_ratio_vol252_z120_neg", c2_vol252, ())
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
BASENAME_TAG: str = "c2_mae40_z120_neg_deep_dive"
TITLE: str = "宽动量基线回测 — C2_MAE40_z120_neg 因子变换"
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-07"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
