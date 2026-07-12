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
    ConditionalFactor,
    NegateFactor,
    TransformFactor,
)
from factors.volatility import Volatility
from factors.volume_family import VolumePriceCorrelation
from factors.price_return import PriceReturn
from factors.price_momentum import HighPointPosition, LowPointPosition
from factors.ma import MAPosition, MADispersion, MADistance


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

lpp_20_bin_0 = TransformFactor(dependency=LowPointPosition(window=20), transform="binarize_winrate", threshold=0.0)
# 最大有利偏移
mfe_20 = MaxFavorableExcursion(window=20)
mfe_20_delta_10 = TransformFactor(dependency=mfe_20, transform="delta", window=10)

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

condition_c2_lpp20_bin05 = ConditionalFactor(
    signal=c2_factor,
    condition=lpp_20_bin_0,
    op = "gt",
    threshold = 0.5,
)
condition_c2_lpp20_bin08 = ConditionalFactor(
    signal=c2_factor,
    condition=lpp_20_bin_0,
    op = "gt",
    threshold = 0.8,
)
condition_c2_lpp20_bin03 = ConditionalFactor(
    signal=c2_factor,
    condition=lpp_20_bin_0,
    op = "gt",
    threshold = 0.3,
)
condition_c2_lpp20_bin_minus05 = ConditionalFactor(
    signal=c2_factor,
    condition=lpp_20_bin_0,
    op = "lt",
    threshold = 0.5,
)

product_c2_lpp20_bin05 = CombineFactor(
    factor_a=c2_factor,
    factor_b=condition_c2_lpp20_bin05,
    method="product",
    normalize=False,
)

condition_c2_mfe20_delta10_gt_0 = ConditionalFactor(
    signal=c2_factor,
    condition=mfe_20_delta_10,
    op = "gt",
    threshold = 0.0,
)

condition_c2_mfe20_delta10_gt_05 = ConditionalFactor(
    signal=c2_factor,
    condition=mfe_20_delta_10,
    op = "gt",
    threshold = 0.05,
)
condition_c2_mfe20_delta10_gt_02 = ConditionalFactor(
    signal=c2_factor,
    condition=mfe_20_delta_10,
    op = "gt",
    threshold = 0.02,
)
product_c2_mfe20_delta10 = CombineFactor(
    factor_a=c2_factor,
    factor_b=mfe_20_delta_10,
    method="product",
    normalize=False,
)
sum_c2_mfe20_delta10 = CombineFactor(
    factor_a=c2_factor,
    factor_b=mfe_20_delta_10,
    method="weighted_sum",
    normalize=True,
)

ratio_c2_mfe20_delta10 = CombineFactor(
    factor_a=mfe_20_delta_10,
    factor_b=c2_factor,
    method="ratio",
    normalize=False,
)

volume_price_corr_20__binarize_winrate_10_0 = TransformFactor(
    dependency=VolumePriceCorrelation(window=20),
    transform="binarize_winrate",
    window=10,
    threshold=0.0,
)

trendR2_240_r2__binarize_winrate_20_05 = TransformFactor(
    dependency=TrendR2Factor(window=240, output="r2"),
    transform="binarize_winrate",
    window=20,
    threshold=0.5,
)

highPointPosition_20__weighted_sum_MADistance_close_20_60 = CombineFactor(
    factor_a=HighPointPosition(window=20),
    factor_b=MADistance(short_window=20, long_window=60),
    method="weighted_sum",
    normalize=True,
    
)

trendR2_60_slope__zscore_120_neg = NegateFactor(
    TransformFactor(
        dependency=TrendR2Factor(window=60, output="slope"),
        transform="zscore",
        window=120,
    )
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
    # ("C2_MAE40_z120_neg", c2_factor, ()),
    # ("C2_MAE40_z120_neg_cond_LPP20_bin0.5", condition_c2_lpp20_bin05, ()),
    # ("C2_MAE40_z120_neg_cond_LPP20_bin0.8", condition_c2_lpp20_bin08, ()),
    # ("C2_MAE40_z120_neg_cond_LPP20_bin0.3", condition_c2_lpp20_bin03, ()),
    # ("C2_MAE40_z120_neg_cond_LPP20_bin-0.5", condition_c2_lpp20_bin_minus05, ())
    # ("C2_MAE40_z120_neg_cond_MFE20_delta10_gt_0", condition_c2_mfe20_delta10_gt_0, ()),
    # ("C2_MAE40_z120_neg_cond_MFE20_delta10_gt_0_5", condition_c2_mfe20_delta10_gt_05, ()),
    # ("C2_MAE40_z120_neg_cond_MFE20_delta10_gt_0_2", condition_c2_mfe20_delta10_gt_02, ()),
    # ("C2_MAE40_z120_neg_prod_MFE20_delta10", product_c2_mfe20_delta10, ()),
    # ("C2_MAE40_z120_neg_sum_MFE20_delta10", sum_c2_mfe20_delta10, ()),
    # ("C2_MAE40_z120_neg_ratio_MFE20_delta10", ratio_c2_mfe20_delta10, ())
    # ("VolumePriceCorrelation_20_binarize_winrate_10_0", volume_price_corr_20__binarize_winrate_10_0, ()),
    # ("TrendR2_240_r2_binarize_winrate_20_0.5", trendR2_240_r2__binarize_winrate_20_05, ()),
    ("HighPointPosition_20_weighted_sum_MADistance_close_20_60", highPointPosition_20__weighted_sum_MADistance_close_20_60, ())
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
