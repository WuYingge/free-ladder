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


# ====================================================================
# HTML 选出的 50 个因子（全部 NegateFactor 变换，IC Mean < 0 → 取反变正）
# ====================================================================

# 1. MAE_40__zscore_120 → neg（同 c2_factor，复用）
f1_mae_40_z120_neg = c2_factor

# 2. TrendR2_60_slope__zscore_120 → neg（已有 trendR2_60_slope__zscore_120_neg，复用）
f2_tr2_60s_z120_neg = trendR2_60_slope__zscore_120_neg

# 3. MADistance_close_20_60__zscore_120 → neg
f3_madist_20_60_z120_neg = NegateFactor(
    TransformFactor(
        dependency=MADistance(short_window=20, long_window=60, price_column="close"),
        transform="zscore", window=120,
    )
)

# 4. AvgDrawdown_40__zscore_120 → neg
f4_avgdd_40_z120_neg = NegateFactor(
    TransformFactor(
        dependency=AvgDrawdown(window=40),
        transform="zscore", window=120,
    )
)

# 5. TrendR2_120_slope__delta_5 → neg
f5_tr2_120s_d5_neg = NegateFactor(
    TransformFactor(
        dependency=TrendR2Factor(window=120, output="slope"),
        transform="delta", window=5,
    )
)

# 6. MASlope_close_40_10__zscore_120 → neg
f6_maslope_40_10_z120_neg = NegateFactor(
    TransformFactor(
        dependency=MASlope(ma_window=40, slope_window=10, price_column="close"),
        transform="zscore", window=120,
    )
)

# 7. TrendR2_120_slope__delta_10 → neg
f7_tr2_120s_d10_neg = NegateFactor(
    TransformFactor(
        dependency=TrendR2Factor(window=120, output="slope"),
        transform="delta", window=10,
    )
)

# 8. MADistance_close_20_120__zscore_120 → neg
f8_madist_20_120_z120_neg = NegateFactor(
    TransformFactor(
        dependency=MADistance(short_window=20, long_window=120, price_column="close"),
        transform="zscore", window=120,
    )
)

# 9. BIAS_close_10__delta_5 → neg
f9_bias_10_d5_neg = NegateFactor(
    TransformFactor(
        dependency=BIAS(window=10, price_column="close"),
        transform="delta", window=5,
    )
)

# 10. PriceReturn_40__zscore_120 → neg
f10_pr_40_z120_neg = NegateFactor(
    TransformFactor(
        dependency=PriceReturn(window=40),
        transform="zscore", window=120,
    )
)

# 11. MAE_60__zscore_120 → neg
f11_mae_60_z120_neg = NegateFactor(
    TransformFactor(
        dependency=MaxAdverseExcursion(window=60),
        transform="zscore", window=120,
    )
)

# 12. LowPointPosition_10__binarize_winrate_10_0.0 → neg
f12_lpp_10_bw10_neg = NegateFactor(
    TransformFactor(
        dependency=LowPointPosition(window=10),
        transform="binarize_winrate", window=10, threshold=0.0,
    )
)

# 13. MASlope_close_40_5__zscore_120 → neg
f13_maslope_40_5_z120_neg = NegateFactor(
    TransformFactor(
        dependency=MASlope(ma_window=40, slope_window=5, price_column="close"),
        transform="zscore", window=120,
    )
)

# 14. MFE_40__zscore_120 → neg
f14_mfe_40_z120_neg = NegateFactor(
    TransformFactor(
        dependency=MaxFavorableExcursion(window=40),
        transform="zscore", window=120,
    )
)

# 15. ShortTermReversal_10__rolling_mean_10 → neg
f15_str_10_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=ShortTermReversal(window=10),
        transform="rolling_mean", window=10,
    )
)

# 16. ShortTermReversal_5__binarize_winrate_10_0.0 → neg
f16_str_5_bw10_neg = NegateFactor(
    TransformFactor(
        dependency=ShortTermReversal(window=5),
        transform="binarize_winrate", window=10, threshold=0.0,
    )
)

# 17. DailyRebound → neg
f17_daily_rebound_neg = NegateFactor(DailyRebound())

# 18. VolumeReversal_5_10__binarize_winrate_10_0.0 → neg
f18_vr_5_10_bw10_neg = NegateFactor(
    TransformFactor(
        dependency=VolumeReversal(ret_window=5, vol_window=10),
        transform="binarize_winrate", window=10, threshold=0.0,
    )
)

# 19. LowPointPosition_10__rolling_mean_10 → neg
f19_lpp_10_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=LowPointPosition(window=10),
        transform="rolling_mean", window=10,
    )
)

# 20. MASlope_close_40_3__zscore_120 → neg
f20_maslope_40_3_z120_neg = NegateFactor(
    TransformFactor(
        dependency=MASlope(ma_window=40, slope_window=3, price_column="close"),
        transform="zscore", window=120,
    )
)

# 21. VolumeReversal_5_20__binarize_winrate_10_0.0 → neg
f21_vr_5_20_bw10_neg = NegateFactor(
    TransformFactor(
        dependency=VolumeReversal(ret_window=5, vol_window=20),
        transform="binarize_winrate", window=10, threshold=0.0,
    )
)

# 22. VolumeStd_20 → neg
f22_volstd_20_neg = NegateFactor(VolumeStd(window=20))

# 23. VolumeStd_20__rolling_mean_5 → neg
f23_volstd_20_rm5_neg = NegateFactor(
    TransformFactor(
        dependency=VolumeStd(window=20),
        transform="rolling_mean", window=5,
    )
)

# 24. ShortTermReversal_5__rolling_mean_10 → neg
f24_str_5_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=ShortTermReversal(window=5),
        transform="rolling_mean", window=10,
    )
)

# 25. ShortTermReversal_10__rolling_mean_5 → neg
f25_str_10_rm5_neg = NegateFactor(
    TransformFactor(
        dependency=ShortTermReversal(window=10),
        transform="rolling_mean", window=5,
    )
)

# 26. VolumeStd_20__rolling_mean_10 → neg
f26_volstd_20_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=VolumeStd(window=20),
        transform="rolling_mean", window=10,
    )
)

# 27. VolumeStd_40 → neg
f27_volstd_40_neg = NegateFactor(VolumeStd(window=40))

# 28. VolumeStd_10 → neg
f28_volstd_10_neg = NegateFactor(VolumeStd(window=10))

# 29. MADistance_close_20_40__zscore_120 → neg
f29_madist_20_40_z120_neg = NegateFactor(
    TransformFactor(
        dependency=MADistance(short_window=20, long_window=40, price_column="close"),
        transform="zscore", window=120,
    )
)

# 30. VolumeStd_10__rolling_mean_10 → neg
f30_volstd_10_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=VolumeStd(window=10),
        transform="rolling_mean", window=10,
    )
)

# 31. VolumeStd_10__rolling_std_10 → neg
f31_volstd_10_rstd10_neg = NegateFactor(
    TransformFactor(
        dependency=VolumeStd(window=10),
        transform="rolling_std", window=10,
    )
)

# 32. ExtremeReversal_20_p15__rolling_mean_10 → neg
f32_er_20_p15_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=ExtremeReversal(window=20, tail_pct=0.15),
        transform="rolling_mean", window=10,
    )
)

# 33. VolumeStd_40__rolling_mean_5 → neg
f33_volstd_40_rm5_neg = NegateFactor(
    TransformFactor(
        dependency=VolumeStd(window=40),
        transform="rolling_mean", window=5,
    )
)

# 34. VolumeStd_10__rolling_mean_5 → neg
f34_volstd_10_rm5_neg = NegateFactor(
    TransformFactor(
        dependency=VolumeStd(window=10),
        transform="rolling_mean", window=5,
    )
)

# 35. AverageAmount_10__rolling_std_10 → neg
f35_avgamt_10_rstd10_neg = NegateFactor(
    TransformFactor(
        dependency=AverageAmount(window=10),
        transform="rolling_std", window=10,
    )
)

# 36. LowPointPosition_10__rolling_mean_5 → neg
f36_lpp_10_rm5_neg = NegateFactor(
    TransformFactor(
        dependency=LowPointPosition(window=10),
        transform="rolling_mean", window=5,
    )
)

# 37. VolumeStd_40__rolling_mean_10 → neg
f37_volstd_40_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=VolumeStd(window=40),
        transform="rolling_mean", window=10,
    )
)

# 38. ExtremeReversal_20_p05__rolling_mean_10 → neg
f38_er_20_p05_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=ExtremeReversal(window=20, tail_pct=0.05),
        transform="rolling_mean", window=10,
    )
)

# 39. MADistance_close_20_60__rolling_mean_10 → neg
f39_madist_20_60_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=MADistance(short_window=20, long_window=60, price_column="close"),
        transform="rolling_mean", window=10,
    )
)

# 40. AverageAmount_5 → neg
f40_avgamt_5_neg = NegateFactor(AverageAmount(window=5))

# 41. AverageAmount_20 → neg
f41_avgamt_20_neg = NegateFactor(AverageAmount(window=20))

# 42. AverageAmount_10 → neg
f42_avgamt_10_neg = NegateFactor(AverageAmount(window=10))

# 43. AverageAmount_5__rolling_mean_5 → neg
f43_avgamt_5_rm5_neg = NegateFactor(
    TransformFactor(
        dependency=AverageAmount(window=5),
        transform="rolling_mean", window=5,
    )
)

# 44. VolumeStd_20__rolling_std_10 → neg
f44_volstd_20_rstd10_neg = NegateFactor(
    TransformFactor(
        dependency=VolumeStd(window=20),
        transform="rolling_std", window=10,
    )
)

# 45. ExtremeReversal_20_p1__rolling_mean_10 → neg  (p1 = tail_pct=0.01)
f45_er_20_p1_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=ExtremeReversal(window=20, tail_pct=0.01),
        transform="rolling_mean", window=10,
    )
)

# 46. MAE_40__product_TimeSeriesMomentum_252 → neg
f46_mae40_prod_tsm252_neg = NegateFactor(
    CombineFactor(
        factor_a=MaxAdverseExcursion(window=40),
        factor_b=TimeSeriesMomentum(window=252),
        method="product",
        normalize=False,
    )
)

# 47. ShortTermReversal_1__binarize_winrate_20_0.0 → neg
f47_str_1_bw20_neg = NegateFactor(
    TransformFactor(
        dependency=ShortTermReversal(window=1),
        transform="binarize_winrate", window=20, threshold=0.0,
    )
)

# 48. AverageAmount_10__rolling_mean_10 → neg
f48_avgamt_10_rm10_neg = NegateFactor(
    TransformFactor(
        dependency=AverageAmount(window=10),
        transform="rolling_mean", window=10,
    )
)

# 49. DailyRebound__delta_5 → neg
f49_daily_rebound_d5_neg = NegateFactor(
    TransformFactor(
        dependency=DailyRebound(),
        transform="delta", window=5,
    )
)

# 50. AverageAmount_20__rolling_mean_5 → neg
f50_avgamt_20_rm5_neg = NegateFactor(
    TransformFactor(
        dependency=AverageAmount(window=20),
        transform="rolling_mean", window=5,
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
# 组定义
# ====================================================================
GROUPS: list[tuple[str, object, tuple[ThresholdFilter, ...]]] = [
    # ── 原有因子 ──
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
    # ("HighPointPosition_20_weighted_sum_MADistance_close_20_60", highPointPosition_20__weighted_sum_MADistance_close_20_60, ()),

    # ── HTML 选出的 50 个因子 ──
    ("MAE_40__zscore_120__neg", f1_mae_40_z120_neg, ()),
    ("TrendR2_60_slope__zscore_120__neg", f2_tr2_60s_z120_neg, ()),
    ("MADistance_close_20_60__zscore_120__neg", f3_madist_20_60_z120_neg, ()),
    ("AvgDrawdown_40__zscore_120__neg", f4_avgdd_40_z120_neg, ()),
    ("TrendR2_120_slope__delta_5__neg", f5_tr2_120s_d5_neg, ()),
    ("MASlope_close_40_10__zscore_120__neg", f6_maslope_40_10_z120_neg, ()),
    ("TrendR2_120_slope__delta_10__neg", f7_tr2_120s_d10_neg, ()),
    ("MADistance_close_20_120__zscore_120__neg", f8_madist_20_120_z120_neg, ()),
    ("BIAS_close_10__delta_5__neg", f9_bias_10_d5_neg, ()),
    ("PriceReturn_40__zscore_120__neg", f10_pr_40_z120_neg, ()),
    ("MAE_60__zscore_120__neg", f11_mae_60_z120_neg, ()),
    ("LowPointPosition_10__binarize_winrate_10_0.0__neg", f12_lpp_10_bw10_neg, ()),
    ("MASlope_close_40_5__zscore_120__neg", f13_maslope_40_5_z120_neg, ()),
    ("MFE_40__zscore_120__neg", f14_mfe_40_z120_neg, ()),
    ("ShortTermReversal_10__rolling_mean_10__neg", f15_str_10_rm10_neg, ()),
    ("ShortTermReversal_5__binarize_winrate_10_0.0__neg", f16_str_5_bw10_neg, ()),
    ("DailyRebound__neg", f17_daily_rebound_neg, ()),
    ("VolumeReversal_5_10__binarize_winrate_10_0.0__neg", f18_vr_5_10_bw10_neg, ()),
    ("LowPointPosition_10__rolling_mean_10__neg", f19_lpp_10_rm10_neg, ()),
    ("MASlope_close_40_3__zscore_120__neg", f20_maslope_40_3_z120_neg, ()),
    ("VolumeReversal_5_20__binarize_winrate_10_0.0__neg", f21_vr_5_20_bw10_neg, ()),
    ("VolumeStd_20__neg", f22_volstd_20_neg, ()),
    ("VolumeStd_20__rolling_mean_5__neg", f23_volstd_20_rm5_neg, ()),
    ("ShortTermReversal_5__rolling_mean_10__neg", f24_str_5_rm10_neg, ()),
    ("ShortTermReversal_10__rolling_mean_5__neg", f25_str_10_rm5_neg, ()),
    ("VolumeStd_20__rolling_mean_10__neg", f26_volstd_20_rm10_neg, ()),
    ("VolumeStd_40__neg", f27_volstd_40_neg, ()),
    ("VolumeStd_10__neg", f28_volstd_10_neg, ()),
    ("MADistance_close_20_40__zscore_120__neg", f29_madist_20_40_z120_neg, ()),
    ("VolumeStd_10__rolling_mean_10__neg", f30_volstd_10_rm10_neg, ()),
    ("VolumeStd_10__rolling_std_10__neg", f31_volstd_10_rstd10_neg, ()),
    ("ExtremeReversal_20_p15__rolling_mean_10__neg", f32_er_20_p15_rm10_neg, ()),
    ("VolumeStd_40__rolling_mean_5__neg", f33_volstd_40_rm5_neg, ()),
    ("VolumeStd_10__rolling_mean_5__neg", f34_volstd_10_rm5_neg, ()),
    ("AverageAmount_10__rolling_std_10__neg", f35_avgamt_10_rstd10_neg, ()),
    ("LowPointPosition_10__rolling_mean_5__neg", f36_lpp_10_rm5_neg, ()),
    ("VolumeStd_40__rolling_mean_10__neg", f37_volstd_40_rm10_neg, ()),
    ("ExtremeReversal_20_p05__rolling_mean_10__neg", f38_er_20_p05_rm10_neg, ()),
    ("MADistance_close_20_60__rolling_mean_10__neg", f39_madist_20_60_rm10_neg, ()),
    ("AverageAmount_5__neg", f40_avgamt_5_neg, ()),
    ("AverageAmount_20__neg", f41_avgamt_20_neg, ()),
    ("AverageAmount_10__neg", f42_avgamt_10_neg, ()),
    ("AverageAmount_5__rolling_mean_5__neg", f43_avgamt_5_rm5_neg, ()),
    ("VolumeStd_20__rolling_std_10__neg", f44_volstd_20_rstd10_neg, ()),
    ("ExtremeReversal_20_p1__rolling_mean_10__neg", f45_er_20_p1_rm10_neg, ()),
    ("MAE_40__product_TimeSeriesMomentum_252__neg", f46_mae40_prod_tsm252_neg, ()),
    ("ShortTermReversal_1__binarize_winrate_20_0.0__neg", f47_str_1_bw20_neg, ()),
    ("AverageAmount_10__rolling_mean_10__neg", f48_avgamt_10_rm10_neg, ()),
    ("DailyRebound__delta_5__neg", f49_daily_rebound_d5_neg, ()),
    ("AverageAmount_20__rolling_mean_5__neg", f50_avgamt_20_rm5_neg, ()),
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
