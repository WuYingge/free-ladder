"""宽动量基线回测 — 全量因子扫描（来自 factors_to_analyze.csv）。

包含 145 个排名因子（纯排名，无过滤）。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.factors0706_all
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)

from factors.breakout_family import DonchianChannelPosition, NewLowContinuous
from factors.daily_rebound import DailyRebound
from factors.distribution_family import MaxAdverseExcursion, MaxFavorableExcursion
from factors.ma import BIAS, BollingerBandPosition, MADispersion, MADistance, MAFactor, MAPosition, MASlope
from factors.meta_factor import CombineFactor, ConditionSpec, ConditionalFactor, MultiConditionalFactor, NegateFactor, SwitchFactor, TransformFactor
from factors.oscillator import CCI, MFI, RSI, Stochastic, WilliamsR
from factors.price_momentum import HighPointPosition, LowPointPosition, TimeSeriesMomentum
from factors.price_return import PriceReturn
from factors.reversal import ShortTermReversal, VolumeReversal
from factors.trend_quality import KaufmanEfficiencyRatio
from factors.trend_r2 import TrendR2Factor
from factors.volatility_family import AvgDrawdown, DownsideVolatility
from factors.volatility import Volatility


# ====================================================================
# 因子定义
# ====================================================================

# 共 145 个因子

# ── [1] AvgDrawdown_120 ──
f1 = AvgDrawdown(window=120)

# ── [2] AvgDrawdown_120__binarize_winrate_10_-2.0 ──
f2 = TransformFactor(dependency=f1, transform="binarize_winrate", window=10, threshold=-2.0)

# ── [3] AvgDrawdown_120__delta_10 ──
f3 = TransformFactor(dependency=f1, transform="delta", window=10)

# ── [4] AvgDrawdown_120__rolling_mean_10 ──
f4 = TransformFactor(dependency=f1, transform="rolling_mean", window=10)

# ── [5] AvgDrawdown_120__rolling_mean_5 ──
f5 = TransformFactor(dependency=f1, transform="rolling_mean", window=5)

# ── [6] AvgDrawdown_120__zscore_252 ──
f6 = TransformFactor(dependency=f1, transform="zscore", window=252)

# ── [7] AvgDrawdown_250__rolling_mean_5 ──
f7 = AvgDrawdown(window=250)
f8 = TransformFactor(dependency=f7, transform="rolling_mean", window=5)

# ── [8] AvgDrawdown_40__rolling_mean_5 ──
f9 = AvgDrawdown(window=40)
f10 = TransformFactor(dependency=f9, transform="rolling_mean", window=5)

# ── [9] AvgDrawdown_40__zscore_120 ──
f11 = TransformFactor(dependency=f9, transform="zscore", window=120)

# ── [10] BIAS_close_10__binarize_winrate_10_0.0 ──
f12 = BIAS(window=10, price_column="close")
f13 = TransformFactor(dependency=f12, transform="binarize_winrate", window=10, threshold=0.0)

# ── [11] BIAS_close_10__delta_5 ──
f14 = TransformFactor(dependency=f12, transform="delta", window=5)

# ── [12] BIAS_close_10__rolling_mean_10 ──
f15 = TransformFactor(dependency=f12, transform="rolling_mean", window=10)

# ── [13] BollingerBandPosition_close_10_1.5__binarize_winrate_10_0.0 ──
f16 = BollingerBandPosition(window=10, k=1.5, price_column="close")
f17 = TransformFactor(dependency=f16, transform="binarize_winrate", window=10, threshold=0.0)

# ── [14] BollingerBandPosition_close_10_2.0__binarize_winrate_10_0.0 ──
f18 = BollingerBandPosition(window=10, k=2.0, price_column="close")
f19 = TransformFactor(dependency=f18, transform="binarize_winrate", window=10, threshold=0.0)

# ── [15] BollingerBandPosition_close_10_2.5__binarize_winrate_10_0.0 ──
f20 = BollingerBandPosition(window=10, k=2.5, price_column="close")
f21 = TransformFactor(dependency=f20, transform="binarize_winrate", window=10, threshold=0.0)

# ── [16] BollingerBandPosition_close_20_2.0__delta_5 ──
f22 = BollingerBandPosition(window=20, k=2.0, price_column="close")
f23 = TransformFactor(dependency=f22, transform="delta", window=5)

# ── [17] BollingerBandPosition_close_20_2.0__zscore_120 ──
f24 = TransformFactor(dependency=f22, transform="zscore", window=120)

# ── [18] CCI_10__binarize_winrate_10_0.0 ──
f25 = CCI(window=10)
f26 = TransformFactor(dependency=f25, transform="binarize_winrate", window=10, threshold=0.0)

# ── [19] CCI_20__rolling_mean_5 ──
f27 = CCI(window=20)
f28 = TransformFactor(dependency=f27, transform="rolling_mean", window=5)

# ── [20] CCI_20__zscore_120 ──
f29 = TransformFactor(dependency=f27, transform="zscore", window=120)

# ── [21] DailyRebound ──
f30 = DailyRebound()

# ── [22] DonchianPosition_10__rolling_mean_10 ──
f31 = DonchianChannelPosition(window=10)
f32 = TransformFactor(dependency=f31, transform="rolling_mean", window=10)

# ── [23] DonchianPosition_20__delta_10 ──
f33 = DonchianChannelPosition(window=20)
f34 = TransformFactor(dependency=f33, transform="delta", window=10)

# ── [24] DonchianPosition_20__rolling_mean_5 ──
f35 = TransformFactor(dependency=f33, transform="rolling_mean", window=5)

# ── [25] DonchianPosition_20__zscore_120 ──
f36 = TransformFactor(dependency=f33, transform="zscore", window=120)

# ── [26] HighPointPosition_10__rolling_mean_10 ──
f37 = HighPointPosition(window=10)
f38 = TransformFactor(dependency=f37, transform="rolling_mean", window=10)

# ── [27] HighPointPosition_20 ──
f39 = HighPointPosition(window=20)

# ── [28] HighPointPosition_20__delta_5 ──
f40 = TransformFactor(dependency=f39, transform="delta", window=5)

# ── [29] HighPointPosition_20__product_TimeSeriesMomentum_252 ──
f41 = TimeSeriesMomentum(window=252)
f42 = CombineFactor(factor_a=f39, factor_b=f41, method="product")

# ── [30] HighPointPosition_20__rolling_mean_5 ──
f43 = TransformFactor(dependency=f39, transform="rolling_mean", window=5)

# ── [31] HighPointPosition_20__weighted_sum_AvgDrawdown_120 ──
f44 = CombineFactor(factor_a=f39, factor_b=f1, method="weighted_sum")

# ── [32] HighPointPosition_20__weighted_sum_MADistance_close_20_60 ──
f45 = MADistance(short_window=20, long_window=60, price_column="close")
f46 = CombineFactor(factor_a=f39, factor_b=f45, method="weighted_sum")

# ── [33] HighPointPosition_20__weighted_sum_TimeSeriesMomentum_252 ──
f47 = CombineFactor(factor_a=f39, factor_b=f41, method="weighted_sum")

# ── [34] HighPointPosition_20__weighted_sum_TimeSeriesMomentum_252__if_MAPosition_close_200_gt_0_and_TrendR2_120_r2_gt_0.3__else_HighPointPosition_20__weighted_sum_AvgDrawdown_120 ──
f48 = CombineFactor(factor_a=f39, factor_b=f41, method="weighted_sum")
f49 = CombineFactor(factor_a=f39, factor_b=f1, method="weighted_sum")
f51 = MAPosition(window=200, price_column="close")
f52 = TrendR2Factor(window=120, output="r2")
# SwitchFactor (multi-cond, and): HighPointPosition_20__weighted_sum_TimeSeriesMomentum_252 / HighPointPosition_20__weighted_sum_AvgDrawdown_120
f50 = SwitchFactor(
    signal_true=f48,
    signal_false=f49,
    conditions=[
        ConditionSpec(condition=f51, op="gt", threshold=0.0),
        ConditionSpec(condition=f52, op="gt", threshold=0.3),
    ],
    logic="and",
)

# ── [35] HighPointPosition_20__weighted_sum_TimeSeriesMomentum_252__if_MAPosition_close_200_gt_0_and_TrendR2_120_r2_gt_0.3_and_MAE_40__zscore_120_lt_1.5 ──
f55 = CombineFactor(factor_a=f39, factor_b=f41, method="weighted_sum")
f53 = MaxAdverseExcursion(window=40)
f54 = TransformFactor(dependency=f53, transform="zscore", window=120)
# MultiConditionalFactor (and): HighPointPosition_20__weighted_sum_TimeSeriesMomentum_252
f56 = MultiConditionalFactor(
    signal=f55,
    conditions=[
        ConditionSpec(condition=f51, op="gt", threshold=0.0),
        ConditionSpec(condition=f52, op="gt", threshold=0.3),
        ConditionSpec(condition=f54, op="lt", threshold=1.5),
    ],
    logic="and",
)

# ── [36] HighPointPosition_20__zscore_252 ──
f57 = TransformFactor(dependency=f39, transform="zscore", window=252)

# ── [37] HighPointPosition_20__zscore_252__if_MAE_40__zscore_120_lt_1.5 ──
f59 = TransformFactor(dependency=f39, transform="zscore", window=252)
f58 = TransformFactor(dependency=f53, transform="zscore", window=120)
# ConditionalFactor: HighPointPosition_20__zscore_252 if MAE_40__zscore_120 lt 1.5
f60 = ConditionalFactor(
    signal=f59,
    condition=f58,
    op="lt",
    threshold=1.5,
)

# ── [38] HighPointPosition_20__zscore_252__if_TrendR2_120_r2_gt_0.3__else_TimeSeriesMomentum_252 ──
f61 = TransformFactor(dependency=f39, transform="zscore", window=252)
# SwitchFactor: HighPointPosition_20__zscore_252 / TimeSeriesMomentum_252
f62 = SwitchFactor(
    signal_true=f61,
    signal_false=f41,
    condition=f52,
    op="gt",
    threshold=0.3,
)

# ── [39] HighPointPosition_20__zscore_252__if_TrendR2_120_r2_gt_0.5__else_AvgDrawdown_120__neg ──
f63 = TransformFactor(dependency=f39, transform="zscore", window=252)
# SwitchFactor: HighPointPosition_20__zscore_252 / AvgDrawdown_120
f64 = SwitchFactor(
    signal_true=f63,
    signal_false=f1,
    condition=f52,
    op="gt",
    threshold=0.5,
)
f65 = NegateFactor(f64)

# ── [40] HighPointPosition_20__zscore_252__if_TrendR2_120_r2_gt_0.5__else_TimeSeriesMomentum_252 ──
f66 = TransformFactor(dependency=f39, transform="zscore", window=252)
# SwitchFactor: HighPointPosition_20__zscore_252 / TimeSeriesMomentum_252
f67 = SwitchFactor(
    signal_true=f66,
    signal_false=f41,
    condition=f52,
    op="gt",
    threshold=0.5,
)

# ── [41] LowPointPosition_10__binarize_winrate_10_0.0 ──
f68 = LowPointPosition(window=10)
f69 = TransformFactor(dependency=f68, transform="binarize_winrate", window=10, threshold=0.0)

# ── [42] LowPointPosition_10__rolling_mean_10 ──
f70 = TransformFactor(dependency=f68, transform="rolling_mean", window=10)

# ── [43] MADispersion_close_5_10_20_60__delta_10 ──
f71 = MADispersion(windows=(5, 10, 20, 60), price_column="close")
f72 = TransformFactor(dependency=f71, transform="delta", window=10)

# ── [44] MADispersion_close_5_10_20_60__delta_5 ──
f73 = TransformFactor(dependency=f71, transform="delta", window=5)

# ── [45] MADistance_close_10_20__zscore_120 ──
f74 = MADistance(short_window=10, long_window=20, price_column="close")
f75 = TransformFactor(dependency=f74, transform="zscore", window=120)

# ── [46] MADistance_close_10_40__delta_10 ──
f76 = MADistance(short_window=10, long_window=40, price_column="close")
f77 = TransformFactor(dependency=f76, transform="delta", window=10)

# ── [47] MADistance_close_10_40__delta_5 ──
f78 = TransformFactor(dependency=f76, transform="delta", window=5)

# ── [48] MADistance_close_10_60__delta_10 ──
f79 = MADistance(short_window=10, long_window=60, price_column="close")
f80 = TransformFactor(dependency=f79, transform="delta", window=10)

# ── [49] MADistance_close_20_120__zscore_120 ──
f81 = MADistance(short_window=20, long_window=120, price_column="close")
f82 = TransformFactor(dependency=f81, transform="zscore", window=120)

# ── [50] MADistance_close_20_40__delta_5 ──
f83 = MADistance(short_window=20, long_window=40, price_column="close")
f84 = TransformFactor(dependency=f83, transform="delta", window=5)

# ── [51] MADistance_close_20_60__delta_10 ──
f85 = TransformFactor(dependency=f45, transform="delta", window=10)

# ── [52] MADistance_close_20_60__delta_5 ──
f86 = TransformFactor(dependency=f45, transform="delta", window=5)

# ── [53] MADistance_close_20_60__rolling_mean_5 ──
f87 = TransformFactor(dependency=f45, transform="rolling_mean", window=5)

# ── [54] MADistance_close_20_60__zscore_120 ──
f88 = TransformFactor(dependency=f45, transform="zscore", window=120)

# ── [55] MADistance_close_5_40__delta_10 ──
f89 = MADistance(short_window=5, long_window=40, price_column="close")
f90 = TransformFactor(dependency=f89, transform="delta", window=10)

# ── [56] MAE_20__zscore_120 ──
f91 = MaxAdverseExcursion(window=20)
f92 = TransformFactor(dependency=f91, transform="zscore", window=120)

# ── [57] MAE_40__binarize_winrate_10_-1.0 ──
f93 = TransformFactor(dependency=f53, transform="binarize_winrate", window=10, threshold=-1.0)

# ── [58] MAE_40__delta_10 ──
f94 = TransformFactor(dependency=f53, transform="delta", window=10)

# ── [59] MAE_40__delta_5 ──
f95 = TransformFactor(dependency=f53, transform="delta", window=5)

# ── [60] MAE_40__product_TimeSeriesMomentum_252 ──
f96 = CombineFactor(factor_a=f53, factor_b=f41, method="product")

# ── [61] MAE_40__rolling_mean_10 ──
f97 = TransformFactor(dependency=f53, transform="rolling_mean", window=10)

# ── [62] MAE_40__rolling_mean_5 ──
f98 = TransformFactor(dependency=f53, transform="rolling_mean", window=5)

# ── [63] MAE_40__zscore_120 ──
f99 = TransformFactor(dependency=f53, transform="zscore", window=120)

# ── [64] MAE_60__zscore_120 ──
f100 = MaxAdverseExcursion(window=60)
f101 = TransformFactor(dependency=f100, transform="zscore", window=120)

# ── [65] MAPosition_close_200__pct_change_10 ──
f102 = TransformFactor(dependency=f51, transform="pct_change", window=10)

# ── [66] MAPosition_close_200__rolling_mean_10 ──
f103 = TransformFactor(dependency=f51, transform="rolling_mean", window=10)

# ── [67] MAPosition_close_200__rolling_mean_5 ──
f104 = TransformFactor(dependency=f51, transform="rolling_mean", window=5)

# ── [68] MAPosition_close_250 ──
f105 = MAPosition(window=250, price_column="close")

# ── [69] MAPosition_close_250__binarize_winrate_10_0.0 ──
f106 = TransformFactor(dependency=f105, transform="binarize_winrate", window=10, threshold=0.0)

# ── [70] MAPosition_close_250__rolling_mean_10 ──
f107 = TransformFactor(dependency=f105, transform="rolling_mean", window=10)

# ── [71] MAPosition_close_250__rolling_mean_5 ──
f108 = TransformFactor(dependency=f105, transform="rolling_mean", window=5)

# ── [72] MASlope_close_10_10__delta_10 ──
f109 = MASlope(ma_window=10, slope_window=10, price_column="close")
f110 = TransformFactor(dependency=f109, transform="delta", window=10)

# ── [73] MASlope_close_10_10__rolling_mean_5 ──
f111 = TransformFactor(dependency=f109, transform="rolling_mean", window=5)

# ── [74] MASlope_close_10_3__rolling_mean_10 ──
f112 = MASlope(ma_window=10, slope_window=3, price_column="close")
f113 = TransformFactor(dependency=f112, transform="rolling_mean", window=10)

# ── [75] MASlope_close_10_3__rolling_mean_5 ──
f114 = TransformFactor(dependency=f112, transform="rolling_mean", window=5)

# ── [76] MASlope_close_10_5__rolling_mean_10 ──
f115 = MASlope(ma_window=10, slope_window=5, price_column="close")
f116 = TransformFactor(dependency=f115, transform="rolling_mean", window=10)

# ── [77] MASlope_close_10_5__rolling_mean_5 ──
f117 = TransformFactor(dependency=f115, transform="rolling_mean", window=5)

# ── [78] MASlope_close_10_5__weighted_sum_MAE_40 ──
f118 = CombineFactor(factor_a=f115, factor_b=f53, method="weighted_sum")

# ── [79] MASlope_close_20_10__delta_10 ──
f119 = MASlope(ma_window=20, slope_window=10, price_column="close")
f120 = TransformFactor(dependency=f119, transform="delta", window=10)

# ── [80] MASlope_close_20_10__delta_5 ──
f121 = TransformFactor(dependency=f119, transform="delta", window=5)

# ── [81] MASlope_close_20_3__delta_10 ──
f122 = MASlope(ma_window=20, slope_window=3, price_column="close")
f123 = TransformFactor(dependency=f122, transform="delta", window=10)

# ── [82] MASlope_close_20_5__delta_10 ──
f124 = MASlope(ma_window=20, slope_window=5, price_column="close")
f125 = TransformFactor(dependency=f124, transform="delta", window=10)

# ── [83] MASlope_close_40_10__zscore_120 ──
f126 = MASlope(ma_window=40, slope_window=10, price_column="close")
f127 = TransformFactor(dependency=f126, transform="zscore", window=120)

# ── [84] MASlope_close_40_3__zscore_120 ──
f128 = MASlope(ma_window=40, slope_window=3, price_column="close")
f129 = TransformFactor(dependency=f128, transform="zscore", window=120)

# ── [85] MASlope_close_40_5__zscore_120 ──
f130 = MASlope(ma_window=40, slope_window=5, price_column="close")
f131 = TransformFactor(dependency=f130, transform="zscore", window=120)

# ── [86] MA_close_10__pct_change_10 ──
f132 = MAFactor(window=10, price_column="close")
f133 = TransformFactor(dependency=f132, transform="pct_change", window=10)

# ── [87] MFE_20__delta_10 ──
f134 = MaxFavorableExcursion(window=20)
f135 = TransformFactor(dependency=f134, transform="delta", window=10)

# ── [88] MFE_40__zscore_120 ──
f136 = MaxFavorableExcursion(window=40)
f137 = TransformFactor(dependency=f136, transform="zscore", window=120)

# ── [89] MFI_14 ──
f138 = MFI(window=14)

# ── [90] MFI_14__delta_10 ──
f139 = TransformFactor(dependency=f138, transform="delta", window=10)

# ── [91] MFI_14__delta_5 ──
f140 = TransformFactor(dependency=f138, transform="delta", window=5)

# ── [92] MFI_14__if_HighPointPosition_20_lt_0.8_and_TrendR2_120_r2_gt_0.3__else_AvgDrawdown_120__neg ──
# SwitchFactor (multi-cond, and): MFI_14 / AvgDrawdown_120
f141 = SwitchFactor(
    signal_true=f138,
    signal_false=f1,
    conditions=[
        ConditionSpec(condition=f39, op="lt", threshold=0.8),
        ConditionSpec(condition=f52, op="gt", threshold=0.3),
    ],
    logic="and",
)
f142 = NegateFactor(f141)

# ── [93] MFI_14__if_MAPosition_close_200_gt_0.0 ──
# ConditionalFactor: MFI_14 if MAPosition_close_200 gt 0.0
f143 = ConditionalFactor(
    signal=f138,
    condition=f51,
    op="gt",
    threshold=0.0,
)

# ── [94] MFI_14__product_TimeSeriesMomentum_252 ──
f144 = CombineFactor(factor_a=f138, factor_b=f41, method="product")

# ── [95] MFI_14__rolling_mean_5 ──
f145 = TransformFactor(dependency=f138, transform="rolling_mean", window=5)

# ── [96] MFI_14__weighted_sum_HighPointPosition_20 ──
f146 = CombineFactor(factor_a=f138, factor_b=f39, method="weighted_sum")

# ── [97] MFI_14__weighted_sum_TimeSeriesMomentum_252 ──
f147 = CombineFactor(factor_a=f138, factor_b=f41, method="weighted_sum")

# ── [98] MFI_7__rolling_mean_10 ──
f148 = MFI(window=7)
f149 = TransformFactor(dependency=f148, transform="rolling_mean", window=10)

# ── [99] NewLowContinuous_50__delta_10 ──
f150 = NewLowContinuous(window=50)
f151 = TransformFactor(dependency=f150, transform="delta", window=10)

# ── [100] PriceReturn_10__rolling_mean_10 ──
f152 = PriceReturn(window=10)
f153 = TransformFactor(dependency=f152, transform="rolling_mean", window=10)

# ── [101] PriceReturn_20__diff_PriceReturn_60 ──
f154 = PriceReturn(window=20)
f155 = PriceReturn(window=60)
f156 = CombineFactor(factor_a=f154, factor_b=f155, method="diff")

# ── [102] PriceReturn_20__if_TrendR2_120_r2_gt_0.5 ──
# ConditionalFactor: PriceReturn_20 if TrendR2_120_r2 gt 0.5
f157 = ConditionalFactor(
    signal=f154,
    condition=f52,
    op="gt",
    threshold=0.5,
)

# ── [103] PriceReturn_20__product_KaufmanER_20 ──
f158 = KaufmanEfficiencyRatio(window=20)
f159 = CombineFactor(factor_a=f154, factor_b=f158, method="product")

# ── [104] PriceReturn_20__ratio_DownsideVolatility_20 ──
f160 = DownsideVolatility(window=20)
f161 = CombineFactor(factor_a=f154, factor_b=f160, method="ratio")

# ── [105] PriceReturn_40__zscore_120 ──
f162 = PriceReturn(window=40)
f163 = TransformFactor(dependency=f162, transform="zscore", window=120)

# ── [106] RSI_7__rolling_mean_5 ──
f164 = RSI(window=7)
f165 = TransformFactor(dependency=f164, transform="rolling_mean", window=5)

# ── [107] ShortTermReversal_10__rolling_mean_10 ──
f166 = ShortTermReversal(window=10)
f167 = TransformFactor(dependency=f166, transform="rolling_mean", window=10)

# ── [108] ShortTermReversal_5__binarize_winrate_10_0.0 ──
f168 = ShortTermReversal(window=5)
f169 = TransformFactor(dependency=f168, transform="binarize_winrate", window=10, threshold=0.0)

# ── [109] ShortTermReversal_5__delta_5 ──
f170 = TransformFactor(dependency=f168, transform="delta", window=5)

# ── [110] Stochastic_14_3_K__rolling_mean_10 ──
f171 = Stochastic(n=14, m=3, output="K")
f172 = TransformFactor(dependency=f171, transform="rolling_mean", window=10)

# ── [111] Stochastic_14_3_K__rolling_mean_5 ──
f173 = TransformFactor(dependency=f171, transform="rolling_mean", window=5)

# ── [112] Stochastic_14_5_K__rolling_mean_10 ──
f174 = Stochastic(n=14, m=5, output="K")
f175 = TransformFactor(dependency=f174, transform="rolling_mean", window=10)

# ── [113] Stochastic_14_5_K__rolling_mean_5 ──
f176 = TransformFactor(dependency=f174, transform="rolling_mean", window=5)

# ── [114] Stochastic_21_3_K__rolling_mean_5 ──
f177 = Stochastic(n=21, m=3, output="K")
f178 = TransformFactor(dependency=f177, transform="rolling_mean", window=5)

# ── [115] Stochastic_21_5_K__rolling_mean_5 ──
f179 = Stochastic(n=21, m=5, output="K")
f180 = TransformFactor(dependency=f179, transform="rolling_mean", window=5)

# ── [116] Stochastic_7_3_K__rolling_mean_10 ──
f181 = Stochastic(n=7, m=3, output="K")
f182 = TransformFactor(dependency=f181, transform="rolling_mean", window=10)

# ── [117] Stochastic_7_5_K__rolling_mean_10 ──
f183 = Stochastic(n=7, m=5, output="K")
f184 = TransformFactor(dependency=f183, transform="rolling_mean", window=10)

# ── [118] TimeSeriesMomentum_120__delta_10 ──
f185 = TimeSeriesMomentum(window=120)
f186 = TransformFactor(dependency=f185, transform="delta", window=10)

# ── [119] TimeSeriesMomentum_240 ──
f187 = TimeSeriesMomentum(window=240)

# ── [120] TimeSeriesMomentum_240__rolling_mean_10 ──
f188 = TransformFactor(dependency=f187, transform="rolling_mean", window=10)

# ── [121] TimeSeriesMomentum_240__rolling_mean_5 ──
f189 = TransformFactor(dependency=f187, transform="rolling_mean", window=5)

# ── [122] TimeSeriesMomentum_252 ──


# ── [123] TimeSeriesMomentum_252__binarize_winrate_10_0.0 ──
f190 = TransformFactor(dependency=f41, transform="binarize_winrate", window=10, threshold=0.0)

# ── [124] TimeSeriesMomentum_252__binarize_winrate_20_0.0 ──
f191 = TransformFactor(dependency=f41, transform="binarize_winrate", window=20, threshold=0.0)

# ── [125] TimeSeriesMomentum_252__delta_10 ──
f192 = TransformFactor(dependency=f41, transform="delta", window=10)

# ── [126] TimeSeriesMomentum_252__if_TrendR2_120_r2_gt_0.3 ──
# ConditionalFactor: TimeSeriesMomentum_252 if TrendR2_120_r2 gt 0.3
f193 = ConditionalFactor(
    signal=f41,
    condition=f52,
    op="gt",
    threshold=0.3,
)

# ── [127] TimeSeriesMomentum_252__rolling_mean_10 ──
f194 = TransformFactor(dependency=f41, transform="rolling_mean", window=10)

# ── [128] TimeSeriesMomentum_252__rolling_mean_5 ──
f195 = TransformFactor(dependency=f41, transform="rolling_mean", window=5)

# ── [129] TimeSeriesMomentum_252__zscore_252 ──
f196 = TransformFactor(dependency=f41, transform="zscore", window=252)

# ── [130] TimeSeriesMomentum_252__zscore_252__if_MADistance_close_20_60__zscore_120_lt_0.0__else_MFI_14 ──
f197 = TransformFactor(dependency=f41, transform="zscore", window=252)
f199 = TransformFactor(dependency=f45, transform="zscore", window=120)
# SwitchFactor: TimeSeriesMomentum_252__zscore_252 / MFI_14
f198 = SwitchFactor(
    signal_true=f197,
    signal_false=f138,
    condition=f199,
    op="lt",
    threshold=0.0,
)

# ── [131] TimeSeriesMomentum_252__zscore_252__if_TrendR2_120_r2_gt_0.5_and_MAE_40__zscore_120_lt_1.0 ──
f201 = TransformFactor(dependency=f41, transform="zscore", window=252)
f200 = TransformFactor(dependency=f53, transform="zscore", window=120)
# MultiConditionalFactor (and): TimeSeriesMomentum_252__zscore_252
f202 = MultiConditionalFactor(
    signal=f201,
    conditions=[
        ConditionSpec(condition=f52, op="gt", threshold=0.5),
        ConditionSpec(condition=f200, op="lt", threshold=1.0),
    ],
    logic="and",
)

# ── [132] TrendR2_120_r2__rolling_mean_10 ──
f203 = TransformFactor(dependency=f52, transform="rolling_mean", window=10)

# ── [133] TrendR2_120_r2__rolling_mean_5 ──
f204 = TransformFactor(dependency=f52, transform="rolling_mean", window=5)

# ── [134] TrendR2_120_slope__delta_10 ──
f205 = TrendR2Factor(window=120, output="slope")
f206 = TransformFactor(dependency=f205, transform="delta", window=10)

# ── [135] TrendR2_120_slope__delta_5 ──
f207 = TransformFactor(dependency=f205, transform="delta", window=5)

# ── [136] TrendR2_240_r2__rolling_mean_10 ──
f208 = TrendR2Factor(window=240, output="r2")
f209 = TransformFactor(dependency=f208, transform="rolling_mean", window=10)

# ── [137] TrendR2_240_slope__rolling_mean_10 ──
f210 = TrendR2Factor(window=240, output="slope")
f211 = TransformFactor(dependency=f210, transform="rolling_mean", window=10)

# ── [138] TrendR2_240_slope__rolling_mean_5 ──
f212 = TransformFactor(dependency=f210, transform="rolling_mean", window=5)

# ── [139] TrendR2_60_slope__zscore_120 ──
f213 = TrendR2Factor(window=60, output="slope")
f214 = TransformFactor(dependency=f213, transform="zscore", window=120)

# ── [140] VolumeReversal_5_10__binarize_winrate_10_0.0 ──
f215 = VolumeReversal(ret_window=5, vol_window=10)
f216 = TransformFactor(dependency=f215, transform="binarize_winrate", window=10, threshold=0.0)

# ── [141] VolumeReversal_5_20__binarize_winrate_10_0.0 ──
f217 = VolumeReversal(ret_window=5, vol_window=20)
f218 = TransformFactor(dependency=f217, transform="binarize_winrate", window=10, threshold=0.0)

# ── [142] WilliamsR_14__rolling_mean_10 ──
f219 = WilliamsR(window=14)
f220 = TransformFactor(dependency=f219, transform="rolling_mean", window=10)

# ── [143] WilliamsR_14__rolling_mean_5 ──
f221 = TransformFactor(dependency=f219, transform="rolling_mean", window=5)

# ── [144] WilliamsR_21__rolling_mean_5 ──
f222 = WilliamsR(window=21)
f223 = TransformFactor(dependency=f222, transform="rolling_mean", window=5)

# ── [145] WilliamsR_7__rolling_mean_10 ──
f224 = WilliamsR(window=7)
f225 = TransformFactor(dependency=f224, transform="rolling_mean", window=10)


# ====================================================================
# 共享管道因子
# ====================================================================
vol20 = Volatility(window=20)

SHARED_PIPELINE: tuple = (vol20,)


# ====================================================================
# 组定义: 145 组（145 因子，无过滤器）
# ====================================================================
# (label, ranking_factor, builtin_filters)
GROUPS: list[tuple[str, object, tuple]] = [
    # [1] AvgDrawdown_120
    ("AvgDrawdown_120",          f1, ()),
    # [2] AvgDrawdown_120__binarize_winrate_10_-2.0
    ("AvgDrawdown_120_binarize_winrate_10__2_0",          f2, ()),
    # [3] AvgDrawdown_120__delta_10
    ("AvgDrawdown_120_delta_10",          f3, ()),
    # [4] AvgDrawdown_120__rolling_mean_10
    ("AvgDrawdown_120_rolling_mean_10",          f4, ()),
    # [5] AvgDrawdown_120__rolling_mean_5
    ("AvgDrawdown_120_rolling_mean_5",          f5, ()),
    # [6] AvgDrawdown_120__zscore_252
    ("AvgDrawdown_120_zscore_252",          f6, ()),
    # [7] AvgDrawdown_250__rolling_mean_5
    ("AvgDrawdown_250_rolling_mean_5",          f8, ()),
    # [8] AvgDrawdown_40__rolling_mean_5
    ("AvgDrawdown_40_rolling_mean_5",          f10, ()),
    # [9] AvgDrawdown_40__zscore_120
    ("AvgDrawdown_40_zscore_120",          f11, ()),
    # [10] BIAS_close_10__binarize_winrate_10_0.0
    ("BIAS_close_10_binarize_winrate_10_0_0",          f13, ()),
    # [11] BIAS_close_10__delta_5
    ("BIAS_close_10_delta_5",          f14, ()),
    # [12] BIAS_close_10__rolling_mean_10
    ("BIAS_close_10_rolling_mean_10",          f15, ()),
    # [13] BollingerBandPosition_close_10_1.5__binarize_winrate_10_0.0
    ("BollingerBandPosition_close_10_1_5_binarize_winrate_10_0_0",          f17, ()),
    # [14] BollingerBandPosition_close_10_2.0__binarize_winrate_10_0.0
    ("BollingerBandPosition_close_10_2_0_binarize_winrate_10_0_0",          f19, ()),
    # [15] BollingerBandPosition_close_10_2.5__binarize_winrate_10_0.0
    ("BollingerBandPosition_close_10_2_5_binarize_winrate_10_0_0",          f21, ()),
    # [16] BollingerBandPosition_close_20_2.0__delta_5
    ("BollingerBandPosition_close_20_2_0_delta_5",          f23, ()),
    # [17] BollingerBandPosition_close_20_2.0__zscore_120
    ("BollingerBandPosition_close_20_2_0_zscore_120",          f24, ()),
    # [18] CCI_10__binarize_winrate_10_0.0
    ("CCI_10_binarize_winrate_10_0_0",          f26, ()),
    # [19] CCI_20__rolling_mean_5
    ("CCI_20_rolling_mean_5",          f28, ()),
    # [20] CCI_20__zscore_120
    ("CCI_20_zscore_120",          f29, ()),
    # [21] DailyRebound
    ("DailyRebound",          f30, ()),
    # [22] DonchianPosition_10__rolling_mean_10
    ("DonchianPosition_10_rolling_mean_10",          f32, ()),
    # [23] DonchianPosition_20__delta_10
    ("DonchianPosition_20_delta_10",          f34, ()),
    # [24] DonchianPosition_20__rolling_mean_5
    ("DonchianPosition_20_rolling_mean_5",          f35, ()),
    # [25] DonchianPosition_20__zscore_120
    ("DonchianPosition_20_zscore_120",          f36, ()),
    # [26] HighPointPosition_10__rolling_mean_10
    ("HighPointPosition_10_rolling_mean_10",          f38, ()),
    # [27] HighPointPosition_20
    ("HighPointPosition_20",          f39, ()),
    # [28] HighPointPosition_20__delta_5
    ("HighPointPosition_20_delta_5",          f40, ()),
    # [29] HighPointPosition_20__product_TimeSeriesMomentum_252
    ("HighPointPosition_20_product_TimeSeriesMomentum_252",          f42, ()),
    # [30] HighPointPosition_20__rolling_mean_5
    ("HighPointPosition_20_rolling_mean_5",          f43, ()),
    # [31] HighPointPosition_20__weighted_sum_AvgDrawdown_120
    ("HighPointPosition_20_weighted_sum_AvgDrawdown_120",          f44, ()),
    # [32] HighPointPosition_20__weighted_sum_MADistance_close_20_60
    ("HighPointPosition_20_weighted_sum_MADistance_close_20_60",          f46, ()),
    # [33] HighPointPosition_20__weighted_sum_TimeSeriesMomentum_252
    ("HighPointPosition_20_weighted_sum_TimeSeriesMomentum_252",          f47, ()),
    # [34] HighPointPosition_20__weighted_sum_TimeSeriesMomentum_252__if_MAPosition_close_200_gt_0_and_TrendR2_120_r2_gt_0.3__else_HighPointPosition_20__weighted_sum_AvgDrawdown_120
    ("HighPointPosition_20_weighted_sum_TimeSeriesMomentum_252_if_MAPosition_clos___",          f50, ()),
    # [35] HighPointPosition_20__weighted_sum_TimeSeriesMomentum_252__if_MAPosition_close_200_gt_0_and_TrendR2_120_r2_gt_0.3_and_MAE_40__zscore_120_lt_1.5
    ("HighPointPosition_20_weighted_sum_TimeSeriesMomentum_252_if_MAPosition_clos___",          f56, ()),
    # [36] HighPointPosition_20__zscore_252
    ("HighPointPosition_20_zscore_252",          f57, ()),
    # [37] HighPointPosition_20__zscore_252__if_MAE_40__zscore_120_lt_1.5
    ("HighPointPosition_20_zscore_252_if_MAE_40_zscore_120_lt_1_5",          f60, ()),
    # [38] HighPointPosition_20__zscore_252__if_TrendR2_120_r2_gt_0.3__else_TimeSeriesMomentum_252
    ("HighPointPosition_20_zscore_252_if_TrendR2_120_r2_gt_0_3_else_TimeSeriesMo___",          f62, ()),
    # [39] HighPointPosition_20__zscore_252__if_TrendR2_120_r2_gt_0.5__else_AvgDrawdown_120__neg
    ("HighPointPosition_20_zscore_252_if_TrendR2_120_r2_gt_0_5_else_AvgDrawdown____",          f65, ()),
    # [40] HighPointPosition_20__zscore_252__if_TrendR2_120_r2_gt_0.5__else_TimeSeriesMomentum_252
    ("HighPointPosition_20_zscore_252_if_TrendR2_120_r2_gt_0_5_else_TimeSeriesMo___",          f67, ()),
    # [41] LowPointPosition_10__binarize_winrate_10_0.0
    ("LowPointPosition_10_binarize_winrate_10_0_0",          f69, ()),
    # [42] LowPointPosition_10__rolling_mean_10
    ("LowPointPosition_10_rolling_mean_10",          f70, ()),
    # [43] MADispersion_close_5_10_20_60__delta_10
    ("MADispersion_close_5_10_20_60_delta_10",          f72, ()),
    # [44] MADispersion_close_5_10_20_60__delta_5
    ("MADispersion_close_5_10_20_60_delta_5",          f73, ()),
    # [45] MADistance_close_10_20__zscore_120
    ("MADistance_close_10_20_zscore_120",          f75, ()),
    # [46] MADistance_close_10_40__delta_10
    ("MADistance_close_10_40_delta_10",          f77, ()),
    # [47] MADistance_close_10_40__delta_5
    ("MADistance_close_10_40_delta_5",          f78, ()),
    # [48] MADistance_close_10_60__delta_10
    ("MADistance_close_10_60_delta_10",          f80, ()),
    # [49] MADistance_close_20_120__zscore_120
    ("MADistance_close_20_120_zscore_120",          f82, ()),
    # [50] MADistance_close_20_40__delta_5
    ("MADistance_close_20_40_delta_5",          f84, ()),
    # [51] MADistance_close_20_60__delta_10
    ("MADistance_close_20_60_delta_10",          f85, ()),
    # [52] MADistance_close_20_60__delta_5
    ("MADistance_close_20_60_delta_5",          f86, ()),
    # [53] MADistance_close_20_60__rolling_mean_5
    ("MADistance_close_20_60_rolling_mean_5",          f87, ()),
    # [54] MADistance_close_20_60__zscore_120
    ("MADistance_close_20_60_zscore_120",          f88, ()),
    # [55] MADistance_close_5_40__delta_10
    ("MADistance_close_5_40_delta_10",          f90, ()),
    # [56] MAE_20__zscore_120
    ("MAE_20_zscore_120",          f92, ()),
    # [57] MAE_40__binarize_winrate_10_-1.0
    ("MAE_40_binarize_winrate_10__1_0",          f93, ()),
    # [58] MAE_40__delta_10
    ("MAE_40_delta_10",          f94, ()),
    # [59] MAE_40__delta_5
    ("MAE_40_delta_5",          f95, ()),
    # [60] MAE_40__product_TimeSeriesMomentum_252
    ("MAE_40_product_TimeSeriesMomentum_252",          f96, ()),
    # [61] MAE_40__rolling_mean_10
    ("MAE_40_rolling_mean_10",          f97, ()),
    # [62] MAE_40__rolling_mean_5
    ("MAE_40_rolling_mean_5",          f98, ()),
    # [63] MAE_40__zscore_120
    ("MAE_40_zscore_120",          f99, ()),
    # [64] MAE_60__zscore_120
    ("MAE_60_zscore_120",          f101, ()),
    # [65] MAPosition_close_200__pct_change_10
    ("MAPosition_close_200_pct_change_10",          f102, ()),
    # [66] MAPosition_close_200__rolling_mean_10
    ("MAPosition_close_200_rolling_mean_10",          f103, ()),
    # [67] MAPosition_close_200__rolling_mean_5
    ("MAPosition_close_200_rolling_mean_5",          f104, ()),
    # [68] MAPosition_close_250
    ("MAPosition_close_250",          f105, ()),
    # [69] MAPosition_close_250__binarize_winrate_10_0.0
    ("MAPosition_close_250_binarize_winrate_10_0_0",          f106, ()),
    # [70] MAPosition_close_250__rolling_mean_10
    ("MAPosition_close_250_rolling_mean_10",          f107, ()),
    # [71] MAPosition_close_250__rolling_mean_5
    ("MAPosition_close_250_rolling_mean_5",          f108, ()),
    # [72] MASlope_close_10_10__delta_10
    ("MASlope_close_10_10_delta_10",          f110, ()),
    # [73] MASlope_close_10_10__rolling_mean_5
    ("MASlope_close_10_10_rolling_mean_5",          f111, ()),
    # [74] MASlope_close_10_3__rolling_mean_10
    ("MASlope_close_10_3_rolling_mean_10",          f113, ()),
    # [75] MASlope_close_10_3__rolling_mean_5
    ("MASlope_close_10_3_rolling_mean_5",          f114, ()),
    # [76] MASlope_close_10_5__rolling_mean_10
    ("MASlope_close_10_5_rolling_mean_10",          f116, ()),
    # [77] MASlope_close_10_5__rolling_mean_5
    ("MASlope_close_10_5_rolling_mean_5",          f117, ()),
    # [78] MASlope_close_10_5__weighted_sum_MAE_40
    ("MASlope_close_10_5_weighted_sum_MAE_40",          f118, ()),
    # [79] MASlope_close_20_10__delta_10
    ("MASlope_close_20_10_delta_10",          f120, ()),
    # [80] MASlope_close_20_10__delta_5
    ("MASlope_close_20_10_delta_5",          f121, ()),
    # [81] MASlope_close_20_3__delta_10
    ("MASlope_close_20_3_delta_10",          f123, ()),
    # [82] MASlope_close_20_5__delta_10
    ("MASlope_close_20_5_delta_10",          f125, ()),
    # [83] MASlope_close_40_10__zscore_120
    ("MASlope_close_40_10_zscore_120",          f127, ()),
    # [84] MASlope_close_40_3__zscore_120
    ("MASlope_close_40_3_zscore_120",          f129, ()),
    # [85] MASlope_close_40_5__zscore_120
    ("MASlope_close_40_5_zscore_120",          f131, ()),
    # [86] MA_close_10__pct_change_10
    ("MA_close_10_pct_change_10",          f133, ()),
    # [87] MFE_20__delta_10
    ("MFE_20_delta_10",          f135, ()),
    # [88] MFE_40__zscore_120
    ("MFE_40_zscore_120",          f137, ()),
    # [89] MFI_14
    ("MFI_14",          f138, ()),
    # [90] MFI_14__delta_10
    ("MFI_14_delta_10",          f139, ()),
    # [91] MFI_14__delta_5
    ("MFI_14_delta_5",          f140, ()),
    # [92] MFI_14__if_HighPointPosition_20_lt_0.8_and_TrendR2_120_r2_gt_0.3__else_AvgDrawdown_120__neg
    ("MFI_14_if_HighPointPosition_20_lt_0_8_and_TrendR2_120_r2_gt_0_3_else_AvgDra___",          f142, ()),
    # [93] MFI_14__if_MAPosition_close_200_gt_0.0
    ("MFI_14_if_MAPosition_close_200_gt_0_0",          f143, ()),
    # [94] MFI_14__product_TimeSeriesMomentum_252
    ("MFI_14_product_TimeSeriesMomentum_252",          f144, ()),
    # [95] MFI_14__rolling_mean_5
    ("MFI_14_rolling_mean_5",          f145, ()),
    # [96] MFI_14__weighted_sum_HighPointPosition_20
    ("MFI_14_weighted_sum_HighPointPosition_20",          f146, ()),
    # [97] MFI_14__weighted_sum_TimeSeriesMomentum_252
    ("MFI_14_weighted_sum_TimeSeriesMomentum_252",          f147, ()),
    # [98] MFI_7__rolling_mean_10
    ("MFI_7_rolling_mean_10",          f149, ()),
    # [99] NewLowContinuous_50__delta_10
    ("NewLowContinuous_50_delta_10",          f151, ()),
    # [100] PriceReturn_10__rolling_mean_10
    ("PriceReturn_10_rolling_mean_10",          f153, ()),
    # [101] PriceReturn_20__diff_PriceReturn_60
    ("PriceReturn_20_diff_PriceReturn_60",          f156, ()),
    # [102] PriceReturn_20__if_TrendR2_120_r2_gt_0.5
    ("PriceReturn_20_if_TrendR2_120_r2_gt_0_5",          f157, ()),
    # [103] PriceReturn_20__product_KaufmanER_20
    ("PriceReturn_20_product_KaufmanER_20",          f159, ()),
    # [104] PriceReturn_20__ratio_DownsideVolatility_20
    ("PriceReturn_20_ratio_DownsideVolatility_20",          f161, ()),
    # [105] PriceReturn_40__zscore_120
    ("PriceReturn_40_zscore_120",          f163, ()),
    # [106] RSI_7__rolling_mean_5
    ("RSI_7_rolling_mean_5",          f165, ()),
    # [107] ShortTermReversal_10__rolling_mean_10
    ("ShortTermReversal_10_rolling_mean_10",          f167, ()),
    # [108] ShortTermReversal_5__binarize_winrate_10_0.0
    ("ShortTermReversal_5_binarize_winrate_10_0_0",          f169, ()),
    # [109] ShortTermReversal_5__delta_5
    ("ShortTermReversal_5_delta_5",          f170, ()),
    # [110] Stochastic_14_3_K__rolling_mean_10
    ("Stochastic_14_3_K_rolling_mean_10",          f172, ()),
    # [111] Stochastic_14_3_K__rolling_mean_5
    ("Stochastic_14_3_K_rolling_mean_5",          f173, ()),
    # [112] Stochastic_14_5_K__rolling_mean_10
    ("Stochastic_14_5_K_rolling_mean_10",          f175, ()),
    # [113] Stochastic_14_5_K__rolling_mean_5
    ("Stochastic_14_5_K_rolling_mean_5",          f176, ()),
    # [114] Stochastic_21_3_K__rolling_mean_5
    ("Stochastic_21_3_K_rolling_mean_5",          f178, ()),
    # [115] Stochastic_21_5_K__rolling_mean_5
    ("Stochastic_21_5_K_rolling_mean_5",          f180, ()),
    # [116] Stochastic_7_3_K__rolling_mean_10
    ("Stochastic_7_3_K_rolling_mean_10",          f182, ()),
    # [117] Stochastic_7_5_K__rolling_mean_10
    ("Stochastic_7_5_K_rolling_mean_10",          f184, ()),
    # [118] TimeSeriesMomentum_120__delta_10
    ("TimeSeriesMomentum_120_delta_10",          f186, ()),
    # [119] TimeSeriesMomentum_240
    ("TimeSeriesMomentum_240",          f187, ()),
    # [120] TimeSeriesMomentum_240__rolling_mean_10
    ("TimeSeriesMomentum_240_rolling_mean_10",          f188, ()),
    # [121] TimeSeriesMomentum_240__rolling_mean_5
    ("TimeSeriesMomentum_240_rolling_mean_5",          f189, ()),
    # [122] TimeSeriesMomentum_252
    ("TimeSeriesMomentum_252",          f41, ()),
    # [123] TimeSeriesMomentum_252__binarize_winrate_10_0.0
    ("TimeSeriesMomentum_252_binarize_winrate_10_0_0",          f190, ()),
    # [124] TimeSeriesMomentum_252__binarize_winrate_20_0.0
    ("TimeSeriesMomentum_252_binarize_winrate_20_0_0",          f191, ()),
    # [125] TimeSeriesMomentum_252__delta_10
    ("TimeSeriesMomentum_252_delta_10",          f192, ()),
    # [126] TimeSeriesMomentum_252__if_TrendR2_120_r2_gt_0.3
    ("TimeSeriesMomentum_252_if_TrendR2_120_r2_gt_0_3",          f193, ()),
    # [127] TimeSeriesMomentum_252__rolling_mean_10
    ("TimeSeriesMomentum_252_rolling_mean_10",          f194, ()),
    # [128] TimeSeriesMomentum_252__rolling_mean_5
    ("TimeSeriesMomentum_252_rolling_mean_5",          f195, ()),
    # [129] TimeSeriesMomentum_252__zscore_252
    ("TimeSeriesMomentum_252_zscore_252",          f196, ()),
    # [130] TimeSeriesMomentum_252__zscore_252__if_MADistance_close_20_60__zscore_120_lt_0.0__else_MFI_14
    ("TimeSeriesMomentum_252_zscore_252_if_MADistance_close_20_60_zscore_120_lt____",          f198, ()),
    # [131] TimeSeriesMomentum_252__zscore_252__if_TrendR2_120_r2_gt_0.5_and_MAE_40__zscore_120_lt_1.0
    ("TimeSeriesMomentum_252_zscore_252_if_TrendR2_120_r2_gt_0_5_and_MAE_40_zsco___",          f202, ()),
    # [132] TrendR2_120_r2__rolling_mean_10
    ("TrendR2_120_r2_rolling_mean_10",          f203, ()),
    # [133] TrendR2_120_r2__rolling_mean_5
    ("TrendR2_120_r2_rolling_mean_5",          f204, ()),
    # [134] TrendR2_120_slope__delta_10
    ("TrendR2_120_slope_delta_10",          f206, ()),
    # [135] TrendR2_120_slope__delta_5
    ("TrendR2_120_slope_delta_5",          f207, ()),
    # [136] TrendR2_240_r2__rolling_mean_10
    ("TrendR2_240_r2_rolling_mean_10",          f209, ()),
    # [137] TrendR2_240_slope__rolling_mean_10
    ("TrendR2_240_slope_rolling_mean_10",          f211, ()),
    # [138] TrendR2_240_slope__rolling_mean_5
    ("TrendR2_240_slope_rolling_mean_5",          f212, ()),
    # [139] TrendR2_60_slope__zscore_120
    ("TrendR2_60_slope_zscore_120",          f214, ()),
    # [140] VolumeReversal_5_10__binarize_winrate_10_0.0
    ("VolumeReversal_5_10_binarize_winrate_10_0_0",          f216, ()),
    # [141] VolumeReversal_5_20__binarize_winrate_10_0.0
    ("VolumeReversal_5_20_binarize_winrate_10_0_0",          f218, ()),
    # [142] WilliamsR_14__rolling_mean_10
    ("WilliamsR_14_rolling_mean_10",          f220, ()),
    # [143] WilliamsR_14__rolling_mean_5
    ("WilliamsR_14_rolling_mean_5",          f221, ()),
    # [144] WilliamsR_21__rolling_mean_5
    ("WilliamsR_21_rolling_mean_5",          f223, ()),
    # [145] WilliamsR_7__rolling_mean_10
    ("WilliamsR_7_rolling_mean_10",          f225, ()),
]

GROUPS = GROUPS[33:]


# ====================================================================
# Grid Search 参数
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1, 5, 10)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10)
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

def _adaptive_tiered(candidates):
    """自适应分档：前 40% 权重 1.5，后 60% 权重 1.0。"""
    if not candidates:
        return {}
    n = len(candidates)
    top_count = max(1, round(n * 0.4))
    weights = {}
    for i, c in enumerate(candidates):
        weights[c.symbol] = 1.5 if i < top_count else 1.0
    return weights
_adaptive_tiered.__name__ = "tiered"
alloc_tiered = _adaptive_tiered


WEIGHT_ALLOCATORS: tuple = (
    alloc_inv_vol,
)


# ====================================================================
# 执行参数
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
TITLE: str              = "宽动量基线回测 — 全量因子扫描 (factors0706_all)"
START_DATE: str         = "2020-01-01"
END_DATE: str           = "2026-05-29"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None