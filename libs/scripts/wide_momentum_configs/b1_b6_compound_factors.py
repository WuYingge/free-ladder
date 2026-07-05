"""B1~B6 复合/条件因子回测 — 12 组回测配置。

6 排名因子 × 2 过滤器配置（纯排名 / 三层过滤）。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.b1_b6_compound_factors

因子清单:
    B1  `MASlope_close_10_5__weighted_sum_MAE_40`                         复合（加权和）
    B2  `HighPointPosition_20__weighted_sum_MADistance_close_20_60`       复合（加权和）
    B3  `HighPointPosition_20__product_TimeSeriesMomentum_252`            复合（乘积）
    B4  `MFI_14__product_TimeSeriesMomentum_252`                          复合（乘积）
    B5  `TimeSeriesMomentum_252__if_TrendR2_120_r2_gt_0.3`                条件
    B6  `HighPointPosition_20__weighted_sum_TimeSeriesMomentum_252__if_...` 多条件 AND
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)
from factors.price_momentum import HighPointPosition, TimeSeriesMomentum
from factors.oscillator import MFI
from factors.trend_r2 import TrendR2Factor
from factors.rsrs import RsrsFactor
from factors.volatility import Volatility
from factors.ma import MAPosition, MASlope, MADistance
from factors.distribution_family import MaxAdverseExcursion
from factors.meta_factor import (
    TransformFactor,
    CombineFactor,
    ConditionalFactor,
    MultiConditionalFactor,
    ConditionSpec,
)


# ====================================================================
# 因子定义
# ====================================================================

# ── 基础因子 ──
maslope_10_5  = MASlope(ma_window=10, slope_window=5, price_column="close")
mae_40        = MaxAdverseExcursion(window=40)
hpp20         = HighPointPosition(window=20)
tsm252        = TimeSeriesMomentum(window=252)
mfi14         = MFI(window=14)
madist_20_60  = MADistance(short_window=20, long_window=60, price_column="close")
trend_r2_120  = TrendR2Factor(window=120, output="r2")
ma200         = MAPosition(window=200, price_column="close")

# ── 变换因子（B6 条件）──
mae40_zscore = TransformFactor(dependency=mae_40, transform="zscore", window=120)

# ── B1: MASlope_close_10_5__weighted_sum_MAE_40 ──
# 公式: 0.5 × rank(MASlope) + 0.5 × rank(-MAE_40)
b1_factor = CombineFactor(
    factor_a=maslope_10_5,
    factor_b=mae_40,
    method="weighted_sum",
    weight_b=-0.5,
)

# ── B2: HighPointPosition_20__weighted_sum_MADistance_close_20_60 ──
# 公式: 0.5 × rank(HPP_20) + 0.5 × rank(-MADistance_20_60)
b2_factor = CombineFactor(
    factor_a=hpp20,
    factor_b=madist_20_60,
    method="weighted_sum",
    weight_b=-0.5,
)

# ── B3: HighPointPosition_20__product_TimeSeriesMomentum_252 ──
b3_factor = CombineFactor(
    factor_a=hpp20,
    factor_b=tsm252,
    method="product",
)

# ── B4: MFI_14__product_TimeSeriesMomentum_252 ──
b4_factor = CombineFactor(
    factor_a=mfi14,
    factor_b=tsm252,
    method="product",
)

# ── B5: TimeSeriesMomentum_252__if_TrendR2_120_r2_gt_0.3 ──
b5_factor = ConditionalFactor(
    signal=tsm252,
    condition=trend_r2_120,
    op="gt",
    threshold=0.3,
)

# ── B6: HighPointPosition_20__weighted_sum_TimeSeriesMomentum_252
#        __if_MAPosition_close_200_gt_0.0_and_TrendR2_120_r2_gt_0.3_and_MAE_40__zscore_120_lt_1.5 ──
b6_signal = CombineFactor(
    factor_a=hpp20,
    factor_b=tsm252,
    method="weighted_sum",
)

b6_factor = MultiConditionalFactor(
    signal=b6_signal,
    conditions=[
        ConditionSpec(condition=ma200, op="gt", threshold=0.0),
        ConditionSpec(condition=trend_r2_120, op="gt", threshold=0.3),
        ConditionSpec(condition=mae40_zscore, op="lt", threshold=1.5),
    ],
    logic="and",
)


# ====================================================================
# 共享管道因子（三层过滤器用）
# ====================================================================
trend_r2 = TrendR2Factor(window=120, output="r2")
rsrs     = RsrsFactor(regression_window=14, zscore_window=600, output="zscore")
vol20    = Volatility(window=20)

# 共享管道（不含排名因子 — 排名因子由各组单独指定）
SHARED_PIPELINE: tuple = (rsrs, trend_r2, ma200, vol20)


# ====================================================================
# 过滤器定义
# ====================================================================
FILT_MA200 = ThresholdFilter(field=ma200.get_output_name(), operator=">=", value=0)
FILT_RSRS  = ThresholdFilter(field=rsrs.get_output_name(), operator=">=", value=0)
FILT_TR2R2 = ThresholdFilter(field=trend_r2.get_output_name(), operator=">=", value=0.5)

_FULL_FILTERS = (FILT_MA200, FILT_RSRS, FILT_TR2R2)


# ====================================================================
# 组定义: 12 组（6 因子 × 2 过滤器配置）
# ====================================================================
# (label, ranking_factor, builtin_filters)
GROUPS: list[tuple[str, object, tuple[ThresholdFilter, ...]]] = [
    # ── B1: MASlope_close_10_5__weighted_sum_MAE_40 ──
    ("B1_MASlope_wsum_MAE40",       b1_factor, ()),
    ("B1_MASlope_wsum_MAE40_filt",  b1_factor, _FULL_FILTERS),
    # ── B2: HighPointPosition_20__weighted_sum_MADistance_close_20_60 ──
    ("B2_HPP20_wsum_MADist",        b2_factor, ()),
    ("B2_HPP20_wsum_MADist_filt",   b2_factor, _FULL_FILTERS),
    # ── B3: HighPointPosition_20__product_TimeSeriesMomentum_252 ──
    ("B3_HPP20_prod_TSM252",        b3_factor, ()),
    ("B3_HPP20_prod_TSM252_filt",   b3_factor, _FULL_FILTERS),
    # ── B4: MFI_14__product_TimeSeriesMomentum_252 ──
    ("B4_MFI14_prod_TSM252",        b4_factor, ()),
    ("B4_MFI14_prod_TSM252_filt",   b4_factor, _FULL_FILTERS),
    # ── B5: TimeSeriesMomentum_252__if_TrendR2_120_r2_gt_0.3 ──
    ("B5_TSM252_if_TR2R2_gt03",     b5_factor, ()),
    ("B5_TSM252_if_TR2R2_gt03_filt", b5_factor, _FULL_FILTERS),
    # ── B6: 复合信号 + 多条件 AND ──
    ("B6_HPP_TSM_if_3cond",         b6_factor, ()),
    ("B6_HPP_TSM_if_3cond_filt",    b6_factor, _FULL_FILTERS),
]


# ====================================================================
# Grid Search 参数
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1, 5, 10, 20)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 20)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (True, False)
GRID_HOLD_OVERLAP: tuple[bool, ...] = (False, True)


# ====================================================================
# 权重分配器
# ====================================================================
# 方案 0: 等权（基线）
alloc_equal = equal_weight_allocator

# 方案 1: 动量加权 — 权重 ∝ 排名因子 score
alloc_momentum = score_proportional_allocator
alloc_momentum.__name__ = "momentum"

# 方案 2: 反波动率加权 — 权重 ∝ 1/σ_20d
alloc_inv_vol = make_factor_weighted_allocator(vol20.get_output_name(), inverse=True)
alloc_inv_vol.__name__ = "invvol"

# 方案 3: 分档加权 — 自适应比例：前 40% 权重×1.5，后 60% 权重×1.0
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

# 方案 4: RSRS 强度加权 — 权重 ∝ RSRS zscore
alloc_rsrs = make_factor_weighted_allocator(rsrs.get_output_name())
alloc_rsrs.__name__ = "rsrs"

# 当前启用的分配器
WEIGHT_ALLOCATORS: tuple = (
    # alloc_equal,
    # alloc_momentum,
    alloc_inv_vol,
    # alloc_tiered,
    # alloc_rsrs,
)


# ====================================================================
# 执行参数
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
TITLE: str              = "宽动量基线回测 — B1~B6 复合/条件因子测试"
START_DATE: str         = "2020-01-01"
END_DATE: str           = "2026-05-29"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
