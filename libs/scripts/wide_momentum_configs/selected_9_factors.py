"""9 因子回测 — 18 组回测配置。

9 排名因子 × 2 过滤器配置（纯排名 / 三层过滤）。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.selected_9_factors

因子清单:
    F1  TSM_240                                       时序动量（240 日）
    F2  TSM_240__rolling_mean_5                       时序动量 → 5 日均值平滑
    F3  TrendR2_120_r2__rolling_mean_5                趋势 R² → 5 日均值平滑
    F4  DailyRebound                                  日内反弹
    F5  MAPosition_close_250                          250 日均线位置
    F6  AvgDrawdown_120                               平均回撤（120 日，raw）
    F7  DonchianPosition_20__zscore_120               唐奇安通道位置 → zscore(120)
    F8  BollingerBandPosition_close_20_2.0__zscore_120 布林带位置 → zscore(120)
    F9  MADistance_close_20_60__delta_5               均线距离 → delta(5)
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)
from factors.price_momentum import TimeSeriesMomentum
from factors.trend_r2 import TrendR2Factor
from factors.daily_rebound import DailyRebound
from factors.ma import MAPosition, BollingerBandPosition, MADistance
from factors.volatility_family import AvgDrawdown
from factors.breakout_family import DonchianChannelPosition
from factors.meta_factor import TransformFactor
from factors.rsrs import RsrsFactor
from factors.volatility import Volatility


# ====================================================================
# 因子定义
# ====================================================================

# ── F1: TSM_240 ──
tsm240 = TimeSeriesMomentum(window=240)

# ── F2: TSM_240__rolling_mean_5 ──
f2_factor = TransformFactor(dependency=tsm240, transform="rolling_mean", window=5)

# ── F3: TrendR2_120_r2__rolling_mean_5 ──
trend_r2_120 = TrendR2Factor(window=120, output="r2")
f3_factor = TransformFactor(dependency=trend_r2_120, transform="rolling_mean", window=5)

# ── F4: DailyRebound ──
f4_factor = DailyRebound()

# ── F5: MAPosition_close_250 ──
f5_factor = MAPosition(window=250, price_column="close")

# ── F6: AvgDrawdown_120 (raw) ──
f6_factor = AvgDrawdown(window=120)

# ── F7: DonchianPosition_20__zscore_120 ──
dcp20 = DonchianChannelPosition(window=20)
f7_factor = TransformFactor(dependency=dcp20, transform="zscore", window=120)

# ── F8: BollingerBandPosition_close_20_2.0__zscore_120 ──
bbp_20_2 = BollingerBandPosition(window=20, k=2.0, price_column="close")
f8_factor = TransformFactor(dependency=bbp_20_2, transform="zscore", window=120)

# ── F9: MADistance_close_20_60__delta_5 ──
madist_20_60 = MADistance(short_window=20, long_window=60, price_column="close")
f9_factor = TransformFactor(dependency=madist_20_60, transform="delta", window=5)


# ====================================================================
# 共享管道因子（三层过滤器用）
# ====================================================================
trend_r2 = TrendR2Factor(window=120, output="r2")
rsrs     = RsrsFactor(regression_window=14, zscore_window=600, output="zscore")
vol20    = Volatility(window=20)
ma200    = MAPosition(window=200, price_column="close")

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
# 组定义: 18 组（9 因子 × 2 过滤器配置）
# ====================================================================
# (label, ranking_factor, builtin_filters)
GROUPS: list[tuple[str, object, tuple[ThresholdFilter, ...]]] = [
    # ── F1: TSM_240 ──
    ("F1_TSM240",           tsm240,  ()),
    ("F1_TSM240_filt",      tsm240,  _FULL_FILTERS),
    # ── F2: TSM_240__rolling_mean_5 ──
    ("F2_TSM240_rm5",       f2_factor, ()),
    ("F2_TSM240_rm5_filt",  f2_factor, _FULL_FILTERS),
    # ── F3: TrendR2_120_r2__rolling_mean_5 ──
    ("F3_TR2R2_rm5",        f3_factor, ()),
    ("F3_TR2R2_rm5_filt",   f3_factor, _FULL_FILTERS),
    # ── F4: DailyRebound ──
    ("F4_DailyRebound",      f4_factor, ()),
    ("F4_DailyRebound_filt", f4_factor, _FULL_FILTERS),
    # ── F5: MAPosition_close_250 ──
    ("F5_MAPos250",          f5_factor, ()),
    ("F5_MAPos250_filt",     f5_factor, _FULL_FILTERS),
    # ── F6: AvgDrawdown_120 (raw) ──
    ("F6_AvgDD120",          f6_factor, ()),
    ("F6_AvgDD120_filt",     f6_factor, _FULL_FILTERS),
    # ── F7: DonchianPosition_20__zscore_120 ──
    ("F7_DCP20_z120",        f7_factor, ()),
    ("F7_DCP20_z120_filt",   f7_factor, _FULL_FILTERS),
    # ── F8: BollingerBandPosition_close_20_2.0__zscore_120 ──
    ("F8_BBP20_z120",        f8_factor, ()),
    ("F8_BBP20_z120_filt",   f8_factor, _FULL_FILTERS),
    # ── F9: MADistance_close_20_60__delta_5 ──
    ("F9_MADist_d5",         f9_factor, ()),
    ("F9_MADist_d5_filt",    f9_factor, _FULL_FILTERS),
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
    alloc_inv_vol,
)


# ====================================================================
# 执行参数
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
TITLE: str              = "宽动量基线回测 — 9 因子扫描"
START_DATE: str         = "2020-01-01"
END_DATE: str           = "2026-05-29"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
