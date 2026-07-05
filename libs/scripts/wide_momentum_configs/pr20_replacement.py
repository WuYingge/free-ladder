"""PR20 最优替代者搜索 — 12 组回测配置。

6 排名因子 × 2 过滤器配置（纯排名 / 三层过滤）。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.pr20_replacement
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)
from factors.price_return import PriceReturn
from factors.oscillator import MFI
from factors.price_momentum import HighPointPosition, TimeSeriesMomentum
from factors.trend_quality import KaufmanEfficiencyRatio
from factors.rsrs import RsrsFactor
from factors.trend_r2 import TrendR2Factor
from factors.volatility import Volatility
from factors.ma import MAPosition, MASlope
from factors.meta_factor import TransformFactor


# ====================================================================
# 因子定义
# ====================================================================

# ── A0~A5 排名因子 ──
pr20   = PriceReturn(window=20)                       # A0 基线对照
mfi14  = MFI(window=14)                               # A1 资金流量指标
hpp20  = HighPointPosition(window=20)                 # A2 高点位置
ker20  = KaufmanEfficiencyRatio(window=20)            # A3 Kaufman 效率比
tsm252 = TimeSeriesMomentum(window=252)               # A5 长程时序动量

# A4 排名因子: MASlope(ma_window=10, slope_window=5) → rolling_mean(5)
maslope_10_5_rm5 = TransformFactor(
    dependency=MASlope(ma_window=10, slope_window=5),
    transform="rolling_mean", window=5,
)

# ── 共享管道因子（三层过滤器用）──
trend_r2 = TrendR2Factor(window=120, output="r2")
rsrs     = RsrsFactor(regression_window=14, zscore_window=600, output="zscore")
ma200    = MAPosition(window=200, price_column="close")
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
    # ── A0: PriceReturn_20 ──
    ("A0_PR20",        pr20,   ()),
    ("A0_PR20_filt",   pr20,   _FULL_FILTERS),
    # ── A1: MFI_14 ──
    ("A1_MFI14",       mfi14,  ()),
    ("A1_MFI14_filt",  mfi14,  _FULL_FILTERS),
    # ── A2: HighPointPosition_20 ──
    ("A2_HPP20",       hpp20,  ()),
    ("A2_HPP20_filt",  hpp20,  _FULL_FILTERS),
    # ── A3: KaufmanER_20 ──
    ("A3_KER20",       ker20,  ()),
    ("A3_KER20_filt",  ker20,  _FULL_FILTERS),
    # ── A4: MASlope_close_10_5__rolling_mean_5 ──
    ("A4_MASlope",     maslope_10_5_rm5,  ()),
    ("A4_MASlope_filt",maslope_10_5_rm5,  _FULL_FILTERS),
    # ── A5: TimeSeriesMomentum_252 ──
    ("A5_TSM252",      tsm252, ()),
    ("A5_TSM252_filt", tsm252, _FULL_FILTERS),
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
TITLE: str              = "宽动量基线回测 — PR20 最优替代者搜索"
START_DATE: str         = "2020-01-01"
END_DATE: str           = "2026-05-29"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
