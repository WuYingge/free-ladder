"""C1/C2 因子回测 — 2 组回测配置。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.c1_c2_factors0706

因子清单:
    C1  `AvgDrawdown_120__rolling_mean_5`   平均回撤平滑
    C2  `MAE_40__zscore_120__neg`           最大不利偏移 zscore（负向取反）
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)
from factors.trend_r2 import TrendR2Factor
from factors.rsrs import RsrsFactor
from factors.volatility import Volatility
from factors.volatility_family import AvgDrawdown
from factors.ma import MAPosition
from factors.distribution_family import MaxAdverseExcursion
from factors.meta_factor import TransformFactor
from factors.base_factor import BaseFactor
from factors.derived_factor import DerivedFactor


# ====================================================================
# 小工具：取反包装
# ====================================================================

class _Negate(DerivedFactor):
    """对依赖因子取反（乘 -1），用于反转因子方向。"""
    name = "_Negate"

    def __init__(self, dependency: BaseFactor) -> None:
        super().__init__()
        self.add_dependency(dependency)
        self.warmup_period = dependency.get_max_warmup_period()

    def get_output_name(self) -> str:
        dep_name = self._dependencies[0].get_output_name()
        return f"{dep_name}__neg"

    def compute_from_frame(self, frame):
        import pandas as pd
        dep_name = self._dependencies[0].get_output_name()
        result = -frame[dep_name]
        result.name = self.get_output_name()
        return result


# ====================================================================
# 因子定义
# ====================================================================

# ── C1: AvgDrawdown_120__rolling_mean_5 ──
avgdd_120 = AvgDrawdown(window=120)
c1_factor = TransformFactor(dependency=avgdd_120, transform="rolling_mean", window=5)

# ── C2: MAE_40__zscore_120（负向，取反）──
mae_40 = MaxAdverseExcursion(window=40)
mae_zscore = TransformFactor(dependency=mae_40, transform="zscore", window=120)
c2_factor = _Negate(mae_zscore)


# ====================================================================
# 共享管道因子（过滤器用）
# ====================================================================
trend_r2 = TrendR2Factor(window=120, output="r2")
rsrs     = RsrsFactor(regression_window=14, zscore_window=600, output="zscore")
vol20    = Volatility(window=20)
ma200    = MAPosition(window=200, price_column="close")

# 共享管道（不含排名因子 — 排名因子由各组单独指定）
SHARED_PIPELINE: tuple = (rsrs, trend_r2, ma200, vol20)


# ====================================================================
# 过滤器定义（暂不使用，保留以备后续扩展）
# ====================================================================
FILT_MA200 = ThresholdFilter(field=ma200.get_output_name(), operator=">=", value=0)
FILT_RSRS  = ThresholdFilter(field=rsrs.get_output_name(), operator=">=", value=0)
FILT_TR2R2 = ThresholdFilter(field=trend_r2.get_output_name(), operator=">=", value=0.5)

_FULL_FILTERS = (FILT_MA200, FILT_RSRS, FILT_TR2R2)


# ====================================================================
# 组定义: 2 组（2 因子，无过滤器）
# ====================================================================
# (label, ranking_factor, builtin_filters)
GROUPS: list[tuple[str, object, tuple[ThresholdFilter, ...]]] = [
    ("C1_AvgDD120_rm5",     c1_factor, ()),
    ("C1_AvgDD120_rm5_filt",c1_factor, _FULL_FILTERS),
    ("C2_MAE40_z120_neg",   c2_factor, ()),
    ("C2_MAE40_z120_neg_filt",c2_factor, _FULL_FILTERS),
]


# ====================================================================
# Grid Search 参数（不变）
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1, 5, 10, 20)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 20)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (True, False)
GRID_HOLD_OVERLAP: tuple[bool, ...] = (False, True)


# ====================================================================
# 权重分配器（不变）
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

alloc_rsrs = make_factor_weighted_allocator(rsrs.get_output_name())
alloc_rsrs.__name__ = "rsrs"

# 当前启用的分配器
WEIGHT_ALLOCATORS: tuple = (
    alloc_inv_vol,
)


# ====================================================================
# 执行参数（不变）
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
TITLE: str              = "宽动量基线回测 — C1/C2 因子测试"
START_DATE: str         = "2020-01-01"
END_DATE: str           = "2026-05-29"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
