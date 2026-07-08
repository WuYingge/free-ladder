"""F3/C2 ICIR 动态因子轮动回测。

轮动因子: F3_TR2R2_rm5 / C2_MAE40_z120_neg
调仓参数: top5 / rebal10 / no_bond / wt_invvol

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.f3_c2_icir
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
from factors.meta_factor import NegateFactor, TransformFactor
from factors.volatility import Volatility


# ====================================================================
# 因子定义: F3 (TrendR2 rolling_mean 5) + C2 (MAE zscore neg)
# ====================================================================

# ── F3: TrendR2_120_r2__rolling_mean_5 ──
trend_r2_120 = TrendR2Factor(window=120, output="r2")
f3_factor = TransformFactor(dependency=trend_r2_120, transform="rolling_mean", window=5)

# ── C2: MAE_40__zscore_120__neg ──
mae_40    = MaxAdverseExcursion(window=40)
mae_zscore = TransformFactor(dependency=mae_40, transform="zscore", window=120)
c2_factor = NegateFactor(mae_zscore)

# ── 反波动率加权所用的波动率因子 ──
vol20 = Volatility(window=20)

# ====================================================================
# 共享管道（所有因子必须在此，才能被预计算 & ICIR 使用）
# ====================================================================
SHARED_PIPELINE: tuple = (f3_factor, c2_factor, vol20)

# ====================================================================
# ICIR 动态因子选择
# ====================================================================
RANKING_FACTOR_CANDIDATES: tuple = (f3_factor, c2_factor)
IC_WINDOW: int = 120
IC_SELECTION_MODE: str = "icir"


# ====================================================================
# 组定义: 单组 — ICIR 自适应轮动
# ====================================================================
# ranking_factor 作为 ICIR 全部不可用时的 fallback
GROUPS: list[tuple[str, object, tuple[ThresholdFilter, ...]]] = [
    ("F3_C2_icir", f3_factor, ()),
]


# ====================================================================
# Grid Search 参数（固定到用户指定的唯一组合）
# ====================================================================
GRID_TOP_N: tuple[int, ...]               = (5,)
GRID_MIN_MOMENTUM: tuple                  = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...]  = (10,)
GRID_EXCLUDE_BONDS: tuple[bool, ...]      = (True,)
GRID_HOLD_OVERLAP: tuple[bool, ...]       = (False,)


# ====================================================================
# 权重分配器: invvol（仅此一项）
# ====================================================================
alloc_inv_vol = make_factor_weighted_allocator(vol20.get_output_name(), inverse=True)
alloc_inv_vol.__name__ = "invvol"

WEIGHT_ALLOCATORS: tuple = (alloc_inv_vol,)


# ====================================================================
# 执行参数
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
BASENAME_TAG: str       = "f3_c2_icir"
TITLE: str              = "宽动量基线回测 — F3/C2 ICIR 动态因子轮动"
START_DATE: str         = "2020-01-01"
END_DATE: str           = "2026-05-29"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None