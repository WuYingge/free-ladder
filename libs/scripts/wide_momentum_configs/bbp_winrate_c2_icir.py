"""BBP_winrate / C2 ICIR 动态因子轮动回测。

轮动因子: BollingerBandPosition_close_10_1_5_binarize_winrate_10_0_0 / C2_MAE40_z120_neg
调仓参数: top5 / rebal10 / no_bond / wt_invvol

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.bbp_winrate_c2_icir
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)
from factors.ma import BollingerBandPosition
from factors.distribution_family import MaxAdverseExcursion
from factors.meta_factor import NegateFactor, TransformFactor
from factors.volatility import Volatility


# ====================================================================
# 因子定义
# ====================================================================

# ── BBP_winrate: BollingerBandPosition_close_10_1.5__binarize_winrate_10_0.0 ──
bbp_10_1_5 = BollingerBandPosition(window=10, k=1.5, price_column="close")
bbp_winrate = TransformFactor(dependency=bbp_10_1_5, transform="binarize_winrate", window=10, threshold=0.0)

# ── C2: MAE_40__zscore_120__neg ──
mae_40    = MaxAdverseExcursion(window=40)
mae_zscore = TransformFactor(dependency=mae_40, transform="zscore", window=120)
c2_factor = NegateFactor(mae_zscore)

# ── 反波动率加权所用的波动率因子 ──
vol20 = Volatility(window=20)

# ====================================================================
# 共享管道
# ====================================================================
SHARED_PIPELINE: tuple = (bbp_winrate, c2_factor, vol20)

# ====================================================================
# ICIR 动态因子选择
# ====================================================================
RANKING_FACTOR_CANDIDATES: tuple = (bbp_winrate, c2_factor)
IC_WINDOW: int = 120
IC_SELECTION_MODE: str = "ic"


# ====================================================================
# 组定义: 单组 — ICIR 自适应轮动
# ====================================================================
GROUPS: list[tuple[str, object, tuple[ThresholdFilter, ...]]] = [
    ("BBP_winrate_C2_icir", bbp_winrate, ()),
]


# ====================================================================
# Grid Search 参数
# ====================================================================
GRID_TOP_N: tuple[int, ...]               = (5,)
GRID_MIN_MOMENTUM: tuple                  = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...]  = (10,)
GRID_EXCLUDE_BONDS: tuple[bool, ...]      = (True,)
GRID_HOLD_OVERLAP: tuple[bool, ...]       = (False,)


# ====================================================================
# 权重分配器: invvol
# ====================================================================
alloc_inv_vol = make_factor_weighted_allocator(vol20.get_output_name(), inverse=True)
alloc_inv_vol.__name__ = "invvol"

WEIGHT_ALLOCATORS: tuple = (alloc_inv_vol,)


# ====================================================================
# 执行参数
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
BASENAME_TAG: str       = "bbp_winrate_c2_icir"
TITLE: str              = "宽动量基线回测 — BBP_winrate/C2 ICIR 动态因子轮动"
START_DATE: str         = "2020-01-01"
END_DATE: str           = "2026-05-29"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None