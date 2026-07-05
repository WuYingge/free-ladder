"""PR20 × 4标的极简回测 — 1 组配置。

PriceReturn(20) 在 510300/518880/513100/511010 四标的上 Top-1 择时。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.pr20_four_symbols
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)
from factors.price_return import PriceReturn
from factors.rsrs import RsrsFactor
from factors.volatility import Volatility
from factors.ma import MAPosition
from factors.trend_r2 import TrendR2Factor


# ====================================================================
# 因子定义
# ====================================================================
pr20 = PriceReturn(window=20)


# ====================================================================
# 共享管道因子
# ====================================================================
trend_r2 = TrendR2Factor(window=120, output="r2")
rsrs     = RsrsFactor(regression_window=14, zscore_window=600, output="zscore")
vol20    = Volatility(window=20)
ma200    = MAPosition(window=200, price_column="close")

SHARED_PIPELINE: tuple = (rsrs, trend_r2, ma200, vol20)


# ====================================================================
# 标的池：4 个代表性 ETF
# ====================================================================
SYMBOLS: list[str] = [
    "510300",  # 沪深300
    "518880",  # 黄金ETF
    "513100",  # 纳指ETF
    "511010",  # 国债ETF
]


# ====================================================================
# 组定义: 1 组（纯排名，无过滤器）
# ====================================================================
GROUPS: list[tuple[str, object, tuple[ThresholdFilter, ...]]] = [
    ("PR20_4symbols", pr20, ()),
]


# ====================================================================
# Grid Search 参数（极简：只测必要组合）
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1,)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5,)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (False,)
GRID_HOLD_OVERLAP: tuple[bool, ...] = (False,)


# ====================================================================
# 权重分配器
# ====================================================================
alloc_inv_vol = make_factor_weighted_allocator(vol20.get_output_name(), inverse=True)
alloc_inv_vol.__name__ = "invvol"

WEIGHT_ALLOCATORS: tuple = (alloc_inv_vol,)


# ====================================================================
# 执行参数
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
TITLE: str              = "PR20 × 4标的 Top-1 回测"
START_DATE: str         = "2020-01-01"
END_DATE: str           = "2026-05-29"
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
