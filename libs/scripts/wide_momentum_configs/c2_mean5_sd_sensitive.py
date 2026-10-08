"""C2.mean(5) vs C2 × 489 宽池 起始日敏感性回测。

背景
----
9/5 的 C2.mean(5) 对比回测显示：C2_mean5 + RankFilter(vol120,b0.3,a0) 的
top1_rebal10 大幅跑赢（Sharpe 1.674→1.931），但实盘冠军组合 top2_rebal10 明显受伤
（1.883→1.506）。需验证 C2_mean5_rf 的优势是否起点依赖、是否虚高。

本配置：8 起始日 × rebal 5/10 × top1/2/5 × invvol × 含债，489 池。
组（2 组，对照组 + 本次主角）：
  C2_rf_b0.3_a0.0        —— 实盘基线 RankFilter(vol120, b0.3, a0.0)
  C2_mean5_rf_b0.3_a0.0  —— C2.mean(5) + RankFilter(vol120, b0.3, a0.0)

口径：与 c2_mean5_smooth_compare 和 c2_489pool_sd_sensitive 完全一致。

用法:
    cd /root/.openclaw/workspace/opengouzi/free-ladder
    DATA_DIR=/root/.openclaw/workspace/opengouzi/incoming \
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.c2_mean5_sd_sensitive
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    StopRuleSpec,
    ThresholdFilter,
    equal_weight_allocator,
    score_proportional_allocator,
    make_factor_weighted_allocator,
    RankFilter,
)
from factors.distribution_family import MaxAdverseExcursion
from factors.meta_factor import NegateFactor, TransformFactor
from factors.volatility import Volatility

vol20 = Volatility(window=20)
vol120 = Volatility(window=120)

mae_40 = MaxAdverseExcursion(window=40)
c2 = NegateFactor(
    TransformFactor(dependency=mae_40, transform="zscore", window=120)
)
c2_mean5 = TransformFactor(dependency=c2, transform="rolling_mean", window=5)

SHARED_PIPELINE: tuple = (vol20, vol120)

NO_FILTERS = ()
GROUPS: list[tuple] = [
    ("C2_rf_b0.3_a0.0", c2, NO_FILTERS, (RankFilter(vol120, 0.3, 0.0, name="rf_vol120_b0.3_a0.0"),)),
    ("C2_mean5_rf_b0.3_a0.0", c2_mean5, NO_FILTERS, (RankFilter(vol120, 0.3, 0.0, name="rf_vol120_b0.3_a0.0"),)),
]

GRID_TOP_N: tuple[int, ...] = (1, 2, 5)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (False,)
GRID_HOLD_OVERLAP: tuple[bool, ...] = (False,)

alloc_inv_vol = make_factor_weighted_allocator(vol20.get_output_name(), inverse=True)
alloc_inv_vol.__name__ = "invvol"
WEIGHT_ALLOCATORS: tuple = (alloc_inv_vol,)

OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
BASENAME_TAG: str = "c2_mean5_sd_sensitive_489pool"
TITLE: str = "C2.mean(5) vs C2 × 489宽池 × 8起始日 Top-1/2/5 回测"
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-17"

START_DATES: tuple[str, ...] | None = (
    "2020-01-07", "2020-01-08",
)
END_DATES: tuple[str, ...] | None = None

SYMBOLS: tuple[str, ...] | None = None   # 默认 489 宽池

MAX_WORKERS: int | None = 2
CROSS_GROUP_PARALLEL: bool = True

PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
