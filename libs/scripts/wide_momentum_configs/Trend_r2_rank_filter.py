"""TrendR2_120_r2 × Time Series Momentum 乘积因子回测。

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
    RankFilter,
    make_rank_filters
)
from factors.rsrs import RsrsFactor
from factors.trend_r2 import TrendR2Factor
from factors.distribution_family import MaxAdverseExcursion, MaxFavorableExcursion
from factors.meta_factor import (
    CombineFactor,
    CompositeRankFactor,
    ConditionSpec,
    ConditionalFactor,
    NegateFactor,
    TransformFactor,
    MultiConditionalFactor,
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
# 因子定义: TrendR2_120_r2 × TimeSeriesMomentum
# ====================================================================

trend_r2_120 = TrendR2Factor(window=120, output="r2")
tsm_120 = TimeSeriesMomentum(window=120)

trendR2_condition_tsm120 = ConditionalFactor(
    signal=TrendR2Factor(window=120, output="r2"),
    condition=TimeSeriesMomentum(window=120),
    op="gt",
    threshold=0.0
)
trendR2_condition_tsm252 = ConditionalFactor(
    signal=TrendR2Factor(window=120, output="r2"),
    condition=TimeSeriesMomentum(window=252),
    op="gt",
    threshold=0.0
)
trendR2_condition_tsm20 = ConditionalFactor(
    signal=TrendR2Factor(window=120, output="r2"),
    condition=TimeSeriesMomentum(window=20),
    op="gt",
    threshold=0.0
)


# ── 反波动率加权所用的波动率因子 ──
vol20 = Volatility(window=20)

# ── RankFilters ──
vol60 = Volatility(window=60, annualize=False)
vol120 = Volatility(window=120, annualize=False)
vol252 = Volatility(window=252, annualize=False)
RANK_FILTERS: tuple[RankFilter, ...] = make_rank_filters(vol252, (0.0, 0.1, 0.2, 0.3))
RANK_FILTERS_60: tuple[RankFilter, ...] = make_rank_filters(vol60, (0.0, 0.1, 0.2, 0.3))
RANK_FILTERS_120: tuple[RankFilter, ...] = make_rank_filters(vol120, (0.0, 0.1, 0.2, 0.3))
RANK_FILTERS_60_120 = RANK_FILTERS_60 + RANK_FILTERS_120
ALL_RANK_FILTERS = RANK_FILTERS + RANK_FILTERS_60 + RANK_FILTERS_120

RANK_FILTERS_120_high = (
    RankFilter(vol120, 0.4, 0.1),
    RankFilter(vol120, 0.5, 0.1),
    RankFilter(vol120, 0.6, 0.1),
    RankFilter(vol120, 0.4, 0),
    RankFilter(vol120, 0.5, 0),
    RankFilter(vol120, 0.6, 0),
)

# ====================================================================
# 共享管道（所有因子必须在此，才能被预计算）
# ====================================================================
vol90 = Volatility(window=90)
vol105 = Volatility(window=105)
vol135 = Volatility(window=135)
vol150 = Volatility(window=150)
vol60 = Volatility(window=60)
SHARED_PIPELINE: tuple = (
    vol60,
    vol120,
    vol252
)



# ====================================================================
# 组定义
# ====================================================================
GROUPS: list[tuple] = [
    # test b
    # (f"c2_rank_filter_120_b02_a01", c2_factor, (), (RankFilter(vol120, 0.2, 0.1),),),
    (f"trendR2_condition_tsm120", trendR2_condition_tsm120, (), (rf,)) for rf in ALL_RANK_FILTERS
] + [
    (f"trendR2_condition_tsm252", trendR2_condition_tsm252, (), (rf,)) for rf in ALL_RANK_FILTERS
    ] + [
        (f"trendR2_condition_tsm20", trendR2_condition_tsm20, (), (rf,)) for rf in ALL_RANK_FILTERS
        ] + [
    (f"trendR2_condition_tsm120", trendR2_condition_tsm120, (), ()),
    (f"trendR2_condition_tsm252", trendR2_condition_tsm252, (), ()),
    (f"trendR2_condition_tsm20", trendR2_condition_tsm20, (), ()),
    ]


# ====================================================================
# Grid Search 参数
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1, 3, 5, 10)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 20, 40, 60)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (False,)
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
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest/Trend_r2_tsm_rank_filter"
BASENAME_TAG: str = "Trend_r2_tsm_rank_filter"
TITLE: str = "宽动量基线回测 — Trend_r2_tsm_rank_filter 分组变换"
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-17"
# START_DATES = ("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08")
# END_DATES = ("2026-07-10", "2026-07-11", "2026-07-12", "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17")
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
CROSS_GROUP_PARALLEL = True