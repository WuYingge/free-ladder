"""C2 因子 × ATR RankFilter 回测。

排名因子: C2 (MAE_40__zscore_120__neg = MaxAdverseExcursion(40) zscore(120) 取负)
RankFilter 排序因子: TR/close 逐日归一化后滚动 (ATR_{w}_tr_pct = mean(TR_t/close_t))，
替代 C2_rank_filter.py 中的 Volatility 版。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.c2_atr_rank_filter
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    RankFilter,
    StopRuleSpec,
    ThresholdFilter,
    equal_weight_allocator,
    factor_threshold_stop,
    make_factor_weighted_allocator,
    make_rank_filters,
    score_proportional_allocator,
)
from factors.average_amount import AverageAmount
from factors.average_true_range import AverageTrueRange
from factors.daily_rebound import DailyRebound
from factors.distribution_family import MaxAdverseExcursion, MaxFavorableExcursion
from factors.ma import BIAS, MADispersion, MADistance, MAPosition, MASlope
from factors.meta_factor import (
    CombineFactor,
    CompositeRankFactor,
    ConditionalFactor,
    ConditionSpec,
    MultiConditionalFactor,
    NegateFactor,
    TransformFactor,
)
from factors.price_momentum import (
    HighPointPosition,
    LowPointPosition,
    TimeSeriesMomentum,
)
from factors.price_return import PriceReturn
from factors.reversal import ExtremeReversal, ShortTermReversal, VolumeReversal
from factors.rsrs import RsrsFactor
from factors.trend_r2 import TrendR2Factor
from factors.volatility import Volatility
from factors.volatility_family import AvgDrawdown
from factors.volume_family import VolumePriceCorrelation, VolumeStd

# ====================================================================
# 因子定义: C2 (MAE_40__zscore_120__neg)
# ====================================================================

# ── C2: MAE_40__zscore_120__neg ──
mae_20 = MaxAdverseExcursion(window=20)
mae_40 = MaxAdverseExcursion(window=40)
mae_zscore = TransformFactor(dependency=mae_40, transform="zscore", window=120)
c2_factor = NegateFactor(mae_zscore)

pr_20 = PriceReturn(window=20)

vol252 = Volatility(window=252, annualize=True)

# ── 波动率环境过滤器（用于 builtin_filters，天然 point-in-time）──
FILT_VOL_BOND  = ThresholdFilter(
    field=vol252.get_output_name(), operator="<",  value=0.06,
    name="vol<6%"
)
FILT_VOL_LOW_A = ThresholdFilter(
    field=vol252.get_output_name(), operator=">=", value=0.06,
    name="vol>=6%"
)
FILT_VOL_LOW_B = ThresholdFilter(
    field=vol252.get_output_name(), operator="<",  value=0.20,
    name="vol<20%"
)
FILT_VOL_MID_A = ThresholdFilter(
    field=vol252.get_output_name(), operator=">=", value=0.20,
    name="vol>=20%"
)
FILT_VOL_MID_B = ThresholdFilter(
    field=vol252.get_output_name(), operator="<",  value=0.36,
    name="vol<36%"
)
FILT_VOL_HIGH  = ThresholdFilter(
    field=vol252.get_output_name(), operator=">=", value=0.36,
    name="vol>=36%"
)

FILT_PR_20 = ThresholdFilter(
    field=pr_20.get_output_name(), operator=">", value=0.0,
    name="pr_20>0"
)



pr_20_3_filter = MultiConditionalFactor(
    signal=pr_20,
    conditions=[
        ConditionSpec(
            condition=RsrsFactor(regression_window=14, zscore_window=600, output="zscore"),
            op="gt", threshold=0.0,
        ),
        ConditionSpec(
            condition=MAPosition(window=200, price_column="close"),
            op="gt", threshold=0.0,
        ),
        ConditionSpec(
            condition=TrendR2Factor(window=120, output="r2"),
            op="gt", threshold=0.5,
        ),
    ],
)




# ── 反波动率加权所用的波动率因子 ──
vol20 = Volatility(window=20)

# ── RankFilters (TR/close 逐日归一化: mean(TR_t/close_t), 标准 ATR%) ──
atr25_tr_pct = AverageTrueRange(window=25, normalize_by_close=True, normalize_first=True)
atr60_tr_pct = AverageTrueRange(window=60, normalize_by_close=True, normalize_first=True)
atr120_tr_pct = AverageTrueRange(window=120, normalize_by_close=True, normalize_first=True)
atr252_tr_pct = AverageTrueRange(window=252, normalize_by_close=True, normalize_first=True)

RANK_FILTERS_ATR_TR_PCT25: tuple[RankFilter, ...] = make_rank_filters(atr25_tr_pct, (0.0, 0.1, 0.2, 0.3))
RANK_FILTERS_ATR_TR_PCT60: tuple[RankFilter, ...] = make_rank_filters(atr60_tr_pct, (0.0, 0.1, 0.2, 0.3))
RANK_FILTERS_ATR_TR_PCT120: tuple[RankFilter, ...] = make_rank_filters(atr120_tr_pct, (0.0, 0.1, 0.2, 0.3))
RANK_FILTERS_ATR_TR_PCT252: tuple[RankFilter, ...] = make_rank_filters(atr252_tr_pct, (0.0, 0.1, 0.2, 0.3))
RANK_FILTERS_ATR_TR_PCT = RANK_FILTERS_ATR_TR_PCT25 + RANK_FILTERS_ATR_TR_PCT60 + RANK_FILTERS_ATR_TR_PCT120 + RANK_FILTERS_ATR_TR_PCT252

# ====================================================================
# 共享管道（所有因子必须在此，才能被预计算）
# ====================================================================
trend_r2_20 = TrendR2Factor(window=20, output="r2")
SHARED_PIPELINE: tuple = (
    vol252,
    trend_r2_20,
    atr25_tr_pct,
    atr60_tr_pct,
    atr120_tr_pct,
    atr252_tr_pct,
)


# ====================================================================
# 止损/止盈规则
# ====================================================================
SHARED_STOP_RULES: tuple[StopRuleSpec, ...] = (
)
c2_cond = ConditionalFactor(
    signal=c2_factor,
    condition=trend_r2_20,
    op="lt",
    threshold=0.5
)



# ====================================================================
# 组定义（测试目标: TR/close 归一化 RankFilter）
# ====================================================================
GROUPS: list[tuple] = [
    # ── TR/close 逐日归一化后滚动 (mean(TR_t/close_t)) 全组合（16 组/窗口 × 4 窗口 = 64 组）──
    *[(f"c2_atr_tr_pct_rank_filter_25_{rf.name}", c2_factor, (), (rf,),) for rf in RANK_FILTERS_ATR_TR_PCT25],
    *[(f"c2_atr_tr_pct_rank_filter_60_{rf.name}", c2_factor, (), (rf,),) for rf in RANK_FILTERS_ATR_TR_PCT60],
    *[(f"c2_atr_tr_pct_rank_filter_120_{rf.name}", c2_factor, (), (rf,),) for rf in RANK_FILTERS_ATR_TR_PCT120],
    *[(f"c2_atr_tr_pct_rank_filter_252_{rf.name}", c2_factor, (), (rf,),) for rf in RANK_FILTERS_ATR_TR_PCT252],
]


# ====================================================================
# Grid Search 参数
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1, 2, 3, 5,)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 15, 20)
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
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest/c2_atr_rank_filter"
BASENAME_TAG: str = "c2_atr_rank_filter"
TITLE: str = "宽动量基线回测 — c2_atr_rank_filter"
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-17"
# START_DATES = ("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08")
# END_DATES = ("2026-07-10", "2026-07-11", "2026-07-12", "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17")
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
CROSS_GROUP_PARALLEL = True
