"""C2_MAE40_z120_neg 宽动量回测 — rank filter 因子替换为 ATR/Close 对照版。

对照目的：原 C2_rank_filter.py 用 Volatility(120) 做横截面 rank filter（剔波动率尾部），
本配置将 rank filter 因子换成 ATRRatio(120) = ATR/close（相对波幅），
其余结构（c2_cond 条件化、止损规则、权重、grid）完全不变，用于验证
vol → atr 替换是否产生实质差异。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.C2_rank_filter_atr
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    StopRuleSpec,
    ThresholdFilter,
    factor_threshold_stop,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
    RankFilter,
    make_rank_filters,
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
from factors.breakout_family import ATRRatio
from factors.volume_family import VolumePriceCorrelation, VolumeStd
from factors.price_return import PriceReturn
from factors.price_momentum import HighPointPosition, LowPointPosition, TimeSeriesMomentum
from factors.ma import MAPosition, MADispersion, MADistance, MASlope, BIAS
from factors.reversal import ShortTermReversal, ExtremeReversal, VolumeReversal
from factors.daily_rebound import DailyRebound
from factors.average_amount import AverageAmount


# ====================================================================
# 因子定义: TrendR2_120_r2_rm5 × C2_MAE40_z120_neg
# ====================================================================

# ── F3: TrendR2_120_r2__rolling_mean_5 ──
trend_r2_120 = TrendR2Factor(window=120, output="r2")
f3_factor = TransformFactor(dependency=trend_r2_120, transform="rolling_mean", window=5)


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

# ── RankFilters ──
# ★ ATR 对照版：横截面 rank filter 因子从 Volatility 换成 ATRRatio(ATR/close)
#   窗口统一取 120，与 vol120 对齐；ATRRatio 为相对量纲，跨标的可比。
vol60 = Volatility(window=60, annualize=False)
vol120 = Volatility(window=120, annualize=False)
atr_ratio_120 = ATRRatio(window=120)
RANK_FILTERS: tuple[RankFilter, ...] = make_rank_filters(atr_ratio_120, (0.0, 0.1, 0.2, 0.3))
RANK_FILTERS_60: tuple[RankFilter, ...] = make_rank_filters(atr_ratio_120, (0.0, 0.1, 0.2, 0.3))
RANK_FILTERS_120: tuple[RankFilter, ...] = make_rank_filters(atr_ratio_120, (0.0, 0.1, 0.2, 0.3))
RANK_FILTERS_60_120 = RANK_FILTERS_60 + RANK_FILTERS_120

RANK_FILTERS_120_high = (
    RankFilter(atr_ratio_120, 0.4, 0.1),
    RankFilter(atr_ratio_120, 0.5, 0.1),
    RankFilter(atr_ratio_120, 0.6, 0.1),
    RankFilter(atr_ratio_120, 0.4, 0),
    RankFilter(atr_ratio_120, 0.5, 0),
    RankFilter(atr_ratio_120, 0.6, 0),
)

# ====================================================================
# 共享管道（所有因子必须在此，才能被预计算）
# ====================================================================
vol90 = Volatility(window=90)
vol105 = Volatility(window=105)
vol135 = Volatility(window=135)
vol150 = Volatility(window=150)
trend_r2_20 = TrendR2Factor(window=20, output="r2")
SHARED_PIPELINE: tuple = (
    vol90,
    vol105,
    vol120,
    vol135,
    vol150,
    vol252,
    atr_ratio_120,
    trend_r2_20,
)


# ====================================================================
# 止损/止盈规则
# ====================================================================
SHARED_STOP_RULES: tuple[StopRuleSpec, ...] = (
    StopRuleSpec(
        rule=factor_threshold_stop(
            factor_col=trend_r2_20.get_output_name(),
            operator=">",
            threshold=0.5,
        ),
        name="tr2_gt_0.5",
    ),
)
c2_cond = ConditionalFactor(
    signal=c2_factor,
    condition=trend_r2_20,
    op="lt",
    threshold=0.5
)



# ====================================================================
# 组定义
# ====================================================================
GROUPS: list[tuple] = [
    # test b
    # (f"c2_rank_filter_120_b02_a01", c2_factor, (), (RankFilter(vol120, 0.2, 0.1),),),
    # (f"c2_rank_filter_120_b025_a01", c2_factor, (), (RankFilter(vol120, 0.25, 0.1),),),
    (f"c2_rank_filter_atr120_b03_a01", c2_cond, (), (RankFilter(atr_ratio_120, 0.3, 0.1),),),
    (f"c2_rank_filter_atr120_b03_a0", c2_cond, (), (RankFilter(atr_ratio_120, 0.3, 0),),),
    # (f"c2_rank_filter_120_b035_a01", c2_factor, (), (RankFilter(vol120, 0.35, 0.1),),),
    # (f"c2_rank_filter_120_b04_a01", c2_factor, (), (RankFilter(vol120, 0.4, 0.1),),),
    # # test vol window
    # (f"c2_rank_filter_90_b03_a01", c2_factor, (), (RankFilter(vol90, 0.3, 0.1),),),
    # (f"c2_rank_filter_105_b03_a01", c2_factor, (), (RankFilter(vol105, 0.3, 0.1),),),
    # (f"c2_rank_filter_135_b03_a01", c2_factor, (), (RankFilter(vol135, 0.3, 0.1),),),
    # (f"c2_rank_filter_150_b03_a01", c2_factor, (), (RankFilter(vol150, 0.3, 0.1),),),
    # # test a
    # (f"c2_rank_filter_120_b03_a005", c2_factor, (), (RankFilter(vol120, 0.3, 0.05),),),
    # (f"c2_rank_filter_120_b03_a0075", c2_factor, (), (RankFilter(vol120, 0.3, 0.075),),),
    # (f"c2_rank_filter_120_b03_a0125", c2_factor, (), (RankFilter(vol120, 0.3, 0.125),),),
    # (f"c2_rank_filter_120_b03_a015", c2_factor, (), (RankFilter(vol120, 0.3, 0.15),),),
] + [("c2", c2_cond, (FILT_VOL_LOW_A,), (), )]


# ====================================================================
# Grid Search 参数
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1, 2, 3, 5,)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 20)
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
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest/c2_atr_rank_filter_对照"
BASENAME_TAG: str = "c2_atr_rank_filter_对照"
TITLE: str = "宽动量基线回测 — C2 rank filter vol→ATR/Close 对照"
# 与 vol 版同区间同参数，保证对照可比
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-17"
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-17"
# START_DATES = ("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08")
# END_DATES = ("2026-07-10", "2026-07-11", "2026-07-12", "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17")
MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
CROSS_GROUP_PARALLEL = True