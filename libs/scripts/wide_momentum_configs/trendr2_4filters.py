"""TrendR2 因子 + 4 个硬过滤器回测配置。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.trendr2_4filters

过滤器（全部为 ThresholdFilter 硬性过滤器，AND 关系）:
    1. r2_gt_0.4        R² 过滤:   趋势稳定性必须 > 0.4（正常期）
    2. close_gt_ma10    均线过滤:  价格必须在 MA10 上方（走弱期）
    3. volume_lt_18x    成交量过滤: 当日成交量 < 历史均量(20日) × 1.8
    4. drop3_gt_3pct    短期风控:  近 3 日单日跌幅不得跌破 3%（MinDailyReturn_3 > -0.03）

其余参数（Grid / 权重 / 执行参数）与 _template.py 保持一致。
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    StopRuleSpec,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)
from factors.trend_quality import KaufmanEfficiencyRatio
from factors.trend_r2 import TrendR2Factor
from factors.ma import MAPosition
from factors.volume_family import VolumeRatio
from factors.min_daily_return import MinDailyReturn
from factors.volatility import Volatility
from factors.meta_factor import CombineFactor
from factors.price_return import PriceReturn
from factors.oscillator import MFI


# ====================================================================
# 1. 因子定义
# ====================================================================
# 排名因子: TrendR2(120) 的 R² 输出（趋势稳定性，越高越稳）
trend_r2 = TrendR2Factor(window=120, output="r2")
trend_slope = TrendR2Factor(window=120, output="slope")
trend_prod = CombineFactor(
    factor_a = trend_r2,
    factor_b = trend_slope,
    method = "product"
)
pr20 = PriceReturn(window=20)
pr60 = PriceReturn(window=60)
mfi14 = MFI(window=14)

ker5 = KaufmanEfficiencyRatio(window=5)
ker20 = KaufmanEfficiencyRatio(window=20)
    

# 过滤器字段因子
ma10 = MAPosition(window=10, price_column="close")   # close>MA10 ⟺ 输出 > 0
vol_ratio20 = VolumeRatio(window=20)                 # 当日量 / 20日均量
drop3 = MinDailyReturn(window=3)                     # 近 3 日最小单日收益率

# 反波动率加权所用（alloc_inv_vol 需要预计算）
vol20 = Volatility(window=20)


# ====================================================================
# 2. 共享管道
# ====================================================================
# 排名因子 (trend_r2) 由引擎自动预计算，无需放入；
# 其余过滤器字段因子与权重因子必须在此列出。
SHARED_PIPELINE: tuple = (
    # vol20,
    ma10,
    vol_ratio20,
    # drop3,
    trend_r2,
    mfi14
)


# ====================================================================
# 3. 过滤器定义（ThresholdFilter 硬性过滤器）
# ====================================================================
FILT_R2 = ThresholdFilter(
    field=trend_r2.get_output_name(), operator=">", value=0.5,
    name="r2_gt_0.5",
)
FILT_MA10 = ThresholdFilter(
    field=ma10.get_output_name(), operator=">", value=0,
    name="close_gt_ma10",
)
FILT_VOL = ThresholdFilter(
    field=vol_ratio20.get_output_name(), operator="<", value=1.8,
    name="volume_lt_1.8x_ma20",
)
FILT_DROP = ThresholdFilter(
    field=drop3.get_output_name(), operator=">", value=-0.03,
    name="min_daily_drop3_gt_-3pct",
)
FILT_MFI = ThresholdFilter(
    field=mfi14.get_output_name(), operator="<", value=80,
    name="mfi14_lt_80",
)

FULL_FILTERS = (
    FILT_R2, 
    FILT_MFI,
    # FILT_MA10, 
    FILT_VOL, 
    # FILT_DROP
)


# ====================================================================
# 4. 止损/止盈规则（与 template 一致: 不启用）
# ====================================================================
SHARED_STOP_RULES: tuple[StopRuleSpec, ...] = ()


# ====================================================================
# 5. ICIR 动态因子选择（与 template 一致: 不启用）
# ====================================================================
RANKING_FACTOR_CANDIDATES: tuple = ()
IC_WINDOW: int = 120
IC_SELECTION_MODE: str = "icir"


# ====================================================================
# 6. 组定义
# ====================================================================
# (label, ranking_factor, builtin_filters, cross_sectional_filters)
GROUPS: list[tuple] = [
    # ("MFI14_R2GT0.5_MFI_LT80", mfi14, (FILT_R2, FILT_VOL,), ()),
    # ("PR20_R2GT0.5_MFI_LT80", pr20, (FILT_R2, FILT_VOL,), ()),
    ("ker5", ker5, (), ()),
    ("ker20", ker20, (), ()),
    ("ker5_R2GT0.5_MFI_LT80", ker5, (FILT_R2, FILT_VOL,), ()),
    ("ker20_R2GT0.5_MFI_LT80", ker20, (FILT_R2, FILT_VOL,), ()),
    ("ker5_MA10", ker5, (FILT_MA10,), ()),
    ("ker20_MA10", ker20, (FILT_MA10,), ()),
]
# for fil in FULL_FILTERS:
#     GROUPS += [(f"TrendR2XSlope_{fil.name}", trend_prod, (fil,), ()),
#         (f"TrendSlope_{fil.name}", trend_slope, (fil,), ()),
#         (f"PR20_{fil.name}", pr20, (fil,), ()),
#         (f"MFI14_{fil.name}", mfi14, (fil,), ()),
#         (f"PR60_{fil.name}", pr60, (fil,), ()),
#         ]


# ====================================================================
# 7. Grid Search 参数（与 template 保持一致）
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (
    1,
    3,
    5,
    10,
    20,
)

GRID_MIN_MOMENTUM: tuple = (
    None,
)

GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (
    0,
)

GRID_REBALANCE_INTERVAL: tuple[int, ...] = (
    5,
    10,
    15,
    20,
)

GRID_EXCLUDE_BONDS: tuple[bool, ...] = (
    # True,
    False,
)

GRID_HOLD_OVERLAP: tuple[bool, ...] = (
    False,
)


# ====================================================================
# 8. 权重分配器（与 template 保持一致: 反波动率加权）
# ====================================================================
alloc_equal = equal_weight_allocator
alloc_momentum = score_proportional_allocator
alloc_momentum.__name__ = "momentum"

alloc_inv_vol = make_factor_weighted_allocator(
    vol20.get_output_name(), inverse=True
)
alloc_inv_vol.__name__ = "invvol"

WEIGHT_ALLOCATORS: tuple = (
    alloc_inv_vol,
)


# ====================================================================
# 9. 执行参数（与 template 保持一致, 仅输出 tag/标题不同）
# ====================================================================
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
TITLE: str = "宽动量基线回测 — mfi14"
BASENAME_TAG: str = "mfi14"

START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-17"

START_DATES: tuple[str, ...] | None = None
END_DATES: tuple[str, ...] | None = None

SYMBOLS: tuple[str, ...] | None = None

MAX_WORKERS: int | None = None
CROSS_GROUP_PARALLEL: bool = True

PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
