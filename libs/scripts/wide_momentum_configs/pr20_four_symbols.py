"""PR20 × 4标的极简回测 — 1 组配置。

PriceReturn(20) 在 510300/518880/513100/511010 四标的上 Top-1 择时。

用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.pr20_four_symbols \
        [--start-dates 2019-01-01,2021-01-01] \
        [--end-dates 2026-05-29,2026-05-29]
"""
from __future__ import annotations

# ====================================================================
# 0. 通用导入
# ====================================================================
from backtesting.wide_momentum_baseline import (
    # 核心配置
    ThresholdFilter,
    StopRuleSpec,
    # 权重分配器
    equal_weight_allocator,
    score_proportional_allocator,
    make_factor_weighted_allocator,
)

# ── 因子 ──
from factors.meta_factor import CombineFactor
from factors.price_return import PriceReturn
from factors.rsrs import RsrsFactor
from factors.volatility import Volatility
from factors.ma import MAPosition
from factors.trend_r2 import TrendR2Factor
from factors.price_momentum import (
    RiskAdjustedReturn,
    TimeSeriesMomentum,
    IntradayMomentum,
    OvernightReturn,
    HighPointPosition,
    LowPointPosition,
)
from factors.breakout_family import (
    NewHighContinuous,
    NewLowContinuous,
    DonchianChannelPosition,
    ChandelierExit,
)
from factors.new_high import NewHigh
from factors.change_since_new_high import ChangeSinceNewHigh


# ====================================================================
# 1. 因子定义
# ====================================================================
pr20 = PriceReturn(window=20)

# ── 价格动量（收益类）──
rar20         = RiskAdjustedReturn(window=20)   # 滚动 Sharpe：N 日收益 / N 日波动率
tsm252        = TimeSeriesMomentum(window=252)  # 时序动量二元：N 日收益 > 0 → 1
intraday_mom  = IntradayMomentum()              # 日内动量 (close-open)/open
overnight_ret = OvernightReturn()               # 隔夜跳空收益

# ── 路径动量（位置类）──
hpp20      = HighPointPosition(window=20)        # 高点位置：N 日最高价出现在第几天（0~1）
lpp20      = LowPointPosition(window=20)         # 低点位置：值越高低点越近（弱势），Top-N 选强应取负
donchian20 = DonchianChannelPosition(window=20)  # 唐奇安通道位置（0~1）

# ── 新高突破类 ──
nh50      = NewHigh(high_window=50, low_window=25)  # 输出状态 {2=首买, 1=买, 0=持有, -1=卖}
csnh50_25 = ChangeSinceNewHigh(long_period=50, short_period=25)  # 自首个新高以来的涨幅（值稀疏）
nhc50     = NewHighContinuous(window=50)           # 收盘相对 N 日最高点的距离（连续）
nlc50     = NewLowContinuous(window=50)            # 收盘相对 N 日最低点的距离（反向信号）
chandelier = ChandelierExit(n=22, atr_window=22)   # 吊灯止损偏离（ATR 单位）


# ====================================================================
# 2. 共享管道
# ====================================================================
# 所有需要提前计算的因子（排名因子以外的），必须在此列出。
# 排名因子（pr20）由引擎自动纳入预计算，不必重复。
trend_r2 = TrendR2Factor(window=120, output="r2")
trend_slope = TrendR2Factor(window=120, output="slope")
rsrs     = RsrsFactor(regression_window=14, zscore_window=600, output="zscore")
vol20    = Volatility(window=20)
ma200    = MAPosition(window=200, price_column="close")

SHARED_PIPELINE: tuple = (vol20,)

SlopeR2 = CombineFactor(
    factor_a = trend_slope,
    factor_b = trend_r2,
    method = "product",
)


# ====================================================================
# 3. 过滤器定义
# ====================================================================
# 组合好的过滤器包（可选）；本配置不用过滤器。
FULL_FILTERS = ()
NO_FILTERS = ()


# ====================================================================
# 4. 止损/止盈规则
# ====================================================================
# 空元组 = 不启用止损。
SHARED_STOP_RULES: tuple[StopRuleSpec, ...] = ()


# ====================================================================
# 5. ICIR 动态因子选择（可选 → 默认不启用）
# ====================================================================
RANKING_FACTOR_CANDIDATES: tuple = ()   # 示例: (pr20, ...)
IC_WINDOW: int = 120                     # IC 回溯窗口（交易日）
IC_SELECTION_MODE: str = "icir"          # "icir"=IC_IR 或 "ic"=IC 均值


# ====================================================================
# 6. 组定义
# ====================================================================
# 每组格式: (label, ranking_factor, builtin_filters, cross_sectional_filters)
# - label: str — 仅用于输出目录和报告，无业务含义
# - ranking_factor: BaseFactor 子类实例
# - builtin_filters: tuple[ThresholdFilter, ...] — 硬性过滤器
# - cross_sectional_filters: tuple[RankFilter, ...] — 可选，横截面 rank 过滤器
GROUPS: list[tuple] = [
    ("PR20", pr20, NO_FILTERS, ()),
    ("SlopeR2", SlopeR2, NO_FILTERS, ()),
    # ── 价格动量（收益类）──
    ("RiskAdjustedReturn20", rar20, NO_FILTERS, ()),
    ("TSM252", tsm252, NO_FILTERS, ()),
    ("IntradayMomentum", intraday_mom, NO_FILTERS, ()),
    ("OvernightReturn", overnight_ret, NO_FILTERS, ()),
    # ── 路径动量（位置类）──
    ("HPP20", hpp20, NO_FILTERS, ()),
    ("LPP20", lpp20, NO_FILTERS, ()),
    ("Donchian20", donchian20, NO_FILTERS, ()),
    # ── 新高突破类 ──
    ("NewHigh_50_25", nh50, NO_FILTERS, ()),
    ("ChangeSinceNewHigh", csnh50_25, NO_FILTERS, ()),
    ("NewHighContinuous50", nhc50, NO_FILTERS, ()),
    ("NewLowContinuous50", nlc50, NO_FILTERS, ()),
    ("ChandelierExit", chandelier, NO_FILTERS, ()),
]


# ====================================================================
# 7. Grid Search 参数
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (1,)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 15, 20)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (False,)
GRID_HOLD_OVERLAP: tuple[bool, ...] = (False,)


# ====================================================================
# 8. 权重分配器
# ====================================================================
alloc_equal = equal_weight_allocator                                     # 等权
alloc_momentum = score_proportional_allocator                            # 动量加权
alloc_momentum.__name__ = "momentum"

alloc_inv_vol = make_factor_weighted_allocator(                          # 反波动率加权
    vol20.get_output_name(), inverse=True
)
alloc_inv_vol.__name__ = "invvol"

# 当前启用的分配器（= grid 的 allocator 维度）
WEIGHT_ALLOCATORS: tuple = (
    alloc_inv_vol,
)


# ====================================================================
# 9. 执行参数
# ====================================================================

# ── 输出路径 ──
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"

# ── 报告标题 ──
TITLE: str = "PR20 × 4标的 Top-1 回测"

# ── 输出目录名称 tag（替换默认的 "{n}g"）──
BASENAME_TAG: str = "several_factor_for_simple_symbols"

# ── 回测日期区间（单区间模式）──
START_DATE: str = "2020-01-01"
END_DATE: str   = "2026-07-17"

# ── 多日期区间模式（优先级: 命令行 > 此处 > 单日期）──
START_DATES: tuple[str, ...] | None = None
END_DATES:   tuple[str, ...] | None = None

# ── 标的池 ──
SYMBOLS: tuple[str, ...] | None = (
    # "510300",  # 沪深300
    "518880",  # 黄金ETF
    "513100",  # 纳指ETF
    "511010",  # 国债ETF
    "515180", # 中证红利
    "515000", # 科技ETF
    "513180", # 恒生科技ETF
    "159928", # 消费ETF
    "159981", # 能源化工ETF
    # "512800", # 银行ETF
)

# ── 并发 ──
MAX_WORKERS: int | None = None       # None = 自动（CPU 核数）
CROSS_GROUP_PARALLEL: bool = True    # True = 所有组的 grid 变体统一并行

# ── 分段统计 ──
PERIOD_FREQ: str | None = None        # 'YE' 年 / 'QE' 季 / 'ME' 月 / 'W' 周
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
