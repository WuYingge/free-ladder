"""宽动量基线回测 — 配置模板。

直接复制本文件，按需修改 → 运行：
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.<your_config_name> \
        [--start-dates 2019-01-01,2021-01-01] \
        [--end-dates 2026-05-01,2026-05-01]

============ 各节说明 ============

[因子定义]      定义排名因子和管道因子。
[共享管道]      所有组公用的"预计算因子"元组。排名因子本身自动预计算，
                无需放入；只放过滤器/止损/ICIR/RankFilter/权重所用的因子。
[过滤器]        ThresholdFilter — 硬性过滤器（必须满足）。
                RankFilter      — 横截面 rank 过滤器（软性，淘汰尾部）。
[止损/止盈]     SHARED_STOP_RULES — 逐日检查的止损/止盈规则。
[ICIR 动态选择] 在多个候选因子间按 ICIR 自适应轮动。
[组定义]        GROUPS — 每组 = (标签, 排名因子, builtin_filters, [cross_sectional_filters])。
[Grid Search]   各维度的网格值，引擎会做笛卡尔积遍历。
[权重方案]      WEIGHT_ALLOCATORS — 下标对应 grid 的 allocator 维度。
[执行参数]      输出路径、日期、并发等。
"""
from __future__ import annotations

# ====================================================================
# 0. 通用导入
# ====================================================================
from backtesting.wide_momentum_baseline import (
    # 核心配置
    ThresholdFilter,
    RankFilter,
    make_rank_filters,
    StopRuleSpec,
    factor_threshold_stop,
    # 权重分配器
    equal_weight_allocator,
    score_proportional_allocator,
    make_factor_weighted_allocator,
)

# ── 因子基类 ──
from factors.price_return import PriceReturn
from factors.price_momentum import HighPointPosition, LowPointPosition, TimeSeriesMomentum
from factors.trend_r2 import TrendR2Factor
from factors.ma import MAPosition, MASlope, MADistance, MADispersion, BIAS, BollingerBandPosition
from factors.rsrs import RsrsFactor
from factors.reversal import ShortTermReversal, ExtremeReversal, VolumeReversal
from factors.volatility import Volatility
from factors.volatility_family import AvgDrawdown
from factors.distribution_family import MaxAdverseExcursion, MaxFavorableExcursion
from factors.volume_family import VolumePriceCorrelation, VolumeStd
from factors.daily_rebound import DailyRebound
from factors.average_amount import AverageAmount
from factors.oscillator import MFI
from factors.trend_quality import KaufmanEfficiencyRatio

# ── 元因子（按需选择，不需要的可以注释掉）──
from factors.meta_factor import (
    CombineFactor,        # 乘积/比值/加权和
    CompositeRankFactor,  # 加权 rank 合成
    ConditionalFactor,    # 条件因子 (if condition > threshold, keep signal else 0)
    NegateFactor,         # 取负
    TransformFactor,      # zscore / delta / rolling_mean / binarize_winrate 等
)


# ====================================================================
# 1. 因子定义
# ====================================================================
# 示例排名因子 — 按需替换/增删：
vol20 = Volatility(20)

# ====================================================================
# 2. 共享管道
# ====================================================================
# 所有需要提前计算的因子（排名因子以外的），必须在此列出。
# 排名因子由引擎自动纳入预计算，不必重复。
SHARED_PIPELINE: tuple = (
    vol20,
)


# ====================================================================
# 3. 过滤器定义
# ====================================================================

# ── 3a. 硬性过滤器 (ThresholdFilter) ──
# 语法: ThresholdFilter(field="因子输出名", operator=">=" | ">" | "<=" | "<" | "==", value=阈值)
# FILT_MA200 = ThresholdFilter(
#     field=ma200.get_output_name(), operator=">=", value=0,
#     name="ma200>=0",
# )

# 组合好的过滤器包（可选）
FULL_FILTERS = ()
NO_FILTERS = ()

# ── 3b. 横截面 rank 过滤器 (RankFilter, 软性过滤器) ──
# 用法1: 手动构造
# rank_filter_sample = RankFilter(
#     factor=vol252,        # 排序依据的因子
#     bottom=0.3,           # 淘汰底部 30%
#     allow_after=0.1,      # 额外淘汰紧接着 10%（宽松区）
# )
# 用法2: 批量生成
# RANK_FILTERS = make_rank_filters(vol252, (0.0, 0.1, 0.2, 0.3))
#     → 生成 8 个 RankFilter（bottom ∈ {0,0.1,0.2,0.3} × allow_after ∈ {0.1,0.2,0.3} 交叉）


# ====================================================================
# 4. 止损/止盈规则
# ====================================================================
# 内置示例: factor_threshold_stop — 当某因子值越过阈值时清仓。
# 空元组 = 不启用止损。
SHARED_STOP_RULES: tuple[StopRuleSpec, ...] = (
    # 示例（注释掉）:
    # StopRuleSpec(
    #     rule=factor_threshold_stop(
    #         factor_col=tr2.get_output_name(),
    #         operator="<",
    #         threshold=0.5,
    #     ),
    #     name="tr2_lt_0.5_清仓",
    # ),
)
# 若不需要止损，保持空元组：
# SHARED_STOP_RULES: tuple[StopRuleSpec, ...] = ()


# ====================================================================
# 5. ICIR 动态因子选择（可选 → 默认不启用）
# ====================================================================
# 若配置了 RANKING_FACTOR_CANDIDATES，引擎会在每次调仓时从候选因子池中
# 按 ICIR（或 IC）选出最优因子作为当期的排名因子。
# GROUPS 中的 ranking_factor 仍需要填一个"默认因子"，它只在第一个
# warm-up 期内用到；后续会被动态选择覆盖。
RANKING_FACTOR_CANDIDATES: tuple = ()   # 示例: (pr20, hpp20, tsm252)
IC_WINDOW: int = 120                     # IC 回溯窗口（交易日）
IC_SELECTION_MODE: str = "icir"          # "icir"=IC_IR 或 "ic"=IC 均值


# ====================================================================
# 6. 组定义
# ====================================================================
# 每组格式: (label, ranking_factor, builtin_filters, [cross_sectional_filters])
# - label: str — 仅用于输出目录和报告，无业务含义
# - ranking_factor: BaseFactor 子类实例
# - builtin_filters: tuple[ThresholdFilter, ...] — 硬性过滤器
# - cross_sectional_filters: tuple[RankFilter, ...] — 可选，横截面过滤器
#
# 后两种格式兼容:
#   3 元素: (label, ranking_factor, builtin_filters)
#   4 元素: (label, ranking_factor, builtin_filters, cross_sectional_filters)
GROUPS: list[tuple] = [
    # ── 单组示例 ──
    # ("PR20_3filters", pr20, FULL_FILTERS),

    # ── 多组示例 ──
    # ("HPP20",         hpp20,  NO_FILTERS),
    # ("HPP20_filt",    hpp20,  FULL_FILTERS),
    # ("TSM252",        tsm252, NO_FILTERS),
    # ("TSM252_filt",   tsm252, FULL_FILTERS),

    # ── 带横截面过滤器的示例 ──
    # ("PR20_rankF", pr20, NO_FILTERS, (
    #     RankFilter(vol252, 0.3, 0.1),
    # )),
]


# ====================================================================
# 7. Grid Search 参数
# ====================================================================
# 引擎会做 7 个维度的笛卡尔积（外加 WEIGHT_ALLOCATORS 的下标维度）。
# 单值用 `(val,)`，多值遍历。

GRID_TOP_N: tuple[int, ...] = (
    1,      # 持仓数量
    5,
    10,
    20,
)

GRID_MIN_MOMENTUM: tuple = (
    None,   # 不对排名分数设最低门槛
    # 0.0,  # 只要 >0 的标的
    # 0.02, # 排名分 > 2%
)

GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (
    0,      # 0 = 不限制每个 cluster 的持仓数
    # 3,    # 每个 cluster 最多 3 个
    # 5,
)

GRID_REBALANCE_INTERVAL: tuple[int, ...] = (
    5,      # 每 5 个交易日调仓
    10,
    20,
)

GRID_EXCLUDE_BONDS: tuple[bool, ...] = (
    True,   # 剔除债券类 ETF (cluster 43/44)
    False,  # 保留债券
)

GRID_HOLD_OVERLAP: tuple[bool, ...] = (
    False,  # 全部平仓再开仓
    # True, # 已持仓且仍在 Top-N 内的，不动
)


# ====================================================================
# 8. 权重分配器
# ====================================================================
# 每个分配器就是一个下标维度，参与 grid 遍历。
# 常用模式:
alloc_equal = equal_weight_allocator                                     # 等权
alloc_momentum = score_proportional_allocator                            # 动量加权
alloc_momentum.__name__ = "momentum"

alloc_inv_vol = make_factor_weighted_allocator(                          # 反波动率加权
    vol20.get_output_name(), inverse=True
)
alloc_inv_vol.__name__ = "invvol"

# ── 自定义分配器示例 ──
# def my_tiered_allocator(candidates):
#     """前 40% 权重 ×1.5，其余 ×1.0。"""
#     if not candidates:
#         return {}
#     n = len(candidates)
#     top_count = max(1, round(n * 0.4))
#     weights = {}
#     for i, c in enumerate(candidates):
#         weights[c.symbol] = 1.5 if i < top_count else 1.0
#     return weights
# my_tiered_allocator.__name__ = "tiered"

# 当前启用的分配器（= grid 的 allocator 维度）
WEIGHT_ALLOCATORS: tuple = (
    # alloc_equal,
    # alloc_momentum,
    alloc_inv_vol,
)


# ====================================================================
# 9. 执行参数
# ====================================================================

# ── 输出路径 ──
OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
# OUTPUT_BASE_DIR: str = "data/backtest_results"

# ── 报告标题 ──
TITLE: str = "宽动量基线回测 — vol_study"

# ── 输出目录名称 tag（替换默认的 "{n}g"）──
BASENAME_TAG: str = "vol_study"

# ── 回测日期区间（单区间模式）──
START_DATE: str = "2020-01-01"
END_DATE: str   = "2026-07-17"

# ── 多日期区间模式（优先级: 命令行 > 此处 > 单日期）──
# 用法: 分别给 START_DATES / END_DATES 赋值即可。
# 注意: 它们必须在 START_DATE / END_DATE 之后定义，否则会互相覆盖。
START_DATES: tuple[str, ...] | None = None
# 示例: ("2020-01-01", "2021-01-01", "2022-01-01")
END_DATES:   tuple[str, ...] | None = None
# 示例: ("2024-12-31", "2025-12-31", "2026-05-29")

# ── 标的池（不设置 = 全市场 ETF）──
SYMBOLS: tuple[str, ...] | None = None
# 示例: ("510050", "510300", "159915")

# ── 并发 ──
MAX_WORKERS: int | None = None       # None = 自动（CPU 核数）
CROSS_GROUP_PARALLEL: bool = True    # True = 所有组的 grid 变体统一并行

# ── 分段统计 ──
PERIOD_FREQ: str | None = None        # 'YE' 年 / 'QE' 季 / 'ME' 月 / 'W' 周
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
# 示例: (("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"))
