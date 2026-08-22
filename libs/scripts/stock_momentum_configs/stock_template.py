"""股票动量基线回测 — 配置模板。

直接复制本文件，按需修改 → 运行：
    uv run python libs/scripts/run_stock_momentum_custom.py \
        --config libs.scripts.stock_momentum_configs.<your_config_name> \
        [--start-dates 2016-01-01] \
        [--end-dates 2026-08-08]

============ 股票数据注意事项 ============
- 数据源: data/stock_data/*.csv（东财后复权价，全量约 5300 只）
- 后复权价计算收益/因子正确；持仓股数为浮点（无 100 股整手约束，
  复权价下无法正确取整，属已知限制）
- 新股随数据起点自动进入候选池（MIN_LISTING_DAYS=0 时数据第 2 天即可选，
  靠因子 warmup 期 NaN 自然过滤）
- 无涨跌停/ST 过滤（已知限制）
- 无 cluster/债券维度（股票无 cluster 概念，grid 不包含该维度）

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

# ── 现成因子（全部基于 OHLCV，可直接用于股票；按需保留/注释）──
from factors.price_return import PriceReturn
from factors.price_momentum import HighPointPosition, LowPointPosition, TimeSeriesMomentum
from factors.trend_r2 import TrendR2Factor
from factors.rsrs import RsrsFactor
from factors.new_high import NewHigh
from factors.ma import (
    MAPosition,
    MASlope,
    MADistance,
    MADispersion,
    BIAS,
    BollingerBandPosition,
)
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
# 默认排名因子：20 日动量（月动量，跳过最近 1 日）
pr20 = PriceReturn(window=20, skip_recent=1)

# 其他常用因子示例（取消注释即可启用）：
# pr60   = PriceReturn(window=60, skip_recent=1)
# pr120  = PriceReturn(window=120, skip_recent=1)
tr2    = TrendR2Factor(window=120, output="r2")          # 趋势 R²（20 日线性回归拟合优度）
# rsrs   = RsrsFactor(window=18)             # 阻力支撑相对强弱
# nh50   = NewHigh(high_window=50)           # 50 日新高 + 长期均线过滤
# vol20  = Volatility(20)                    # 20 日波动率
# vol60  = Volatility(60)
# hpp20  = HighPointPosition(20)             # 价格在 20 日区间中的位置
# tsm252 = TimeSeriesMomentum(window=252)
# ma20   = MAPosition(20)                    # 价格相对 MA20 的位置
# bias20 = BIAS(20)                          # 乖离率


# ====================================================================
# 2. 共享管道
# ====================================================================
# 所有需要提前计算的因子（排名因子以外的），必须在此列出。
# 排名因子由引擎自动纳入预计算，不必重复。
vol20 = Volatility(window=20)  # 反波动率加权所用（alloc_inv_vol 需要预计算）
SHARED_PIPELINE: tuple = (
    vol20,   # 若 RankFilter/权重分配器用到 vol20，需在此列出
)


# ====================================================================
# 3. 过滤器定义
# ====================================================================

# ── 3a. 硬性过滤器 (ThresholdFilter) ──
# 语法: ThresholdFilter(field="因子输出名", operator=">=" | ">" | "<=" | "<" | "==", value=阈值)
# 示例：要求 20 日动量 > 0
# FILT_MOM_POS = ThresholdFilter(
#     field=pr20.get_output_name(), operator=">=", value=0.0,
#     name="pr20>=0",
# )

# 组合好的过滤器包（可选）
FULL_FILTERS = ()
NO_FILTERS = ()

# ── 3b. 横截面 rank 过滤器 (RankFilter, 软性过滤器) ──
# 用法1: 手动构造
# rank_filter_sample = RankFilter(
#     factor=vol60,          # 排序依据的因子
#     exclude_below_pct=0.3, # 淘汰底部 30%
#     exclude_above_pct=0.1, # 额外淘汰顶部 10%（宽松区）
# )
# 用法2: 批量生成
# RANK_FILTERS = make_rank_filters(vol60, (0.0, 0.1, 0.2, 0.3))


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
    # ── 默认：PR20 单组 ──
    ("PR20", pr20, NO_FILTERS),
    ("TrendR2", tr2, NO_FILTERS),
    # ── 多因子示例（取消注释）──
    # ("TrendR2", tr2, NO_FILTERS),
    # ("RSRS",    rsrs, NO_FILTERS),
    # ("NewHigh", nh50, NO_FILTERS),
    # ("PR20_filt", pr20, (FILT_MOM_POS,)),
]


# ====================================================================
# 7. Grid Search 参数
# ====================================================================
# 引擎会做 5 个维度的笛卡尔积（外加 WEIGHT_ALLOCATORS 的下标维度）：
#   top_n × min_momentum × rebalance × hold_overlap × allocator
# 单值用 `(val,)`，多值遍历。

GRID_TOP_N: tuple[int, ...] = (
    10,
    20,
    30,
    50,
)

GRID_MIN_MOMENTUM: tuple = (
    None,   # 不对排名分数设最低门槛
    # 0.0,  # 只要 >0 的标的
)

GRID_REBALANCE_INTERVAL: tuple[int, ...] = (
    5, # 每 5 个交易日调仓
    10, 
    20,
)

GRID_HOLD_OVERLAP: tuple[bool, ...] = (
    False,  # 全部平仓再开仓
    # True, # 已持仓且仍在 Top-N 内的，不动
)


# ====================================================================
# 8. 权重分配器
# ====================================================================
# 每个分配器就是一个下标维度，参与 grid 遍历。
alloc_equal = equal_weight_allocator                                     # 等权
alloc_momentum = score_proportional_allocator                            # 动量加权

# 反波动率加权示例（需要把 vol20 加入 SHARED_PIPELINE）：
# alloc_inv_vol = make_factor_weighted_allocator(
#     vol20.get_output_name(), inverse=True
# )
# alloc_inv_vol.__name__ = "invvol"

# 当前启用的分配器（= grid 的 allocator 维度）
WEIGHT_ALLOCATORS: tuple = (
    alloc_equal,
    # alloc_momentum,
)


# ====================================================================
# 9. 执行参数
# ====================================================================

# ── 输出路径 ──
OUTPUT_BASE_DIR: str = "data/backtest_results"

# ── 报告标题 ──
TITLE: str = "股票动量基线回测 — PR20"

# ── 输出目录名称 tag（替换默认的 "{n}g"）──
BASENAME_TAG: str = "try_stock"

# ── 回测日期区间（单区间模式）──
START_DATE: str = "2020-01-01"
END_DATE: str   = "2026-07-17"

# ── 多日期区间模式（优先级: 命令行 > 此处 > 单日期）──
# 用法: 分别给 START_DATES / END_DATES 赋值即可。
# 注意: 它们必须在 START_DATE / END_DATE 之后定义，否则会互相覆盖。
START_DATES: tuple[str, ...] | None = None
END_DATES:   tuple[str, ...] | None = None

# ── 标的池（不设置 = 全市场股票 STOCK_LIST）──
SYMBOLS: tuple[str, ...] | None = None
# 示例: ("600519", "000001", "300750")

# ── 新股可交易门槛 ──
# 数据起点（≈上市日）后多少自然日才可被选中。0 = 数据第 2 天即可选。
MIN_LISTING_DAYS: int = 0

# ── 交易成本（单边费率，买卖双向同费率）──
# A 股近似：佣金万 2.5 双向 + 卖出印花税万 5 摊薄 ≈ 万 8
COMMISSION: float = 0.0008

# ── 起始资金（仅影响金额刻度，不影响收益率指标）──
CASH: float = 1_000_000

# ── 并发 ──
# MAX_WORKERS 只作为 worker 数上限；引擎会按可用内存自动收缩
# （全量 5300+ 只股票时 universe ≈ 1.3-2.5GB，每个 fork worker 需一份，
#   直接按 CPU 核数开 worker 会 OOM）。None = 不设上限（仍受内存约束）。
MAX_WORKERS: int | None = None
CROSS_GROUP_PARALLEL: bool = True    # True = 所有组的 grid 变体统一并行

# ── 分段统计 ──
PERIOD_FREQ: str | None = None        # 'YE' 年 / 'QE' 季 / 'ME' 月 / 'W' 周
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None
# 示例: (("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"))
