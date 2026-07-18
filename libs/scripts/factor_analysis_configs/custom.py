"""
批量因子分析 — 自定义因子配置

在此文件中通过 FACTORS 列表直接创建因子实例（BaseFactor 子类），
与 wide_momentum_configs 的写法完全一致。

用法:
    uv run python libs/scripts/run_batch_factor_analysis.py \
        --config libs.scripts.factor_analysis_configs.custom \
        --mode standard \
        --force \
        --parallel 4
"""

from __future__ import annotations

from factors.meta_factor import (
    TransformFactor,
    NegateFactor,
    CombineFactor,
    ConditionalFactor,
    CompositeRankFactor,
    MultiConditionalFactor,
    ConditionSpec,
)
from factors.distribution_family import MaxAdverseExcursion, MaxFavorableExcursion
from factors.price_return import PriceReturn
from factors.price_momentum import LowPointPosition
from factors.trend_r2 import TrendR2Factor
from factors.ma import MAPosition, MADispersion, MAAlignment, MASlope, MADistance, MADispersion as _MAD, LogBIAS
from factors.volatility import Volatility
from factors.rsrs import RsrsFactor

# ═══════════════════════════════════════════════════════════════════════════════
# ★ FACTORS — 在此添加你要分析的所有因子实例
# ═══════════════════════════════════════════════════════════════════════════════

# ---- 基础因子 ----
mae_40 = MaxAdverseExcursion(window=40)
mae_20 = MaxAdverseExcursion(window=20)
mfe_20 = MaxFavorableExcursion(window=20)
pr_20 = PriceReturn(window=20)
lpp_20 = LowPointPosition(window=20)
trend_r2_120 = TrendR2Factor(window=120, output="r2")
ma_dispersion = MADispersion()

# ---- 三过滤器 ------
pr_20_3_filter = MultiConditionalFactor(
    signal=pr_20,
    conditions=[
        ConditionSpec(condition=RsrsFactor(regression_window=14, zscore_window=600, output="zscore"), op="gt", threshold=0.0),
        ConditionSpec(condition=MAPosition(window=200, price_column="close"),  op="gt", threshold=0.0),
        ConditionSpec(condition=TrendR2Factor(window=120, output="r2"), op="gt", threshold=0.5),
    ],
)

# ---- C2 系列: MAE zscore → negate ----
c2_40_z120 = TransformFactor(dependency=mae_40, transform="zscore", window=120)
c2 = NegateFactor(c2_40_z120)

# ---- LPP 二值化 ----
lpp_20_bin = TransformFactor(dependency=lpp_20, transform="binarize_winrate", window=10, threshold=0.0)

# ---- 条件因子 ----
c2_cond_lpp_bin_05 = ConditionalFactor(signal=c2, condition=lpp_20_bin, op="gt", threshold=0.5)

logBias = LogBIAS(window=20)

logBias_zscore = TransformFactor(dependency=logBias, transform="zscore", window=252)

condition_c2_logBias = ConditionalFactor(
    signal=c2,
    condition=logBias,
    op = "gt",
    threshold=0.05,
)

condition_logBias_07 = ConditionalFactor(
    signal=logBias,
    condition=logBias,
    op = "lt",
    threshold=0.7
)

pr20_condition_logBias = MultiConditionalFactor(
    signal=PriceReturn(window=20),
    conditions=[
        ConditionSpec(logBias, "lt", 0.15),
        ConditionSpec(logBias, "gt", 0.05),
    ]
)

pr20_condition_logBias2 = MultiConditionalFactor(
    signal=PriceReturn(window=20),
    conditions=[
        ConditionSpec(logBias, "lt", 0.10),
        ConditionSpec(logBias, "gt", 0.05),
    ]
)

trendR2_condition_logBias2 = MultiConditionalFactor(
    signal=TrendR2Factor(window=120, output="r2"),
    conditions=[
        ConditionSpec(logBias, "lt", 0.10),
        ConditionSpec(logBias, "gt", 0.05),
    ]
)

# ── 汇总列表 ──
FACTORS = [

    # C2 系列
    c2,

    # 变换衍生
    c2_cond_lpp_bin_05,
    # Pr20 + 三过滤器
    pr_20_3_filter,
    # LogBIAS 相关因子
    logBias,
    logBias_zscore,
    condition_c2_logBias,
    condition_logBias_07,
    pr20_condition_logBias,
    pr20_condition_logBias2,
    trendR2_condition_logBias2,
]
