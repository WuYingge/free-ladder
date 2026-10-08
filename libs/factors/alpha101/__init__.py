"""
Alpha101 横截面因子实现包。

WorldQuant《101 Formulaic Alphas》的公式化 alpha 在「date × symbol」面板上的实现，
**101 个公式已全部落地**，包含：
  - operators.py   : 横截面(axis=1)/时序(axis=0) 算子库
  - universe.py    : 抽象数据源（ETF / 股票）
  - panel.py       : 原始输入面板构建（含 vwap 复权口径、行业 PIT 标签）+ FactorPanel
  - formulas.py    : 101 个 alpha 公式本体（论文原文逐一实现）
  - alphas.py      : 注册表 ALPHA101_REGISTRY + 数据依赖标记（cap/vwap/adv/行业）
  - adapter.py     : Alpha101Factor 适配类（对接 factor_analysis 报告/配置命名）

设计原则：
  * 所有面板/算子输入输出统一为 ``date(Index) x symbol(columns)`` 的 float DataFrame。
  * 横截面算子沿 axis=1（同一时刻不同标的），时序算子沿 axis=0（同一标的沿时间）。
  * 算子区分两种聚合方向；嵌套组合（如 decay_linear(rank(...), d)）依赖面板数据，
    不能逐标的重叠计算 —— 这正是本包相对 BaseFactor(逐标的时序) 的关键差异。
  * 数据依赖显式化：``AlphaSpec.uses_cap / needs_vwap / needs_industry`` 供扫描器
    按可用性启用/跳过（45 个公式用 vwap、18 个需申万行业、1 个用 cap）。
  * 价格口径提醒：面板价格是**后复权**（东财仿射口径 ``H=a·P+b``，日收益被压缩
    ``a·P/(a·P+b)``）；vwap 已用 data/adj_factor 搬回同一尺度。详见
    ``docs/eastmoney_hfq_convention.md``。
"""

from __future__ import annotations

from factors.alpha101.operators import (
    abs_,
    adv,
    correlation,
    covariance,
    decay_linear,
    delay,
    delta,
    indneutralize,
    log,
    max_,
    min_,
    product,
    rank,
    scale,
    sign,
    signedpower,
    ts_argmax,
    ts_argmin,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_stddev,
    ts_sum,
)
from factors.alpha101.adapter import Alpha101Factor
from factors.alpha101.formulas import FORMULAS
from factors.alpha101.alphas import (
    ALPHA101_REGISTRY,
    AlphaSpec,
    data_requirements_summary,
    get_alpha_spec,
    get_computable_alpha_ids,
)
from factors.alpha101.panel import (
    Alpha101Inputs,
    build_alpha101_inputs,
    build_alpha101_panel,
)
from factors.alpha101.universe import (
    Alpha101Universe,
    EtfAlpha101Universe,
    StockAlpha101Universe,
)

__all__ = [
    # operators
    "abs_",
    "adv",
    "correlation",
    "covariance",
    "decay_linear",
    "delay",
    "delta",
    "indneutralize",
    "log",
    "max_",
    "min_",
    "product",
    "rank",
    "scale",
    "sign",
    "signedpower",
    "ts_argmax",
    "ts_argmin",
    "ts_max",
    "ts_mean",
    "ts_min",
    "ts_rank",
    "ts_stddev",
    "ts_sum",
    # universe
    "Alpha101Universe",
    "EtfAlpha101Universe",
    "StockAlpha101Universe",
    # panel
    "Alpha101Inputs",
    "build_alpha101_inputs",
    "build_alpha101_panel",
    # alphas
    "ALPHA101_REGISTRY",
    "AlphaSpec",
    "FORMULAS",
    "data_requirements_summary",
    "get_alpha_spec",
    "get_computable_alpha_ids",
    # adapter
    "Alpha101Factor",
]
