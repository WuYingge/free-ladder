"""
Alpha101 横截面因子实现包。

WorldQuant《101 Formulaic Alphas》的公式化 alpha 在「date × symbol」面板上的实现，
包含：
  - operators.py   : 横截面(axis=1)/时序(axis=0) 算子库
  - universe.py    : 抽象数据源（ETF / 股票）
  - panel.py       : 原始输入面板构建 + FactorPanel 构建
  - alphas.py      : 手写 alpha 公式 + 注册表 ALPHA101_REGISTRY
  - adapter.py     : Alpha101Factor 适配类（对接 factor_analysis 报告/配置命名）

设计原则：
  * 所有面板/算子输入输出统一为 ``date(Index) x symbol(columns)`` 的 float DataFrame。
  * 横截面算子沿 axis=1（同一时刻不同标的），时序算子沿 axis=0（同一标的沿时间）。
  * 算子区分两种聚合方向；嵌套组合（如 decay_linear(rank(...), d)）依赖面板数据，
    不能逐标的重叠计算 —— 这正是本包相对 BaseFactor(逐标的时序) 的关键差异。
"""

from __future__ import annotations

from factors.alpha101.operators import (
    abs_,
    adv,
    correlation,
    decay_linear,
    delay,
    delta,
    indneutralize,
    log,
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
from factors.alpha101.alphas import (
    ALPHA101_REGISTRY,
    AlphaSpec,
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
    "decay_linear",
    "delay",
    "delta",
    "indneutralize",
    "log",
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
    "get_alpha_spec",
    "get_computable_alpha_ids",
    # adapter
    "Alpha101Factor",
]
