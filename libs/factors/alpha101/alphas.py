"""
Alpha101 公式实现 (Alpha101 Formula Implementations)

手写实现 WorldQuant《101 Formulaic Alphas》中的代表性 alpha，配合 operators 算子库
在 date x symbol 面板上计算。本次先落地 5 个代表因子（#1/#2/#3/#4/#101）以打通链路，
后续可向 ALPHA101_REGISTRY 逐一补全。

依赖 cap（市值）的 alpha 在注册表标记 uses_cap=True；data/daily_basic
可用（cap 面板非 None）时由扫描侧启用，缺失时跳过。

每个 alpha 函数签名: (Alpha101Inputs) -> date x symbol 的因子矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from factors.alpha101.operators import (
    correlation,
    delta,
    log,
    rank,
    signedpower,
    ts_argmax,
    ts_rank,
    ts_stddev,
)
from factors.alpha101.panel import Alpha101Inputs


@dataclass(frozen=True)
class AlphaSpec:
    """单个 Alpha101 因子的元数据。"""

    alpha_id: str
    func: Callable[[Alpha101Inputs], pd.DataFrame]
    warmup: int = 0
    uses_cap: bool = False          # 是否依赖市值（当前数据源缺失则不可计算）
    needs_vwap: bool = False
    needs_adv: bool = False
    needs_industry: bool = False
    description: str = ""


# ── 5 个代表因子 ──────────────────────────────────────────────────────────────


def alpha_001(inp: Alpha101Inputs) -> pd.DataFrame:
    """#1: rank(ts_argmax(signedpower(returns<0 ? stddev(returns,20) : close, 2), 5)) - 0.5"""
    std = ts_stddev(inp.returns, 20)
    base = std.where(inp.returns < 0, inp.close)
    sp = signedpower(base, 2.0)
    return rank(ts_argmax(sp, 5)) - 0.5


def alpha_002(inp: Alpha101Inputs) -> pd.DataFrame:
    """#2: -1 * correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6)"""
    a = rank(delta(log(inp.volume), 2))
    b = rank((inp.close - inp.open) / inp.open)
    return -1.0 * correlation(a, b, 6)


def alpha_003(inp: Alpha101Inputs) -> pd.DataFrame:
    """#3: -1 * correlation(rank(open), rank(volume), 10)"""
    return -1.0 * correlation(rank(inp.open), rank(inp.volume), 10)


def alpha_004(inp: Alpha101Inputs) -> pd.DataFrame:
    """#4: -1 * ts_rank(rank(low), 9)"""
    return -1.0 * ts_rank(rank(inp.low), 9)


def alpha_101(inp: Alpha101Inputs) -> pd.DataFrame:
    """#101: (close - open) / ((high - low) + 0.001)"""
    return (inp.close - inp.open) / ((inp.high - inp.low) + 0.001)


# ── 注册表 ────────────────────────────────────────────────────────────────────


ALPHA101_REGISTRY: dict[str, AlphaSpec] = {
    "001": AlphaSpec(
        alpha_id="001",
        func=alpha_001,
        warmup=25,
        description="量价波动/新高位置型：近期收益波动与价格高点的截面排序。",
    ),
    "002": AlphaSpec(
        alpha_id="002",
        func=alpha_002,
        warmup=8,
        needs_vwap=False,
        description="量价相关型：成交量变化与当日涨跌的相关性。",
    ),
    "003": AlphaSpec(
        alpha_id="003",
        func=alpha_003,
        warmup=11,
        description="开量与价格排名的相关性（量价配合）。",
    ),
    "004": AlphaSpec(
        alpha_id="004",
        func=alpha_004,
        warmup=10,
        description="低位时序排名（反转/超跌）。",
    ),
    "101": AlphaSpec(
        alpha_id="101",
        func=alpha_101,
        warmup=1,
        description="单位振幅价格位移（动能/动量）。",
    ),
}


def get_computable_alpha_ids(exclude_cap: bool = True) -> list[str]:
    """返回可计算的 alpha id 列表（exclude_cap=True 时排除依赖 cap 的，保持旧默认）。"""
    ids = []
    for alpha_id, spec in ALPHA101_REGISTRY.items():
        if exclude_cap and spec.uses_cap:
            continue
        ids.append(alpha_id)
    return ids


def get_alpha_spec(alpha_id: str) -> AlphaSpec:
    """按 id 取 AlphaSpec，未知 id 抛 ValueError。"""
    if alpha_id not in ALPHA101_REGISTRY:
        raise ValueError(f"未知 Alpha101 因子: {alpha_id!r}，可用: {sorted(ALPHA101_REGISTRY)}")
    return ALPHA101_REGISTRY[alpha_id]
