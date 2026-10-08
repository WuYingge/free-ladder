"""
Alpha101 自检用合成面板 (Synthetic Panel Builder)

生成满足 :class:`~factors.alpha101.panel.Alpha101Inputs` 契约的随机面板，供
算子单测、101 公式冒烟、新公式快速验证使用。

数据是**几何随机游走**，只有"形状与不变量"是真的：

  * 价格为正，且 ``high >= max(open, close) >= min(open, close) >= low``；
  * ``volume``(手) / ``value``(元) 为正，且 ``value/(volume*100)`` 落在当日
    ``[low, high]`` 内（与真实面板的 vwap 口径一致）；
  * ``cap`` / 行业标签可选（行业标签按 symbol 固定，仅用于跑通链路）。

**不要**把这里的数值当作行情真值 —— 它只保证公式能算、不变量成立。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.alpha101.panel import Alpha101Inputs

__all__ = ["make_synthetic_inputs"]


def make_synthetic_inputs(
    *,
    n_dates: int = 400,
    n_symbols: int = 25,
    seed: int = 7,
    start: str = "2020-01-01",
    with_cap: bool = True,
    with_industry: bool = True,
    n_industries: int = 5,
) -> Alpha101Inputs:
    """构造一个可复现的合成面板（默认 400 交易日 × 25 标的）。

    :param n_dates: 交易日数
    :param n_symbols: 标的数（**横截面 rank 的粒度**：太少会让 ts_rank 长期
                      pin 在极值、相关系数出现零方差洞，公式覆盖率虚低）
    :param seed: 随机种子（可复现）
    :param with_cap: 是否给 cap（缺失 → 依赖 cap 的 #56 应报错）
    :param with_industry: 是否给申万 1/2/3 级行业标签
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n_dates, name="date")
    columns = [f"S{i:03d}" for i in range(n_symbols)]

    returns = pd.DataFrame(
        rng.normal(0.0003, 0.018, (n_dates, n_symbols)), index=index, columns=columns
    )
    close = 100.0 * np.exp(returns.cumsum())
    spread = abs(rng.normal(0, 0.008, (n_dates, n_symbols)))
    open_ = close * (1 + rng.normal(0, 0.005, (n_dates, n_symbols)))
    high = pd.DataFrame(
        np.maximum(close.to_numpy(), open_.to_numpy()) * (1 + spread),
        index=index, columns=columns,
    )
    low = pd.DataFrame(
        np.minimum(close.to_numpy(), open_.to_numpy()) * (1 - spread),
        index=index, columns=columns,
    )
    volume = pd.DataFrame(
        rng.lognormal(15, 0.5, (n_dates, n_symbols)), index=index, columns=columns
    )
    value = volume * 100.0 * close          # 成交均价 ≈ close（落在 [low, high] 内）

    industry = None
    if with_industry:
        codes = np.array(
            [f"{10 + (i % n_industries):02d}0101" for i in range(n_symbols)]
        )[None, :].repeat(n_dates, axis=0)
        industry = {
            level: pd.DataFrame(codes, index=index, columns=columns).apply(
                lambda col, n=n: col.str[:n]
            )
            for level, n in ((1, 2), (2, 4), (3, 6))
        }

    return Alpha101Inputs(
        open=open_,
        high=high,
        low=low,
        close=close,
        value=value,
        volume=volume,
        returns=close.pct_change(),
        vwap=value.div(volume * 100.0),
        cap=pd.DataFrame(1e10, index=index, columns=columns) if with_cap else None,
        industry_levels=industry,
    )
