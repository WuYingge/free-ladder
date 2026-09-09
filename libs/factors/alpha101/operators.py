"""
Alpha101 算子库 (Alpha101 Operator Library)

在 ``date(Index) x symbol(columns)`` 的 float DataFrame 上实现 WorldQuant
《101 Formulaic Alphas》所需的两类算子：

  横截面算子 (axis=1, 同时处理当天所有标的):
      rank / scale / indneutralize
  时序算子 (axis=0, 处理单个标的自身沿时间):
      ts_rank / ts_mean / ts_stddev / ts_sum / ts_argmax / ts_argmin
      ts_min / ts_max / correlation / decay_linear / delta / delay
      product
  通用函数:
      signedpower / sign / abs_ / log / adv(成交额滑动均值)

约定:
  * 输入/输出均为 DataFrame（index=date, columns=symbol）, dtype=float。
  * NaN 语义：warmup 期内（不足 window 个观测）产出 NaN；
    单标的缺失不会污染其它标的的横截面运算。
  * 所有窗口算子默认 ``min_periods=window``，即必须满窗才产生有效值，
    与论文的 `d` 日窗口语义一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
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
]


# ── 基础工具 ──────────────────────────────────────────────────────────────────


def _as_float_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """确保输入是 date(Index) x symbol(columns) 的 float DataFrame。"""
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.astype(float)


# ── 通用函数 ──────────────────────────────────────────────────────────────────


def sign(x: pd.DataFrame) -> pd.DataFrame:
    """符号函数：sign(x)。"""
    return np.sign(_as_float_matrix(x))


def abs_(x: pd.DataFrame) -> pd.DataFrame:
    """绝对值：|x|。命名带下划线以避免遮蔽 Python 内建 abs。"""
    return _as_float_matrix(x).abs()


def log(x: pd.DataFrame) -> pd.DataFrame:
    """自然对数：log(x)。非正数 -> NaN（log 对负值无意义）。"""
    m = _as_float_matrix(x)
    out = np.log(m.where(m > 0))
    return out.replace([np.inf, -np.inf], np.nan)


def signedpower(x: pd.DataFrame, a: float) -> pd.DataFrame:
    """带符号幂：sign(x) * |x|^a。"""
    m = _as_float_matrix(x)
    return np.sign(m) * m.abs().pow(float(a))


def adv(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """平均日成交额 adv{d} = x(成交额) 的 d 日滚动均值。"""
    return ts_mean(x, d)


# ── 横截面算子 (axis=1) ──────────────────────────────────────────────────────


def rank(x: pd.DataFrame) -> pd.DataFrame:
    """横截面排名（百分比秩，[0,1]）。

    同一时刻不同标的之间比较：pd.rank(axis=1, pct=True)。
    """
    m = _as_float_matrix(x)
    return m.rank(axis=1, pct=True)


def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """横截面缩放：scale(x, a) = a * x / sum(|x|)。

    沿 axis=1 对每个时间截面做归一化，使 x 的绝对值之和为 a。
    """
    m = _as_float_matrix(x)
    row_abs_sum = m.abs().sum(axis=1)
    row_abs_sum = row_abs_sum.where(row_abs_sum != 0)
    return m.div(row_abs_sum, axis=0) * float(a)


def indneutralize(x: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """行业中性化：减去行业均值，返回残差。

    Parameters
    ----------
    x: date x symbol 的因子值。
    group: index=symbol, values=行业/分组标签 的 Series（不随时间变化）。

    实现为"行业均值中性化"：对每个时间截面，减去该标的所属行业当日均值。
    """
    m = _as_float_matrix(x)
    group_map = group.reindex(m.columns)
    if group_map.isna().any():
        raise ValueError("indneutralize: group 缺失部分标的的行业标签。")

    result = m.copy()
    for g in group_map.drop_duplicates():
        cols = group_map[group_map == g].index.tolist()
        group_cols = [c for c in cols if c in m.columns]
        if not group_cols:
            continue
        result[group_cols] = m[group_cols].sub(m[group_cols].mean(axis=1), axis=0)
    return result


# ── 时序算子 (axis=0) ────────────────────────────────────────────────────────


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序排名：当前值在近 d 日内的百分比秩（[0,1]）。

    对每只标的独立计算：(窗口内 <= 当前值的数量) / d。
    当前值为窗口内最大值时 -> 1.0，最小值时 -> 1/d。
    """
    m = _as_float_matrix(x)
    d = int(d)
    return m.rolling(d, min_periods=d).apply(
        lambda w: (w <= w[-1]).sum() / d, raw=True
    )


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序均值（过去 d 日）。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=int(d)).mean()


def ts_stddev(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序标准差（过去 d 日）。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=int(d)).std()


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序求和（过去 d 日）。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=int(d)).sum()


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序最小值（过去 d 日）。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=int(d)).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序最大值（过去 d 日）。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=int(d)).max()


def ts_argmax(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序最大值位置：过去 d 日内最大值出现的偏移位置（0-based）。"""
    m = _as_float_matrix(x)
    return m.rolling(int(d), min_periods=int(d)).apply(np.argmax, raw=True)


def ts_argmin(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序最小值位置：过去 d 日内最小值出现的偏移位置（0-based）。"""
    m = _as_float_matrix(x)
    return m.rolling(int(d), min_periods=int(d)).apply(np.argmin, raw=True)


def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序相关系数：corr(x, y, d) = 过去 d 日 x 与 y 的滚动 Pearson 相关。"""
    m1 = _as_float_matrix(x)
    m2 = _as_float_matrix(y)
    return m1.rolling(int(d), min_periods=int(d)).corr(m2)


def decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """线性加权移动平均：近值权重更大，权重 = d, d-1, ..., 1。

    decay_linear(x, d) = sum_{i=1..d} i * x_{t-d+i} / sum(i)
    """
    m = _as_float_matrix(x)
    d = int(d)
    weights = np.arange(1, d + 1, dtype=float)
    wsum = weights.sum()
    return m.rolling(d, min_periods=d).apply(
        lambda w: float(np.dot(w, weights) / wsum), raw=True
    )


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序差分：x - x.shift(d)。"""
    return _as_float_matrix(x) - _as_float_matrix(x).shift(int(d))


def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序滞后：x.shift(d)。"""
    return _as_float_matrix(x).shift(int(d))


def product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序连乘：过去 d 日的乘积。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=int(d)).apply(
        np.prod, raw=True
    )
