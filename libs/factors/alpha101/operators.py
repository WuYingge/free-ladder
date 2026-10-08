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
  * NaN 语义：所有窗口算子按 ``min_periods=_min_obs(d)`` 计算，即窗口内**至少
    80% 有效观测**即可产出值（不足 → NaN），缺失值不参与计算。理由：A 股面板
    存在停牌缺口，且相关系数在零方差窗口、ts_rank 在连续 pin 极值时会产生洞；
    若要求严格满窗，长链条公式（如 #96 需要 40 天连续无洞）覆盖率会塌到 ~3%。
    **数据无缺失时该容忍不改变任何结果**（min_periods 只在有空位时生效）。
  * 当日值缺失时，依赖"当日值"的算子（ts_rank/decay_linear/product）返回 NaN。
  * 单标的缺失不会污染其它标的的横截面运算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 窗口算子的缺失容忍：窗口内至少 ceil(d * (1 - WINDOW_MISSING_TOLERANCE)) 个有效观测
#: （即允许约 20% 空位；无空位时与"严格满窗"结果完全一致）
WINDOW_MISSING_TOLERANCE = 0.2


def _min_obs(d: int) -> int:
    """窗口 d 所需的最少有效观测数（见 WINDOW_MISSING_TOLERANCE）。"""
    d = int(d)
    return max(1, d - max(1, int(round(d * WINDOW_MISSING_TOLERANCE))))

__all__ = [
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
]

# ── 基础工具 ──────────────────────────────────────────────────────────────────


def _as_float_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """确保输入是 date(Index) x symbol(columns) 的 float DataFrame。"""
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.astype(float)


def _finite(m: pd.DataFrame) -> pd.DataFrame:
    """±inf → NaN（零方差窗口、除零产生的无意义值）。"""
    return m.replace([np.inf, -np.inf], np.nan)


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


def signedpower(x: pd.DataFrame, a: float | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """带符号幂：sign(x) * |x|^a。

    ``a`` 可以是标量，也可以是 Series/DataFrame（论文里存在
    ``SignedPower(ts_rank(...), delta(close, d))`` 这类**指数为序列**的用法）。

    注：底数接近 0 且指数为大负数时会溢出成 ±inf（如 #84），统一抹成 NaN。
    """
    m = _as_float_matrix(x)
    if isinstance(a, pd.DataFrame):
        a = _as_float_matrix(a).reindex(index=m.index, columns=m.columns)
    return _finite(np.sign(m) * m.abs().pow(a))


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


def indneutralize(
    x: pd.DataFrame,
    group: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """行业中性化：减去行业均值，返回残差（point-in-time 安全）。

    Parameters
    ----------
    x: date x symbol 的因子值。
    group:
      * ``Series``（index=symbol）：静态分组标签（不随时间变化）；
      * ``DataFrame``（date x symbol）：**逐日**分组标签，用于时点行业归属
        （申万分类会变，必须用当日生效的分类，见 sw_industry_provider）。

    实现为"行业均值中性化"：对每个时间截面，减去该标的所属行业当日均值。
    标签缺失（NaN）的单元格不做中性化，原值返回。
    """
    m = _as_float_matrix(x)

    if isinstance(group, pd.Series):
        group_map = group.reindex(m.columns)
        if group_map.isna().any():
            raise ValueError("indneutralize: group 缺失部分标的的行业标签。")
        result = m.copy()
        for g in group_map.drop_duplicates():
            group_cols = [c for c in group_map[group_map == g].index if c in m.columns]
            if not group_cols:
                continue
            result[group_cols] = m[group_cols].sub(m[group_cols].mean(axis=1), axis=0)
        return result

    # 逐日标签：按 (日期, 标签) 分组去均值；stack 后一次算完，避免逐日循环
    labels = group.reindex(index=m.index, columns=m.columns)
    stacked = m.stack(future_stack=True)
    label_stack = labels.stack(future_stack=True).reindex(stacked.index)
    valid = label_stack.notna() & stacked.notna()
    if not valid.any():
        return m.copy()
    key = pd.MultiIndex.from_arrays(
        [stacked.index.get_level_values(0), label_stack], names=["date", "label"]
    )
    means = stacked[valid].groupby(key[valid]).transform("mean")
    resid = stacked.copy()
    resid[valid] = stacked[valid] - means
    return resid.unstack()


def min_(x: pd.DataFrame, y: pd.DataFrame | float) -> pd.DataFrame:
    """逐元素最小值：min(x, y)（论文里 min(x, y) 的语义；NaN 传染）。"""
    return pd.DataFrame(
        np.minimum(_as_float_matrix(x).to_numpy(), _other_matrix(y, x).to_numpy()),
        index=x.index,
        columns=x.columns,
    )


def max_(x: pd.DataFrame, y: pd.DataFrame | float) -> pd.DataFrame:
    """逐元素最大值：max(x, y)（论文里 max(x, y) 的语义；NaN 传染）。"""
    return pd.DataFrame(
        np.maximum(_as_float_matrix(x).to_numpy(), _other_matrix(y, x).to_numpy()),
        index=x.index,
        columns=x.columns,
    )


def _other_matrix(y: pd.DataFrame | float, like: pd.DataFrame) -> pd.DataFrame:
    """把标量/Series/DataFrame 广播成与 like 同形的矩阵。"""
    if isinstance(y, pd.DataFrame):
        return _as_float_matrix(y).reindex(index=like.index, columns=like.columns)
    if isinstance(y, pd.Series):
        return pd.DataFrame(
            np.repeat(y.reindex(like.columns).to_numpy()[None, :], len(like.index), axis=0),
            index=like.index,
            columns=like.columns,
        )
    return pd.DataFrame(
        float(y), index=like.index, columns=like.columns
    )


# ── 时序算子 (axis=0) ────────────────────────────────────────────────────────


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序排名：当前值在近 d 日内的百分比秩（[0,1]）。

    对每只标的独立计算：(窗口内 <= 当前值的数量) / 有效观测数。
    当前值为窗口内最大值时 -> 1.0。

    NaN 语义：当日值缺失 → NaN；窗口内至少 d 个有效观测即可计算
    （缺失值不参与计数，也不会把结果整段抹成 NaN）。
    """
    m = _as_float_matrix(x)
    d = int(d)

    def _rank(w: np.ndarray) -> float:
        valid = ~np.isnan(w)
        n_valid = int(valid.sum())
        if np.isnan(w[-1]) or n_valid < _min_obs(d):
            return np.nan
        return float(((w <= w[-1]) & valid).sum() / n_valid)

    return m.rolling(d, min_periods=_min_obs(d)).apply(_rank, raw=True)


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序均值（过去 d 日）。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=_min_obs(d)).mean()


def ts_stddev(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序标准差（过去 d 日）。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=_min_obs(d)).std()


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序求和（过去 d 日）。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=_min_obs(d)).sum()


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序最小值（过去 d 日）。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=_min_obs(d)).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序最大值（过去 d 日）。"""
    return _as_float_matrix(x).rolling(int(d), min_periods=_min_obs(d)).max()


def ts_argmax(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序最大值位置：过去 d 日内最大值出现的偏移位置（0-based，窗口内日历位置）。

    NaN 语义与 pandas 原生窗口算子一致：窗口内至少 d 个有效观测即可计算，
    缺失值不参与比较（因此不会因停牌/单日缺失把结果整段抹成 NaN）。
    """
    m = _as_float_matrix(x)
    d = int(d)

    def _argmax(w: np.ndarray) -> float:
        valid = ~np.isnan(w)
        if valid.sum() < _min_obs(d):
            return np.nan
        return float(np.argmax(np.where(valid, w, -np.inf)))

    return m.rolling(d, min_periods=_min_obs(d)).apply(_argmax, raw=True)


def ts_argmin(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序最小值位置：过去 d 日内最小值出现的偏移位置（0-based，窗口内日历位置）。"""
    m = _as_float_matrix(x)
    d = int(d)

    def _argmin(w: np.ndarray) -> float:
        valid = ~np.isnan(w)
        if valid.sum() < _min_obs(d):
            return np.nan
        return float(np.argmin(np.where(valid, w, np.inf)))

    return m.rolling(d, min_periods=_min_obs(d)).apply(_argmin, raw=True)


def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序相关系数：corr(x, y, d) = 过去 d 日 x 与 y 的滚动 Pearson 相关。"""
    m1 = _as_float_matrix(x)
    m2 = _as_float_matrix(y)
    # 零方差窗口 pandas 可能返回 ±inf（0/0 之外的病态情形）→ 统一抹成 NaN
    return _finite(m1.rolling(int(d), min_periods=_min_obs(d)).corr(m2))


def covariance(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序协方差：cov(x, y, d) = 过去 d 日 x 与 y 的滚动协方差（论文 #13/#16 用）。"""
    m1 = _as_float_matrix(x)
    m2 = _as_float_matrix(y)
    return _finite(m1.rolling(int(d), min_periods=_min_obs(d)).cov(m2))


def decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """线性加权移动平均：近值权重更大，权重 = d, d-1, ..., 1。

    decay_linear(x, d) = sum_{i=1..d} i * x_{t-d+i} / sum(i)

    NaN 语义：当日值缺失 → NaN；窗口内至少 ``_min_obs(d)`` 个有效观测即可计算，
    缺失位置的权重被剔除并按剩余权重重新归一（不会整段抹成 NaN）。
    """
    m = _as_float_matrix(x)
    d = int(d)

    def _decay(w: np.ndarray) -> float:
        # pandas 在序列开头的"部分窗口"会传入短于 d 的数组，故权重按实际长度生成
        n = w.size
        weights = np.arange(1, n + 1, dtype=float)
        valid = ~np.isnan(w)
        if np.isnan(w[-1]) or valid.sum() < _min_obs(d):
            return np.nan
        return float(np.dot(w[valid], weights[valid]) / weights[valid].sum())

    return m.rolling(d, min_periods=_min_obs(d)).apply(_decay, raw=True)


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序差分：x - x.shift(d)。"""
    return _as_float_matrix(x) - _as_float_matrix(x).shift(int(d))


def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序滞后：x.shift(d)。"""
    return _as_float_matrix(x).shift(int(d))


def product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序连乘：过去 d 日的乘积（NaN 语义同 ts_rank：按有效观测计算）。"""
    m = _as_float_matrix(x)
    d = int(d)

    def _prod(w: np.ndarray) -> float:
        valid = ~np.isnan(w)
        if np.isnan(w[-1]) or valid.sum() < _min_obs(d):
            return np.nan
        return float(np.prod(w[valid]))

    return m.rolling(d, min_periods=_min_obs(d)).apply(_prod, raw=True)
