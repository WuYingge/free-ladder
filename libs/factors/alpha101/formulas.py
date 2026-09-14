"""
Alpha101 公式实现 (Alpha101 Formula Implementations)

WorldQuant《101 Formulaic Alphas》全部 101 个公式在 ``date x symbol`` 面板上的实现。
公式文本与符号约定见文末引用；每个函数签名 ``(Alpha101Inputs) -> DataFrame``。

约定与实现取舍（务必先读）：

1. **窗口取整**：论文里窗口常带小数（``3.92795``/``16.1219``/``250``…），本实现按
   四舍五入取整日（``_w``），这是业界通行做法；取整口径在测试里被固定，不随平台变化。
2. **三元条件**：用 ``.where(cond, other)`` 实现，与原式一致（含 ``min(x, d)`` 这类
   第二参数是**标量窗口**的写法 → 当作 ``ts_min(x, d)``；第二参数是**表达式**时才是
   逐元素 ``min_``，见 #29 / #71 / #73 / #77 / #88 / #92 / #96）。
3. **比较型 alpha**（#61/#62/#64/#65/#68/#74/#75/#79/#81/#86/#95/#99 等）原式返回
   boolean，本实现统一转 **float（1.0/0.0）**，以便进入 IC/分组分析。
4. **幂运算**：一律走 ``signedpower``（``sign(x)·|x|^a``），指数可以是序列
   （如 #84 ``SignedPower(ts_rank(...), delta(close, d))``）。
5. **除零**：``(high-low)``、``(vwap-close)`` 等作分母处用 ``_clean`` 把 ±inf 抹成 NaN。
6. **行业中性化** ``IndNeutralize(x, IndClass.xxx)`` 用**当日生效**的申万分类做时点
   中性化（``inp.industry(level)``）：sector→一级、industry→二级、subindustry→三级。
   需要 ``build_alpha101_inputs(..., needs_industry=True)``。
7. **数据口径提醒**：面板价格来自本地**后复权**序列，而东财后复权是**仿射口径**
   （``H = a·P + b``），其日收益被压缩 ``a·P/(a·P+b)``（逐股不同）。因此含
   ``returns/stddev/correlation`` 的因子（原文即以复权价构造）在**跨股幅度**上会带上
   该口径效应；vwap 已用 ``data/adj_factor`` 搬回同一尺度，同日横截面量不受影响。
   详见 ``docs/eastmoney_hfq_convention.md``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
from factors.alpha101.panel import Alpha101Inputs

# ── 工具 ──────────────────────────────────────────────────────────────────────


def _w(d: float) -> int:
    """论文里的分数窗口按四舍五入取整成整数交易日（最小 1）。"""
    return max(1, int(round(float(d))))


def _clean(x: pd.DataFrame) -> pd.DataFrame:
    """±inf → NaN（分母可能为 0 的除法结果）。"""
    return x.replace([np.inf, -np.inf], np.nan)


def _adv(inp: Alpha101Inputs, d: float) -> pd.DataFrame:
    """adv{d} = 过去 d 日平均成交额。"""
    return adv(inp.value, _w(d))


def _ind(inp: Alpha101Inputs, x: pd.DataFrame, level: int) -> pd.DataFrame:
    """按当日生效的申万 level 档位做行业中性化（point-in-time）。"""
    return indneutralize(x, inp.industry(level))


def _b(x: pd.DataFrame) -> pd.DataFrame:
    """比较型布尔结果 → float（1.0/0.0）。"""
    return x.astype(float)


# ── #1 ~ #20 ─────────────────────────────────────────────────────────────────


def alpha_001(inp: Alpha101Inputs) -> pd.DataFrame:
    """#1: rank(ts_argmax(signedpower(returns<0 ? stddev(returns,20) : close, 2), 5)) - 0.5"""
    std = ts_stddev(inp.returns, 20)
    base = std.where(inp.returns < 0, inp.close)
    return rank(ts_argmax(signedpower(base, 2.0), 5)) - 0.5


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


def alpha_005(inp: Alpha101Inputs) -> pd.DataFrame:
    """#5: rank(open - sum(vwap,10)/10) * -1*abs(rank(close - vwap))"""
    o, c, vw = inp.open, inp.close, inp.vwap
    return rank(o - ts_sum(vw, 10) / 10.0) * (-1.0 * abs_(rank(c - vw)))


def alpha_006(inp: Alpha101Inputs) -> pd.DataFrame:
    """#6: -1 * correlation(open, volume, 10)"""
    return -1.0 * correlation(inp.open, inp.volume, 10)


def alpha_007(inp: Alpha101Inputs) -> pd.DataFrame:
    """#7: (adv20 < volume) ? (-1*ts_rank(|delta(close,7)|,60) * sign(delta(close,7))) : -1"""
    d7 = delta(inp.close, 7)
    val = (-1.0 * ts_rank(abs_(d7), 60)) * sign(d7)
    return val.where(_adv(inp, 20) < inp.volume, -1.0)


def alpha_008(inp: Alpha101Inputs) -> pd.DataFrame:
    """#8: -1 * rank(sum(open,5)*sum(returns,5) - delay(sum(open,5)*sum(returns,5), 10))"""
    inner = ts_sum(inp.open, 5) * ts_sum(inp.returns, 5)
    return -1.0 * rank(inner - delay(inner, 10))


def alpha_009(inp: Alpha101Inputs) -> pd.DataFrame:
    """#9: (0 < ts_min(delta(close,1),5)) ? delta : ((ts_max(delta,5) < 0) ? delta : -delta)"""
    d1 = delta(inp.close, 1)
    up = ts_min(d1, 5) > 0
    down = ts_max(d1, 5) < 0
    return d1.where(up | down, -1.0 * d1)


def alpha_010(inp: Alpha101Inputs) -> pd.DataFrame:
    """#10: rank(同 #9 但窗口为 4)"""
    d1 = delta(inp.close, 1)
    up = ts_min(d1, 4) > 0
    down = ts_max(d1, 4) < 0
    return rank(d1.where(up | down, -1.0 * d1))


def alpha_011(inp: Alpha101Inputs) -> pd.DataFrame:
    """#11: (rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3))"""
    vc = inp.vwap - inp.close
    return (rank(ts_max(vc, 3)) + rank(ts_min(vc, 3))) * rank(delta(inp.volume, 3))


def alpha_012(inp: Alpha101Inputs) -> pd.DataFrame:
    """#12: sign(delta(volume,1)) * (-1 * delta(close,1))"""
    return sign(delta(inp.volume, 1)) * (-1.0 * delta(inp.close, 1))


def alpha_013(inp: Alpha101Inputs) -> pd.DataFrame:
    """#13: -1 * rank(covariance(rank(close), rank(volume), 5))"""
    return -1.0 * rank(covariance(rank(inp.close), rank(inp.volume), 5))


def alpha_014(inp: Alpha101Inputs) -> pd.DataFrame:
    """#14: (-1 * rank(delta(returns,3))) * correlation(open, volume, 10)"""
    return (-1.0 * rank(delta(inp.returns, 3))) * correlation(inp.open, inp.volume, 10)


def alpha_015(inp: Alpha101Inputs) -> pd.DataFrame:
    """#15: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""
    return -1.0 * ts_sum(rank(correlation(rank(inp.high), rank(inp.volume), 3)), 3)


def alpha_016(inp: Alpha101Inputs) -> pd.DataFrame:
    """#16: -1 * rank(covariance(rank(high), rank(volume), 5))"""
    return -1.0 * rank(covariance(rank(inp.high), rank(inp.volume), 5))


def alpha_017(inp: Alpha101Inputs) -> pd.DataFrame:
    """#17: (-1*rank(ts_rank(close,10))) * rank(delta(delta(close,1),1)) * rank(ts_rank(volume/adv20,5))"""
    return (
        (-1.0 * rank(ts_rank(inp.close, 10)))
        * rank(delta(delta(inp.close, 1), 1))
        * rank(ts_rank(inp.volume / _adv(inp, 20), 5))
    )


def alpha_018(inp: Alpha101Inputs) -> pd.DataFrame:
    """#18: -1 * rank(stddev(|close-open|,5) + (close-open) + correlation(close,open,10))"""
    co = inp.close - inp.open
    return -1.0 * rank(ts_stddev(abs_(co), 5) + co + correlation(inp.close, inp.open, 10))


def alpha_019(inp: Alpha101Inputs) -> pd.DataFrame:
    """#19: (-1*sign((close-delay(close,7)) + delta(close,7))) * (1 + rank(1+sum(returns,250)))"""
    return (-1.0 * sign((inp.close - delay(inp.close, 7)) + delta(inp.close, 7))) * (
        1.0 + rank(1.0 + ts_sum(inp.returns, 250))
    )


def alpha_020(inp: Alpha101Inputs) -> pd.DataFrame:
    """#20: (-1*rank(open-delay(high,1))) * rank(open-delay(close,1)) * rank(open-delay(low,1))"""
    o = inp.open
    return (
        (-1.0 * rank(o - delay(inp.high, 1)))
        * rank(o - delay(inp.close, 1))
        * rank(o - delay(inp.low, 1))
    )


# ── #21 ~ #40 ────────────────────────────────────────────────────────────────


def alpha_021(inp: Alpha101Inputs) -> pd.DataFrame:
    """#21: 依 (均线±stddev) 与量比判定 ±1 的条件因子"""
    c, v = inp.close, inp.volume
    m8, s8, m2 = ts_mean(c, 8), ts_stddev(c, 8), ts_mean(c, 2)
    ratio = v / _adv(inp, 20)
    cond_lo = m8 + s8 < m2
    cond_hi = m2 < m8 - s8
    out = pd.DataFrame(-1.0, index=c.index, columns=c.columns)
    out = out.where(~cond_hi, 1.0)
    out = out.where(~(~cond_lo & ~cond_hi & (ratio >= 1.0)), 1.0)
    return out.where(~cond_lo, -1.0)


def alpha_022(inp: Alpha101Inputs) -> pd.DataFrame:
    """#22: -1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))"""
    return -1.0 * (delta(correlation(inp.high, inp.volume, 5), 5) * rank(ts_stddev(inp.close, 20)))


def alpha_023(inp: Alpha101Inputs) -> pd.DataFrame:
    """#23: (sum(high,20)/20 < high) ? (-1*delta(high,2)) : 0"""
    h = inp.high
    return (-1.0 * delta(h, 2)).where(ts_mean(h, 20) < h, 0.0)


def alpha_024(inp: Alpha101Inputs) -> pd.DataFrame:
    """#24: 100 日均线相对 100 日前涨幅 <= 5% ? -1*(close-ts_min(close,100)) : -1*delta(close,3)"""
    c = inp.close
    trend = delta(ts_mean(c, 100), 100) / delay(c, 100)
    return (-1.0 * (c - ts_min(c, 100))).where(trend <= 0.05, -1.0 * delta(c, 3))


def alpha_025(inp: Alpha101Inputs) -> pd.DataFrame:
    """#25: rank((-1*returns*adv20*vwap) * (high - close))"""
    return rank(((-1.0 * inp.returns) * _adv(inp, 20) * inp.vwap) * (inp.high - inp.close))


def alpha_026(inp: Alpha101Inputs) -> pd.DataFrame:
    """#26: -1 * ts_max(correlation(ts_rank(volume,5), ts_rank(high,5), 5), 3)"""
    return -1.0 * ts_max(
        correlation(ts_rank(inp.volume, 5), ts_rank(inp.high, 5), 5), 3
    )


def alpha_027(inp: Alpha101Inputs) -> pd.DataFrame:
    """#27: rank(sum(correlation(rank(volume), rank(vwap), 6), 2)/2) > 0.5 ? -1 : 1"""
    inner = ts_sum(correlation(rank(inp.volume), rank(inp.vwap), 6), 2) / 2.0
    out = pd.DataFrame(1.0, index=inner.index, columns=inner.columns)
    return out.where(~(rank(inner) > 0.5), -1.0)


def alpha_028(inp: Alpha101Inputs) -> pd.DataFrame:
    """#28: scale(correlation(adv20, low, 5) + (high+low)/2 - close)"""
    return scale(
        correlation(_adv(inp, 20), inp.low, 5) + (inp.high + inp.low) / 2.0 - inp.close
    )


def alpha_029(inp: Alpha101Inputs) -> pd.DataFrame:
    """#29: ts_min(product(rank(rank(scale(log(sum(ts_min(rank(rank(-rank(delta(close-1,5)))),2),1))))),1),5)
    + ts_rank(delay(-returns,6), 5)"""
    inner = -1.0 * rank(delta(inp.close - 1.0, 5))
    x = ts_sum(ts_min(rank(rank(inner)), 2), 1)
    x = product(rank(rank(scale(log(x)))), 1)
    return ts_min(x, 5) + ts_rank(delay(-1.0 * inp.returns, 6), 5)


def alpha_030(inp: Alpha101Inputs) -> pd.DataFrame:
    """#30: (1 - rank(Σsign(delay(close,i)-delay(close,i+1)))) * sum(volume,5) / sum(volume,20)"""
    c = inp.close
    s = sign(c - delay(c, 1)) + sign(delay(c, 1) - delay(c, 2)) + sign(delay(c, 2) - delay(c, 3))
    return ((1.0 - rank(s)) * ts_sum(inp.volume, 5)) / ts_sum(inp.volume, 20)


def alpha_031(inp: Alpha101Inputs) -> pd.DataFrame:
    """#31: rank(rank(rank(decay_linear(-rank(rank(delta(close,10))),10)))) + rank(-delta(close,3))
    + sign(scale(correlation(adv20, low, 12)))"""
    return (
        rank(rank(rank(decay_linear(-1.0 * rank(rank(delta(inp.close, 10))), 10))))
        + rank(-1.0 * delta(inp.close, 3))
        + sign(scale(correlation(_adv(inp, 20), inp.low, 12)))
    )


def alpha_032(inp: Alpha101Inputs) -> pd.DataFrame:
    """#32: scale(sum(close,7)/7 - close) + 20*scale(correlation(vwap, delay(close,5), 230))"""
    return scale(ts_mean(inp.close, 7) - inp.close) + 20.0 * scale(
        correlation(inp.vwap, delay(inp.close, 5), 230)
    )


def alpha_033(inp: Alpha101Inputs) -> pd.DataFrame:
    """#33: rank(-1 * (1 - open/close)^1)"""
    return rank(-1.0 * (1.0 - inp.open / inp.close))


def alpha_034(inp: Alpha101Inputs) -> pd.DataFrame:
    """#34: rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1))))"""
    ratio = ts_stddev(inp.returns, 2) / ts_stddev(inp.returns, 5)
    return rank((1.0 - rank(ratio)) + (1.0 - rank(delta(inp.close, 1))))


def alpha_035(inp: Alpha101Inputs) -> pd.DataFrame:
    """#35: ts_rank(volume,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(returns,32))"""
    return (
        ts_rank(inp.volume, 32)
        * (1.0 - ts_rank((inp.close + inp.high) - inp.low, 16))
        * (1.0 - ts_rank(inp.returns, 32))
    )


def alpha_036(inp: Alpha101Inputs) -> pd.DataFrame:
    """#36: 2.21*rank(corr(close-open, delay(volume,1),15)) + 0.7*rank(open-close)
    + 0.73*rank(ts_rank(delay(-returns,6),5)) + rank(|corr(vwap, adv20, 6)|)
    + 0.6*rank((sum(close,200)/200 - open) * (close-open))"""
    c, o = inp.close, inp.open
    return (
        2.21 * rank(correlation(c - o, delay(inp.volume, 1), 15))
        + 0.7 * rank(o - c)
        + 0.73 * rank(ts_rank(delay(-1.0 * inp.returns, 6), 5))
        + rank(abs_(correlation(inp.vwap, _adv(inp, 20), 6)))
        + 0.6 * rank((ts_mean(c, 200) - o) * (c - o))
    )


def alpha_037(inp: Alpha101Inputs) -> pd.DataFrame:
    """#37: rank(correlation(delay(open-close,1), close, 200)) + rank(open - close)"""
    return rank(correlation(delay(inp.open - inp.close, 1), inp.close, 200)) + rank(
        inp.open - inp.close
    )


def alpha_038(inp: Alpha101Inputs) -> pd.DataFrame:
    """#38: (-1*rank(ts_rank(close,10))) * rank(close/open)"""
    return (-1.0 * rank(ts_rank(inp.close, 10))) * rank(inp.close / inp.open)


def alpha_039(inp: Alpha101Inputs) -> pd.DataFrame:
    """#39: (-1*rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20, 9))))) * (1+rank(sum(returns,250)))"""
    return (
        -1.0
        * rank(delta(inp.close, 7) * (1.0 - rank(decay_linear(inp.volume / _adv(inp, 20), 9))))
        * (1.0 + rank(ts_sum(inp.returns, 250)))
    )


def alpha_040(inp: Alpha101Inputs) -> pd.DataFrame:
    """#40: (-1*rank(stddev(high,10))) * correlation(high, volume, 10)"""
    return (-1.0 * rank(ts_stddev(inp.high, 10))) * correlation(inp.high, inp.volume, 10)


# ── #41 ~ #60 ────────────────────────────────────────────────────────────────


def alpha_041(inp: Alpha101Inputs) -> pd.DataFrame:
    """#41: sqrt(high*low) - vwap"""
    return _clean((inp.high * inp.low) ** 0.5 - inp.vwap)


def alpha_042(inp: Alpha101Inputs) -> pd.DataFrame:
    """#42: rank(vwap - close) / rank(vwap + close)"""
    return _clean(rank(inp.vwap - inp.close) / rank(inp.vwap + inp.close))


def alpha_043(inp: Alpha101Inputs) -> pd.DataFrame:
    """#43: ts_rank(volume/adv20, 20) * ts_rank(-delta(close,7), 8)"""
    return ts_rank(inp.volume / _adv(inp, 20), 20) * ts_rank(-1.0 * delta(inp.close, 7), 8)


def alpha_044(inp: Alpha101Inputs) -> pd.DataFrame:
    """#44: -1 * correlation(high, rank(volume), 5)"""
    return -1.0 * correlation(inp.high, rank(inp.volume), 5)


def alpha_045(inp: Alpha101Inputs) -> pd.DataFrame:
    """#45: -1 * rank(sum(delay(close,5),20)/20) * corr(close,volume,2) * rank(corr(sum(close,5),sum(close,20),2))"""
    c = inp.close
    return (
        -1.0
        * rank(ts_mean(delay(c, 5), 20))
        * correlation(c, inp.volume, 2)
        * rank(correlation(ts_sum(c, 5), ts_sum(c, 20), 2))
    )


def alpha_046(inp: Alpha101Inputs) -> pd.DataFrame:
    """#46: 依 (delay20-delay10)/10 - (delay10-close)/10 判定的条件因子"""
    c = inp.close
    slope = (delay(c, 20) - delay(c, 10)) / 10.0 - (delay(c, 10) - c) / 10.0
    out = pd.DataFrame(-1.0, index=c.index, columns=c.columns)
    out = out.where(~(slope < 0), 1.0)
    out = out.where(~(slope > 0.25), (-1.0) * (c - delay(c, 1)))
    return out


def alpha_047(inp: Alpha101Inputs) -> pd.DataFrame:
    """#47: (rank(1/close)*volume/adv20) * (high*rank(high-close)/(sum(high,5)/5)) - rank(vwap - delay(vwap,5))"""
    inner = (rank(1.0 / inp.close) * inp.volume) / _adv(inp, 20)
    return _clean(
        inner * ((inp.high * rank(inp.high - inp.close)) / ts_mean(inp.high, 5))
        - rank(inp.vwap - delay(inp.vwap, 5))
    )


def alpha_048(inp: Alpha101Inputs) -> pd.DataFrame:
    """#48: indneutralize(corr(delta(close,1), delta(delay(close,1),1), 250) * delta(close,1) / close, subindustry)
    / sum((delta(close,1)/delay(close,1))^2, 250)"""
    c = inp.close
    d1 = delta(c, 1)
    numer = _ind(inp, _clean(correlation(d1, delta(delay(c, 1), 1), 250) * d1 / c), 3)
    denom = ts_sum(signedpower(_clean(d1 / delay(c, 1)), 2.0), 250)
    return _clean(numer / denom)


def alpha_049(inp: Alpha101Inputs) -> pd.DataFrame:
    """#49: 斜率 < -0.1 ? 1 : -1*(close - delay(close,1))"""
    c = inp.close
    slope = (delay(c, 20) - delay(c, 10)) / 10.0 - (delay(c, 10) - c) / 10.0
    return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
        slope >= -0.1, (-1.0) * (c - delay(c, 1))
    )


def alpha_050(inp: Alpha101Inputs) -> pd.DataFrame:
    """#50: -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)"""
    return -1.0 * ts_max(rank(correlation(rank(inp.volume), rank(inp.vwap), 5)), 5)


def alpha_051(inp: Alpha101Inputs) -> pd.DataFrame:
    """#51: 斜率 < -0.05 ? 1 : -1*(close - delay(close,1))"""
    c = inp.close
    slope = (delay(c, 20) - delay(c, 10)) / 10.0 - (delay(c, 10) - c) / 10.0
    return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
        slope >= -0.05, (-1.0) * (c - delay(c, 1))
    )


def alpha_052(inp: Alpha101Inputs) -> pd.DataFrame:
    """#52: ((-1*ts_min(low,5) + delay(ts_min(low,5),5)) * rank((sum(returns,240)-sum(returns,20))/220)) * ts_rank(volume,5)"""
    m5 = ts_min(inp.low, 5)
    return (
        ((-1.0 * m5) + delay(m5, 5))
        * rank((ts_sum(inp.returns, 240) - ts_sum(inp.returns, 20)) / 220.0)
        * ts_rank(inp.volume, 5)
    )


def alpha_053(inp: Alpha101Inputs) -> pd.DataFrame:
    """#53: -1 * delta(((close-low) - (high-close)) / (close-low), 9)"""
    c, h, low = inp.close, inp.high, inp.low
    return -1.0 * delta(_clean(((c - low) - (h - c)) / (c - low)), 9)


def alpha_054(inp: Alpha101Inputs) -> pd.DataFrame:
    """#54: (-1 * (low-close)*open^5) / ((low-high)*close^5)"""
    c, o, h, low = inp.close, inp.open, inp.high, inp.low
    return _clean(
        (-1.0 * (low - c) * signedpower(o, 5.0)) / ((low - h) * signedpower(c, 5.0))
    )


def alpha_055(inp: Alpha101Inputs) -> pd.DataFrame:
    """#55: -1 * correlation(rank((close - ts_min(low,12)) / (ts_max(high,12) - ts_min(low,12))), rank(volume), 6)"""
    c = inp.close
    pos = _clean((c - ts_min(inp.low, 12)) / (ts_max(inp.high, 12) - ts_min(inp.low, 12)))
    return -1.0 * correlation(rank(pos), rank(inp.volume), 6)


def alpha_056(inp: Alpha101Inputs) -> pd.DataFrame:
    """#56: -(rank(sum(returns,10)/sum(sum(returns,2),3)) * rank(returns*cap))（唯一用 cap 的公式）"""
    ratio = _clean(ts_sum(inp.returns, 10) / ts_sum(ts_sum(inp.returns, 2), 3))
    return -1.0 * (rank(ratio) * rank(inp.returns * inp.cap))


def alpha_057(inp: Alpha101Inputs) -> pd.DataFrame:
    """#57: -((close - vwap) / decay_linear(rank(ts_argmax(close,30)), 2))"""
    return -1.0 * _clean(
        (inp.close - inp.vwap) / decay_linear(rank(ts_argmax(inp.close, 30)), 2)
    )


def alpha_058(inp: Alpha101Inputs) -> pd.DataFrame:
    """#58: -1*ts_rank(decay_linear(corr(indneutralize(vwap, sector), volume, 3.92795), 7.89291), 5.50322)"""
    return -1.0 * ts_rank(
        decay_linear(correlation(_ind(inp, inp.vwap, 1), inp.volume, _w(3.92795)), _w(7.89291)),
        _w(5.50322),
    )


def alpha_059(inp: Alpha101Inputs) -> pd.DataFrame:
    """#59: -1*ts_rank(decay_linear(corr(indneutralize(vwap*0.728317+vwap*0.271683, industry), volume, 4.25197), 16.2289), 8.19648)"""
    vw = inp.vwap * 0.728317 + inp.vwap * (1 - 0.728317)
    return -1.0 * ts_rank(
        decay_linear(correlation(_ind(inp, vw, 2), inp.volume, _w(4.25197)), _w(16.2289)),
        _w(8.19648),
    )


def alpha_060(inp: Alpha101Inputs) -> pd.DataFrame:
    """#60: -(2*scale(rank(((close-low)-(high-close))/(high-low) * volume)) - scale(rank(ts_argmax(close,10))))"""
    c, h, low = inp.close, inp.high, inp.low
    body = _clean(((c - low) - (h - c)) / (h - low)) * inp.volume
    return -1.0 * (2.0 * scale(rank(body)) - scale(rank(ts_argmax(c, 10))))


# ── #61 ~ #80 ────────────────────────────────────────────────────────────────


def alpha_061(inp: Alpha101Inputs) -> pd.DataFrame:
    """#61: rank(vwap - ts_min(vwap, 16.1219)) < rank(correlation(vwap, adv180, 17.9282))"""
    return _b(
        rank(inp.vwap - ts_min(inp.vwap, _w(16.1219)))
        < rank(correlation(inp.vwap, _adv(inp, 180), _w(17.9282)))
    )


def alpha_062(inp: Alpha101Inputs) -> pd.DataFrame:
    """#62: -1*(rank(corr(vwap, sum(adv20,22.4101), 9.91009)) < rank((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high))))"""
    inner = (rank(inp.open) + rank(inp.open)) < (
        rank((inp.high + inp.low) / 2.0) + rank(inp.high)
    )
    return -1.0 * _b(
        rank(correlation(inp.vwap, ts_sum(_adv(inp, 20), _w(22.4101)), _w(9.91009)))
        < rank(_b(inner))
    )


def alpha_063(inp: Alpha101Inputs) -> pd.DataFrame:
    """#63: -1*(rank(decay_linear(delta(indneutralize(close, industry), 2.25164), 8.22237))
    - rank(decay_linear(corr(vwap*0.318108+open*0.681892, sum(adv180,37.2467), 13.557), 12.2883)))"""
    lhs = rank(decay_linear(delta(_ind(inp, inp.close, 2), _w(2.25164)), _w(8.22237)))
    mix = inp.vwap * 0.318108 + inp.open * (1 - 0.318108)
    rhs = rank(
        decay_linear(
            correlation(mix, ts_sum(_adv(inp, 180), _w(37.2467)), _w(13.557)), _w(12.2883)
        )
    )
    return -1.0 * (lhs - rhs)


def alpha_064(inp: Alpha101Inputs) -> pd.DataFrame:
    """#64: -1*(rank(corr(sum(open*0.178404+low*0.821596, 12.7054), sum(adv120,12.7054), 16.6208))
    < rank(delta((high+low)/2*0.178404 + vwap*0.821596, 3.69741)))"""
    mix1 = inp.open * 0.178404 + inp.low * (1 - 0.178404)
    left = rank(
        correlation(
            ts_sum(mix1, _w(12.7054)), ts_sum(_adv(inp, 120), _w(12.7054)), _w(16.6208)
        )
    )
    mix2 = (inp.high + inp.low) / 2.0 * 0.178404 + inp.vwap * (1 - 0.178404)
    return -1.0 * _b(left < rank(delta(mix2, _w(3.69741))))


def alpha_065(inp: Alpha101Inputs) -> pd.DataFrame:
    """#65: -1*(rank(corr(open*0.00817205 + vwap*0.99182795, sum(adv60,8.6911), 6.40374))
    < rank(open - ts_min(open, 13.635)))"""
    mix = inp.open * 0.00817205 + inp.vwap * (1 - 0.00817205)
    left = rank(
        correlation(mix, ts_sum(_adv(inp, 60), _w(8.6911)), _w(6.40374))
    )
    return -1.0 * _b(left < rank(inp.open - ts_min(inp.open, _w(13.635))))


def alpha_066(inp: Alpha101Inputs) -> pd.DataFrame:
    """#66: -1*(rank(decay_linear(delta(vwap, 3.51013), 7.23052))
    + ts_rank(decay_linear((low - vwap)/(open - (high+low)/2), 11.4157), 6.72611))"""
    num = inp.low * 0.96633 + inp.low * (1 - 0.96633) - inp.vwap
    den = inp.open - (inp.high + inp.low) / 2.0
    return -1.0 * (
        rank(decay_linear(delta(inp.vwap, _w(3.51013)), _w(7.23052)))
        + ts_rank(decay_linear(_clean(num / den), _w(11.4157)), _w(6.72611))
    )


def alpha_067(inp: Alpha101Inputs) -> pd.DataFrame:
    """#67: -1*(rank(high - ts_min(high, 2.14593)) ^ rank(corr(indneutralize(vwap, sector), indneutralize(adv20, subindustry), 6.02936)))"""
    left = rank(inp.high - ts_min(inp.high, _w(2.14593)))
    right = rank(
        correlation(
            _ind(inp, inp.vwap, 1), _ind(inp, _adv(inp, 20), 3), _w(6.02936)
        )
    )
    return -1.0 * signedpower(left, right)


def alpha_068(inp: Alpha101Inputs) -> pd.DataFrame:
    """#68: -1*(ts_rank(corr(rank(high), rank(adv15), 8.91644), 13.9333)
    < rank(delta(close*0.518371 + low*0.481629, 1.06157)))"""
    left = ts_rank(
        correlation(rank(inp.high), rank(_adv(inp, 15)), _w(8.91644)), _w(13.9333)
    )
    mix = inp.close * 0.518371 + inp.low * (1 - 0.518371)
    return -1.0 * _b(left < rank(delta(mix, _w(1.06157))))


def alpha_069(inp: Alpha101Inputs) -> pd.DataFrame:
    """#69: -1*(rank(ts_max(delta(indneutralize(vwap, industry), 2.72412), 4.79344))
    ^ ts_rank(corr(close*0.490655 + vwap*0.509345, adv20, 4.92416), 9.0615))"""
    left = rank(
        ts_max(delta(_ind(inp, inp.vwap, 2), _w(2.72412)), _w(4.79344))
    )
    mix = inp.close * 0.490655 + inp.vwap * (1 - 0.490655)
    right = ts_rank(
        correlation(mix, _adv(inp, 20), _w(4.92416)), _w(9.0615)
    )
    return -1.0 * signedpower(left, right)


def alpha_070(inp: Alpha101Inputs) -> pd.DataFrame:
    """#70: -1*(rank(delta(vwap, 1.29456)) ^ ts_rank(corr(indneutralize(close, industry), adv50, 17.8256), 17.9171))"""
    left = rank(delta(inp.vwap, _w(1.29456)))
    right = ts_rank(
        correlation(_ind(inp, inp.close, 2), _adv(inp, 50), _w(17.8256)), _w(17.9171)
    )
    return -1.0 * signedpower(left, right)


def alpha_071(inp: Alpha101Inputs) -> pd.DataFrame:
    """#71: max(ts_rank(decay_linear(corr(ts_rank(close,3.43976), ts_rank(adv180,12.0647), 18.0175), 4.20501), 15.6948),
    ts_rank(decay_linear(rank((low+open) - 2*vwap)^2, 16.4662), 4.4388))"""
    lhs = ts_rank(
        decay_linear(
            correlation(
                ts_rank(inp.close, _w(3.43976)),
                ts_rank(_adv(inp, 180), _w(12.0647)),
                _w(18.0175),
            ),
            _w(4.20501),
        ),
        _w(15.6948),
    )
    rhs = ts_rank(
        decay_linear(signedpower(rank((inp.low + inp.open) - (inp.vwap + inp.vwap)), 2.0), _w(16.4662)),
        _w(4.4388),
    )
    return max_(lhs, rhs)


def alpha_072(inp: Alpha101Inputs) -> pd.DataFrame:
    """#72: rank(decay_linear(corr((high+low)/2, adv40, 8.93345), 10.1519))
    / rank(decay_linear(corr(ts_rank(vwap,3.72469), ts_rank(volume,18.5188), 6.86671), 2.95011))"""
    numer = rank(
        decay_linear(
            correlation((inp.high + inp.low) / 2.0, _adv(inp, 40), _w(8.93345)), _w(10.1519)
        )
    )
    denom = rank(
        decay_linear(
            correlation(
                ts_rank(inp.vwap, _w(3.72469)),
                ts_rank(inp.volume, _w(18.5188)),
                _w(6.86671),
            ),
            _w(2.95011),
        )
    )
    return _clean(numer / denom)


def alpha_073(inp: Alpha101Inputs) -> pd.DataFrame:
    """#73: -1*max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
    ts_rank(decay_linear(-delta(open*0.147155+low*0.852845, 2.03608)/(open*0.147155+low*0.852845), 3.33829), 16.7411))"""
    lhs = rank(decay_linear(delta(inp.vwap, _w(4.72775)), _w(2.91864)))
    mix = inp.open * 0.147155 + inp.low * (1 - 0.147155)
    rhs = ts_rank(
        decay_linear(_clean(-1.0 * delta(mix, _w(2.03608)) / mix), _w(3.33829)), _w(16.7411)
    )
    return -1.0 * max_(lhs, rhs)


def alpha_074(inp: Alpha101Inputs) -> pd.DataFrame:
    """#74: -1*(rank(corr(close, sum(adv30,37.4843), 15.1365))
    < rank(corr(rank(high*0.0261661 + vwap*0.9738339), rank(volume), 11.4791)))"""
    left = rank(correlation(inp.close, ts_sum(_adv(inp, 30), _w(37.4843)), _w(15.1365)))
    mix = inp.high * 0.0261661 + inp.vwap * (1 - 0.0261661)
    right = rank(correlation(rank(mix), rank(inp.volume), _w(11.4791)))
    return -1.0 * _b(left < right)


def alpha_075(inp: Alpha101Inputs) -> pd.DataFrame:
    """#75: rank(corr(vwap, volume, 4.24304)) < rank(corr(rank(low), rank(adv50), 12.4413))"""
    return _b(
        rank(correlation(inp.vwap, inp.volume, _w(4.24304)))
        < rank(correlation(rank(inp.low), rank(_adv(inp, 50)), _w(12.4413)))
    )


def alpha_076(inp: Alpha101Inputs) -> pd.DataFrame:
    """#76: -1*max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
    ts_rank(decay_linear(ts_rank(corr(indneutralize(low, sector), adv81, 8.14941), 19.569), 17.1543), 19.383))"""
    lhs = rank(decay_linear(delta(inp.vwap, _w(1.24383)), _w(11.8259)))
    rhs = ts_rank(
        decay_linear(
            ts_rank(
                correlation(_ind(inp, inp.low, 1), _adv(inp, 81), _w(8.14941)), _w(19.569)
            ),
            _w(17.1543),
        ),
        _w(19.383),
    )
    return -1.0 * max_(lhs, rhs)


def alpha_077(inp: Alpha101Inputs) -> pd.DataFrame:
    """#77: min(rank(decay_linear((high+low)/2 + high - (vwap+high), 20.0451)),
    rank(decay_linear(corr((high+low)/2, adv40, 3.1614), 5.64125)))"""
    lhs = rank(
        decay_linear(((inp.high + inp.low) / 2.0 + inp.high) - (inp.vwap + inp.high), _w(20.0451))
    )
    rhs = rank(
        decay_linear(
            correlation((inp.high + inp.low) / 2.0, _adv(inp, 40), _w(3.1614)), _w(5.64125)
        )
    )
    return min_(lhs, rhs)


def alpha_078(inp: Alpha101Inputs) -> pd.DataFrame:
    """#78: rank(corr(sum(low*0.352233 + vwap*0.647767, 19.7428), sum(adv40,19.7428), 6.83313))
    ^ rank(corr(rank(vwap), rank(volume), 5.77492))"""
    mix = inp.low * 0.352233 + inp.vwap * (1 - 0.352233)
    left = rank(
        correlation(
            ts_sum(mix, _w(19.7428)), ts_sum(_adv(inp, 40), _w(19.7428)), _w(6.83313)
        )
    )
    right = rank(correlation(rank(inp.vwap), rank(inp.volume), _w(5.77492)))
    return signedpower(left, right)


def alpha_079(inp: Alpha101Inputs) -> pd.DataFrame:
    """#79: rank(delta(indneutralize(close*0.60733 + open*0.39267, sector), 1.23438))
    < rank(corr(ts_rank(vwap,3.60973), ts_rank(adv150,9.18637), 14.6644))"""
    mix = inp.close * 0.60733 + inp.open * (1 - 0.60733)
    left = rank(delta(_ind(inp, mix, 1), _w(1.23438)))
    right = rank(
        correlation(
            ts_rank(inp.vwap, _w(3.60973)), ts_rank(_adv(inp, 150), _w(9.18637)), _w(14.6644)
        )
    )
    return _b(left < right)


def alpha_080(inp: Alpha101Inputs) -> pd.DataFrame:
    """#80: -1*(rank(sign(delta(indneutralize(open*0.868128 + high*0.131872, industry), 4.04545)))
    ^ ts_rank(corr(high, adv10, 5.11456), 5.53756))"""
    mix = inp.open * 0.868128 + inp.high * (1 - 0.868128)
    left = rank(sign(delta(_ind(inp, mix, 2), _w(4.04545))))
    right = ts_rank(correlation(inp.high, _adv(inp, 10), _w(5.11456)), _w(5.53756))
    return -1.0 * signedpower(left, right)


# ── #81 ~ #101 ───────────────────────────────────────────────────────────────


def alpha_081(inp: Alpha101Inputs) -> pd.DataFrame:
    """#81: -1*(rank(log(product(rank(rank(corr(vwap, sum(adv10,49.6054), 8.47743))^4), 14.9655)))
    < rank(corr(rank(vwap), rank(volume), 5.07914)))"""
    corr4 = signedpower(
        rank(correlation(inp.vwap, ts_sum(_adv(inp, 10), _w(49.6054)), _w(8.47743))), 4.0
    )
    left = rank(log(product(rank(rank(corr4)), _w(14.9655))))
    right = rank(correlation(rank(inp.vwap), rank(inp.volume), _w(5.07914)))
    return -1.0 * _b(left < right)


def alpha_082(inp: Alpha101Inputs) -> pd.DataFrame:
    """#82: -1*min(rank(decay_linear(delta(open, 1.46063), 14.8717)),
    ts_rank(decay_linear(corr(indneutralize(volume, sector), open, 17.4842), 6.92131), 13.4283))"""
    lhs = rank(decay_linear(delta(inp.open, _w(1.46063)), _w(14.8717)))
    mix = inp.open * 0.634196 + inp.open * (1 - 0.634196)
    rhs = ts_rank(
        decay_linear(
            correlation(_ind(inp, inp.volume, 1), mix, _w(17.4842)), _w(6.92131)
        ),
        _w(13.4283),
    )
    return -1.0 * min_(lhs, rhs)


def alpha_083(inp: Alpha101Inputs) -> pd.DataFrame:
    """#83: (rank(delay((high-low)/(sum(close,5)/5), 2)) * rank(rank(volume)))
    / (((high-low)/(sum(close,5)/5)) / (vwap - close))"""
    hl = (inp.high - inp.low) / ts_mean(inp.close, 5)
    return _clean(
        (rank(delay(hl, 2)) * rank(rank(inp.volume))) / _clean(hl / (inp.vwap - inp.close))
    )


def alpha_084(inp: Alpha101Inputs) -> pd.DataFrame:
    """#84: signedpower(ts_rank(vwap - ts_max(vwap, 15.3217), 20.7127), delta(close, 4.96796))"""
    return signedpower(
        ts_rank(inp.vwap - ts_max(inp.vwap, _w(15.3217)), _w(20.7127)),
        delta(inp.close, _w(4.96796)),
    )


def alpha_085(inp: Alpha101Inputs) -> pd.DataFrame:
    """#85: rank(corr(high*0.876703 + close*0.123297, adv30, 9.61331))
    ^ rank(corr(ts_rank((high+low)/2, 3.70596), ts_rank(volume, 10.1595), 7.11408))"""
    mix = inp.high * 0.876703 + inp.close * (1 - 0.876703)
    left = rank(correlation(mix, _adv(inp, 30), _w(9.61331)))
    right = rank(
        correlation(
            ts_rank((inp.high + inp.low) / 2.0, _w(3.70596)),
            ts_rank(inp.volume, _w(10.1595)),
            _w(7.11408),
        )
    )
    return signedpower(left, right)


def alpha_086(inp: Alpha101Inputs) -> pd.DataFrame:
    """#86: -1*(ts_rank(corr(close, sum(adv20,14.7444), 6.00049), 20.4195)
    < rank((open+close) - (vwap+open)))"""
    left = ts_rank(
        correlation(inp.close, ts_sum(_adv(inp, 20), _w(14.7444)), _w(6.00049)), _w(20.4195)
    )
    right = rank((inp.open + inp.close) - (inp.vwap + inp.open))
    return -1.0 * _b(left < right)


def alpha_087(inp: Alpha101Inputs) -> pd.DataFrame:
    """#87: -1*max(rank(decay_linear(delta(close*0.369701 + vwap*0.630299, 1.91233), 2.65461)),
    ts_rank(decay_linear(|corr(indneutralize(adv81, industry), close, 13.4132)|, 4.89768), 14.4535))"""
    mix = inp.close * 0.369701 + inp.vwap * (1 - 0.369701)
    lhs = rank(decay_linear(delta(mix, _w(1.91233)), _w(2.65461)))
    rhs = ts_rank(
        decay_linear(
            abs_(correlation(_ind(inp, _adv(inp, 81), 2), inp.close, _w(13.4132))), _w(4.89768)
        ),
        _w(14.4535),
    )
    return -1.0 * max_(lhs, rhs)


def alpha_088(inp: Alpha101Inputs) -> pd.DataFrame:
    """#88: min(rank(decay_linear((rank(open)+rank(low)) - (rank(high)+rank(close)), 8.06882)),
    ts_rank(decay_linear(corr(ts_rank(close,8.44728), ts_rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))"""
    lhs = rank(
        decay_linear(
            (rank(inp.open) + rank(inp.low)) - (rank(inp.high) + rank(inp.close)), _w(8.06882)
        )
    )
    rhs = ts_rank(
        decay_linear(
            correlation(
                ts_rank(inp.close, _w(8.44728)),
                ts_rank(_adv(inp, 60), _w(20.6966)),
                _w(8.01266),
            ),
            _w(6.65053),
        ),
        _w(2.61957),
    )
    return min_(lhs, rhs)


def alpha_089(inp: Alpha101Inputs) -> pd.DataFrame:
    """#89: ts_rank(decay_linear(corr(low, adv10, 6.94279), 5.51607), 3.79744)
    - ts_rank(decay_linear(delta(indneutralize(vwap, industry), 3.48158), 10.1466), 15.3012)"""
    lhs = ts_rank(
        decay_linear(correlation(inp.low * 0.967285 + inp.low * (1 - 0.967285), _adv(inp, 10), _w(6.94279)), _w(5.51607)),
        _w(3.79744),
    )
    rhs = ts_rank(
        decay_linear(delta(_ind(inp, inp.vwap, 2), _w(3.48158)), _w(10.1466)), _w(15.3012)
    )
    return lhs - rhs


def alpha_090(inp: Alpha101Inputs) -> pd.DataFrame:
    """#90: -1*(rank(close - ts_max(close, 4.66719)) ^ ts_rank(corr(indneutralize(adv40, subindustry), low, 5.38375), 3.21856))"""
    left = rank(inp.close - ts_max(inp.close, _w(4.66719)))
    right = ts_rank(
        correlation(_ind(inp, _adv(inp, 40), 3), inp.low, _w(5.38375)), _w(3.21856)
    )
    return -1.0 * signedpower(left, right)


def alpha_091(inp: Alpha101Inputs) -> pd.DataFrame:
    """#91: -1*(ts_rank(decay_linear(decay_linear(corr(indneutralize(close, industry), volume, 9.74928), 16.398), 3.83219), 4.8667)
    - rank(decay_linear(corr(vwap, adv30, 4.01303), 2.6809)))"""
    lhs = ts_rank(
        decay_linear(
            decay_linear(
                correlation(_ind(inp, inp.close, 2), inp.volume, _w(9.74928)), _w(16.398)
            ),
            _w(3.83219),
        ),
        _w(4.8667),
    )
    rhs = rank(decay_linear(correlation(inp.vwap, _adv(inp, 30), _w(4.01303)), _w(2.6809)))
    return -1.0 * (lhs - rhs)


def alpha_092(inp: Alpha101Inputs) -> pd.DataFrame:
    """#92: min(ts_rank(decay_linear((high+low)/2 + close < low + open, 14.7221), 18.8683),
    ts_rank(decay_linear(corr(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))"""
    cond = _b((inp.high + inp.low) / 2.0 + inp.close < inp.low + inp.open)
    lhs = ts_rank(decay_linear(cond, _w(14.7221)), _w(18.8683))
    rhs = ts_rank(
        decay_linear(
            correlation(rank(inp.low), rank(_adv(inp, 30)), _w(7.58555)), _w(6.94024)
        ),
        _w(6.80584),
    )
    return min_(lhs, rhs)


def alpha_093(inp: Alpha101Inputs) -> pd.DataFrame:
    """#93: ts_rank(decay_linear(corr(indneutralize(vwap, industry), adv81, 17.4193), 19.848), 7.54455)
    / rank(decay_linear(delta(close*0.524434 + vwap*0.475566, 2.77377), 16.2664))"""
    numer = ts_rank(
        decay_linear(
            correlation(_ind(inp, inp.vwap, 2), _adv(inp, 81), _w(17.4193)), _w(19.848)
        ),
        _w(7.54455),
    )
    mix = inp.close * 0.524434 + inp.vwap * (1 - 0.524434)
    denom = rank(decay_linear(delta(mix, _w(2.77377)), _w(16.2664)))
    return _clean(numer / denom)


def alpha_094(inp: Alpha101Inputs) -> pd.DataFrame:
    """#94: -1*(rank(vwap - ts_min(vwap, 11.5783)) ^ ts_rank(corr(ts_rank(vwap,19.6462), ts_rank(adv60,4.02992), 18.0926), 2.70756))"""
    left = rank(inp.vwap - ts_min(inp.vwap, _w(11.5783)))
    right = ts_rank(
        correlation(
            ts_rank(inp.vwap, _w(19.6462)), ts_rank(_adv(inp, 60), _w(4.02992)), _w(18.0926)
        ),
        _w(2.70756),
    )
    return -1.0 * signedpower(left, right)


def alpha_095(inp: Alpha101Inputs) -> pd.DataFrame:
    """#95: rank(open - ts_min(open, 12.4105))
    < ts_rank(rank(corr(sum((high+low)/2, 19.1351), sum(adv40,19.1351), 12.8742))^5, 11.7584)"""
    left = rank(inp.open - ts_min(inp.open, _w(12.4105)))
    right = ts_rank(
        signedpower(
            rank(
                correlation(
                    ts_sum((inp.high + inp.low) / 2.0, _w(19.1351)),
                    ts_sum(_adv(inp, 40), _w(19.1351)),
                    _w(12.8742),
                )
            ),
            5.0,
        ),
        _w(11.7584),
    )
    return _b(left < right)


def alpha_096(inp: Alpha101Inputs) -> pd.DataFrame:
    """#96: -1*max(ts_rank(decay_linear(corr(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151),
    ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7.45404), ts_rank(adv60,4.13242), 3.65459), 12.6556), 14.0365), 13.4143))"""
    lhs = ts_rank(
        decay_linear(
            correlation(rank(inp.vwap), rank(inp.volume), _w(3.83878)), _w(4.16783)
        ),
        _w(8.38151),
    )
    rhs = ts_rank(
        decay_linear(
            ts_argmax(
                correlation(
                    ts_rank(inp.close, _w(7.45404)),
                    ts_rank(_adv(inp, 60), _w(4.13242)),
                    _w(3.65459),
                ),
                _w(12.6556),
            ),
            _w(14.0365),
        ),
        _w(13.4143),
    )
    return -1.0 * max_(lhs, rhs)


def alpha_097(inp: Alpha101Inputs) -> pd.DataFrame:
    """#97: -1*(rank(decay_linear(delta(indneutralize(low*0.721001 + vwap*0.278999, industry), 3.3705), 20.4523))
    - ts_rank(decay_linear(ts_rank(corr(ts_rank(low,7.87871), ts_rank(adv60,17.255), 4.97547), 18.5925), 15.7152), 6.71659))"""
    mix = inp.low * 0.721001 + inp.vwap * (1 - 0.721001)
    lhs = rank(decay_linear(delta(_ind(inp, mix, 2), _w(3.3705)), _w(20.4523)))
    rhs = ts_rank(
        decay_linear(
            ts_rank(
                correlation(
                    ts_rank(inp.low, _w(7.87871)),
                    ts_rank(_adv(inp, 60), _w(17.255)),
                    _w(4.97547),
                ),
                _w(18.5925),
            ),
            _w(15.7152),
        ),
        _w(6.71659),
    )
    return -1.0 * (lhs - rhs)


def alpha_098(inp: Alpha101Inputs) -> pd.DataFrame:
    """#98: rank(decay_linear(corr(vwap, sum(adv5,26.4719), 4.58418), 7.18088))
    - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206))"""
    lhs = rank(
        decay_linear(
            correlation(inp.vwap, ts_sum(_adv(inp, 5), _w(26.4719)), _w(4.58418)), _w(7.18088)
        )
    )
    rhs = rank(
        decay_linear(
            ts_rank(
                ts_argmin(
                    correlation(rank(inp.open), rank(_adv(inp, 15)), _w(20.8187)), _w(8.62571)
                ),
                _w(6.95668),
            ),
            _w(8.07206),
        )
    )
    return lhs - rhs


def alpha_099(inp: Alpha101Inputs) -> pd.DataFrame:
    """#99: -1*(rank(corr(sum((high+low)/2, 19.8975), sum(adv60,19.8975), 8.8136))
    < rank(corr(low, volume, 6.28259)))"""
    left = rank(
        correlation(
            ts_sum((inp.high + inp.low) / 2.0, _w(19.8975)),
            ts_sum(_adv(inp, 60), _w(19.8975)),
            _w(8.8136),
        )
    )
    right = rank(correlation(inp.low, inp.volume, _w(6.28259)))
    return -1.0 * _b(left < right)


def alpha_100(inp: Alpha101Inputs) -> pd.DataFrame:
    """#100: -(1.5*scale(indneutralize(indneutralize(rank(((close-low)-(high-close))/(high-low)*volume), subindustry), subindustry))
    - scale(indneutralize(corr(close, rank(adv20), 5) - rank(ts_argmin(close,30)), subindustry))) * volume/adv20"""
    c, h, low = inp.close, inp.high, inp.low
    body = _clean(((c - low) - (h - c)) / (h - low)) * inp.volume
    lhs = 1.5 * scale(_ind(inp, _ind(inp, rank(body), 3), 3))
    rhs = scale(
        _ind(inp, correlation(c, rank(_adv(inp, 20)), 5) - rank(ts_argmin(c, 30)), 3)
    )
    return -1.0 * ((lhs - rhs) * (inp.volume / _adv(inp, 20)))


def alpha_101(inp: Alpha101Inputs) -> pd.DataFrame:
    """#101: (close - open) / ((high - low) + 0.001)"""
    return _clean((inp.close - inp.open) / ((inp.high - inp.low) + 0.001))


#: 全部公式（键为 3 位 id）
FORMULAS: dict[str, object] = {
    f"{i:03d}": fn
    for i, fn in enumerate(
        [
            alpha_001, alpha_002, alpha_003, alpha_004, alpha_005, alpha_006, alpha_007,
            alpha_008, alpha_009, alpha_010, alpha_011, alpha_012, alpha_013, alpha_014,
            alpha_015, alpha_016, alpha_017, alpha_018, alpha_019, alpha_020, alpha_021,
            alpha_022, alpha_023, alpha_024, alpha_025, alpha_026, alpha_027, alpha_028,
            alpha_029, alpha_030, alpha_031, alpha_032, alpha_033, alpha_034, alpha_035,
            alpha_036, alpha_037, alpha_038, alpha_039, alpha_040, alpha_041, alpha_042,
            alpha_043, alpha_044, alpha_045, alpha_046, alpha_047, alpha_048, alpha_049,
            alpha_050, alpha_051, alpha_052, alpha_053, alpha_054, alpha_055, alpha_056,
            alpha_057, alpha_058, alpha_059, alpha_060, alpha_061, alpha_062, alpha_063,
            alpha_064, alpha_065, alpha_066, alpha_067, alpha_068, alpha_069, alpha_070,
            alpha_071, alpha_072, alpha_073, alpha_074, alpha_075, alpha_076, alpha_077,
            alpha_078, alpha_079, alpha_080, alpha_081, alpha_082, alpha_083, alpha_084,
            alpha_085, alpha_086, alpha_087, alpha_088, alpha_089, alpha_090, alpha_091,
            alpha_092, alpha_093, alpha_094, alpha_095, alpha_096, alpha_097, alpha_098,
            alpha_099, alpha_100, alpha_101,
        ],
        start=1,
    )
}
