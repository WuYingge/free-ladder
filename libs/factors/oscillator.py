"""超买超卖族因子。

本模块包含 6 个超买超卖族震荡指标：

- **RSI**：Wilder 平滑的相对强弱指标，14 日默认
- **Stochastic**：Fast %K / %D，KDJ 随机指标
- **CCI**：Commodity Channel Index，商品通道指数
- **WilliamsR**：Williams %R，比 KDJ 更敏感
- **MFI**：Money Flow Index，成交量加权版 RSI
- **UltimateOscillator**：7/14/28 三周期加权综合震荡指标
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base_factor import BaseFactor


class RSI(BaseFactor):
    """相对强弱指标：Wilder 平滑版 RSI。

    公式::

        daily_ret = close.pct_change()
        gain = max(daily_ret, 0)
        loss = abs(min(daily_ret, 0))
        avg_gain = WilderSmooth(gain, N)
        avg_loss = WilderSmooth(loss, N)
        RSI = 100 - 100 / (1 + avg_gain / avg_loss)

    Wilder 平滑：首期用 SMA，之后 avg = (prev × (N-1) + cur) / N。

    参数
    ----------
    window : int
        RSI 窗口，默认 14。
    price_column : str
        价格列名，默认 "close"。
    """

    name = "RSI"
    params = {
        "window": 14,
        "price_column": "close",
    }

    def __init__(self, window: int = 14, price_column: str = "close") -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.price_column = price_column
        self.warmup_period = self.window + 1
        self._set_params(window=window, price_column=price_column)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data[self.price_column].astype(float)
        daily_ret = close.pct_change()

        gain = daily_ret.clip(lower=0.0)
        loss = (-daily_ret).clip(lower=0.0)

        # Wilder 平滑
        avg_gain = self._wilder_smooth(gain)
        avg_loss = self._wilder_smooth(loss)

        # 除零保护
        rs = np.full(len(close), np.nan, dtype=float)
        safe_mask = avg_loss > 1e-12
        rs[safe_mask] = avg_gain[safe_mask] / avg_loss[safe_mask]

        result = 100.0 - 100.0 / (1.0 + rs)
        result[avg_loss < 1e-12] = 100.0  # 无下跌 → RSI=100
        return pd.Series(result, index=data.index, name=self.get_output_name())

    def _wilder_smooth(self, series: pd.Series) -> np.ndarray:
        """Wilder 指数平滑：首期 SMA，后续 EMA"""
        arr = series.values.astype(float)
        result = np.full(len(arr), np.nan, dtype=float)
        n = self.window

        if len(arr) <= n:
            return result

        # 首期 SMA
        result[n] = np.nanmean(arr[1 : n + 1])
        # 后续 Wilder EMA
        for i in range(n + 1, len(arr)):
            result[i] = (result[i - 1] * (n - 1) + arr[i]) / n

        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.price_column not in data.columns:
            raise ValueError(
                f"RSI requires column '{self.price_column}'"
            )


class Stochastic(BaseFactor):
    """随机指标（Fast Stochastic）：%K / %D。

    公式::

        %K = (close - low_N) / (high_N - low_N) × 100
        %D = MA(%K, M)

    横盘保护：当 high_N == low_N 时，%K 返回 50（中性）。

    参数
    ----------
    n : int
        %K 的窗口，默认 14。
    m : int
        %D 的平滑窗口，默认 3。
    output : str
        输出列，"K" 或 "D"，默认 "K"。
    """

    name = "Stochastic"
    params = {
        "n": 14,
        "m": 3,
        "output": "K",
    }

    def __init__(
        self,
        n: int = 14,
        m: int = 3,
        output: str = "K",
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.output = output.upper()
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.m < 1:
            raise ValueError("m must be at least 1")
        if self.output not in ("K", "D"):
            raise ValueError("output must be 'K' or 'D'")
        self.warmup_period = self.n + self.m - 1
        self._set_params(n=n, m=m, output=output)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.n}_{self.m}_{self.output}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        high_n = high.rolling(window=self.n).max()
        low_n = low.rolling(window=self.n).min()

        denom = high_n - low_n
        k = pd.Series(np.nan, index=data.index, dtype=float)
        safe_mask = denom > 1e-12
        k[safe_mask] = (close[safe_mask] - low_n[safe_mask]) / denom[safe_mask] * 100.0
        # 横盘：返回 50（中性）
        flat_mask = denom <= 1e-12
        k[flat_mask] = 50.0

        if self.output == "K":
            result = k
        else:
            result = k.rolling(window=self.m).mean()

        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("close", "high", "low"):
            if col not in data.columns:
                raise ValueError(
                    f"Stochastic requires column {col!r}"
                )


class CCI(BaseFactor):
    """商品通道指数（Commodity Channel Index）。

    公式::

        TP = (high + low + close) / 3
        MA_TP = MA(TP, N)
        mean_abs_dev = MA(|TP - MA_TP|, N)
        CCI = (TP - MA_TP) / (0.015 × mean_abs_dev)

    常数 0.015 来自 Lambert 原始定义，使得约 70-80% 的值落在 ±100 之间。

    参数
    ----------
    window : int
        窗口，默认 20。
    """

    name = "CCI"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        tp = (high + low + close) / 3.0
        ma_tp = tp.rolling(window=self.window).mean()
        mad = tp.rolling(window=self.window).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        denom = 0.015 * mad
        denom_safe = denom.replace(0.0, np.nan)
        result = (tp - ma_tp) / denom_safe
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(
                    f"CCI requires column {col!r}"
                )


class WilliamsR(BaseFactor):
    """Williams %R 指标。

    公式::

        %R = (high_N - close) / (high_N - low_N) × (-100)

    范围 [−100, 0]。
    −20 以上 = 超买，−80 以下 = 超卖。
    比 KDJ 更敏感（取值范围更大）。

    参数
    ----------
    window : int
        窗口，默认 14。
    """

    name = "WilliamsR"
    params = {"window": 14}

    def __init__(self, window: int = 14) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window - 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        high_n = high.rolling(window=self.window).max()
        low_n = low.rolling(window=self.window).min()

        denom = high_n - low_n
        result = pd.Series(np.nan, index=data.index, dtype=float)
        safe_mask = denom > 1e-12
        result[safe_mask] = (high_n[safe_mask] - close[safe_mask]) / denom[safe_mask] * (-100.0)
        # 横盘：返回 -50
        flat_mask = denom <= 1e-12
        result[flat_mask] = -50.0

        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(
                    f"WilliamsR requires column {col!r}"
                )


class MFI(BaseFactor):
    """资金流量指标（Money Flow Index）：成交量加权版 RSI。

    公式::

        TP = (high + low + close) / 3
        MF = TP × volume
        Positive MF = MF[TP > TP.shift(1)]，否则 0
        Negative MF = MF[TP < TP.shift(1)]，否则 0
        MR = rolling_sum(Positive MF, N) / rolling_sum(Negative MF, N)
        MFI = 100 - 100 / (1 + MR)

    参数
    ----------
    window : int
        窗口，默认 14。
    """

    name = "MFI"
    params = {"window": 14}

    def __init__(self, window: int = 14) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        tp = (high + low + close) / 3.0
        mf = tp * volume
        tp_prev = tp.shift(1)

        pos_mf = mf.where(tp > tp_prev, 0.0)
        neg_mf = mf.where(tp < tp_prev, 0.0)

        pos_sum = pos_mf.rolling(window=self.window).sum()
        neg_sum = neg_mf.rolling(window=self.window).sum()

        mr = np.full(len(close), np.nan, dtype=float)
        safe_mask = neg_sum > 1e-12
        mr[safe_mask] = pos_sum[safe_mask] / neg_sum[safe_mask]

        mfi = 100.0 - 100.0 / (1.0 + mr)
        mfi[neg_sum < 1e-12] = 100.0  # 无负资金流 → MFI=100
        result = pd.Series(mfi, index=data.index, name=self.get_output_name())
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("high", "low", "close", "volume"):
            if col not in data.columns:
                raise ValueError(
                    f"MFI requires column {col!r}"
                )


class UltimateOscillator(BaseFactor):
    """终极震荡指标（Ultimate Oscillator）：三周期加权综合。

    公式::

        BP = close - min(low, prev_close)
        TR = max(high, prev_close) - min(low, prev_close)
        avg_N = rolling_sum(BP, N) / rolling_sum(TR, N)
        UO = 100 × (4 × avg_7 + 2 × avg_14 + 1 × avg_28) / 7

    三周期 7/14/28，权重 4/2/1。

    参数
    ----------
    short : int
        短期窗口，默认 7。
    mid : int
        中期窗口，默认 14。
    long : int
        长期窗口，默认 28。
    """

    name = "UltimateOscillator"
    params = {
        "short": 7,
        "mid": 14,
        "long": 28,
    }

    def __init__(
        self,
        short: int = 7,
        mid: int = 14,
        long: int = 28,
    ) -> None:
        super().__init__()
        self.short = int(short)
        self.mid = int(mid)
        self.long = int(long)
        self.warmup_period = self.long + 1
        self._set_params(short=short, mid=mid, long=long)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.short}_{self.mid}_{self.long}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)

        bp = close - np.minimum(low, prev_close)
        tr = np.maximum(high, prev_close) - np.minimum(low, prev_close)
        tr_safe = tr.replace(0.0, np.nan)

        def _avg(n: int) -> pd.Series:
            bp_sum = bp.rolling(window=n).sum()
            tr_sum = tr_safe.rolling(window=n).sum()
            return bp_sum / tr_sum

        avg_short = _avg(self.short)
        avg_mid = _avg(self.mid)
        avg_long = _avg(self.long)

        uo = 100.0 * (4.0 * avg_short + 2.0 * avg_mid + 1.0 * avg_long) / 7.0
        uo.name = self.get_output_name()
        return uo

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(
                    f"UltimateOscillator requires column {col!r}"
                )
