"""趋势质量族因子。

本模块包含 6 个趋势质量族因子，补充已有的 `TrendR²` / `RollingOLS`：

- **HurstExponent**：Hurst 指数，重标极差分析，H>0.5 有持续性，H<0.5 均值回归
- **KaufmanEfficiencyRatio**：Kaufman 效率比，净位移 / 路径总长度
- **UpDownRatio**：涨跌比，N 日上涨天数占比
- **ConsecutiveUpDays**：连涨天数，当前连续上涨了多少个交易日
- **ConsecutiveDownDays**：连跌天数，当前连续下跌了多少个交易日
- **ADX**：ADX 趋势强度，Welles Wilder 经典指标，衡量趋势有多强（不是方向）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base_factor import BaseFactor


def _hurst_rs(log_prices: np.ndarray) -> float:
    """对单个窗口计算 Hurst RS 值（重标极差）。

    参数
    ----------
    log_prices : np.ndarray
        窗口内的对数价格序列。

    返回
    -------
    float
        Hurst 指数 H = log(R/S) / log(N)，N 为收益个数。
    """
    n = len(log_prices)
    if n < 4:
        return np.nan

    # 对数收益
    rets = np.diff(log_prices)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 4:
        return np.nan

    mean_ret = rets.mean()
    cumdev = np.cumsum(rets - mean_ret)
    R = float(cumdev.max() - cumdev.min())
    S = float(rets.std(ddof=1))
    if S < 1e-12:
        return np.nan

    return float(np.log(R / S) / np.log(len(rets)))


class HurstExponent(BaseFactor):
    """Hurst 指数因子：重标极差分析 (R/S)。

    对每个滚动窗口，计算对数收益的重标极差统计量的 Hurst 指数：

    - H > 0.5：趋势具有持续性（趋势行情）
    - H < 0.5：均值回归特性（震荡行情）
    - H ≈ 0.5：近似随机游走

    计算通过 ``rolling().apply()`` 实现，窗口较大时注意性能。

    参数
    ----------
    window : int
        滚动窗口天数，默认 120。
    """

    name = "HurstExponent"
    params = {"window": 120}

    def __init__(self, window: int = 120) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 4:
            raise ValueError("window must be at least 4")
        # rolling(window=window, min_periods=window).apply()
        # 第一个有效值在 index=window-1
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"HurstExponent_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        log_price = np.log(close.replace(0.0, np.nan))

        # rolling.apply 在原始值上操作，apply 函数接收 numpy 数组
        result = log_price.rolling(
            window=self.window, min_periods=self.window
        ).apply(_hurst_rs, raw=True)

        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 4:
            raise ValueError("window must be at least 4")
        if "close" not in data.columns:
            raise ValueError(
                f"HurstExponent requires column 'close', got {list(data.columns)}"
            )


class KaufmanEfficiencyRatio(BaseFactor):
    """Kaufman 效率比：净位移除以路径总长度。

    公式::

        net_change = |close − close.shift(N)|
        path_length = Σ |daily_change|  (N 日)
        result = net_change / path_length

    取值 0~1：
    - 1：价格沿直线运动（高效趋势），趋势最"真"
    - 0：原地踏步（震荡），所有位移被来回抵消

    参数
    ----------
    window : int
        滚动窗口天数，默认 20。
    """

    name = "KaufmanER"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 2:
            raise ValueError("window must be at least 2")
        # rolling(window=window, min_periods=window).apply()
        # 第一个有效值在 index=window-1
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"KaufmanER_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)

        net_change = (close - close.shift(self.window)).abs()
        daily_change = close.diff().abs()
        path_length = daily_change.rolling(
            window=self.window, min_periods=self.window
        ).sum()

        # path_length 为零时（完全横盘）结果为 NaN
        result = net_change / path_length.replace(0.0, np.nan)
        # 理论上不应超过 1，但浮点误差可能导致略超，clip 到 [0, 1]
        result = result.clip(0.0, 1.0)
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if "close" not in data.columns:
            raise ValueError(
                f"KaufmanEfficiencyRatio requires column 'close', got {list(data.columns)}"
            )


class UpDownRatio(BaseFactor):
    """涨跌比因子：N 日上涨天数占比。

    公式::

        up_count = Σ (close > close.shift(1))  (N 日)
        result = up_count / N

    取值 0~1：
    - 1：连续 N 天上涨
    - 0：连续 N 天下跌
    - ~0.5：涨跌各半（横盘/震荡）

    参数
    ----------
    window : int
        滚动窗口天数，默认 20。
    """

    name = "UpDownRatio"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        # rolling(window=window, min_periods=window).apply()
        # 第一个有效值在 index=window-1
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"UpDownRatio_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)

        up = (close.diff() > 0).astype(float)
        up.iloc[0] = np.nan  # 首日没有前值可比较
        result = up.rolling(
            window=self.window, min_periods=self.window
        ).sum() / self.window
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if "close" not in data.columns:
            raise ValueError(
                f"UpDownRatio requires column 'close', got {list(data.columns)}"
            )


class ConsecutiveUpDays(BaseFactor):
    """连涨天数因子：当前连续上涨了多少个交易日。

    从当天向前数，直到遇到第一个下跌日（含平盘），计数从 0 开始。

    实现上用 ``(close <= close.shift(1)).cumsum()`` 分组，
    组内 ``cumcount()`` 即为连涨天数。

    参数
    ----------
    无
    """

    name = "ConsecutiveUpDays"
    params = {}

    def __init__(self) -> None:
        super().__init__()
        self.warmup_period = 1

    def get_output_name(self) -> str:
        return "ConsecutiveUpDays"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)

        # 遇到下跌或平盘即断连涨
        is_break = close <= close.shift(1)
        group_id = is_break.cumsum()
        result = close.groupby(group_id).cumcount()
        result.name = self.get_output_name()
        return result.astype(int)

    def _validate_input(self, data: pd.DataFrame) -> None:
        if "close" not in data.columns:
            raise ValueError(
                f"ConsecutiveUpDays requires column 'close', got {list(data.columns)}"
            )


class ConsecutiveDownDays(BaseFactor):
    """连跌天数因子：当前连续下跌了多少个交易日。

    从当天向前数，直到遇到第一个上涨日（含平盘），计数从 0 开始。

    实现上用 ``(close >= close.shift(1)).cumsum()`` 分组，
    组内 ``cumcount()`` 即为连跌天数。

    参数
    ----------
    无
    """

    name = "ConsecutiveDownDays"
    params = {}

    def __init__(self) -> None:
        super().__init__()
        self.warmup_period = 1

    def get_output_name(self) -> str:
        return "ConsecutiveDownDays"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)

        # 遇到上涨或平盘即断连跌
        is_break = close >= close.shift(1)
        group_id = is_break.cumsum()
        result = close.groupby(group_id).cumcount()
        result.name = self.get_output_name()
        return result.astype(int)

    def _validate_input(self, data: pd.DataFrame) -> None:
        if "close" not in data.columns:
            raise ValueError(
                f"ConsecutiveDownDays requires column 'close', got {list(data.columns)}"
            )


class ADX(BaseFactor):
    """ADX 趋势强度因子：Welles Wilder 经典指标。

    计算流程:

        1. +DM = max(high − prev_high, 0) 若 high − prev_high > prev_low − low 否则 0
        2. −DM = max(prev_low − low, 0) 若 prev_low − low > high − prev_high 否则 0
        3. TR  = max(high−low, |high−prev_close|, |low−prev_close|)
        4. 对 +DM / −DM / TR 做 Wilder smoothing（EMA, alpha = 1/window）
        5. +DI = 100 × smoothed(+DM) / smoothed(TR)
        6. −DI = 100 × smoothed(−DM) / smoothed(TR)
        7. DX  = |+DI − −DI| / (+DI + −DI) × 100
        8. ADX = Wilder smoothed DX

    ADX 值越高，趋势越强（无论涨跌）。它不指示方向，只指示强度。

    可通过 ``output`` 参数选择输出：
    - ``"adx"``（默认）：ADX 趋势强度
    - ``"plus_di"``：+DI 正向指标
    - ``"minus_di"``：−DI 负向指标
    - ``"dx"``：DX 方向性动量

    参数
    ----------
    window : int
        Wilder smoothing 窗口，默认定值 14。
    output : str
        输出类型，可选 "adx" / "plus_di" / "minus_di" / "dx"。
    """

    name = "ADX"
    params = {
        "window": 14,
        "output": "adx",
    }

    def __init__(self, window: int = 14, output: str = "adx") -> None:
        super().__init__()
        self.window = int(window)
        self.output = output
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if output not in ("adx", "plus_di", "minus_di", "dx"):
            raise ValueError("output must be one of: adx, plus_di, minus_di, dx")
        # Wilder smoothing 收敛:
        # atr/+DI/−DI 第一步 ewm → index=window-1 有值
        # DX → 第二步 ewm 需要 window 个 DX 非 NaN → index=2*window-2 有值
        self.warmup_period = 2 * self.window - 2
        self._set_params(window=window, output=output)

    def get_output_name(self) -> str:
        return f"ADX_{self.window}_{self.output}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # Step 1: +DM / −DM
        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)

        plus_mask = (up_move > down_move) & (up_move > 0)
        minus_mask = (down_move > up_move) & (down_move > 0)

        plus_dm[plus_mask] = up_move[plus_mask]
        minus_dm[minus_mask] = down_move[minus_mask]

        # Step 2: TR
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Step 3: Wilder smoothing (EMA, alpha = 1/window, adjust=False)
        alpha = 1.0 / self.window
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=self.window).mean()
        smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=self.window).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=self.window).mean()

        # Step 4: +DI / −DI
        plus_di = 100.0 * smoothed_plus_dm / atr.replace(0.0, np.nan)
        minus_di = 100.0 * smoothed_minus_dm / atr.replace(0.0, np.nan)

        # Step 5: DX
        di_sum = plus_di + minus_di
        dx = (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan) * 100.0

        # Step 6: ADX = Wilder smoothed DX
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=self.window).mean()

        if self.output == "adx":
            result = adx
        elif self.output == "plus_di":
            result = plus_di
        elif self.output == "minus_di":
            result = minus_di
        elif self.output == "dx":
            result = dx
        else:
            raise ValueError(f"Unknown output: {self.output}")

        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.output not in ("adx", "plus_di", "minus_di", "dx"):
            raise ValueError("output must be one of: adx, plus_di, minus_di, dx")
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(
                    f"ADX requires column {col!r}, got {list(data.columns)}"
                )
