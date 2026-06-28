"""结构性/突破族扩展因子。

本模块包含 5 个结构性/突破族因子：

- **NewHighContinuous**：N 日新高（连续值），(close − high_N) / high_N
- **NewLowContinuous**：N 日新低（连续值），(close − low_N) / low_N
- **DonchianChannelPosition**：唐奇安通道位置，(close − low_N) / (high_N − low_N)
- **ATRRatio**：ATR 比例，ATR / close，波动率的环境自适应锚点
- **ChandelierExit**：Chandelier Exit，close 偏离 N 日最高的程度，以 ATR 为单位
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base_factor import BaseFactor


class NewHighContinuous(BaseFactor):
    """N 日新高因子（连续值）：收盘价相对过去 N 日最高点的距离。

    ``result = (close - high_N) / high_N``

    正值 = 创新高（突破幅度），0 = 持平，负值 = 在前高下方。
    与 ``NewHigh``（离散信号）不同，此因子返回连续值，可区分微破和大破。

    参数
    ----------
    window : int
        回看窗口天数，默认 50。
    """

    name = "NewHighContinuous"
    params = {"window": 50}

    def __init__(self, window: int = 50) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"NewHighContinuous_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        close = data["close"].astype(float)
        high_n = high.rolling(window=self.window, min_periods=self.window).max()
        denom = high_n.replace(0.0, np.nan)
        result = (close - high_n) / denom
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        for col in ("close", "high"):
            if col not in data.columns:
                raise ValueError(
                    f"NewHighContinuous requires column {col!r}, got {list(data.columns)}"
                )


class NewLowContinuous(BaseFactor):
    """N 日新低因子（连续值）：收盘价相对过去 N 日最低点的距离。

    ``result = (close - low_N) / low_N``

    正值 = 在最低点上方，0 = 持平新低，负值 = 创新低（跌破幅度）。

    参数
    ----------
    window : int
        回看窗口天数，默认 50。
    """

    name = "NewLowContinuous"
    params = {"window": 50}

    def __init__(self, window: int = 50) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"NewLowContinuous_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        low_n = low.rolling(window=self.window, min_periods=self.window).min()
        denom = low_n.replace(0.0, np.nan)
        result = (close - low_n) / denom
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        for col in ("close", "low"):
            if col not in data.columns:
                raise ValueError(
                    f"NewLowContinuous requires column {col!r}, got {list(data.columns)}"
                )


class DonchianChannelPosition(BaseFactor):
    """唐奇安通道位置：收盘价在 N 日高低价箱体中的相对位置。

    ``result = (close - low_N) / (high_N - low_N)``

    取值 0~1：
    - 1 = 触碰 N 日最高（突破上轨）
    - 0 = 触碰 N 日最低（跌破下轨）
    - 0.5 = 箱体中间

    横盘保护：当 high_N == low_N 时返回 0.5（中性）。

    参数
    ----------
    window : int
        回看窗口天数，默认 20。
    """

    name = "DonchianChannelPosition"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"DonchianPosition_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        high_n = high.rolling(window=self.window, min_periods=self.window).max()
        low_n = low.rolling(window=self.window, min_periods=self.window).min()

        denom = high_n - low_n
        result = pd.Series(np.nan, index=data.index, dtype=float)
        safe_mask = denom > 1e-12
        result[safe_mask] = (
            (close[safe_mask] - low_n[safe_mask]) / denom[safe_mask]
        )
        # 横盘：返回 0.5（中性）
        flat_mask = denom <= 1e-12
        result[flat_mask] = 0.5
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        for col in ("close", "high", "low"):
            if col not in data.columns:
                raise ValueError(
                    f"DonchianChannelPosition requires column {col!r}, got {list(data.columns)}"
                )


class ATRRatio(BaseFactor):
    """ATR 比例因子：ATR / close，将绝对波幅转换成价格百分比。

    ``result = ATR / close``

    用于跨标的波动率比较：无论股价是 10 元还是 500 元，
    ATRRatio 给出可横向对比的相对波动率。

    参数
    ----------
    window : int
        ATR 计算的均线窗口，默认 25。
    """

    name = "ATRRatio"
    params = {"window": 25}

    def __init__(self, window: int = 25) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"ATRRatio_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        # True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=self.window, min_periods=self.window).mean()
        safe_close = close.replace(0.0, np.nan)
        result = atr / safe_close
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        for col in ("close", "high", "low"):
            if col not in data.columns:
                raise ValueError(
                    f"ATRRatio requires column {col!r}, got {list(data.columns)}"
                )


class ChandelierExit(BaseFactor):
    """吊灯止损（Chandelier Exit）：从 N 日最高点以 ATR 为单位度量的偏离。

    ``result = (close - high_N) / ATR``

    负值 → 低于最高点，值越小（越负）→ 偏离越多。
    例如 -3 表示从 N 日最高点跌了 3 倍 ATR，可用于移动止损。

    参数
    ----------
    n : int
        计算最高点的窗口天数，默认 22。
    atr_window : int
        ATR 计算的窗口天数，默认 22。
    """

    name = "ChandelierExit"
    params = {
        "n": 22,
        "atr_window": 22,
    }

    def __init__(self, n: int = 22, atr_window: int = 22) -> None:
        super().__init__()
        self.n = int(n)
        self.atr_window = int(atr_window)
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.atr_window < 1:
            raise ValueError("atr_window must be at least 1")
        self.warmup_period = max(self.n, self.atr_window) + 1
        self._set_params(n=n, atr_window=atr_window)

    def get_output_name(self) -> str:
        return f"ChandelierExit_{self.n}_{self.atr_window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        # N 日最高
        high_n = high.rolling(window=self.n, min_periods=self.n).max()

        # ATR
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_window, min_periods=self.atr_window).mean()

        # 除零保护
        safe_atr = atr.replace(0.0, np.nan)
        result = (close - high_n) / safe_atr
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.atr_window < 1:
            raise ValueError("atr_window must be at least 1")
        for col in ("close", "high", "low"):
            if col not in data.columns:
                raise ValueError(
                    f"ChandelierExit requires column {col!r}, got {list(data.columns)}"
                )
