"""分布形态族因子。

本模块包含 7 个分布形态族因子：

- **ReturnSkew**：收益偏度，N 日收益率的三阶矩，正偏=暴涨多，负偏=暴跌多
- **ReturnKurtosis**：收益峰度，N 日收益率的四阶矩（超额峰度），肥尾程度
- **HistoricalVaR**：历史 VaR，N 日收益率的分位数
- **CVaR**：条件 VaR / Expected Shortfall，超过 VaR 的尾部平均损失
- **MaxFavorableExcursion**：MFE，N 日内从起点到最高点的最大收益
- **MaxAdverseExcursion**：MAE，N 日内从起点到最低点的最大亏损
- **InformationDiscreteness**：ID，收益方向反转天数占比（Frog in the Pan）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base_factor import BaseFactor


class ReturnSkew(BaseFactor):
    """收益偏度因子：N 日收益率的三阶矩。

    ``result = daily_ret.rolling(N).skew()``

    正值 → 右偏（偶发暴涨），负值 → 左偏（偶发暴跌），0 → 对称。

    参数
    ----------
    window : int
        滚动窗口天数，默认 60。
    """

    name = "ReturnSkew"
    params = {"window": 60}

    def __init__(self, window: int = 60) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 3:
            raise ValueError("window must be at least 3")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"ReturnSkew_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        daily_ret = close.pct_change()
        result = daily_ret.rolling(
            window=self.window, min_periods=self.window
        ).skew()
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 3:
            raise ValueError("window must be at least 3")
        if "close" not in data.columns:
            raise ValueError(
                f"ReturnSkew requires column 'close', got {list(data.columns)}"
            )


class ReturnKurtosis(BaseFactor):
    """收益峰度因子：N 日收益率的四阶矩（超额峰度）。

    ``result = daily_ret.rolling(N).kurt()``

    > 0 → 肥尾（极端行情比正态分布频繁）
    ≈ 0 → 接近正态分布
    < 0 → 薄尾（行情比正态分布温和）

    参数
    ----------
    window : int
        滚动窗口天数，默认 60。
    """

    name = "ReturnKurtosis"
    params = {"window": 60}

    def __init__(self, window: int = 60) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 4:
            raise ValueError("window must be at least 4")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"ReturnKurtosis_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        daily_ret = close.pct_change()
        result = daily_ret.rolling(
            window=self.window, min_periods=self.window
        ).kurt()
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 4:
            raise ValueError("window must be at least 4")
        if "close" not in data.columns:
            raise ValueError(
                f"ReturnKurtosis requires column 'close', got {list(data.columns)}"
            )


class HistoricalVaR(BaseFactor):
    """历史 VaR 因子：N 日收益率的历史分位数。

    ``result = daily_ret.rolling(N).quantile(q)``

    输出为分位数对应的收益率值（负值），表示在历史上有 (1-q) 的把握
    单日亏损不超过该绝对值。

    参数
    ----------
    window : int
        滚动窗口天数，默认 252（约 1 年）。
    q : float
        分位数，默认 0.05（5% VaR）。
    """

    name = "HistoricalVaR"
    params = {
        "window": 252,
        "q": 0.05,
    }

    def __init__(self, window: int = 252, q: float = 0.05) -> None:
        super().__init__()
        self.window = int(window)
        self.q = float(q)
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if not (0.0 < self.q < 1.0):
            raise ValueError("q must be in (0, 1)")
        self.warmup_period = self.window + 1
        self._set_params(window=window, q=q)

    def _q_label(self) -> str:
        return f"{self.q:.2f}".replace("0.", "").rstrip("0") or "0"

    def get_output_name(self) -> str:
        return f"HistoricalVaR_{self.window}_q{self._q_label()}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        daily_ret = close.pct_change()
        result = daily_ret.rolling(
            window=self.window, min_periods=self.window
        ).quantile(self.q)
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if not (0.0 < self.q < 1.0):
            raise ValueError("q must be in (0, 1)")
        if "close" not in data.columns:
            raise ValueError(
                f"HistoricalVaR requires column 'close', got {list(data.columns)}"
            )


class CVaR(BaseFactor):
    """条件 VaR / Expected Shortfall：超过 VaR 的尾部平均损失。

    对每个滚动窗口：
        1. 计算窗口内收益率的 q 分位数（VaR）
        2. 取 ≤ VaR 的所有收益率的均值

    输出为负值，绝对值 ≥ 对应 VaR，度量尾部风险的深度。

    参数
    ----------
    window : int
        滚动窗口天数，默认 252。
    q : float
        分位数，默认 0.05。
    """

    name = "CVaR"
    params = {
        "window": 252,
        "q": 0.05,
    }

    def __init__(self, window: int = 252, q: float = 0.05) -> None:
        super().__init__()
        self.window = int(window)
        self.q = float(q)
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if not (0.0 < self.q < 1.0):
            raise ValueError("q must be in (0, 1)")
        self.warmup_period = self.window + 1
        self._set_params(window=window, q=q)

    def _q_label(self) -> str:
        return f"{self.q:.2f}".replace("0.", "").rstrip("0") or "0"

    def get_output_name(self) -> str:
        return f"CVaR_{self.window}_q{self._q_label()}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        daily_ret = close.pct_change()
        q = self.q

        def _cvar(arr: np.ndarray) -> float:
            """计算数组的 CVaR：低于 q 分位数的均值。"""
            var = np.quantile(arr, q)
            tail = arr[arr <= var]
            if len(tail) == 0:
                return np.nan
            return float(np.mean(tail))

        result = daily_ret.rolling(
            window=self.window, min_periods=self.window
        ).apply(_cvar, raw=True)
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if not (0.0 < self.q < 1.0):
            raise ValueError("q must be in (0, 1)")
        if "close" not in data.columns:
            raise ValueError(
                f"CVaR requires column 'close', got {list(data.columns)}"
            )


class MaxFavorableExcursion(BaseFactor):
    """最大有利偏移（MFE）：N 日内从起点到最高点的最大收益。

    ``result = rolling_max(close, N) / close.shift(N) - 1``

    正值 → 过程中曾浮盈多少。值大但最终收益小 → 冲高回落。

    参数
    ----------
    window : int
        回看窗口天数，默认 20。
    """

    name = "MaxFavorableExcursion"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"MFE_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        rolling_max = close.rolling(
            window=self.window, min_periods=self.window
        ).max()
        ref_close = close.shift(self.window)
        safe_mask = ref_close.abs() >= 1e-12
        result = pd.Series(np.nan, index=data.index, dtype=float)
        result[safe_mask] = (
            rolling_max[safe_mask] / ref_close[safe_mask] - 1.0
        )
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if "close" not in data.columns:
            raise ValueError(
                f"MaxFavorableExcursion requires column 'close', got {list(data.columns)}"
            )


class MaxAdverseExcursion(BaseFactor):
    """最大不利偏移（MAE）：N 日内从起点到最低点的最大亏损。

    ``result = rolling_min(close, N) / close.shift(N) - 1``

    负值 → 过程中曾浮亏多少。值越小（越负）→ 最大不利偏移越大。

    参数
    ----------
    window : int
        回看窗口天数，默认 20。
    """

    name = "MaxAdverseExcursion"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"MAE_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        rolling_min = close.rolling(
            window=self.window, min_periods=self.window
        ).min()
        ref_close = close.shift(self.window)
        safe_mask = ref_close.abs() >= 1e-12
        result = pd.Series(np.nan, index=data.index, dtype=float)
        result[safe_mask] = (
            rolling_min[safe_mask] / ref_close[safe_mask] - 1.0
        )
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if "close" not in data.columns:
            raise ValueError(
                f"MaxAdverseExcursion requires column 'close', got {list(data.columns)}"
            )


class InformationDiscreteness(BaseFactor):
    """信息离散度（ID）：N 日内收益方向反转的天数占比。

    计算逻辑::

        daily_ret = close.pct_change()
        sign_changes = sign(daily_ret) != sign(daily_ret.shift(1))
        ID = count(sign_changes in window) / (window - 1)

    低 ID → 行情是"一步步走出来"的（扎实趋势），动量更可靠。
    高 ID → 行情是"一跳跳出来的"（可能反转），Frog in the Pan。

    参考：Da, Gurun, and Warachka (2014).

    参数
    ----------
    window : int
        滚动窗口天数，默认 20。
    """

    name = "InformationDiscreteness"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"ID_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        daily_ret = close.pct_change()
        sign_now = np.sign(daily_ret)
        sign_prev = np.sign(daily_ret.shift(1))
        # 排除涉及 NaN 的相邻对（NaN != anything 恒为 True，会引入噪声）
        valid_mask = sign_now.notna() & sign_prev.notna()
        changes = pd.Series(np.nan, index=data.index, dtype=float)
        changes[valid_mask] = (sign_now[valid_mask] != sign_prev[valid_mask]).astype(float)
        # 滚动窗口覆盖 window 个 changes 值（每值对应一天的符号变化）
        roll_sum = changes.rolling(
            window=self.window, min_periods=self.window
        ).sum()
        result = roll_sum / self.window
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if "close" not in data.columns:
            raise ValueError(
                f"InformationDiscreteness requires column 'close', got {list(data.columns)}"
            )
