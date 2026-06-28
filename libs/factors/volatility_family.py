"""波动率族扩展因子。

本模块包含 6 个波动率族因子，补充已有的 `Volatility` 和 `AverageTrueRange`：

- **DownsideVolatility**：下行波动率，只计入负日收益的标准差，区分好坏波动
- **ParkinsonVolatility**：Parkinson 波动率，用日内高低价估计，比收盘价波动率更高效
- **GarmanKlassVolatility**：Garman-Klass 波动率，额外利用开盘信息，比 Parkinson 更准
- **VolOfVol**：波动率之波动率，衡量波动率自身的稳定性
- **MaxDrawdown**：最大回撤，close / running_max(close, N) − 1
- **AvgDrawdown**：平均回撤，N 日内每日回撤的均值
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base_factor import BaseFactor


class DownsideVolatility(BaseFactor):
    """下行波动率：只计入负日收益的标准差。

    将正收益置零后计算标准差，区分"好坏波动"：
    - 上行波动 = 好（上涨过程中的波动）
    - 下行波动 = 坏（下跌过程中的波动）

    该因子对做多策略尤为重要——只关心下跌风险。

    参数
    ----------
    window : int
        滚动窗口天数，默认 20。
    """

    name = "DownsideVolatility"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"DownsideVolatility_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        daily_ret = close.pct_change()
        # 将正收益置零，仅保留下行信息
        downside = daily_ret.clip(upper=0.0)
        result = downside.rolling(
            window=self.window, min_periods=self.window
        ).std(ddof=1)
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if "close" not in data.columns:
            raise ValueError(
                f"DownsideVolatility requires column 'close', got {list(data.columns)}"
            )


class ParkinsonVolatility(BaseFactor):
    """Parkinson 波动率：用日内高低价估计波动率。

    公式::

        daily_estimator = (ln(high/low))²
        result = sqrt( rolling_mean(daily_estimator, N) / (4 × ln2) )

    Parkinson (1980) 证明了高低价波动率估计器比收盘价波动率估计器
    效率高约 5.2 倍——即用更少的数据达到相同的估计精度。

    参数
    ----------
    window : int
        滚动窗口天数，默认 20。
    annualize : bool
        是否年化输出，默认 False。
    """

    name = "ParkinsonVolatility"
    params = {
        "window": 20,
        "annualize": False,
    }

    # 常数 4 × ln(2)
    _DENOM = 4.0 * np.log(2.0)

    def __init__(self, window: int = 20, *, annualize: bool = False) -> None:
        super().__init__()
        self.window = int(window)
        self.annualize = annualize
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self.warmup_period = self.window
        self._set_params(window=window, annualize=annualize)

    def get_output_name(self) -> str:
        suffix = "_ann" if self.annualize else ""
        return f"ParkinsonVolatility_{self.window}{suffix}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        log_hl = np.log(high / low.replace(0.0, np.nan))
        daily_estimator = log_hl ** 2
        rolling_mean = daily_estimator.rolling(
            window=self.window, min_periods=self.window
        ).mean()

        result = np.sqrt(rolling_mean / self._DENOM)
        if self.annualize:
            result = result * np.sqrt(252)
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        for col in ("high", "low"):
            if col not in data.columns:
                raise ValueError(
                    f"ParkinsonVolatility requires column {col!r}, got {list(data.columns)}"
                )


class GarmanKlassVolatility(BaseFactor):
    """Garman-Klass 波动率：扩展 Parkinson，额外利用开盘信息。

    公式::

        daily_var = 0.5 × (ln(high/low))² − (2ln2 − 1) × (ln(close/open))²
        daily_var = max(daily_var, 0)   # 防止数值误差导致的负方差
        result = sqrt( rolling_mean(daily_var, N) )

    Garman & Klass (1980) 证明了该估计器比 Parkinson 更精确，
    效率约为收盘价波动率估计器的 7.4 倍。

    参数
    ----------
    window : int
        滚动窗口天数，默认 20。
    annualize : bool
        是否年化输出，默认 False。
    """

    name = "GarmanKlassVolatility"
    params = {
        "window": 20,
        "annualize": False,
    }

    # 2 × ln(2) − 1
    _COEFF = 2.0 * np.log(2.0) - 1.0

    def __init__(self, window: int = 20, *, annualize: bool = False) -> None:
        super().__init__()
        self.window = int(window)
        self.annualize = annualize
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self.warmup_period = self.window
        self._set_params(window=window, annualize=annualize)

    def get_output_name(self) -> str:
        suffix = "_ann" if self.annualize else ""
        return f"GarmanKlassVolatility_{self.window}{suffix}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        open_ = data["open"].astype(float)
        close = data["close"].astype(float)

        log_hl = np.log(high / low.replace(0.0, np.nan))
        log_co = np.log(close / open_.replace(0.0, np.nan))

        daily_var = 0.5 * log_hl ** 2 - self._COEFF * log_co ** 2
        # 数值误差可能导致极小负值，clip 到 0
        daily_var = daily_var.clip(lower=0.0)

        rolling_var = daily_var.rolling(
            window=self.window, min_periods=self.window
        ).mean()
        result = np.sqrt(rolling_var)
        if self.annualize:
            result = result * np.sqrt(252)
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        for col in ("open", "high", "low", "close"):
            if col not in data.columns:
                raise ValueError(
                    f"GarmanKlassVolatility requires column {col!r}, got {list(data.columns)}"
                )


class VolOfVol(BaseFactor):
    """波动率之波动率：衡量波动率自身的稳定性。

    计算逻辑::

        daily_ret = close.pct_change()
        vol = daily_ret.rolling(vol_window).std()
        result = vol.rolling(std_window).std()

    波动率之波动率高 → 市场从平静到剧烈切换频繁，风险环境不稳定。
    波动率之波动率低 → 波动率本身稳定，无论市场平静还是剧烈都处于稳态。

    参数
    ----------
    vol_window : int
        计算基础波动率的窗口，默认 20。
    std_window : int
        对波动率序列计算标准差的窗口，默认 60。
    """

    name = "VolOfVol"
    params = {
        "vol_window": 20,
        "std_window": 60,
    }

    def __init__(self, vol_window: int = 20, std_window: int = 60) -> None:
        super().__init__()
        self.vol_window = int(vol_window)
        self.std_window = int(std_window)
        if self.vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        if self.std_window < 2:
            raise ValueError("std_window must be at least 2")
        self.warmup_period = self.vol_window + self.std_window
        self._set_params(vol_window=vol_window, std_window=std_window)

    def get_output_name(self) -> str:
        return f"VolOfVol_{self.vol_window}_{self.std_window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)

        daily_ret = close.pct_change()
        vol = daily_ret.rolling(
            window=self.vol_window, min_periods=self.vol_window
        ).std(ddof=1)
        result = vol.rolling(
            window=self.std_window, min_periods=self.std_window
        ).std(ddof=1)
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        if self.std_window < 2:
            raise ValueError("std_window must be at least 2")
        if "close" not in data.columns:
            raise ValueError(
                f"VolOfVol requires column 'close', got {list(data.columns)}"
            )


class MaxDrawdown(BaseFactor):
    """最大回撤因子：当前价格相对过去 N 日最高点的跌幅。

    公式::

        running_max = close.rolling(N).max()
        result = close / running_max − 1

    取值在 (−1, 0] 之间：
    0 = 处于 N 日最高点
    −0.2 = 从最高点跌了 20%

    参数
    ----------
    window : int
        回看窗口天数，默认 60。
    """

    name = "MaxDrawdown"
    params = {"window": 60}

    def __init__(self, window: int = 60) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"MaxDrawdown_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)

        running_max = close.rolling(
            window=self.window, min_periods=self.window
        ).max()
        result = close / running_max.replace(0.0, np.nan) - 1.0
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if "close" not in data.columns:
            raise ValueError(
                f"MaxDrawdown requires column 'close', got {list(data.columns)}"
            )


class AvgDrawdown(BaseFactor):
    """平均回撤因子：N 日内每日回撤的均值。

    公式::

        running_max = close.rolling(N).max()
        daily_drawdown = close / running_max − 1
        result = daily_drawdown.rolling(N).mean()

    相比 MaxDrawdown 更平滑，不易受单个极端点影响。

    参数
    ----------
    window : int
        回看窗口天数，默认 60。
    """

    name = "AvgDrawdown"
    params = {"window": 60}

    def __init__(self, window: int = 60) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"AvgDrawdown_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)

        running_max = close.rolling(
            window=self.window, min_periods=self.window
        ).max()
        daily_drawdown = close / running_max.replace(0.0, np.nan) - 1.0
        result = daily_drawdown.rolling(
            window=self.window, min_periods=self.window
        ).mean()
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if "close" not in data.columns:
            raise ValueError(
                f"AvgDrawdown requires column 'close', got {list(data.columns)}"
            )
