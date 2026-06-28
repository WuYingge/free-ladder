"""成交量/流动性族因子。

本模块包含 7 个成交量/流动性族因子：

- **VolumeRatio**：量比，volume / volume_MA(N)
- **VolumePriceCorrelation**：量价相关系数，volume 与 close 的滚动 Spearman 相关
- **OBV**：On-Balance Volume，能量潮
- **VPT**：Volume Price Trend，量价趋势
- **AmihudIlliquidity**：Amihud 非流动性，abs(return) / value
- **VolumeStd**：成交量标准差，量能波动
- **VolumeSkew**：成交量偏度
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base_factor import BaseFactor


class VolumeRatio(BaseFactor):
    """量比因子：volume / MA(volume, N)。

    值 > 1 = 放量，值 < 1 = 缩量。
    用于衡量当日成交相对近期均量的活跃程度。

    参数
    ----------
    window : int
        均线窗口，默认 5。
    volume_column : str
        成交量列名，默认 "volume"。
    """

    name = "VolumeRatio"
    params = {
        "window": 5,
        "volume_column": "volume",
    }

    def __init__(self, window: int = 5, volume_column: str = "volume") -> None:
        super().__init__()
        self.window = int(window)
        self.volume_column = volume_column
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window
        self._set_params(window=window, volume_column=volume_column)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        vol = data[self.volume_column].astype(float)
        ma_vol = vol.rolling(window=self.window).mean()
        denom = ma_vol.replace(0.0, np.nan)
        result = vol / denom
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.volume_column not in data.columns:
            raise ValueError(
                f"VolumeRatio requires column '{self.volume_column}'"
            )


class VolumePriceCorrelation(BaseFactor):
    """量价相关系数：volume 与 close 的滚动 Spearman 秩相关。

    正值 → 量价同向（放量上涨/缩量下跌，趋势健康）
    负值 → 量价背离（放量下跌/缩量上涨，需要警惕）

    使用滑动窗口内手动计算 Spearman 秩相关（pandas 3.x
    移除了 rolling.corr 的 method 参数）。

    参数
    ----------
    window : int
        滚动窗口，默认 20。
    """

    name = "VolumePriceCorrelation"
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
        close = data["close"].astype(float).values
        vol = data["volume"].astype(float).values
        n = len(close)
        result_arr = np.full(n, np.nan, dtype=float)
        if n >= self.window:
            from numpy.lib.stride_tricks import sliding_window_view
            cw = sliding_window_view(close, self.window)
            vw = sliding_window_view(vol, self.window)
            valid_mask = np.isfinite(cw).all(axis=1) & np.isfinite(vw).all(axis=1)
            for idx in np.where(valid_mask)[0]:
                c_win = cw[idx]
                v_win = vw[idx]
                rc = pd.Series(c_win).rank().values
                rv = pd.Series(v_win).rank().values
                result_arr[idx + self.window - 1] = np.corrcoef(rc, rv)[0, 1]
        result = pd.Series(result_arr, index=data.index, name=self.get_output_name())
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("close", "volume"):
            if col not in data.columns:
                raise ValueError(
                    f"VolumePriceCorrelation requires column {col!r}"
                )


class OBV(BaseFactor):
    """能量潮（On-Balance Volume）：累积成交量方向。

    公式::

        direction = sign(close - prev_close)
        OBV = cumsum(volume × direction)

    涨日累加成交量，跌日累减成交量。
    无滚动窗口，全历史累积。

    OBV 的绝对值受初始值影响，使用时建议关注趋势而非绝对值。
    """

    name = "OBV"
    params = {}
    warmup_period = 1

    def __init__(self) -> None:
        super().__init__()

    def get_output_name(self) -> str:
        return "OBV"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        vol = data["volume"].astype(float)

        direction = np.sign(close.diff())
        # 第一行 diff 为 NaN，方向置 0
        direction.iloc[0] = 0.0

        result = (vol * direction).cumsum()
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("close", "volume"):
            if col not in data.columns:
                raise ValueError(
                    f"OBV requires column {col!r}"
                )


class VPT(BaseFactor):
    """量价趋势（Volume Price Trend）：价格变化率 × 成交量的累积。

    公式::

        pct = (close - prev_close) / prev_close
        VPT = cumsum(volume × pct)

    度量成交量加权的价格变动趋势。
    """

    name = "VPT"
    params = {}
    warmup_period = 1

    def __init__(self) -> None:
        super().__init__()

    def get_output_name(self) -> str:
        return "VPT"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        vol = data["volume"].astype(float)

        pct = close.pct_change()
        # 第一行 pct 为 NaN，置 0
        pct.iloc[0] = 0.0

        result = (vol * pct).cumsum()
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("close", "volume"):
            if col not in data.columns:
                raise ValueError(
                    f"VPT requires column {col!r}"
                )


class AmihudIlliquidity(BaseFactor):
    """Amihud 非流动性指标：度量价格冲击。

    公式::

        daily = abs(daily_return) / value
        result = MA(daily, N)

    值越大 = 流动性越差（单位成交额造成的价格冲击越大）。

    单位取决于 value 的量纲：东方财富数据 value 单位为元，
    所以该因子数值量级很小（~1e-8 ~ 1e-12）。

    参数
    ----------
    window : int
        均值窗口，默认 20。
    """

    name = "AmihudIlliquidity"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
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
        close = data["close"].astype(float)
        value = data["value"].astype(float)

        daily_ret = close.pct_change().abs()
        safe_value = value.replace(0.0, np.nan)
        daily_illiq = daily_ret / safe_value

        result = daily_illiq.rolling(window=self.window).mean()
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("close", "value"):
            if col not in data.columns:
                raise ValueError(
                    f"AmihudIlliquidity requires column {col!r}"
                )


class VolumeStd(BaseFactor):
    """成交量标准差因子：volume 的 N 日 std，度量量能波动。

    值大 = 成交量波动剧烈（可能伴随变盘），值小 = 成交量稳定。

    参数
    ----------
    window : int
        窗口，默认 20。
    volume_column : str
        成交量列名，默认 "volume"。
    """

    name = "VolumeStd"
    params = {
        "window": 20,
        "volume_column": "volume",
    }

    def __init__(self, window: int = 20, volume_column: str = "volume") -> None:
        super().__init__()
        self.window = int(window)
        self.volume_column = volume_column
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self.warmup_period = self.window
        self._set_params(window=window, volume_column=volume_column)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        vol = data[self.volume_column].astype(float)
        result = vol.rolling(window=self.window).std(ddof=1)
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.volume_column not in data.columns:
            raise ValueError(
                f"VolumeStd requires column '{self.volume_column}'"
            )


class VolumeSkew(BaseFactor):
    """成交量偏度因子：volume 的 N 日偏度。

    正值 → 成交量右偏（偶发性放量）
    负值 → 成交量左偏（偶发性缩量）
    零 → 成交量对称分布

    参数
    ----------
    window : int
        窗口，默认 20。
    volume_column : str
        成交量列名，默认 "volume"。
    """

    name = "VolumeSkew"
    params = {
        "window": 20,
        "volume_column": "volume",
    }

    def __init__(self, window: int = 20, volume_column: str = "volume") -> None:
        super().__init__()
        self.window = int(window)
        self.volume_column = volume_column
        if self.window < 3:
            raise ValueError("window must be at least 3")
        self.warmup_period = self.window
        self._set_params(window=window, volume_column=volume_column)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        vol = data[self.volume_column].astype(float)
        result = vol.rolling(window=self.window).skew()
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.volume_column not in data.columns:
            raise ValueError(
                f"VolumeSkew requires column '{self.volume_column}'"
            )