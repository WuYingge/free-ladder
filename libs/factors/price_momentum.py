"""价格动量族因子。

本模块包含 5 个价格动量族因子，用于扩展 `libs/factors/` 的因子库：

- **RiskAdjustedReturn**：N 日收益 / N 日波动率，即滚动 Sharpe 比率
- **IntradayMomentum**：当日 (close − open) / open，纯日内走势
- **OvernightReturn**：当日 open 相对前日 close 的跳空幅度
- **HighPointPosition**：N 日内最高点出现在第几天（归一化 0~1）
- **LowPointPosition**：N 日内最低点出现在第几天（归一化 0~1）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from factors.base_factor import BaseFactor


class RiskAdjustedReturn(BaseFactor):
    """风险调整动量因子：N 日收益除以 N 日波动率（滚动 Sharpe）。

    正数 = 上涨伴随低波动（稳定上行），负数 = 下跌伴随高波动（恐慌下跌）。
    绝对值越大，趋势质量越高。

    计算逻辑::

        N_ret  = close / close.shift(N) - 1
        N_vol  = daily_ret.rolling(N).std() * sqrt(N)
        result = N_ret / N_vol

    Parameters
    ----------
    window:
        滚动窗口天数，默认 20（约 1 个月）。
    """

    name = "RiskAdjustedReturn"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"RiskAdjustedReturn_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        N_ret = close / close.shift(self.window) - 1.0
        daily_ret = close.pct_change()
        daily_vol = daily_ret.rolling(window=self.window, min_periods=self.window).std()
        N_vol = daily_vol * np.sqrt(self.window)
        # 极低波动率（横盘/停牌）时设为 NaN，避免除零产生 inf
        result = np.where(N_vol < 1e-8, np.nan, N_ret / N_vol)
        result = pd.Series(result, index=data.index, name=self.get_output_name())
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if "close" not in data.columns:
            raise ValueError(
                f"RiskAdjustedReturn requires column 'close', got {list(data.columns)}"
            )


class IntradayMomentum(BaseFactor):
    """日内动量因子：当日 (close − open) / open。

    反映纯日内价格变动，排除隔夜跳空影响。

    正值 = 日内上涨，负值 = 日内下跌。

    Examples
    --------
    >>> factor = IntradayMomentum()
    >>> result = factor(data)  # data 需包含 'open' 和 'close' 列
    """

    name = "IntradayMomentum"
    params = {}

    def __init__(self) -> None:
        super().__init__()

    def get_output_name(self) -> str:
        return "IntradayMomentum"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        open_ = data["open"].astype(float)
        close = data["close"].astype(float)
        # open 接近 0 时设为 NaN
        safe_mask = open_.abs() >= 1e-12
        result = pd.Series(np.nan, index=data.index, dtype=float)
        result[safe_mask] = (close[safe_mask] - open_[safe_mask]) / open_[safe_mask]
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("open", "close"):
            if col not in data.columns:
                raise ValueError(
                    f"IntradayMomentum requires column {col!r}, got {list(data.columns)}"
                )


class OvernightReturn(BaseFactor):
    """隔夜收益因子：当日 open 相对前日 close 的跳空幅度。

    ``result = (open - prev_close) / prev_close``

    正值 = 跳空高开（隔夜利好），负值 = 跳空低开（隔夜利空）。

    Examples
    --------
    >>> factor = OvernightReturn()
    >>> result = factor(data)  # data 需包含 'open' 和 'close' 列
    """

    name = "OvernightReturn"
    params = {}
    warmup_period = 1  # 需要前一根 bar 的 close

    def __init__(self) -> None:
        super().__init__()

    def get_output_name(self) -> str:
        return "OvernightReturn"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        open_ = data["open"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)
        safe_mask = prev_close.abs() >= 1e-12
        result = pd.Series(np.nan, index=data.index, dtype=float)
        result[safe_mask] = (open_[safe_mask] - prev_close[safe_mask]) / prev_close[safe_mask]
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        for col in ("open", "close"):
            if col not in data.columns:
                raise ValueError(
                    f"OvernightReturn requires column {col!r}, got {list(data.columns)}"
                )


def _compute_path_position(
    price_arr: np.ndarray,
    window: int,
    index: pd.Index,
    output_name: str,
    use_high: bool = True,
) -> pd.Series:
    """计算路径位置因子的通用实现。

    Parameters
    ----------
    price_arr:
        high 或 low 列的 numpy 数组。
    window:
        滚动窗口大小。
    index:
        输出 Series 的索引。
    output_name:
        输出 Series 的 name 属性。
    use_high:
        True → 找最高点 (argmax)；False → 找最低点 (argmin)。
    """
    n = len(price_arr)
    result_arr = np.full(n, np.nan, dtype=float)

    if n < window:
        return pd.Series(result_arr, index=index, name=output_name)

    windows = sliding_window_view(price_arr, window_shape=window)
    valid = np.isfinite(windows).all(axis=1)

    if not valid.any():
        return pd.Series(result_arr, index=index, name=output_name)

    if use_high:
        positions = np.argmax(windows[valid], axis=1).astype(float)
    else:
        positions = np.argmin(windows[valid], axis=1).astype(float)

    normalized = positions / (window - 1)
    result_arr[window - 1 :][valid] = normalized

    return pd.Series(result_arr, index=index, name=output_name)


class HighPointPosition(BaseFactor):
    """路径动量-高点位置：过去 N 日内最高价出现在第几天（归一化到 0~1）。

    0 = 最高点出现在 N 天前（早期冲高后回落）。
    1 = 最高点出现在最近一天（稳步推升/尾盘发力）。

    Parameters
    ----------
    window:
        滚动窗口天数，默认 20。
    """

    name = "HighPointPosition"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"HighPointPosition_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        high_arr = pd.to_numeric(data["high"], errors="coerce").to_numpy(dtype=float, copy=False)
        return _compute_path_position(
            price_arr=high_arr,
            window=self.window,
            index=data.index,
            output_name=self.get_output_name(),
            use_high=True,
        )

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if "high" not in data.columns:
            raise ValueError(
                f"HighPointPosition requires column 'high', got {list(data.columns)}"
            )


class LowPointPosition(BaseFactor):
    """路径动量-低点位置：过去 N 日内最低价出现在第几天（归一化到 0~1）。

    0 = 最低点出现在 N 天前（早盘探底后回升）。
    1 = 最低点出现在最近一天（持续阴跌/尾盘跳水）。

    Parameters
    ----------
    window:
        滚动窗口天数，默认 20。
    """

    name = "LowPointPosition"
    params = {"window": 20}

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"LowPointPosition_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        low_arr = pd.to_numeric(data["low"], errors="coerce").to_numpy(dtype=float, copy=False)
        return _compute_path_position(
            price_arr=low_arr,
            window=self.window,
            index=data.index,
            output_name=self.get_output_name(),
            use_high=False,
        )

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if "low" not in data.columns:
            raise ValueError(
                f"LowPointPosition requires column 'low', got {list(data.columns)}"
            )


class TimeSeriesMomentum(BaseFactor):
    """时序动量二元因子：N 日收益 > 0 则为 1，否则为 0。

    ``result = (close / close.shift(N) - 1 > 0).astype(int)``

    与 ``PriceReturn``（连续值）不同，此因子直接输出 0/1 二元信号，
    适合用作简单的方向判断或截面分组依据。

    参考：Moskowitz, Ooi, and Pedersen (2012).

    参数
    ----------
    window : int
        回看窗口天数，默认 252（约 12 个月）。
    """

    name = "TimeSeriesMomentum"
    params = {"window": 252}

    def __init__(self, window: int = 252) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"TimeSeriesMomentum_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        ret = close / close.shift(self.window) - 1.0
        # 收益未定义的位置保持 NaN（而非误判为 0）
        result = pd.Series(np.nan, index=data.index, dtype=float)
        valid = ret.notna()
        result[valid] = (ret[valid] > 0).astype(int)
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if "close" not in data.columns:
            raise ValueError(
                f"TimeSeriesMomentum requires column 'close', got {list(data.columns)}"
            )
