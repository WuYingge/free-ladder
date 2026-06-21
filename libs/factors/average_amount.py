"""平均成交额因子。

计算过去 N 日的日均成交额（东方财富数据中的 ``value`` 列），
用于流动性过滤。
"""

from __future__ import annotations

import pandas as pd

from factors.base_factor import BaseFactor


class AverageAmount(BaseFactor):
    """过去 N 日平均成交额。

    典型用法::

        avg_amount = AverageAmount(window=20)
        # 用作 ThresholdFilter:
        # ThresholdFilter(field=avg_amount.get_output_name(), operator=">=", value=5_000_000)

    Parameters
    ----------
    window:
        计算均值所用的滚动窗口天数。
    """

    name = "AverageAmount"
    params = {
        "window": 20,
    }

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = int(window)
        self.warmup_period = self.window
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        if "value" not in data.columns:
            raise ValueError("AverageAmount 需要 'value' 列（成交额），请检查数据源")
        amount = data["value"].astype(float)
        result = amount.rolling(window=self.window).mean()
        result.name = self.get_output_name()
        return result
