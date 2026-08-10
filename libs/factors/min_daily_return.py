"""日收益率族因子。

本模块包含 1 个因子：

- **MinDailyReturn**：近 N 个交易日的最小单日收益率（最大单日跌幅），风控过滤用
"""

from __future__ import annotations

import pandas as pd

from factors.base_factor import BaseFactor


class MinDailyReturn(BaseFactor):
    """近 N 个交易日的最小单日收益率。

    值为 ``min(close.pct_change(), N)`` —— 对每个交易日取其日收益率（当日相对
    前一交易日），再取最近 N 日的滚动最小值。负数表示窗口内存在下跌日，越小
    代表单日跌幅越大；正数表示窗口内每天都上涨。

    典型用法（短期风控过滤）::

        ThresholdFilter(field="MinDailyReturn_3", operator=">", value=-0.03)

    即"近 3 日单日跌幅不得跌破 3%"。

    参数
    ----------
    window : int
        回看窗口天数（交易日），默认 3。
    price_column : str
        价格列名，默认 "close"。
    """

    name = "MinDailyReturn"
    params = {
        "window": 3,
        "price_column": "close",
    }

    def __init__(self, window: int = 3, price_column: str = "close") -> None:
        super().__init__()
        self.window = int(window)
        self.price_column = price_column
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.price_column not in ("close", "open", "high", "low"):
            raise ValueError(
                f"price_column must be one of close/open/high/low, got {price_column!r}"
            )
        # pct_change 需要 1 根 bar，rolling min 需要 window 根 bar
        self.warmup_period = self.window + 1
        self._set_params(window=window, price_column=price_column)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        if self.price_column not in data.columns:
            raise ValueError(
                f"MinDailyReturn requires column '{self.price_column}'"
            )
        price = data[self.price_column].astype(float)
        daily_ret = price.pct_change()
        result = daily_ret.rolling(window=self.window).min()
        result.name = self.get_output_name()
        return result
