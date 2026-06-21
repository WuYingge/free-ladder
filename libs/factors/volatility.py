from __future__ import annotations

import pandas as pd

from factors.base_factor import BaseFactor


class Volatility(BaseFactor):
    """N 日收益率标准差（年化或非年化）。

    用于反波动率加权：波动率越低权重越大，每个标的对组合的风险贡献趋于均衡。
    """

    name = "Volatility"
    params = {
        "window": 20,
        "annualize": False,
    }

    def __init__(self, window: int = 20, *, annualize: bool = False) -> None:
        super().__init__()
        self.window = int(window)
        self.annualize = annualize
        self.warmup_period = self.window + 1
        self._set_params(window=window, annualize=annualize)

    def get_output_name(self) -> str:
        suffix = "_ann" if self.annualize else ""
        return f"Volatility_{self.window}{suffix}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        daily_return = close.pct_change()
        rolling_std = daily_return.rolling(window=self.window, min_periods=self.window).std()
        if self.annualize:
            rolling_std = rolling_std * (252 ** 0.5)
        result = rolling_std
        result.name = self.get_output_name()
        return result
