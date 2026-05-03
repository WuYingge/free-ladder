from __future__ import annotations

import pandas as pd

from factors.base_factor import BaseFactor


class PriceReturn(BaseFactor):
    """N 日价格动量因子：(close - close.shift(N)) / close.shift(N)。

    常用窗口：
    - 20  ≈ 1 个月动量
    - 60  ≈ 3 个月动量
    - 120 ≈ 6 个月动量
    - 240 ≈ 12 个月动量（跳过最近 1 个月可通过 skip_recent 参数实现）
    """

    name = "PriceReturn"
    params = {
        "window": 60,
        "skip_recent": 0,  # 跳过最近 N 根 bar（如经典 12-1 动量跳过最近 1 个月）
    }

    def __init__(self, window: int = 60, skip_recent: int = 0) -> None:
        super().__init__()
        self.window = int(window)
        self.skip_recent = int(skip_recent)
        self.warmup_period = self.window + self.skip_recent + 1
        self._set_params(window=window, skip_recent=skip_recent)

    def get_output_name(self) -> str:
        if self.skip_recent == 0:
            return f"PriceReturn_{self.window}"
        return f"PriceReturn_{self.window}_skip{self.skip_recent}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        if self.skip_recent > 0:
            past_close = close.shift(self.skip_recent + self.window)
            ref_close = close.shift(self.skip_recent)
        else:
            past_close = close.shift(self.window)
            ref_close = close

        result = (ref_close - past_close) / past_close
        result.name = self.get_output_name()
        return result
