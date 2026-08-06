from __future__ import annotations

import pandas as pd
from factors.base_factor import BaseFactor


class AverageTrueRange(BaseFactor):
    """
    计算平均真实波幅（ATR）的类

    归一化两种顺序:
      normalize_by_close=True, normalize_first=False → mean(TR) / close     (ATR_{w}_pct)
      normalize_by_close=True, normalize_first=True  → mean(TR_t / close_t) (ATR_{w}_tr_pct)

    mean(TR)/close 用"当前"close 去除窗口平均 TR，趋势中会带价格路径偏差；
    mean(TR/close) 逐日归一化后再滚动，是无路径偏差的纯波动率代理（标准 ATR%）。
    """

    name = "AverageTrueRange"
    params = {
        "window": 25,  # ATR计算的默认窗口大小，取newHigh短周期同样的值
        "normalize_by_close": False,
        "normalize_first": False,
    }

    def __init__(
        self,
        window: int = 25,  # ATR计算的默认窗口大小，取newHigh短周期同样的值
        normalize_by_close: bool = False,  # 是否用 close 归一化（消除价格水平影响）
        normalize_first: bool = False,  # 归一化顺序: True=先逐日 TR/close 再滚动; False=先滚动再除以 close
    ):
        super().__init__()
        self.window = window
        self.normalize_by_close = normalize_by_close
        self.normalize_first = normalize_first
        self.warmup_period = int(window)
        self._set_params(
            window=window,
            normalize_by_close=normalize_by_close,
            normalize_first=normalize_first,
        )

    def get_output_name(self) -> str:
        if self.normalize_by_close and self.normalize_first:
            return f"ATR_{self.window}_tr_pct"
        if self.normalize_by_close:
            return f"ATR_{self.window}_pct"
        return f"ATR_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']

        # 计算真实波幅（True Range）
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        if self.normalize_by_close and self.normalize_first:
            # 先逐日归一化再滚动: mean(TR_t / close_t)
            atr = (true_range / close).rolling(window=self.window).mean()
        else:
            # 先滚动: mean(TR)；normalize_by_close=True 时再除以 close
            atr = true_range.rolling(window=self.window).mean()
            if self.normalize_by_close:
                atr = atr / close
        atr.name = self.get_output_name()
        return atr
