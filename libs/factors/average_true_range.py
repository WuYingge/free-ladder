from __future__ import annotations
import pandas as pd
from factors.base_factor import BaseFactor

class AverageTrueRange(BaseFactor):
    """
    计算平均真实波幅（ATR）的类
    """
    
    name = "AverageTrueRange"
    
    def __init__(
        self, 
        window: int = 25 # ATR计算的默认窗口大小，取newHigh短周期同样的值
        ):
        super().__init__()
        self.window = window
    
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算真实波幅（True Range）
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算平均真实波幅（ATR）
        atr = true_range.rolling(window=self.window).mean()
        atr.name = f"ATR_{self.window}"
        return atr
    