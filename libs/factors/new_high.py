from typing import Any

from pandas import DataFrame
from pandas.core.api import Series as Series
from factors.base_factor import BaseFactor
from factors.common.ma_filter import long_ma_filter
from factors.average_true_range import AverageTrueRange

"""
1. 创最近50天新高，且50天均线>100天均线的
"""
class NewHigh(BaseFactor):
    
    first_buy = 2 # todo don't know why it is needed leave there for further investigation
    buy = 1
    hold = 0
    sell = -1
    
    name = "NewHigh"
    params = {
        "high_window": 50,
        "low_window": 25
    }
    
    
    def __init__(self, high_window=50, low_window=25) -> None:
        super().__init__()
        self.high_window = high_window
        self.low_window = low_window
        self.params = {
            "high_window": high_window,
            "low_window": low_window
        }
        self.add_dependency(AverageTrueRange(window=50))
        
    def __call__(self, data: DataFrame) -> Series:
        long_filter = long_ma_filter(data)
        new_high = data["close"].rolling(window=self.high_window).max()
        low_recent = data["close"].rolling(window=self.low_window).min()
        temp_df = DataFrame({
            "close": data["close"],
            "new_high": new_high,
            "low_recent": low_recent,
            "long_filter": long_filter
        })
        def judge(row):
            if row["close"] >= row["new_high"] and row["long_filter"]:
                return self.buy
            elif row["close"] <= row["low_recent"]:
                return self.sell
            else:
                return self.hold
        
        res = temp_df.apply(judge, axis=1)
        res.name = self.name
        return res
        