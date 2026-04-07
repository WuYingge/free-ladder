from typing import Any
import pandas as pd
from factors.base_factor import BaseFactor
from factors.utils import sign

class DailyRebound(BaseFactor):
    
    name = "DailyRebound"
    params = {}

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        # std = data["close"].rolling(window=20).std()
        return (data["close"] - data["low"]) / (data["high"] - data["low"])
    
