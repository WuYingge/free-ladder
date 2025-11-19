from typing import Any, List
import pandas as pd
from scipy.stats import linregress
from base_factor import BaseFactor


class RsrsFactor(BaseFactor):
    
    WINDOW: int = 20
    
    def __init__(self) -> None:
        self.name = "RSRS"
    
    def __call__(self, data: pd.DataFrame) -> pd.Series[Any]:
        res = data.rolling(
            window=self.WINDOW
        ).apply(self.fit_k)
        
        
    @staticmethod
    def fit_k(df: pd.DataFrame) -> tuple[float, float]:
        def _fit_k(x, y):
            slope, intercept, r_value, p_value, std_err, intercept_stderr = linregress(x, y)
            result = linregress(x, y)
            return result.slope, result.rvalue
        slope, r_value = _fit_k(df["low"].values.tolist(), df["high"].values.tolist())
        r_square = r_value ** 2
        return slope, r_square
    
    
        
