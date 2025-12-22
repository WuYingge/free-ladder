import numpy as np
import pandas as pd
from factors.base_factor import BaseFactor


def calculate_slope(series):
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series))
    y = series.values
    # 线性回归：y = mx + b
    m, b = np.polyfit(x, y, 1)
    return m

def cal_n_day_linear_slope(series, n=20):
    return series.rolling(window=n).apply(calculate_slope, raw=False)

def cal_volatility_n_day(series: pd.Series, n=20) -> float:
    trading_days = 252
    volatility_series = (series/series.shift(1)).apply(np.log).rolling(window=n).std() * np.sqrt(trading_days)
    return volatility_series.tail(1).values[0]

def cal_predict_increase_percent(df, slope_name):
    df[f"predict_interest_{slope_name}"] = df[slope_name] / df["close"] * 100
    

def cal_ma_n(df, n=20, price_col="close"):
    df[f"ma{n}"] = df["close"].rolling(window=n).mean()
    
def cal_max_loss_of_ma_n(df, n=20):
    cur = (df["close"] - df[f"ma{n}"]) / df["close"] * 100
    df[f"max_loss_ma{n}"] = cur.apply(lambda x: x if x > 0 else np.inf)

def cal_ratio_of_interest_risk(df: pd.DataFrame, k=1, n=20) -> float:
    use_data = df.tail(1)
    c = cal_volatility_n_day(df["close"], n) * 0.5
    slope = use_data[f"predict_interest_{n}_slope"].values[0]
    max_loss = use_data[f"max_loss_ma{n}"].values[0]
    return k*slope / (abs(max_loss) + c)

def cal_one_etf(df: pd.DataFrame, k=1, n=20) -> float:
    slope_name = f"{n}_slope"
    df[slope_name] = cal_n_day_linear_slope(df["close"])
    cal_predict_increase_percent(df, slope_name)
    cal_ma_n(df)
    cal_max_loss_of_ma_n(df)
    return cal_ratio_of_interest_risk(df, k)
    

class SlopeRiskRatio(BaseFactor):
    name = "SlopeRiskRatio"
    
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        data["ratio"] = cal_one_etf(data)
        return data["ratio"]
