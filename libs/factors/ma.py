import numpy as np


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

def cal_predict_increase_percent(df, slope_name):
    df[f"predict_interest_{slope_name}"] = df[slope_name] / df["收盘"] * 100
    

def cal_ma_n(df, n=20, price_col="收盘"):
    df[f"ma{n}"] = df["收盘"].rolling(window=n).mean()
    
def cal_max_loss_of_ma_n(df, n=20):
    df[f"max_loss_ma{n}"] = (df["收盘"] - df[f"ma{n}"]) / df["收盘"] * 100

def cal_ratio_of_interest_risk(df, n=20):
    use_data = df.tail(1)
    return use_data[f"predict_interest_{n}_slope"] / use_data[f"max_loss_ma{n}"]

def cal_one_etf(df, n=20):
    slope_name = f"{n}_slope"
    df[slope_name] = cal_n_day_linear_slope(df["收盘"])
    cal_predict_increase_percent(df, slope_name)
    cal_ma_n(df)
    cal_max_loss_of_ma_n(df)
    return cal_ratio_of_interest_risk(df).values[0]
    

