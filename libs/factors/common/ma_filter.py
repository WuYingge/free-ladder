import pandas as pd

def long_ma_filter(
    data: pd.DataFrame, 
    quick_window: int = 60, 
    slow_window: int = 120,
    column_name = "long_ma_filter"
    ):
    """
    Applies a long moving average filter to the input data.
    Modify the data inplace.
    """
    slow_ma = data['close'].rolling(window=slow_window).mean()
    quick_ma = data['close'].rolling(window=quick_window).mean()
    res = quick_ma > slow_ma
    res.name = column_name
    return res
