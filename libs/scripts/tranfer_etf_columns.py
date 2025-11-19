import os
import sys
import pandas as pd
from data_manager.etf_data_manager import etf_data_iter
from config import DataPath

def transer_etf_origin_to_etf_model(df: pd.DataFrame):
    origin_col = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    target_col = ["date", "open", "close", "high", "low",  "volume", "value", "range", "gain", "change", "turnOver"]
    mapper = dict(zip(origin_col, target_col))
    df.rename(columns=mapper, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", drop=True, inplace=True)

if __name__ == "__main__":
    for symbol, name, df in etf_data_iter():
        transer_etf_origin_to_etf_model(df)
        df.to_csv(os.path.join(DataPath.BAK_PATH, "etf_data", f"{name}-{symbol}.csv"), encoding="utf-8-sig", index=True)
