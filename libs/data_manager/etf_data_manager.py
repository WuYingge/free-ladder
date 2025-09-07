import os
import shutil
import datetime
import pandas as pd

from config import DataPath
from fetcher.etf import get_etf_certain_date_data, get_etf_last_n_day_data
from utils.interval_utils import retry_with_intervals, intervals

def get_symbol_fp(symbol, name):
    return os.path.join(DataPath.DEFAULT_PATH, f"{name}-{symbol}.csv")

def save_etf_data(symbol, name, df):
    df.to_csv(get_symbol_fp(symbol, name), encoding="utf-8_sig", index=False)
    
def get_symbol_name_from_fp(fp):
    pre = os.path.splitext(os.path.basename(fp))[0].split("-")
    name, symbol = pre
    return symbol, name

def sync_backup(src, dest):
    files = shutil.copytree(src, dest, dirs_exist_ok=True)

@retry_with_intervals(interval_func=intervals)
def get_and_save(symbol, name, n=40):
    try: 
        cur_df = get_etf_last_n_day_data(symbol, n)
        save_etf_data(symbol, name, cur_df)
        return True
    except Exception as err:
        print(f"Can't get and save because {err}")
        return False
      
@retry_with_intervals(interval_func=intervals)
def update(code, name):
    try:
        # get the last date
        fp = get_symbol_fp(code, name)
        df = pd.read_csv(fp)
        last_update = pd.to_datetime(df.tail(1)["日期"].values).to_pydatetime()
        now = datetime.datetime.now()
        if now - last_update < datetime.timedelta(days=1):
            return True
        update_df = get_etf_certain_date_data(code, last_update, now)
        pd.concat([df, update_df]).drop_duplicates(subset=["日期"], ignore_index=True).to_csv(fp, index=False, encoding="utf-8-sig")
        return True
    except Exception as err:
        print(f"Can't update {code}-{name} because: {err}")
        return False
    