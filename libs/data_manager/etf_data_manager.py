import os
import shutil
import datetime
import tqdm
import traceback
import pandas as pd
from typing import Iterator

from config import DataPath
from core.models.etf_daily_data import EtfData
from fetcher.etf import get_etf_certain_date_data, get_etf_last_n_day_data, get_all_etf_code
from utils.interval_utils import retry_with_intervals, intervals


def transer_em_etf_to_model(df: pd.DataFrame) -> pd.DataFrame:
    origin_col = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    target_col = ["date", "open", "close", "high", "low",  "volume", "value", "range", "gain", "change", "turnOver"]
    mapper = dict(zip(origin_col, target_col))
    res = df.rename(columns=mapper)
    res["date"] = pd.to_datetime(res["date"])
    return res.set_index("date", drop=True)

def get_symbol_fp(symbol, name):
    return os.path.join(DataPath.DEFAULT_PATH, f"{name}-{symbol}.csv")

def save_etf_data(symbol, name, df):
    transer_em_etf_to_model(df).to_csv(get_symbol_fp(symbol, name), encoding="utf-8_sig", index=True)
    
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
        df = pd.read_csv(fp, parse_dates=True, index_col=0)
        last_update = df.tail(1).index.date[0] # type: ignore
        now = datetime.datetime.now()
        if now.date() - last_update < datetime.timedelta(days=1):
            return True
        update_df = get_etf_certain_date_data(code, last_update, now)
        update_df = transer_em_etf_to_model(update_df)
        update_df = update_df.combine_first(df,)
        update_df.drop_duplicates().to_csv(fp, index=True, encoding="utf-8-sig")
        print(f"update {code}-{name} in {fp}")
        return True
    except pd.errors.EmptyDataError as err:
        return get_and_save(code, name, 120)
    except Exception as err:
        traceback.print_exc()
        print(f"Can't update {code}-{name} because: {err}")
        return False
    
def etf_data_iter() -> Iterator[EtfData]:
    files = os.listdir(DataPath.DEFAULT_PATH)
    for f in tqdm.tqdm_notebook(files):
        if not f.endswith(".csv"):
            continue
        fp = os.path.join(DataPath.DEFAULT_PATH, f)
        try:
            yield EtfData.from_csv(fp)
        except Exception as err:
            print(f"can't iter {fp} due to {err}")
            continue
    
def save_res_df_to_windows(df, relative_fp):
    df.to_excel(os.path.join(DataPath.DEFAULT_WINDOWS_PATH, relative_fp), index=False)

def update_etf_data():
    all_etf = get_all_etf_code()[["代码", "名称"]]
    for code, name in tqdm.tqdm_notebook(all_etf.values):
        fp = get_symbol_fp(code, name)
        if os.path.exists(fp):
            update(code, name)
        else:
            get_and_save(code, name, 120)
            intervals(0.01)
