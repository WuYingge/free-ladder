import os
import shutil
import datetime
import tqdm
import traceback
import pandas as pd
from typing import Iterator
from multiprocessing import Pool

from config import DataPath
from core.models.etf_daily_data import EtfData
from fetcher.etf import get_etf_certain_date_data, get_etf_last_n_day_data, get_all_etf_code
from utils.interval_utils import retry_with_intervals, intervals
from fetcher.utils import generate_time_slices_alternative
from data_manager.providers.etf_list_provider import ETF_LIST


def transer_em_etf_to_model(df: pd.DataFrame) -> pd.DataFrame:
    origin_col = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    target_col = ["date", "open", "close", "high", "low",  "volume", "value", "range", "gain", "change", "turnOver"]
    if len(df) == 0:
        res = pd.DataFrame(columns=target_col)
        res["date"] = pd.to_datetime(res["date"])
        return res.set_index("date", drop=True)
    mapper = dict(zip(origin_col, target_col))
    res = df.rename(columns=mapper)
    res["date"] = pd.to_datetime(res["date"])
    return res.set_index("date", drop=True)

def get_symbol_fp(symbol):
    return os.path.join(DataPath.DEFAULT_PATH, f"{symbol}.csv")

def save_etf_data(symbol, df):
    transer_em_etf_to_model(df).to_csv(get_symbol_fp(symbol), encoding="utf-8_sig", index=True)
    
def sync_backup(src, dest):
    files = shutil.copytree(src, dest, dirs_exist_ok=True)

@retry_with_intervals(interval_func=intervals)
def get_and_save(symbol, n=40):
    try: 
        cur_df = get_etf_last_n_day_data(symbol, n)
        save_etf_data(symbol, cur_df)
        return True
    except Exception as err:
        print(f"Can't get and save because {err}")
        return False
      
@retry_with_intervals(max_retries=1000, interval_func=intervals)
def update(code) -> bool:
    try:
        # get the last date
        fp = get_symbol_fp(code)
        df = pd.read_csv(fp, parse_dates=True, index_col=0)
        if len(df) == 0: # some of the etf has no data because it is newly listed
            raise pd.errors.EmptyDataError("Empty data")
        last_update = df.tail(1).index.date[0] # type: ignore
        now = datetime.datetime.now()
        if now.date() - last_update < datetime.timedelta(days=1):
            print(f"already updated {code}")
            return True
        update_df = get_etf_certain_date_data(code, last_update, now)
        update_df = transer_em_etf_to_model(update_df)
        update_df = update_df.combine_first(df,)
        update_df.drop_duplicates().to_csv(fp, index=True, encoding="utf-8-sig")
        print(f"update {code} in {fp}")
        return True
    except pd.errors.EmptyDataError as err:
        df = get_with_retry(code, 4000)
        if df is not None:
            save_etf_data(code, df)
            return True
        return False
    except Exception as err:
        traceback.print_exc()
        print(f"Can't update {code} because: {err}")
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
        
def get_etf_data_by_symbol(symbol: str) -> EtfData:
    fp = get_symbol_fp(symbol)
    return EtfData.from_csv(fp)

def get_etf_data_by_symbols(symbols: list[str]) -> list[EtfData]:
    res = []
    for symbol in symbols:
        res.append(get_etf_data_by_symbol(symbol))
    return res
    
def save_res_df_to_windows(df, relative_fp):
    df.to_excel(os.path.join(DataPath.DEFAULT_WINDOWS_PATH, relative_fp), index=False)

def update_etf_data():
    all_etf = ETF_LIST.get_all_symbol()
    with Pool(15) as p:
        res = p.map(update_single_etf_data, all_etf)
    for code, result in res:
        if result:
            print(f"Successfully updated data for {code}")
        else:
            print(f"Failed to update data for {code}")
    
            
def update_single_etf_data(code: str) -> tuple[str, bool]:
    fp = get_symbol_fp(code)
    if os.path.exists(fp):
        update(code)
        return code, True
    else:
        df = get_with_retry(code, 4000)
        if df is not None:
            save_etf_data(code, df)
            return code, True
    return code, False


def batch_acquire_etf_data(codes: list[str], last_n_days: int, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    with Pool(15) as p:
        results = p.starmap(save_etf_data_to_path, [(code, last_n_days, save_path, True) for code in codes])
    with open(os.path.join(save_path, "acquire_results.txt"), "w") as f:
        for code, res in zip(codes, results):
            f.write(f"{code}: {res}\n")
        
def save_etf_data_to_path(code: str, last_n_days: int, save_path: str, check_exists: bool = True) -> str:
    try:
        if check_exists and os.path.exists(os.path.join(save_path, f"{code}.csv")):
            return "Already exists"
        df = get_with_retry(code, last_n_days)
        if df is None:
            return f"Can't acquire data for {code} due to empty data"
        df = transer_em_etf_to_model(df)
        df.to_csv(os.path.join(save_path, f"{code}.csv"), encoding="utf-8-sig", index=True)
        return "OK"
    except Exception as err:
        return f"Can't acquire data for {code} due to {err}"
    
        
def get_with_retry(code, last_n_days: int) -> pd.DataFrame | None:
    print("Acquiring data for", code, "for last", last_n_days, "days")
    count = 1000
    dfs = []
    for s, e in generate_time_slices_alternative(last_n_days):
        df = None
        while count:
            try:
                df = get_etf_certain_date_data(code, datetime.datetime.strptime(s, "%Y%m%d"), datetime.datetime.strptime(e, "%Y%m%d"))
                dfs.append(df)
                break
            except Exception as err:
                count -= 1
                if count < 990:
                    print(f"Retrying to get data for {code} from {s} to {e} due to {err}, {count} retries left")
    if not dfs:
        return None
    return pd.concat(dfs).drop_duplicates()
