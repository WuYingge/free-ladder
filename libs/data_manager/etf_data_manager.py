import os
import shutil
import datetime
import tqdm
import traceback
import pandas as pd
from functools import lru_cache
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


def get_etf_last_local_date(symbol: str) -> datetime.date | None:
    """Return the last available local data date for a symbol, or None if unavailable."""
    fp = get_symbol_fp(symbol)
    if not os.path.exists(fp):
        return None

    try:
        df = pd.read_csv(fp, parse_dates=True, index_col=0)
    except Exception:
        return None

    if df.empty:
        return None

    try:
        return pd.to_datetime(df.index.max()).date()
    except Exception:
        return None


def is_etf_data_updated_to_date(
    symbol: str,
    target_date: str | datetime.date | datetime.datetime | pd.Timestamp,
) -> bool:
    """Check if local ETF data has been updated to target_date (inclusive)."""
    last_date = get_etf_last_local_date(symbol)
    if last_date is None:
        return False

    try:
        expected_date = pd.to_datetime(target_date).date()
    except Exception:
        return False

    return last_date >= expected_date


def batch_check_etf_data_updated(
    symbols: list[str],
    target_date: str | datetime.date | datetime.datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return per-symbol local update status for a target date (default: today)."""
    if target_date is None:
        expected_date = datetime.date.today()
    else:
        expected_date = pd.to_datetime(target_date).date()

    rows = []
    for symbol in symbols:
        fp = get_symbol_fp(symbol)
        last_local_date = get_etf_last_local_date(symbol)
        is_updated = bool(last_local_date and last_local_date >= expected_date)
        rows.append(
            {
                "symbol": str(symbol).zfill(6),
                "exists": os.path.exists(fp),
                "last_local_date": last_local_date,
                "target_date": expected_date,
                "is_updated": is_updated,
            }
        )

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(["is_updated", "symbol"], ascending=[True, True]).reset_index(drop=True)
    return res

def save_etf_data(symbol, df):
    transer_em_etf_to_model(df).to_csv(get_symbol_fp(symbol), encoding="utf-8_sig", index=True)
    
def sync_backup(src, dest):
    files = shutil.copytree(src, dest, dirs_exist_ok=True)


@lru_cache(maxsize=1)
def _get_listed_etf_symbols() -> set[str]:
    """Cache currently listed ETF symbols to quickly skip not-yet-listed codes."""
    try:
        df = get_all_etf_code()
    except Exception as err:
        # Fail open: if listing check fails, keep normal fetching path.
        print(f"Failed to load listed ETF symbols due to {err}, fallback to normal retry")
        return set()

    if df is None or df.empty:
        return set()

    symbol_col = "基金代码" if "基金代码" in df.columns else ("代码" if "代码" in df.columns else None)
    if symbol_col is None:
        return set()
    return set(df[symbol_col].astype(str).str.zfill(6).tolist())


def _is_not_listed_yet(code: str) -> bool:
    listed_symbols = _get_listed_etf_symbols()
    # If listed_symbols is empty, we don't have reliable listing info.
    if not listed_symbols:
        return False
    return str(code).zfill(6) not in listed_symbols

@retry_with_intervals(interval_func=intervals)
def get_and_save(symbol, n=40):
    try: 
        cur_df = get_etf_last_n_day_data(symbol, n)
        save_etf_data(symbol, cur_df)
        return True
    except Exception as err:
        print(f"Can't get and save because {err}")
        return False
      
@retry_with_intervals(max_retries=1, interval_func=intervals)
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
        if _is_not_listed_yet(code):
            print(f"Skip update {code}: symbol not listed yet")
            return True

        df = get_with_retry(code, 4000)
        if df is not None:
            save_etf_data(code, df)
            return True
        print(f"Failed update {code}: fetch failed, will retry")
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
        return code, update(code)
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
    if _is_not_listed_yet(code):
        print(f"Skip {code}: symbol not listed yet, stop retry")
        return None

    max_retries_per_slice = 5
    dfs = []
    for s, e in generate_time_slices_alternative(last_n_days):
        success = False
        for retry_idx in range(max_retries_per_slice):
            try:
                df = get_etf_certain_date_data(code, datetime.datetime.strptime(s, "%Y%m%d"), datetime.datetime.strptime(e, "%Y%m%d"))
                dfs.append(df)
                success = True
                break
            except Exception as err:
                retries_left = max_retries_per_slice - retry_idx - 1
                print(f"Retrying {code} {s}-{e} due to {err}, {retries_left} retries left")
                if retries_left > 0:
                    intervals(0.3)
        if not success:
            print(f"Failed to get {code} for slice {s}-{e}, skip this symbol")
            return None

    if not dfs:
        return None
    return pd.concat(dfs).drop_duplicates()
