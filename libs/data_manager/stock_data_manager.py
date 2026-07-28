from __future__ import annotations

import datetime
import os
import traceback
from multiprocessing import Pool, current_process
from typing import Iterator

import pandas as pd
import tqdm

from config import DataPath
from core.models.stock_daily_data import StockDailyData
from data_manager.providers.stock_list_provider import STOCK_LIST
from fetcher.stock import get_stock_certain_date_data
from fetcher.utils import generate_time_slices_alternative
from proxy.proxy import initialize_proxy_pool
from utils.interval_utils import intervals


STOCK_UPDATE_POOL_SIZE = 15
STOCK_HISTORY_START = datetime.datetime(1990, 1, 1)
STOCK_HISTORY_SLICE_DAYS = 365
PROXY_INIT_STAGGER_SECONDS = max(float(os.getenv("PROXY_INIT_STAGGER_SECONDS") or "2.0"), 0.0)


# ---------------------------------------------------------------------------
# Column translation
# ---------------------------------------------------------------------------

STANDARD_STOCK_COLUMNS = [
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌幅",
    "涨跌额",
    "换手率",
]

TARGET_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "value",
    "range",
    "gain",
    "change",
    "turnOver",
]


def transer_stock_to_model(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        res = pd.DataFrame(columns=TARGET_COLUMNS)
        res["date"] = pd.to_datetime(res["date"])
        return res.set_index("date", drop=True)

    mapper = dict(zip(STANDARD_STOCK_COLUMNS, TARGET_COLUMNS))
    res = df.rename(columns=mapper).copy()
    res["date"] = pd.to_datetime(res["date"])
    return res.set_index("date", drop=True)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_symbol_fp(symbol: str) -> str:
    os.makedirs(DataPath.STOCK_PATH, exist_ok=True)
    return os.path.join(DataPath.STOCK_PATH, f"{str(symbol).zfill(6)}.csv")


# ---------------------------------------------------------------------------
# Local date check
# ---------------------------------------------------------------------------

def get_stock_last_local_date(symbol: str) -> datetime.date | None:
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


def is_stock_data_updated_to_date(
    symbol: str,
    target_date: str | datetime.date | datetime.datetime | pd.Timestamp,
) -> bool:
    last_date = get_stock_last_local_date(symbol)
    if last_date is None:
        return False
    try:
        expected_date = pd.to_datetime(target_date).date()
    except Exception:
        return False
    return last_date >= expected_date


def batch_check_stock_data_updated(
    symbols: list[str],
    target_date: str | datetime.date | datetime.datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    expected_date = (
        datetime.date.today()
        if target_date is None
        else pd.to_datetime(target_date).date()
    )
    rows = []
    for symbol in symbols:
        normalized = str(symbol).zfill(6)
        fp = get_symbol_fp(normalized)
        last_local_date = get_stock_last_local_date(normalized)
        is_updated = bool(last_local_date and last_local_date >= expected_date)
        rows.append(
            {
                "symbol": normalized,
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


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_stock_data(symbol: str, df: pd.DataFrame) -> None:
    transer_stock_to_model(df).to_csv(
        get_symbol_fp(symbol),
        encoding="utf-8_sig",
        index=True,
    )


# ---------------------------------------------------------------------------
# History fetch helpers
# ---------------------------------------------------------------------------

def _empty_stock_history_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_STOCK_COLUMNS)


def _fetch_stock_history(
    symbol: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> pd.DataFrame:
    return get_stock_certain_date_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )


def _generate_stock_history_slices(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    max_slice_days: int = STOCK_HISTORY_SLICE_DAYS,
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    if max_slice_days <= 0:
        raise ValueError("max_slice_days must be positive")
    if end_date < start_date:
        raise ValueError("end_date must be later than or equal to start_date")
    slices: list[tuple[datetime.datetime, datetime.datetime]] = []
    slice_start = start_date
    while slice_start <= end_date:
        slice_end = min(
            slice_start + datetime.timedelta(days=max_slice_days - 1),
            end_date,
        )
        slices.append((slice_start, slice_end))
        slice_start = slice_end + datetime.timedelta(days=1)
    return slices


def _concat_stock_history_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [f.copy() for f in frames if f is not None and not f.empty]
    if not valid_frames:
        return _empty_stock_history_frame()
    combined = pd.concat(valid_frames, ignore_index=True)
    combined["_sort_date"] = pd.to_datetime(combined["日期"], errors="coerce")
    combined = combined.drop_duplicates(subset=["日期"], keep="last")
    combined = combined.sort_values(["_sort_date", "日期"]).drop(columns=["_sort_date"])
    return combined.reset_index(drop=True)


def _fetch_stock_history_in_slices(
    symbol: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    max_slice_days: int = STOCK_HISTORY_SLICE_DAYS,
) -> pd.DataFrame:
    total_days = (end_date.date() - start_date.date()).days + 1
    if total_days <= max_slice_days:
        return _fetch_stock_history(symbol=symbol, start_date=start_date, end_date=end_date)

    frames: list[pd.DataFrame] = []
    for slice_start, slice_end in _generate_stock_history_slices(
        start_date=start_date, end_date=end_date, max_slice_days=max_slice_days,
    ):
        frame = _fetch_stock_history(symbol=symbol, start_date=slice_start, end_date=slice_end)
        if frame is not None and not frame.empty:
            frames.append(frame)
    return _concat_stock_history_frames(frames)


def _fetch_full_stock_history(symbol: str) -> pd.DataFrame:
    return _fetch_stock_history_in_slices(
        symbol=symbol,
        start_date=STOCK_HISTORY_START,
        end_date=datetime.datetime.now(),
    )


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

def _get_proxy_init_delay(base_delay_sec: float | None = None) -> float:
    delay_step = PROXY_INIT_STAGGER_SECONDS if base_delay_sec is None else max(base_delay_sec, 0.0)
    if delay_step <= 0:
        return 0.0
    identity = getattr(current_process(), "_identity", ())
    worker_index = max(int(identity[0]) - 1, 0) if identity else 0
    return worker_index * delay_step


def _initialize_stock_update_worker(base_delay_sec: float | None = None) -> None:
    delay_sec = _get_proxy_init_delay(base_delay_sec)
    if delay_sec > 0:
        intervals(delay_sec)
    try:
        initialize_proxy_pool()
    except Exception as err:
        print(f"Proxy pool warm-up failed in {current_process().name}: {err}")


def update(code: str) -> bool:
    try:
        fp = get_symbol_fp(code)
        df = pd.read_csv(fp, parse_dates=True, index_col=0)
        if len(df) == 0:
            raise pd.errors.EmptyDataError("Empty data")

        last_update = pd.to_datetime(df.index.max()).date()
        now = datetime.datetime.now()
        if now.date() - last_update < datetime.timedelta(days=1):
            print(f"already updated {code}")
            return True

        update_df = _fetch_stock_history_in_slices(
            symbol=code,
            start_date=datetime.datetime.combine(last_update, datetime.time.min),
            end_date=now,
        )
        update_df = transer_stock_to_model(update_df)
        merged_df = update_df.combine_first(df)
        merged_df = merged_df[~merged_df.index.duplicated(keep="last")].sort_index()
        merged_df.to_csv(fp, index=True, encoding="utf-8-sig")
        print(f"update {code} in {fp}")
        return True
    except pd.errors.EmptyDataError:
        df = _fetch_full_stock_history(code)
        if df is not None and not df.empty:
            save_stock_data(code, df)
            return True
        print(f"Failed update {code}: fetch returned empty history")
        return False
    except Exception as err:
        traceback.print_exc()
        print(f"Can't update {code} because: {err}")
        return False


def update_single_stock_data(code: str) -> tuple[str, bool]:
    normalized = str(code).zfill(6)
    fp = get_symbol_fp(normalized)
    if os.path.exists(fp):
        return normalized, update(normalized)
    try:
        df = _fetch_full_stock_history(normalized)
        if df is not None and not df.empty:
            save_stock_data(normalized, df)
            return normalized, True
    except Exception as err:
        print(f"Can't acquire full history for {normalized} because {err}")
    return normalized, False


def update_stock_data(symbols: list[str] | None = None) -> None:
    all_stocks = (
        STOCK_LIST.get_all_symbol()
        if symbols is None
        else [str(s).zfill(6) for s in symbols]
    )
    if not all_stocks:
        print("No stock symbols to update")
        return

    with Pool(STOCK_UPDATE_POOL_SIZE, initializer=_initialize_stock_update_worker) as pool:
        results = pool.map(update_single_stock_data, all_stocks)

    for code, result in results:
        if result:
            print(f"Successfully updated data for {code}")
        else:
            print(f"Failed to update data for {code}")


# ---------------------------------------------------------------------------
# Batch acquire (full history)
# ---------------------------------------------------------------------------

def batch_acquire_stock_data(
    codes: list[str],
    last_n_days: int,
    save_path: str,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    with Pool(STOCK_UPDATE_POOL_SIZE) as pool:
        results = pool.starmap(
            _save_stock_data_to_path,
            [(code, last_n_days, save_path) for code in codes],
        )
    result_path = os.path.join(save_path, "acquire_results.txt")
    with open(result_path, "w") as f:
        for code, res in zip(codes, results):
            f.write(f"{code}: {res}\n")


def _save_stock_data_to_path(code: str, last_n_days: int, save_path: str) -> str:
    try:
        fp = os.path.join(save_path, f"{code}.csv")
        if os.path.exists(fp):
            return "Already exists"
        df = _get_stock_with_retry(code, last_n_days)
        if df is None:
            return f"Can't acquire data for {code} due to empty data"
        df = transer_stock_to_model(df)
        df.to_csv(fp, encoding="utf-8-sig", index=True)
        return "OK"
    except Exception as err:
        return f"Can't acquire data for {code} due to {err}"


def _get_stock_with_retry(code: str, last_n_days: int) -> pd.DataFrame | None:
    print(f"Acquiring data for {code} for last {last_n_days} days")
    max_retries_per_slice = 5
    dfs = []
    for s, e in generate_time_slices_alternative(last_n_days):
        success = False
        for retry_idx in range(max_retries_per_slice):
            try:
                df = get_stock_certain_date_data(
                    code,
                    datetime.datetime.strptime(s, "%Y%m%d"),
                    datetime.datetime.strptime(e, "%Y%m%d"),
                )
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


# ---------------------------------------------------------------------------
# Iteration / loading
# ---------------------------------------------------------------------------

def stock_data_iter() -> Iterator[StockDailyData]:
    os.makedirs(DataPath.STOCK_PATH, exist_ok=True)
    files = os.listdir(DataPath.STOCK_PATH)
    for file_name in tqdm.tqdm_notebook(files):
        if not file_name.endswith(".csv"):
            continue
        fp = os.path.join(DataPath.STOCK_PATH, file_name)
        try:
            yield StockDailyData.from_csv(fp)
        except Exception as err:
            print(f"can't iter {fp} due to {err}")
            continue


def get_stock_data_by_symbol(symbol: str) -> StockDailyData:
    fp = get_symbol_fp(symbol)
    return StockDailyData.from_csv(fp)


def get_stock_data_by_symbols(symbols: list[str]) -> list[StockDailyData]:
    return [get_stock_data_by_symbol(s) for s in symbols]


# ---------------------------------------------------------------------------
# Stock name list generator
# ---------------------------------------------------------------------------

_STOCK_NAME_LIST_POOL_SIZE = int(os.getenv("STOCK_NAME_LIST_POOL_SIZE") or "8")


def _generate_proxy_init_delay(base_delay_sec: float | None = None) -> float:
    delay_step = PROXY_INIT_STAGGER_SECONDS if base_delay_sec is None else max(base_delay_sec, 0.0)
    if delay_step <= 0:
        return 0.0
    identity = getattr(current_process(), "_identity", ())
    worker_index = max(int(identity[0]) - 1, 0) if identity else 0
    return worker_index * delay_step


def _generate_init_worker() -> None:
    delay = _generate_proxy_init_delay()
    if delay > 0:
        intervals(delay)
    try:
        initialize_proxy_pool()
    except Exception as err:
        print(f"Proxy pool warm-up failed in {current_process().name}: {err}")


def _resolve_market(symbol: str) -> str:
    code = str(symbol).zfill(6)
    if code.startswith("6"):
        return "SH"
    return "SZ"


def _fetch_individual_info(symbol: str) -> tuple[str, str]:
    """Return (symbol, list_date_str). list_date_str is empty on failure."""
    try:
        from fetcher.stock import get_stock_individual_info_em

        df = get_stock_individual_info_em(symbol)
        row = df[df["item"] == "上市时间"]
        if not row.empty:
            val = str(row["value"].values[0]).strip()
            return symbol, val
    except Exception as e:
        # Log first few failures so user can diagnose proxy / API issues.
        if not hasattr(_fetch_individual_info, "_error_count"):
            _fetch_individual_info._error_count = 0  # type: ignore[attr-defined]
        _fetch_individual_info._error_count += 1  # type: ignore[attr-defined]
        if _fetch_individual_info._error_count <= 3:  # type: ignore[attr-defined]
            print(f"[{_fetch_individual_info._error_count}] get_stock_individual_info_em({symbol}) failed: {e}")  # type: ignore[attr-defined]
    return symbol, ""


def generate_stock_name_list() -> pd.DataFrame:
    """Generate or refresh stock_name_list.csv.

    Returns the DataFrame that was written to disk.
    """
    import akshare as ak

    output_path = os.path.join(DataPath.DATA_DIR, "const", "stock_name_list.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. 获取沪/深全量 A 股（排除 B 股和北交所）
    print("Fetching all A-share codes from akshare ...")
    all_df = ak.stock_info_a_code_name()
    if all_df is None or all_df.empty:
        raise RuntimeError("stock_info_a_code_name returned empty data")

    codes_raw = all_df["code"].astype(str).str.zfill(6).tolist()
    codes = [c for c in codes_raw if not c.startswith(("2", "4", "8", "9"))]
    names = dict(zip(all_df["code"].astype(str).str.zfill(6), all_df["name"].astype(str)))
    print(f"Total A-share symbols (沪/深): {len(codes)}")

    # 2. 获取退市列表
    delist_map: dict[str, dict] = {}
    for source_name, fetch_fn in [
        ("上交所退市", lambda: ak.stock_info_sh_delist("全部")),
        ("深交所退市(终止)", lambda: ak.stock_info_sz_delist("终止上市公司")),
    ]:
        try:
            df = fetch_fn()
            if df is not None and not df.empty:
                code_col = "公司代码" if "公司代码" in df.columns else None
                if code_col:
                    for _, row in df.iterrows():
                        code = str(row[code_col]).zfill(6)
                        delist_map[code] = {
                            "name": str(row.get("公司简称", "")),
                            "list_date": str(row.get("上市日期", "")),
                            "delist_date": str(row.get("暂停上市日期", "")),
                        }
            print(f"{source_name}: {len(df) if df is not None else 0} 条")
        except Exception as exc:
            print(f"{source_name} 获取失败: {exc}")
            raise

    # 3. 合并
    rows = []
    merged_codes = set(codes) | set(delist_map.keys())
    for code in sorted(merged_codes):
        if code in delist_map:
            d = delist_map[code]
            rows.append(
                {
                    "symbol": code,
                    "name": d["name"],
                    "market": _resolve_market(code),
                    "list_date": d["list_date"],
                    "delist_date": d["delist_date"],
                }
            )
        else:
            rows.append(
                {
                    "symbol": code,
                    "name": names.get(code, ""),
                    "market": _resolve_market(code),
                    "list_date": "",
                    "delist_date": "",
                }
            )

    base_df = pd.DataFrame(rows)
    need_list_date = base_df[base_df["list_date"].str.strip() == ""]
    print(f"在市的股票: {len(base_df) - len(need_list_date)}, "
          f"需补上市日期的: {len(need_list_date)}")

    # 4. 批量获取上市日期
    if len(need_list_date) > 0:
        print(f"通过 stock_individual_info_em 补上市日期 (Pool={_STOCK_NAME_LIST_POOL_SIZE}) ...")
        symbols_to_query = need_list_date["symbol"].tolist()
        with Pool(_STOCK_NAME_LIST_POOL_SIZE, initializer=_generate_init_worker) as pool:
            results = pool.map(_fetch_individual_info, symbols_to_query)

        list_date_updates = {sym: d for sym, d in results if d}
        base_df.loc[base_df["symbol"].isin(list_date_updates), "list_date"] = base_df["symbol"].map(
            list_date_updates
        )
        still_empty = (base_df["list_date"].str.strip() == "").sum()
        print(f"补上市日期完成，仍有 {still_empty} 只留空")

    # 5. 原子写入
    tmp_path = output_path + ".tmp"
    base_df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, output_path)
    print(f"已写入: {output_path} ({len(base_df)} 条)")

    return base_df
