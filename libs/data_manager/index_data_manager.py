from __future__ import annotations

import datetime
import os
import traceback
from multiprocessing import Pool
from typing import Iterator

import pandas as pd
import tqdm

from config import DataPath
from core.models.index_daily_data import IndexDailyData
from data_manager.providers.index_list_provider import INDEX_LIST
from fetcher.index import STANDARD_HISTORY_COLUMNS, get_index_certain_date_data


INDEX_UPDATE_POOL_SIZE = 8
INDEX_HISTORY_START = datetime.datetime(1990, 1, 1)
INDEX_HISTORY_SLICE_DAYS = 365


def transer_index_to_model(df: pd.DataFrame) -> pd.DataFrame:
    origin_col = [
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
    target_col = [
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
    if len(df) == 0:
        res = pd.DataFrame(columns=target_col)
        res["date"] = pd.to_datetime(res["date"])
        return res.set_index("date", drop=True)

    mapper = dict(zip(origin_col, target_col))
    res = df.rename(columns=mapper).copy()
    res["date"] = pd.to_datetime(res["date"])
    return res.set_index("date", drop=True)


def get_symbol_fp(symbol: str) -> str:
    os.makedirs(DataPath.INDEX_PATH, exist_ok=True)
    return os.path.join(DataPath.INDEX_PATH, f"{str(symbol).zfill(6)}.csv")


def get_index_last_local_date(symbol: str) -> datetime.date | None:
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


def is_index_data_updated_to_date(
    symbol: str,
    target_date: str | datetime.date | datetime.datetime | pd.Timestamp,
) -> bool:
    last_date = get_index_last_local_date(symbol)
    if last_date is None:
        return False

    try:
        expected_date = pd.to_datetime(target_date).date()
    except Exception:
        return False

    return last_date >= expected_date


def batch_check_index_data_updated(
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
        normalized_symbol = str(symbol).zfill(6)
        fp = get_symbol_fp(normalized_symbol)
        last_local_date = get_index_last_local_date(normalized_symbol)
        is_updated = bool(last_local_date and last_local_date >= expected_date)
        rows.append(
            {
                "symbol": normalized_symbol,
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


def save_index_data(symbol: str, df: pd.DataFrame) -> None:
    transer_index_to_model(df).to_csv(
        get_symbol_fp(symbol),
        encoding="utf-8_sig",
        index=True,
    )


def _get_index_source(symbol: str) -> str:
    source = INDEX_LIST.get_source(symbol)
    return source or "index_zh_a_hist"


def _fetch_index_history(
    symbol: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> pd.DataFrame:
    return get_index_certain_date_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        source=_get_index_source(symbol),
    )


def _empty_index_history_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_HISTORY_COLUMNS)


def _generate_index_history_slices(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    max_slice_days: int = INDEX_HISTORY_SLICE_DAYS,
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


def _concat_index_history_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return _empty_index_history_frame()

    combined = pd.concat(valid_frames, ignore_index=True)
    combined["_sort_date"] = pd.to_datetime(combined["日期"], errors="coerce")
    combined = combined.drop_duplicates(subset=["日期"], keep="last")
    combined = combined.sort_values(["_sort_date", "日期"]).drop(columns=["_sort_date"])
    return combined.reset_index(drop=True)


def _fetch_index_history_in_slices(
    symbol: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    max_slice_days: int = INDEX_HISTORY_SLICE_DAYS,
) -> pd.DataFrame:
    total_days = (end_date.date() - start_date.date()).days + 1
    if total_days <= max_slice_days:
        return _fetch_index_history(symbol=symbol, start_date=start_date, end_date=end_date)

    frames: list[pd.DataFrame] = []
    for slice_start, slice_end in _generate_index_history_slices(
        start_date=start_date,
        end_date=end_date,
        max_slice_days=max_slice_days,
    ):
        frame = _fetch_index_history(
            symbol=symbol,
            start_date=slice_start,
            end_date=slice_end,
        )
        if frame is not None and not frame.empty:
            frames.append(frame)

    return _concat_index_history_frames(frames)


def _fetch_full_index_history(symbol: str) -> pd.DataFrame:
    return _fetch_index_history_in_slices(
        symbol=symbol,
        start_date=INDEX_HISTORY_START,
        end_date=datetime.datetime.now(),
    )


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

        update_df = _fetch_index_history_in_slices(
            symbol=code,
            start_date=datetime.datetime.combine(last_update, datetime.time.min),
            end_date=now,
        )
        update_df = transer_index_to_model(update_df)
        merged_df = update_df.combine_first(df)
        merged_df = merged_df[~merged_df.index.duplicated(keep="last")].sort_index()
        merged_df.to_csv(fp, index=True, encoding="utf-8-sig")
        print(f"update {code} in {fp}")
        return True
    except pd.errors.EmptyDataError:
        df = _fetch_full_index_history(code)
        if df is not None and not df.empty:
            save_index_data(code, df)
            return True
        print(f"Failed update {code}: fetch returned empty history")
        return False
    except Exception as err:
        traceback.print_exc()
        print(f"Can't update {code} because: {err}")
        return False


def update_single_index_data(code: str) -> tuple[str, bool]:
    normalized_code = str(code).zfill(6)
    fp = get_symbol_fp(normalized_code)
    if os.path.exists(fp):
        return normalized_code, update(normalized_code)

    try:
        df = _fetch_full_index_history(normalized_code)
        if df is not None and not df.empty:
            save_index_data(normalized_code, df)
            return normalized_code, True
    except Exception as err:
        print(f"Can't acquire full history for {normalized_code} because {err}")
    return normalized_code, False


def update_index_data(symbols: list[str] | None = None) -> None:
    all_indices = (
        INDEX_LIST.get_all_symbol()
        if symbols is None
        else [str(symbol).zfill(6) for symbol in symbols]
    )
    if not all_indices:
        print("No index symbols to update")
        return

    with Pool(INDEX_UPDATE_POOL_SIZE) as pool:
        results = pool.map(update_single_index_data, all_indices)

    for code, result in results:
        if result:
            print(f"Successfully updated data for {code}")
        else:
            print(f"Failed to update data for {code}")


def index_data_iter() -> Iterator[IndexDailyData]:
    os.makedirs(DataPath.INDEX_PATH, exist_ok=True)
    files = os.listdir(DataPath.INDEX_PATH)
    for file_name in tqdm.tqdm_notebook(files):
        if not file_name.endswith(".csv"):
            continue
        fp = os.path.join(DataPath.INDEX_PATH, file_name)
        try:
            yield IndexDailyData.from_csv(fp)
        except Exception as err:
            print(f"can't iter {fp} due to {err}")
            continue


def get_index_data_by_symbol(symbol: str) -> IndexDailyData:
    fp = get_symbol_fp(symbol)
    return IndexDailyData.from_csv(fp)


def get_index_data_by_symbols(symbols: list[str]) -> list[IndexDailyData]:
    return [get_index_data_by_symbol(symbol) for symbol in symbols]
