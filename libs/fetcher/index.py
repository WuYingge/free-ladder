from __future__ import annotations

import datetime
import os

import pandas as pd

from fetcher.utils import request_get_via_proxy
from utils.interval_utils import intervals


STANDARD_HISTORY_COLUMNS = [
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

SUPPORTED_INDEX_SOURCES = {
    "index_zh_a_hist",
    "stock_zh_index_hist_csindex",
}

INDEX_HISTORY_UT = os.getenv(
    "INDEX_HISTORY_EM_UT",
    "7eea3edcaed734bea9cbfc24409ed989",
)


def _empty_index_history_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_HISTORY_COLUMNS)


def _coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in STANDARD_HISTORY_COLUMNS if column != "日期"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _ensure_standard_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in STANDARD_HISTORY_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    return _coerce_numeric_columns(normalized[STANDARD_HISTORY_COLUMNS].copy())


def _normalize_index_zh_a_hist(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_index_history_frame()
    return _ensure_standard_columns(frame)


def _normalize_csindex_hist(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_index_history_frame()

    normalized = frame.rename(
        columns={
            "成交金额": "成交额",
            "涨跌": "涨跌额",
        }
    ).copy()

    if "成交量" in normalized.columns:
        normalized["成交量"] = pd.to_numeric(normalized["成交量"], errors="coerce") * 10000
    if "成交额" in normalized.columns:
        normalized["成交额"] = pd.to_numeric(normalized["成交额"], errors="coerce") * 100000000

    return _ensure_standard_columns(normalized)


def _fetch_index_zh_a_hist_raw(
    symbol: str,
    period: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
    if period not in period_dict:
        raise ValueError(f"Unsupported period: {period}")

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    candidate_market_ids = ["1", "0", "2", "47"]

    seen_market_ids: set[str] = set()
    data_json: dict | None = None
    for market_id in candidate_market_ids:
        if market_id in seen_market_ids:
            continue
        seen_market_ids.add(market_id)
        params = {
            "secid": f"{market_id}.{symbol}",
            "ut": INDEX_HISTORY_UT,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": period_dict[period],
            "fqt": "0",
            "beg": "0",
            "end": "20500000",
        }
        response = request_get_via_proxy(url, timeout=15, params=params)
        candidate_json = response.json()
        candidate_data = candidate_json.get("data")
        if candidate_data and candidate_data.get("klines"):
            data_json = candidate_json
            break

    if data_json is None or not data_json.get("data"):
        return _empty_index_history_frame()

    klines = data_json["data"].get("klines") or []
    if not klines:
        return _empty_index_history_frame()

    frame = pd.DataFrame([item.split(",") for item in klines])
    frame.columns = STANDARD_HISTORY_COLUMNS
    frame.index = pd.to_datetime(frame["日期"], errors="coerce")
    frame = frame[start_date:end_date]
    frame.reset_index(inplace=True, drop=True)
    return _coerce_numeric_columns(frame)


def _fetch_csindex_hist_raw(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    url = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
    params = {
        "indexCode": str(symbol),
        "startDate": start_date,
        "endDate": end_date,
    }
    response = request_get_via_proxy(url, timeout=15, params=params)
    data_json = response.json()
    rows = data_json.get("data") or []
    if not rows:
        return _empty_index_history_frame()

    frame = pd.DataFrame(rows)
    frame.columns = [
        "日期",
        "指数代码",
        "指数中文全称",
        "指数中文简称",
        "指数英文全称",
        "指数英文简称",
        "开盘",
        "最高",
        "最低",
        "收盘",
        "涨跌",
        "涨跌幅",
        "成交量",
        "成交金额",
        "样本数量",
        "滚动市盈率",
    ]
    return frame


def normalize_index_history_frame(
    frame: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    if source == "index_zh_a_hist":
        return _normalize_index_zh_a_hist(frame)
    if source == "stock_zh_index_hist_csindex":
        return _normalize_csindex_hist(frame)
    raise ValueError(f"Unsupported index source: {source}")


def get_index_certain_date_data(
    symbol: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    source: str = "index_zh_a_hist",
) -> pd.DataFrame:
    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if source == "index_zh_a_hist":
                raw_df = _fetch_index_zh_a_hist_raw(
                    symbol=str(symbol),
                    period="daily",
                    start_date=start,
                    end_date=end,
                )
            elif source == "stock_zh_index_hist_csindex":
                raw_df = _fetch_csindex_hist_raw(
                    symbol=str(symbol),
                    start_date=start,
                    end_date=end,
                )
            else:
                raise ValueError(f"Unsupported index source: {source}")

            return normalize_index_history_frame(raw_df, source=source)
        except Exception as err:
            last_error = err
            if attempt < 2:
                intervals(0.3)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Index history fetch failed without a captured exception")


def get_index_last_n_day_data(
    symbol: str,
    n: int = 40,
    source: str = "index_zh_a_hist",
) -> pd.DataFrame:
    now = datetime.datetime.now()
    start = now - datetime.timedelta(days=n)
    return get_index_certain_date_data(symbol, start, now, source=source)
