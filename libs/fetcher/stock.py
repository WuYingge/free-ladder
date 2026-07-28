from __future__ import annotations

import datetime
import os

import pandas as pd

from fetcher.utils import request_get_via_proxy
from utils.interval_utils import intervals


_STOCK_HIST_EM_UT = os.getenv(
    "STOCK_HIST_EM_UT",
    "7eea3edcaed734bea9cbfc24409ed989",
)


def _resolve_market_code(symbol: str) -> str:
    return "1" if symbol.startswith("6") else "0"


def get_stock_hist_em(
    symbol: str = "000001",
    period: str = "daily",
    start_date: str = "19700101",
    end_date: str = "20500101",
    adjust: str = "hfq",
) -> pd.DataFrame:
    """
    东方财富-A股日行情 （从 akshare stock_zh_a_hist 微调）
    https://quote.eastmoney.com/concept/sh603777.html?from=classic
    :param symbol: 股票代码
    :param period: choice of {'daily', 'weekly', 'monthly'}
    :param start_date: 开始日期 (YYYYMMDD)
    :param end_date: 结束日期 (YYYYMMDD)
    :param adjust: choice of {"qfq": "前复权", "hfq": "后复权", "": "不复权"}
    :return: 每日行情 (11 列，与 ETF/Index 格式一致)
    """
    adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
    period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": _STOCK_HIST_EM_UT,
        "klt": period_dict[period],
        "fqt": adjust_dict[adjust],
        "secid": f"{_resolve_market_code(symbol)}.{symbol}",
        "beg": start_date,
        "end": end_date,
    }
    r = request_get_via_proxy(url, timeout=15, params=params, max_proxy_retries=3)
    data_json = r.json()
    if not (data_json["data"] and data_json["data"]["klines"]):
        return pd.DataFrame()

    temp_df = pd.DataFrame([item.split(",") for item in data_json["data"]["klines"]])
    temp_df.columns = [
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
    temp_df.reset_index(inplace=True, drop=True)
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]:
        temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")
    return temp_df


def get_stock_certain_date_data(
    symbol: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    adjust: str = "hfq",
) -> pd.DataFrame:
    return get_stock_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust=adjust,
    )


def get_stock_last_n_day_data(symbol: str, n: int = 40) -> pd.DataFrame:
    now = datetime.datetime.now()
    start = now - datetime.timedelta(days=n)
    return get_stock_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=now.strftime("%Y%m%d"),
        adjust="hfq",
    )


def get_stock_individual_info_em(symbol: str = "603777") -> pd.DataFrame:
    """
    东方财富-个股信息 （从 akshare stock_individual_info_em 微调，走代理）
    https://quote.eastmoney.com/concept/sh603777.html?from=classic
    :param symbol: 股票代码
    :return: item/value 两列，含"上市时间""总市值"等
    """
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "fltt": "2",
        "invt": "2",
        "fields": "f57,f58,f84,f85,f116,f117,f127,f189,f43",
        "secid": f"{_resolve_market_code(symbol)}.{symbol}",
    }
    r = request_get_via_proxy(url, timeout=15, params=params, max_proxy_retries=3)
    data_json = r.json()
    raw_data = data_json.get("data")
    if not raw_data:
        return pd.DataFrame(columns=["item", "value"])

    # data may be a dict or a Python-repr string like "{'f57': '600519', ...}"
    if isinstance(raw_data, dict):
        parsed = raw_data
    elif isinstance(raw_data, str):
        import ast

        try:
            parsed = ast.literal_eval(raw_data)
        except (ValueError, SyntaxError):
            return pd.DataFrame(columns=["item", "value"])
    else:
        return pd.DataFrame(columns=["item", "value"])

    field_label = {
        "f57": "股票代码",
        "f58": "股票简称",
        "f84": "总股本",
        "f85": "流通股",
        "f127": "行业",
        "f116": "总市值",
        "f117": "流通市值",
        "f189": "上市时间",
        "f43": "最新",
    }
    rows = []
    for key, label in field_label.items():
        if key in parsed:
            rows.append({"item": label, "value": str(parsed[key])})
    return pd.DataFrame(rows)


def get_all_stock_code() -> set[str]:
    """Return the set of currently-listed A-share symbols (online)."""
    try:
        import akshare as ak

        df = ak.stock_info_a_code_name()
    except Exception as err:
        print(f"Failed to load stock list: {err}")
        return set()

    if df is None or df.empty:
        return set()

    code_col = "code" if "code" in df.columns else None
    if code_col is None:
        return set()
    codes = df[code_col].astype(str).str.zfill(6).tolist()
    # exclude B shares (200xxx, 900xxx) and BJ (4xxxxx, 8xxxxx)
    return {
        c for c in codes
        if not c.startswith(("2", "4", "8", "9"))
    }
