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


# ---------------------------------------------------------------------------
# Daily valuation (RPT_VALUEANALYSIS_DET, 东财数据中心)
# ↓ 单个股每日 总市值/流通市值/流通A股股本, 数据自 2018-01-02 起
# ---------------------------------------------------------------------------

_STOCK_DAILY_BASIC_FILE_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_STOCK_DAILY_BASIC_REPORT = "RPT_VALUEANALYSIS_DET"


def get_stock_daily_basic_em(
    symbol: str = "000001",
    start_date: str = "20160101",
    end_date: str = "20500101",
) -> pd.DataFrame:
    """
    东方财富-个股每日市值/股本 (东财数据中心 RPT_VALUEANALYSIS_DET)
    https://data.eastmoney.com/stock/stockdetail/000001.html

    :param symbol: 股票代码 (6 位)
    :param start_date: 开始日期 (YYYYMMDD), 接口实际数据自 2018-01-02 起
    :param end_date: 结束日期 (YYYYMMDD)
    :return: 标准列 date/circ_mv/total_mv/float_share, DatetimeIndex 升序
             (circ_mv=流通市值 元, total_mv=总市值 元, float_share=流通A股 股)
    """
    start_fmt = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_fmt = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    params = {
        "reportName": _STOCK_DAILY_BASIC_REPORT,
        "columns": "ALL",
        "filter": (
            f'(SECURITY_CODE="{str(symbol).zfill(6)}")'
            f"(TRADE_DATE>='{start_fmt}')(TRADE_DATE<='{end_fmt}')"
        ),
        "pageSize": "5000",
        "pageNumber": "1",
    }
    r = request_get_via_proxy(_STOCK_DAILY_BASIC_FILE_URL, timeout=20, params=params, max_proxy_retries=3)
    data_json = r.json()
    result = data_json.get("result") or {}
    rows = result.get("data") or []
    if not rows:
        return pd.DataFrame(
            columns=["circ_mv", "total_mv", "float_share"],
            index=pd.DatetimeIndex([], name="date"),
        )

    temp_df = pd.DataFrame(rows)
    # 返回按 TRADE_DATE 降序, 翻转为升序
    temp_df = temp_df.sort_values("TRADE_DATE", ascending=True)
    # 注意: 先建帧再设索引 (dict 构造时传 index= 会按标签对齐导致 NaN)
    parsed = pd.DataFrame(
        {
            "circ_mv": pd.to_numeric(temp_df["NOTLIMITED_MARKETCAP_A"], errors="coerce"),
            "total_mv": pd.to_numeric(temp_df["TOTAL_MARKET_CAP"], errors="coerce"),
            "float_share": pd.to_numeric(temp_df["FREE_SHARES_A"], errors="coerce"),
        }
    )
    parsed.index = pd.to_datetime(temp_df["TRADE_DATE"])
    parsed.index.name = "date"
    return parsed
