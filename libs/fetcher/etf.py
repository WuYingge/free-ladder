import os
import datetime
import requests
import json
from random import choice
import akshare as ak
import pandas as pd
from proxy.proxy import get_proxy
from fetcher.utils import fetch_paginated_data
from fetcher.consts import headers
from dotenv import load_dotenv

load_dotenv()

from utils.interval_utils import retry_with_intervals, intervals

def get_all_ETF_fund(all_funds):
    return all_funds[all_funds["拼音缩写"].str.endswith("ETF")]

def get_code_of_one_ETF_data(series):
    return series["基金代码"]

def get_all_etf_code(*filters):
    all_etf = None
    while all_etf is None:
        try:
            all_etf = fund_etf_spot_em()
        except Exception as e:
            print(e)
    for _filter in filters:
        all_etf = all_etf[_filter(all_etf)]
    return all_etf

def get_etf_last_n_day_data(symbol, n=40):
    now = datetime.datetime.now()
    start = now - datetime.timedelta(days=n)
    return fund_etf_hist_em(
        symbol = symbol,
        period = "daily",
        start_date = start.strftime("%Y%m%d"),
        end_date = now.strftime("%Y%m%d"),
        adjust = "hfq"
    )

def get_etf_certain_date_data(symbol, s, e):
    return fund_etf_hist_em(
        symbol = symbol,
        period = "daily",
        start_date = s.strftime("%Y%m%d"),
        end_date = e.strftime("%Y%m%d"),
        adjust = "hfq"
    )

def fund_etf_hist_em(
    symbol: str = "159707",
    period: str = "daily",
    start_date: str = "19700101",
    end_date: str = "20500101",
    adjust: str = "",
) -> pd.DataFrame:
    """
    东方财富-ETF行情
    https://quote.eastmoney.com/sz159707.html
    :param symbol: ETF 代码
    :type symbol: str
    :param period: choice of {'daily', 'weekly', 'monthly'}
    :type period: str
    :param start_date: 开始日期
    :type start_date: str
    :param end_date: 结束日期
    :type end_date: str
    :param adjust: choice of {"qfq": "前复权", "hfq": "后复权", "": "不复权"}
    :type adjust: str
    :return: 每日行情
    :rtype: pandas.DataFrame
    """
    adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
    period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": os.getenv("FUND_ETF_HIST_EM_UT"),
        "klt": period_dict[period],
        "fqt": adjust_dict[adjust],
        "beg": start_date,
        "end": end_date,
    }
    try:
        market_id = "0" if symbol[0] == "1" else "1"
        params.update({"secid": f"{market_id}.{symbol}"})
        r = requests.get(url, timeout=15, params=params, proxies=get_proxy())
        data_json = r.json()
    except KeyError:
        print(KeyError)
        market_id = 1
        params.update({"secid": f"{market_id}.{symbol}"})
        r = requests.get(url, timeout=15, params=params, proxies=get_proxy())
        data_json = r.json()
        if not data_json["data"]:
            market_id = 0
            params.update({"secid": f"{market_id}.{symbol}"})
            r = requests.get(url, timeout=15, params=params, proxies=get_proxy())
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
    # temp_df.index = pd.to_datetime(temp_df["日期"], errors="coerce")
    temp_df.reset_index(inplace=True, drop=True)
    temp_df["开盘"] = pd.to_numeric(temp_df["开盘"], errors="coerce")
    temp_df["收盘"] = pd.to_numeric(temp_df["收盘"], errors="coerce")
    temp_df["最高"] = pd.to_numeric(temp_df["最高"], errors="coerce")
    temp_df["最低"] = pd.to_numeric(temp_df["最低"], errors="coerce")
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
    temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
    temp_df["换手率"] = pd.to_numeric(temp_df["换手率"], errors="coerce")
    return temp_df

def get_etf_market_id():
    """
    东方财富-ETF代码和市场标识映射
    https://quote.eastmoney.com/center/gridlist.html#fund_etf
    :return: ETF 代码和市场标识映射
    :rtype: dict
    """
    url = "https://88.push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1",
        "pz": "100",
        "po": "1",
        "np": "1",
        "ut": os.getenv("ETF_MARKET_ID_EM_UT"),
        "fltt": "2",
        "invt": "2",
        "wbp2u": "|0|0|0|web",
        "fid": "f3",
        "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024",
        "fields": "f3,f12,f13",
    }
    temp_df = fetch_paginated_data(url, params)
    temp_dict = dict(zip(temp_df["f12"], temp_df["f13"]))
    return temp_dict


def fund_etf_spot_em() -> pd.DataFrame:
    """
    东方财富-ETF 实时行情
    https://quote.eastmoney.com/center/gridlist.html#fund_etf
    :return: ETF 实时行情
    :rtype: pandas.DataFrame
    """
    url = "https://88.push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1",
        "pz": "100",
        "po": "1",
        "np": "1",
        "ut": os.getenv("FUND_ETF_SPOT_EM_UT"),
        "fltt": "2",
        "invt": "2",
        "wbp2u": "|0|0|0|web",
        "fid": "f12",
        "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827",
        "fields": (
            "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,"
            "f12,f13,f14,f15,f16,f17,f18,f20,f21,"
            "f23,f24,f25,f22,f11,f30,f31,f32,f33,"
            "f34,f35,f38,f62,f63,f64,f65,f66,f69,"
            "f72,f75,f78,f81,f84,f87,f115,f124,f128,"
            "f136,f152,f184,f297,f402,f441"
        ),
    }
    temp_df = fetch_paginated_data(url, params)
    temp_df.rename(
        columns={
            "f12": "代码",
            "f14": "名称",
            "f2": "最新价",
            "f4": "涨跌额",
            "f3": "涨跌幅",
            "f5": "成交量",
            "f6": "成交额",
            "f7": "振幅",
            "f17": "开盘价",
            "f15": "最高价",
            "f16": "最低价",
            "f18": "昨收",
            "f8": "换手率",
            "f10": "量比",
            "f30": "现手",
            "f31": "买一",
            "f32": "卖一",
            "f33": "委比",
            "f34": "外盘",
            "f35": "内盘",
            "f62": "主力净流入-净额",
            "f184": "主力净流入-净占比",
            "f66": "超大单净流入-净额",
            "f69": "超大单净流入-净占比",
            "f72": "大单净流入-净额",
            "f75": "大单净流入-净占比",
            "f78": "中单净流入-净额",
            "f81": "中单净流入-净占比",
            "f84": "小单净流入-净额",
            "f87": "小单净流入-净占比",
            "f38": "最新份额",
            "f21": "流通市值",
            "f20": "总市值",
            "f402": "基金折价率",
            "f441": "IOPV实时估值",
            "f297": "数据日期",
            "f124": "更新时间",
        },
        inplace=True,
    )
    temp_df = temp_df[
        [
            "代码",
            "名称",
            "最新价",
            "IOPV实时估值",
            "基金折价率",
            "涨跌额",
            "涨跌幅",
            "成交量",
            "成交额",
            "开盘价",
            "最高价",
            "最低价",
            "昨收",
            "振幅",
            "换手率",
            "量比",
            "委比",
            "外盘",
            "内盘",
            "主力净流入-净额",
            "主力净流入-净占比",
            "超大单净流入-净额",
            "超大单净流入-净占比",
            "大单净流入-净额",
            "大单净流入-净占比",
            "中单净流入-净额",
            "中单净流入-净占比",
            "小单净流入-净额",
            "小单净流入-净占比",
            "现手",
            "买一",
            "卖一",
            "最新份额",
            "流通市值",
            "总市值",
            "数据日期",
            "更新时间",
        ]
    ]
    temp_df["最新价"] = pd.to_numeric(temp_df["最新价"], errors="coerce")
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
    temp_df["开盘价"] = pd.to_numeric(temp_df["开盘价"], errors="coerce")
    temp_df["最高价"] = pd.to_numeric(temp_df["最高价"], errors="coerce")
    temp_df["最低价"] = pd.to_numeric(temp_df["最低价"], errors="coerce")
    temp_df["昨收"] = pd.to_numeric(temp_df["昨收"], errors="coerce")
    temp_df["换手率"] = pd.to_numeric(temp_df["换手率"], errors="coerce")
    temp_df["量比"] = pd.to_numeric(temp_df["量比"], errors="coerce")
    temp_df["委比"] = pd.to_numeric(temp_df["委比"], errors="coerce")
    temp_df["外盘"] = pd.to_numeric(temp_df["外盘"], errors="coerce")
    temp_df["内盘"] = pd.to_numeric(temp_df["内盘"], errors="coerce")
    temp_df["流通市值"] = pd.to_numeric(temp_df["流通市值"], errors="coerce")
    temp_df["总市值"] = pd.to_numeric(temp_df["总市值"], errors="coerce")
    temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
    temp_df["现手"] = pd.to_numeric(temp_df["现手"], errors="coerce")
    temp_df["买一"] = pd.to_numeric(temp_df["买一"], errors="coerce")
    temp_df["卖一"] = pd.to_numeric(temp_df["卖一"], errors="coerce")
    temp_df["最新份额"] = pd.to_numeric(temp_df["最新份额"], errors="coerce")
    temp_df["IOPV实时估值"] = pd.to_numeric(temp_df["IOPV实时估值"], errors="coerce")
    temp_df["基金折价率"] = pd.to_numeric(temp_df["基金折价率"], errors="coerce")
    temp_df["主力净流入-净额"] = pd.to_numeric(
        temp_df["主力净流入-净额"], errors="coerce"
    )
    temp_df["主力净流入-净占比"] = pd.to_numeric(
        temp_df["主力净流入-净占比"], errors="coerce"
    )
    temp_df["超大单净流入-净额"] = pd.to_numeric(
        temp_df["超大单净流入-净额"], errors="coerce"
    )
    temp_df["超大单净流入-净占比"] = pd.to_numeric(
        temp_df["超大单净流入-净占比"], errors="coerce"
    )
    temp_df["大单净流入-净额"] = pd.to_numeric(
        temp_df["大单净流入-净额"], errors="coerce"
    )
    temp_df["大单净流入-净占比"] = pd.to_numeric(
        temp_df["大单净流入-净占比"], errors="coerce"
    )
    temp_df["中单净流入-净额"] = pd.to_numeric(
        temp_df["中单净流入-净额"], errors="coerce"
    )
    temp_df["中单净流入-净占比"] = pd.to_numeric(
        temp_df["中单净流入-净占比"], errors="coerce"
    )
    temp_df["小单净流入-净额"] = pd.to_numeric(
        temp_df["小单净流入-净额"], errors="coerce"
    )
    temp_df["小单净流入-净占比"] = pd.to_numeric(
        temp_df["小单净流入-净占比"], errors="coerce"
    )
    temp_df["数据日期"] = pd.to_datetime(
        temp_df["数据日期"], format="%Y%m%d", errors="coerce"
    )
    temp_df["更新时间"] = (
        pd.to_datetime(temp_df["更新时间"], unit="s", errors="coerce")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Asia/Shanghai")
    )
    return temp_df

def fund_etf_name_em() -> pd.DataFrame:
    """
    东方财富网站-天天基金网-基金数据-所有基金的名称和类型
    https://fund.eastmoney.com/manager/default.html#dt14;mcreturnjson;ftall;pn20;pi1;scabbname;stasc
    :return: 所有基金的名称和类型
    :rtype: pandas.DataFrame
    """
    url = "https://fund.eastmoney.com/js/fundcode_search.js"
    r = requests.get(url, headers=headers, proxies=get_proxy())
    text_data = r.text.strip("var r = ").strip(";")
    data_json = json.loads(text_data)
    df = pd.DataFrame(data_json, columns=["symbol", "pinyin_abbr", "name", "type", "full_pinyin"])
    df = df[df["name"].str.contains("ETF") & ~df["name"].str.contains("联接") & df["type"].isin(["指数型-股票", "指数型-海外股票", "指数型-固收", "指数型-其他"])]
    df = pd.concat([
        df, 
        pd.DataFrame({
            'symbol': {0: '513170', 1: '588690', 2: '588710'},
            'pinyin_abbr': {0: 'PHHSZGYQQDII', 1: 'KZZZ', 2: 'KBDT'},
            'name': {0: '鹏华恒生中国央企(QDII)', 1: '科综指增', 2: '科半导体'},
            'type': {0: '指数型-海外股票', 1: '指数型-股票', 2: '指数型-股票'},
            'full_pinyin': {0: 'PENGHUAHENGSHENGZHONGGUOYANGQIQDII', 1: 'KEZONGZHIZENG', 2: 'KEBANDAOTI'}
            })
        ], ignore_index=True)
    return df[["symbol", "name", "type"]].reset_index(drop=True)
