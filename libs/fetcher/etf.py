import datetime
import akshare as ak

from fetcher.etf import get_etf_last_n_day_data
from data_manager.etf_data_manager import save_etf_data
from utils.interval_utils import retry_with_intervals, intervals

def get_all_ETF_fund(all_funds):
    return all_funds[all_funds["拼音缩写"].str.endswith("ETF")]

def get_code_of_one_ETF_data(series):
    return series["基金代码"]

def get_all_etf_code(*filters):
    all_etf = ak.fund_etf_spot_em()
    for _filter in filters:
        all_etf = all_etf[_filter(all_etf)]
    return all_etf

def get_etf_last_n_day_data(symbol, n=40):
    now = datetime.datetime.now()
    start = now - datetime.timedelta(days=n)
    return ak.fund_etf_hist_em(
        symbol = symbol,
        period = "daily",
        start_date = start.strftime("%Y%m%d"),
        end_date = now.strftime("%Y%m%d"),
        adjust = "hfq"
    )

def get_etf_certain_date_data(symbol, s, e):
    return ak.fund_etf_hist_em(
        symbol = symbol,
        period = "daily",
        start_date = s.strftime("%Y%m%d"),
        end_date = e.strftime("%Y%m%d"),
        adjust = "hfq"
    )
