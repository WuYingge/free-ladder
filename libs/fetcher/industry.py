"""申万宏源研究-行业分类抓取（从 akshare stock_industry_clf_hist_sw 微调，走项目代理）。

数据源：申万宏源官网 2021 版个股行业分类历史
    https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls

文件语义：全市场（含已退市）每只股票的每次行业分类变动一行，
    计入日期 = 该行业归属生效日（可做时点回溯），更新日期 = 官网维护时间。
    行业代码为 6 位申万分类代码（如 480301），前 2/4/6 位分别对应申万一/二/三级行业档位
    （代码按前缀嵌套，该文件不含"代码→行业名称"标准表，名称映射为后续扩展点）。

注意：申万官网 TLS 证书链在 curl_cffi 下校验失败，本项目统一走代理且固定 verify=False。
"""

from __future__ import annotations

import io

import pandas as pd

from fetcher.utils import request_get_via_proxy

_STOCK_SW_INDUSTRY_CLF_URL = (
    "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
)

_STOCK_CODE_COL = "股票代码"
_START_DATE_COL = "计入日期"
_INDUSTRY_CODE_COL = "行业代码"
_UPDATE_TIME_COL = "更新日期"


def get_stock_sw_industry_clf_hist(
    timeout: int = 20,
    max_attempts: int = 5,
) -> pd.DataFrame:
    """下载申万 2021 版全市场个股行业分类历史（官网 xls，走代理）。

    :param timeout: 单次请求读超时（秒），连接超时由代理层固定 3 秒
    :param max_attempts: 代理轮换失败后的总尝试次数
    :return: 标准列 symbol/start_date/industry_code/update_time，
             按 (symbol, start_date) 升序；symbol/industry_code 为 str，
             start_date/update_time 为 datetime.date
    :raises RuntimeError: 数据为空或多次尝试仍失败
    """
    import time

    last_err: Exception | None = None
    for _ in range(max_attempts):
        try:
            r = request_get_via_proxy(
                _STOCK_SW_INDUSTRY_CLF_URL,
                timeout=timeout,
                max_proxy_retries=3,
                verify=False,  # 见模块 docstring
            )
            raw = pd.read_excel(
                io.BytesIO(r.content),
                dtype={_STOCK_CODE_COL: str, _INDUSTRY_CODE_COL: str},
            )
            break
        except Exception as err:  # noqa: BLE001 — 代理池故障类型多样，统一重试
            last_err = err
            time.sleep(2)
    else:
        raise RuntimeError("get_stock_sw_industry_clf_hist: 多次尝试下载失败") from last_err

    if raw is None or raw.empty:
        raise RuntimeError("get_stock_sw_industry_clf_hist: 官网返回空数据")

    parsed = pd.DataFrame(
        {
            "symbol": raw[_STOCK_CODE_COL].astype(str).str.strip().str.zfill(6),
            "start_date": pd.to_datetime(raw[_START_DATE_COL], errors="coerce"),
            "industry_code": raw[_INDUSTRY_CODE_COL].astype(str).str.strip(),
            "update_time": pd.to_datetime(raw[_UPDATE_TIME_COL], errors="coerce"),
        }
    )
    parsed = parsed.dropna(subset=["start_date", "update_time"])
    parsed = parsed[parsed["industry_code"].ne("") & parsed["symbol"].ne("")]
    # 官网数据可能含重复维护行，同一 (symbol, start_date) 保留末行
    parsed = parsed.drop_duplicates(subset=["symbol", "start_date"], keep="last")
    parsed["start_date"] = parsed["start_date"].dt.date
    parsed["update_time"] = parsed["update_time"].dt.date
    parsed = parsed.sort_values(["symbol", "start_date"], ignore_index=True)
    return parsed
