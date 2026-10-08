"""财报/财务数据抓取 (Financial Statements Fetcher)

数据源与 akshare 等价接口 (列名已实测, 非估计):

| 本模块函数 | 东财/巨潮 原始接口 | akshare 等价 |
|---|---|---|
| ``get_financial_statement_rows`` | ``emweb/PC_HSF10/NewFinanceAnalysis/{lrb,zcfzb,xjllb}AjaxNew`` | ``stock_profit_sheet_by_report_em`` / ``stock_balance_sheet_by_report_em`` / ``stock_cash_flow_sheet_by_report_em`` (**不用**: akshare 版本丢弃 NOTICE_DATE) |
| ``get_period_forecast_rows`` | 东财 ``RPT_PUBLIC_OP_NEWPREDICT`` | ``stock_yjyg_em`` |
| ``get_period_dividend_rows`` | 东财 ``RPT_SHAREBONUS_DET`` | ``stock_fhps_em`` |
| ``get_period_holder_num_rows`` | 东财 ``RPT_HOLDERNUM_DET`` | ``stock_zh_a_gdhs`` |
| ``get_share_change_rows`` | 巨潮 ``webapi.cninfo.com.cn`` | ``stock_share_change_cninfo`` |
| ``get_period_announce_crosscheck_rows`` | 东财 ``RPT_LICO_FN_CPD`` | ``stock_yjbb_em`` (**仅一次性对拍**, 其 "最新公告日期" 随更正前移, 不入库) |

取数通道 (2026-09-11 实测, 拆开"链路延迟"与"请求扇出"两件事):

  * 链路延迟(同一端点各 5 次): 直连 p50=0.078s, 代理 p50=0.419s —— **代理到东财并不慢**,
    只贵约 0.34s/请求。量价类任务每只股票 1 个请求, 这点开销无感;
  * 财务的请求扇出高两个数量级: F10 单次最多 5 期, 全历史 × 三表 ≈ 31~37 请求/股,
    于是 +0.34s/请求被乘成 +10~13s/股。**这是代理相对更慢的唯一原因**。
  * 池被其他任务占满时(代理商回 "您提取且未使用的IP太多"), 每次调用要先在"取代理"上
    失败再回退直连; 本模块对该情形只重试 1 次以尽快回退, 不自旋等待。

**默认 proxy_first** (客户 2026-09-11 指定): 与同仓库其他任务共用代理出口、不暴露本机 IP。
全量历史回填建议临时 `FINANCIAL_FETCH_MODE=direct_first` (省 ~10s/只), 可用环境变量调整:

    FINANCIAL_FETCH_MODE=direct_first|proxy_first|direct_only|proxy_only
    FINANCIAL_FETCH_THREADS=8

本模块只负责"取原始行并整理成落盘 schema 的列", 不做去重/落盘/质检
(见 ``data_manager.financial_manager`` / ``data_manager.financial_schema``)。
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from data_manager.financial_schema import CURRENCY_CNY, report_type_of
from fetcher.utils import request_get_via_proxy
from utils.interval_utils import intervals

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

FINANCIAL_FETCH_MODE = os.getenv("FINANCIAL_FETCH_MODE") or "proxy_first"
FINANCIAL_FETCH_THREADS = max(int(os.getenv("FINANCIAL_FETCH_THREADS") or "8"), 1)

#: 东财 F10 财务分析 (按报告期, reportType=1 = 累计口径)
EMWEB_BASE = "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis"
#: 东财数据中心 (全市场逐期表)
DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
#: 部分 reportName 只在 securities 域名下返回正确字段 (实测: 业绩预告)，见 DATACENTER_HOST_BY_REPORT
DATACENTER_ALT_URL = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
#: reportName → 该走哪个域名
DATACENTER_HOST_BY_REPORT: dict[str, str] = {
    "RPT_PUBLIC_OP_NEWPREDICT": DATACENTER_ALT_URL,
}
#: reportName → 该表 filter 里报告期字段名 (写错字段名时东财会静默返回**别的数据集**,
#: 例如业绩预告写 REPORTDATE 会返回业绩报表字段而 PREDICT_* 全为 None → 必须显式登记)
DATACENTER_DATE_FIELD_BY_REPORT: dict[str, str] = {
    "RPT_PUBLIC_OP_NEWPREDICT": "REPORT_DATE",
}
#: 巨潮资讯 (股本变动)
CNINFO_URL = "http://webapi.cninfo.com.cn/api/stock/p_stock2215"
CNINFO_SCOPE = os.getenv("CNINFO_SCOPE") or "1099"
CNINFO_KEY = os.getenv("CNINFO_KEY") or "0f2f4a9ce5b0be1f1c0f0f1c5c0c0c0c"

#: F10 报表端点单次请求的 dates 批量上限 (硬约束: 超过 5 个日期时服务端静默只回最新 5 期)
F10_STATEMENT_BATCH_SIZE = 5

#: companyType 进程内缓存 (symbol → hidctype); 由 _resolve_company_type 填充
_COMPANY_TYPE_CACHE: dict[str, str] = {}

#: 三张财务报表 → (日期清单端点后缀, 数据端点后缀)
STATEMENT_ENDPOINTS: dict[str, tuple[str, str]] = {
    "income": ("lrbDateAjaxNew", "lrbAjaxNew"),
    "balance": ("zcfzbDateAjaxNew", "zcfzbAjaxNew"),
    "cashflow": ("xjllbDateAjaxNew", "xjllbAjaxNew"),
}

#: F10 原始字段 → 落盘列 (仅含从信源取值/相加的列)
#: 注意: 利润表 (lrb) **没有股本字段** —— 字段名已实测 (2026-09-10, 203 个字段逐个核对),
#: 股本请从 zcfzb 的 SHARE_CAPITAL 取 (见 F10_BALANCE_FIELDS)。
F10_INCOME_FIELDS: dict[str, str] = {
    "net_profit_attr_p": "PARENT_NETPROFIT",
    "revenue": "TOTAL_OPERATE_INCOME",
    "operating_profit": "OPERATE_PROFIT",
    "net_profit": "NETPROFIT",
    "net_profit_deducted": "DEDUCT_PARENT_NETPROFIT",
    "rd_expense": "RESEARCH_EXPENSE",
    # 少数股东损益 (损益科目, 累计口径): 合并净利 = 归母净利 + 少数股东损益。
    # 实测 000002 2026H1: -14.95e9 + (-1.07e9) = -16.02e9 = NETPROFIT (恒等式成立)。
    "minority_interest_profit": "MINORITY_INTEREST",
}
F10_BALANCE_FIELDS: dict[str, str] = {
    "total_assets": "TOTAL_ASSETS",
    "total_equity_attr_p": "TOTAL_PARENT_EQUITY",
    "total_liabilities": "TOTAL_LIABILITIES",
    "total_equity": "TOTAL_EQUITY",
    "monetary_funds": "MONETARYFUNDS",
    "accounts_receivable": "ACCOUNTS_RECE",
    "inventory": "INVENTORY",
    "goodwill": "GOODWILL",
    # 注意: 资产负债表里的 MINORITY_EQUITY 是**少数股东权益(净资产科目)**,
    # 数量级与归母净资产相当; 它不是损益表的"少数股东损益", 二者不可互换
    # (实测 000002 2026H1: 权益 1.136e11 vs 损益 -1.07e9)。
    # "少数股东损益" 属于利润表 → 见 F10_INCOME_FIELDS["minority_interest_profit"]。
    "minority_interest_equity": "MINORITY_EQUITY",
    "total_share": "SHARE_CAPITAL",
}
F10_CASHFLOW_FIELDS: dict[str, str] = {
    "ocf_net": "NETCASH_OPERATE",
}
#: 需要相加的字段 (原始字段实测: 销售/管理费用列名为 SALE_EXPENSE / MANAGE_EXPENSE)
F10_SUM_FIELDS: dict[str, tuple[str, ...]] = {
    "sell_admin_expense": ("SALE_EXPENSE", "MANAGE_EXPENSE"),
}

#: 业绩预告: 只保留"净利润"类预测指标 —— 东财同一报告期会为同一只股票返回多行
#: (归母净利 / 扣非净利 / 营业收入 / 扣除后营业收入 / 每股收益), 其中营业收入与
#: 每股收益的**量纲不是元**, 混入会让 forecast_net_profit_* 列失去单位一致性。
FORECAST_INCLUDE_FINANCE: tuple[str, ...] = ("净利润",)
#: 明确排除的预测指标 (量纲不同或非利润口径)
FORECAST_EXCLUDE_FINANCE: tuple[str, ...] = ("每股收益", "营业收入", "现金流", "净资产", "毛利率")

#: 预告类型英文枚举 (东财 PREDICT_TYPE → 需求清单要求的英文列)
ANNOUNCE_TYPE_EN: dict[str, str] = {
    "预增": "increase",
    "略增": "slightly_increase",
    "续盈": "continue_profit",
    "扭亏": "turnaround",
    "减亏": "narrow_loss",
    "首亏": "first_loss",
    "续亏": "continue_loss",
    "增亏": "widen_loss",
    "预减": "decrease",
    "略减": "slightly_decrease",
    "不确定": "uncertain",
}
FORECAST_STATE_EN: dict[str, str] = {
    "increase": "increase",
    "decrease": "decrease",
    "unknown": "uncertain",
}


def _is_net_profit_indicator(finance: str) -> bool:
    """该预测指标是否为"净利润"口径 (元)。"""
    if not finance:
        return False
    if any(keyword in finance for keyword in FORECAST_EXCLUDE_FINANCE):
        return False
    return any(keyword in finance for keyword in FORECAST_INCLUDE_FINANCE)

#: 分红送转
FHPS_REPORT_NAME = "RPT_SHAREBONUS_DET"
HOLDERNUM_REPORT_NAME = "RPT_HOLDERNUM_DET"
#: 业绩预告 (只在 securities 域名返回 PREDICT_* 字段)
FORECAST_REPORT_NAME = "RPT_PUBLIC_OP_NEWPREDICT"
#: 业绩报表 (一次性对拍信源, 不入库)
ANNOUNCE_REPORT_NAME = "RPT_LICO_FN_CPD"


# ---------------------------------------------------------------------------
# HTTP 通道
# ---------------------------------------------------------------------------


class FinancialFetchError(RuntimeError):
    """取数失败 (网络/接口/解析)。"""


def _direct_get(url: str, params: dict[str, Any] | None, timeout: int) -> Any:
    import requests

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def financial_get(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 20,
    mode: str | None = None,
) -> Any:
    """财务数据 HTTP 入口: 默认直连优先, 失败回退仓库代理池。

    :param mode: ``direct_first``(默认) / ``proxy_first`` / ``direct_only`` / ``proxy_only``
    :raises FinancialFetchError: 所选通道全部失败
    """
    resolved = (mode or FINANCIAL_FETCH_MODE).lower()
    errors: list[str] = []

    def _proxy() -> Any:
        return request_get_via_proxy(url, params=params or {}, timeout=timeout, max_proxy_retries=1)

    def _direct() -> Any:
        return _direct_get(url, params, timeout)

    order: list[tuple[str, Callable[[], Any]]]
    if resolved == "direct_only":
        order = [("direct", _direct)]
    elif resolved == "proxy_only":
        order = [("proxy", _proxy)]
    elif resolved == "proxy_first":
        order = [("proxy", _proxy), ("direct", _direct)]
    else:
        order = [("direct", _direct), ("proxy", _proxy)]

    for name, call in order:
        try:
            return call()
        except Exception as err:  # noqa: BLE001 — 通道切换需要捕获全部异常
            errors.append(f"{name}: {type(err).__name__}: {err}")
    raise FinancialFetchError(f"取数失败 {url}: {' | '.join(errors)}")


def _resolve_market_symbol(symbol: str) -> str:
    """6 位代码 → 东财 F10 代码 (SH600519 / SZ000001 / BJ920025)。"""
    code = str(symbol).zfill(6)
    if code.startswith(("83", "87", "88", "89", "92", "43")):
        return f"BJ{code}"
    if code.startswith(("5", "6", "9")):
        return f"SH{code}"
    return f"SZ{code}"


def _parse_period(date: str) -> pd.Timestamp:
    """20241231 / 2024-12-31 → Timestamp。"""
    ts = pd.to_datetime(date, errors="coerce", format="%Y%m%d")
    if pd.isna(ts):
        ts = pd.to_datetime(date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"无法解析报告期: {date!r}")
    assert isinstance(ts, pd.Timestamp)
    return ts


def period_date_str(date: str) -> str:
    """报告期归一为 YYYY-MM-DD。"""
    return _parse_period(date).strftime("%Y-%m-%d")


def _to_float(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float("nan") if pd.isna(number) else float(number)


# ---------------------------------------------------------------------------
# 东财数据中心: 通用分页
# ---------------------------------------------------------------------------


def datacenter_rows(
    report_name: str,
    filter_str: str,
    sort_columns: str = "NOTICE_DATE,SECURITY_CODE",
    sort_types: str = "1,1",
    page_size: int = 500,
    max_pages: int = 60,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    """按 reportName + filter 拉全量行 (东财数据中心分页)。

    域名按 reportName 选择 (见 ``DATACENTER_HOST_BY_REPORT``): 业绩预告只在
    ``datacenter.eastmoney.com/securities`` 下返回 PREDICT_* 字段, 用错域名会
    拿到业绩报表的字段 (PREDICT_* 全为 None), 必须显式区分。
    """
    endpoint = DATACENTER_HOST_BY_REPORT.get(report_name, DATACENTER_URL)
    params: dict[str, Any] = {
        "sortColumns": sort_columns,
        "sortTypes": sort_types,
        "pageSize": str(page_size),
        "pageNumber": "1",
        "reportName": report_name,
        "columns": "ALL",
        "filter": filter_str,
    }
    response = financial_get(endpoint, params=params, timeout=25, mode=mode)
    payload = response.json()
    result = payload.get("result") or {}
    pages = int(result.get("pages") or 0)
    rows: list[dict[str, Any]] = list(result.get("data") or [])
    for page in range(2, min(pages, max_pages) + 1):
        intervals(0.05)
        params["pageNumber"] = str(page)
        payload = financial_get(endpoint, params=params, timeout=25, mode=mode).json()
        rows.extend((payload.get("result") or {}).get("data") or [])
    return rows


def _a_share_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """剔除北交所/非 A 股 (预告/快报接口无 SECURITY_TYPE_CODE 过滤条件)。"""
    kept: list[dict[str, Any]] = []
    for row in rows:
        code = str(row.get("SECURITY_CODE") or "").strip()
        market = str(row.get("TRADE_MARKET_CODE") or "")
        if not (len(code) == 6 and code.isdigit()):
            continue
        if market == "069001017":  # 北京证券交易所
            continue
        if code.startswith(("4", "8", "9")):
            continue
        kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# 东财 F10: 三张财务报表 (逐股, 一次拿全历史)
# ---------------------------------------------------------------------------


def _resolve_company_type(symbol: str, mode: str | None = None) -> str:
    """读取东财 F10 页面里的 hidctype (**必须按公司取**)。

    实测 (2026-09-10): 通用公司=4, 银行=3 (000001), 证券=1 (000776), 保险=2 (601318)。
    硬编码 companyType=4 会让银行/券商/保险的**资产负债表与现金流量表整表为空**
    (利润表模板共通所以看起来正常) —— 这是必须逐股解析的字段, 不能省这一次请求。
    """
    cached = _COMPANY_TYPE_CACHE.get(symbol)
    if cached:
        return cached
    em_code = _resolve_market_symbol(symbol)
    try:
        response = financial_get(
            f"{EMWEB_BASE}/Index",
            params={"type": "web", "code": em_code.lower()},
            timeout=20,
            mode=mode,
        )
        match = re.search(r'id="hidctype"[^>]*value="(\d+)"', response.text)
        company_type = match.group(1) if match else "4"
    except Exception as err:  # noqa: BLE001 — 解析失败退回通用模板
        print(f"[financial] {symbol} companyType 解析失败, 退回 4: {type(err).__name__}: {err}")
        company_type = "4"
    _COMPANY_TYPE_CACHE[symbol] = company_type
    return company_type


def _statement_dates(symbol: str, statement: str, mode: str | None = None) -> list[str]:
    date_api, _ = STATEMENT_ENDPOINTS[statement]
    em_code = _resolve_market_symbol(symbol)
    params = {
        "companyType": _resolve_company_type(symbol, mode=mode),
        "reportDateType": "0",
        "code": em_code,
    }
    response = financial_get(f"{EMWEB_BASE}/{date_api}", params=params, timeout=25, mode=mode)
    payload = response.json()
    return [
        str(item.get("REPORT_DATE") or "")[:10]
        for item in (payload.get("data") or [])
        if item.get("REPORT_DATE")
    ]


def _statement_rows(
    symbol: str,
    statement: str,
    dates: Sequence[str],
    mode: str | None = None,
) -> list[dict[str, Any]]:
    """分批请求 F10 报表数据 (**服务端硬约束: 单次最多 5 个日期**)。

    实测 (2026-09-11):
      * 单次请求 dates 超过 5 个时, 服务端**静默只返回最新 5 期** (传 103 期也只回 5 期,
        不报错) —— 必须分批累积; akshare 同样按 5 期切分。
      * 服务端**接受任选 5 个日期**, 不要求连续(实测 ``2024-12-31,2023-12-31,...`` 与
        ``2024-06-30,2023-06-30,...`` 都按请求返回)。
        备注: 曾试过"按同月日分组"以压缩批次数, 实测**对全历史无收益** ——
        同月日组各有 24~28 个日期, 分组后 22 批 > 连续切分 21 批 (仅在目标期数很少时
        如日更最近 2 期才从 3 批降到 2 批), 故不采用。
    """
    _, data_api = STATEMENT_ENDPOINTS[statement]
    em_code = _resolve_market_symbol(symbol)
    company_type = _resolve_company_type(symbol, mode=mode)
    ordered = list(dates)
    batches = [
        ordered[start : start + F10_STATEMENT_BATCH_SIZE]
        for start in range(0, len(ordered), F10_STATEMENT_BATCH_SIZE)
    ]

    collected: list[dict[str, Any]] = []
    for batch in batches:
        if not batch:
            continue
        params = {
            "companyType": company_type,
            "reportDateType": "0",
            "reportType": "1",  # 累计口径
            "code": em_code,
            "dates": ",".join(batch),
        }
        response = financial_get(f"{EMWEB_BASE}/{data_api}", params=params, timeout=30, mode=mode)
        payload = response.json()
        collected.extend(payload.get("data") or [])
        if len(batches) > 1:
            intervals(0.02)
    if len(ordered) and not collected:
        print(f"[financial] {symbol} {statement}: {len(ordered)} 期请求后返回 0 行 (检查 companyType/日期清单)")
    return collected


def _parse_date_value(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    return pd.NaT if pd.isna(ts) else pd.Timestamp(ts)


def _pick(row: dict[str, Any], field_map: dict[str, str], sum_map: dict[str, tuple[str, ...]]) -> dict[str, float]:
    picked: dict[str, float] = {col: _to_float(row.get(raw)) for col, raw in field_map.items()}
    for col, raws in sum_map.items():
        values = [_to_float(row.get(raw)) for raw in raws]
        present = [v for v in values if not pd.isna(v)]
        picked[col] = float(sum(present)) if present else float("nan")
    return picked


def _statement_frame(
    symbol: str,
    statement: str,
    raw_rows: Sequence[dict[str, Any]],
    mode: str | None = None,
) -> pd.DataFrame:
    """把 F10 原始行整理成 schema 列 (未含 update_date/basis, 由 manager 补)。"""
    if statement == "income":
        field_map, sum_map = F10_INCOME_FIELDS, F10_SUM_FIELDS
    elif statement == "balance":
        field_map, sum_map = F10_BALANCE_FIELDS, {}
    else:
        field_map, sum_map = F10_CASHFLOW_FIELDS, {}

    records: list[dict[str, Any]] = []
    for row in raw_rows:
        report_date = _parse_date_value(row.get("REPORT_DATE"))
        if pd.isna(report_date):
            continue
        notice = _parse_date_value(row.get("NOTICE_DATE"))
        if statement == "income":
            eps = _to_float(row.get("BASIC_EPS"))
        else:
            eps = float("nan")
        record: dict[str, Any] = {
            "symbol": str(symbol).zfill(6),
            "report_date": report_date,
            "ann_date": notice,
            "report_type": report_type_of(report_date),
            "currency": CURRENCY_CNY,
            "_source_update_date": _parse_date_value(row.get("UPDATE_DATE")),
            "_source_report_name": str(row.get("REPORT_DATE_NAME") or ""),
        }
        if statement == "income":
            record["eps_basic"] = eps
        record.update(_pick(row, field_map, sum_map))
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values("report_date").reset_index(drop=True)


def get_financial_statement_data(
    symbol: str,
    periods: int | None = 2,
    statements: Sequence[str] = ("income", "balance", "cashflow"),
    mode: str | None = None,
) -> dict[str, pd.DataFrame]:
    """单只股票的利润表/资产负债表/现金流量表 (累计口径)。

    :param periods: 只取最近 N 个报告期 (日更用 2); None = 全历史 (回填用)
    :return: ``{"income"|"balance"|"cashflow": DataFrame}``; 失败的表返回空 DataFrame
    """
    result: dict[str, pd.DataFrame] = {}
    dates_cache: dict[str, list[str]] = {}
    for statement in statements:
        try:
            dates = dates_cache.get(statement)
            if dates is None:
                dates = _statement_dates(symbol, statement, mode=mode)
                dates_cache[statement] = dates
            if not dates:
                result[statement] = pd.DataFrame()
                continue
            target = dates if periods is None else dates[: max(int(periods), 1)]
            raw_rows = _statement_rows(symbol, statement, target, mode=mode)
            frame = _statement_frame(symbol, statement, raw_rows, mode=mode)
            if frame.empty and target:
                # 静默空数据必须留痕 (曾掩盖"超 5 期只回最新 5 期"的问题)
                print(
                    f"[financial] {symbol} {statement}: 请求 {len(target)} 期但解析为 0 行 "
                    f"(companyType={_resolve_company_type(symbol, mode=mode)})"
                )
            result[statement] = frame
        except Exception as err:  # noqa: BLE001 — 单股单表失败不影响全局
            print(f"[financial] {symbol} {statement} 取数失败: {type(err).__name__}: {err}")
            result[statement] = pd.DataFrame()
        intervals(0.02)
    return result


def get_financial_statement_rows(
    symbol: str,
    periods: int | None = 2,
    mode: str | None = None,
) -> dict[str, pd.DataFrame]:
    """``get_financial_statement_data`` 的别名 (保留计划中的函数名)。"""
    return get_financial_statement_data(symbol, periods=periods, mode=mode)


def fetch_statement_symbols(
    symbols: Sequence[str],
    periods: int | None = 2,
    statements: Sequence[str] = ("income", "balance", "cashflow"),
    threads: int | None = None,
    mode: str | None = None,
) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str]]]:
    """并发抓取多只股票的三张报表。

    :return: ``({statement: 合并后的原始帧}, [(symbol, reason), ...])``
    """
    pool_size = max(int(threads or FINANCIAL_FETCH_THREADS), 1)
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in statements}
    failures: list[tuple[str, str]] = []

    def _one(code: str) -> tuple[str, dict[str, pd.DataFrame], str | None]:
        try:
            data = get_financial_statement_data(code, periods=periods, statements=statements, mode=mode)
        except Exception as err:  # noqa: BLE001
            return code, {}, f"{type(err).__name__}: {err}"
        empty = all(frame is None or frame.empty for frame in data.values())
        return code, data, ("取数为空" if empty else None)

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        for code, data, reason in executor.map(_one, list(symbols)):
            if reason is not None:
                failures.append((code, reason))
            for name, frame in data.items():
                if frame is not None and not frame.empty:
                    frames[name].append(frame)

    merged: dict[str, pd.DataFrame] = {}
    for name in statements:
        merged[name] = (
            pd.concat(frames[name], ignore_index=True) if frames[name] else pd.DataFrame()
        )
    return merged, failures


# ---------------------------------------------------------------------------
# 东财数据中心: 业绩预告
# ---------------------------------------------------------------------------


def get_period_forecast_rows(period: str, mode: str | None = None) -> pd.DataFrame:
    """业绩预告 (全市场, 单报告期)。列名见 ``FORECAST_COLUMNS``。

    过滤字段用 ``REPORT_DATE`` (不是 ``REPORTDATE``): 写错字段名时东财不报错,
    而是静默返回**业绩报表**的数据 (``PREDICT_*`` 全 None) → 这里额外做字段级自检。
    """
    report_date = _parse_period(period).strftime("%Y-%m-%d")
    date_field = DATACENTER_DATE_FIELD_BY_REPORT.get(FORECAST_REPORT_NAME, "REPORTDATE")
    rows = datacenter_rows(
        FORECAST_REPORT_NAME,
        f"({date_field}='{report_date}')",
        sort_columns="NOTICE_DATE,SECURITY_CODE",
        sort_types="1,1",
        mode=mode,
    )
    if rows and not any("PREDICT" in key for key in rows[0]):
        raise FinancialFetchError(
            "业绩预告接口返回了非预告字段 (疑似 filter 字段名或域名错误导致东财返回其他数据集); "
            f"实际字段: {sorted(rows[0].keys())[:8]}..."
        )
    kept = _a_share_rows(rows)
    records: list[dict[str, Any]] = []
    dropped_indicators: dict[str, int] = {}
    for row in kept:
        finance = str(row.get("PREDICT_FINANCE") or "").strip()
        if not _is_net_profit_indicator(finance):
            dropped_indicators[finance or "<空>"] = dropped_indicators.get(finance or "<空>", 0) + 1
            continue
        notice = _parse_date_value(row.get("NOTICE_DATE"))
        low = _to_float(row.get("PREDICT_AMT_LOWER"))
        high = _to_float(row.get("PREDICT_AMT_UPPER"))
        mid = float("nan")
        if not pd.isna(low) and not pd.isna(high):
            mid = (low + high) / 2.0
        announce_type = str(row.get("PREDICT_TYPE") or "").strip()
        records.append(
            {
                "symbol": str(row.get("SECURITY_CODE")).zfill(6),
                "report_date": _parse_date_value(row.get("REPORT_DATE")),
                "ann_date": notice,
                "report_type": report_type_of(_parse_date_value(row.get("REPORT_DATE"))),
                "forecast_ann_date": notice,
                "forecast_indicator_cn": finance,
                "announce_type": announce_type,
                "announce_type_en": ANNOUNCE_TYPE_EN.get(
                    announce_type, FORECAST_STATE_EN.get(str(row.get("FORECAST_STATE") or ""), "")
                ),
                "forecast_net_profit_low": low,
                "forecast_net_profit_high": high,
                "forecast_net_profit_mid": mid,
                "yoy_low": _to_float(row.get("PREDICT_RATIO_LOWER")) / 100.0,
                "yoy_high": _to_float(row.get("PREDICT_RATIO_UPPER")) / 100.0,
                "_source_update_date": _parse_date_value(row.get("UPDATE_DATE")),
                "_forecast_finance": finance,
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values(["symbol", "ann_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 东财数据中心: 分红送转 / 股东户数
# ---------------------------------------------------------------------------


def _parse_plan_profile(profile: str) -> tuple[float, float, float]:
    """解析东财分红送转方案文本 → (每股现金分红元, 每10股送股, 每10股转增)。

    实测文本形态: ``10派0.80元(含税,扣税后0.72元)`` / ``10转4.50股`` /
    ``10送2转3派1.00元``; 解析失败返回 (NaN, NaN, NaN), 由调用方与原始比率列交叉兜底。
    """
    cash = transfer = bonus = float("nan")
    if not profile:
        return cash, transfer, bonus
    match = re.search(r"派([\d.]+)", profile)
    if match:
        cash = _to_float(match.group(1)) / 10.0  # 每 10 股 → 每股
    match = re.search(r"送([\d.]+)", profile)
    if match:
        bonus = _to_float(match.group(1))
    match = re.search(r"转(?:增)?([\d.]+)", profile)
    if match:
        transfer = _to_float(match.group(1))
    return cash, transfer, bonus


def get_period_dividend_rows(period: str, mode: str | None = None) -> pd.DataFrame:
    """分红送转预案 (全市场, 单报告期)。

    ``ann_date`` 取 ``PLAN_NOTICE_DATE`` (预案公告日/首次公开日, 实测 100% 有值);
    ``NOTICE_DATE`` 是该方案的最新公告日, 会随实施进度前移, **不作 PIT 用途**。
    """
    report_date = _parse_period(period).strftime("%Y-%m-%d")
    rows = datacenter_rows(
        FHPS_REPORT_NAME,
        f"(REPORT_DATE='{report_date}')",
        sort_columns="PLAN_NOTICE_DATE,SECURITY_CODE",
        sort_types="1,1",
        mode=mode,
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        plan_ann = _parse_date_value(row.get("PLAN_NOTICE_DATE"))
        cash_ratio = _to_float(row.get("PRETAX_BONUS_RMB"))
        bonus_field = _to_float(row.get("BONUS_IT_RATIO"))
        transfer_field = _to_float(row.get("IT_RATIO"))
        cash_text, transfer_text, bonus_text = _parse_plan_profile(
            str(row.get("IMPL_PLAN_PROFILE") or "")
        )
        records.append(
            {
                "symbol": str(row.get("SECURITY_CODE") or "").zfill(6),
                "report_date": _parse_date_value(row.get("REPORT_DATE")),
                "ann_date": plan_ann,
                "report_type": report_type_of(_parse_date_value(row.get("REPORT_DATE"))),
                "plan_ann_date": plan_ann,
                "implement_date": _parse_date_value(row.get("EX_DIVIDEND_DATE")),
                "cash_div_per_share": cash_ratio if not pd.isna(cash_ratio) else cash_text,
                "bonus_ratio": bonus_field if not pd.isna(bonus_field) else bonus_text,
                "transfer_ratio": transfer_field if not pd.isna(transfer_field) else transfer_text,
                "_assign_progress": str(row.get("ASSIGN_PROGRESS") or ""),
                "_plan_profile": str(row.get("IMPL_PLAN_PROFILE") or ""),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values(["symbol", "ann_date"]).reset_index(drop=True)


def get_period_holder_num_rows(period: str, mode: str | None = None) -> pd.DataFrame:
    """股东户数 (全市场, 按统计截止日 = 报告期)。

    实测字段: ``END_DATE``(统计截止日) / ``HOLD_NOTICE_DATE``(公告日) /
    ``HOLDER_NUM``(股东户数)。
    """
    end_date = _parse_period(period).strftime("%Y-%m-%d")
    rows = datacenter_rows(
        HOLDERNUM_REPORT_NAME,
        f"(END_DATE='{end_date}')",
        sort_columns="END_DATE,SECURITY_CODE",
        sort_types="-1,1",
        mode=mode,
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        end_ts = _parse_date_value(row.get("END_DATE"))
        records.append(
            {
                "symbol": str(row.get("SECURITY_CODE") or "").zfill(6),
                "report_date": end_ts,
                "ann_date": _parse_date_value(row.get("HOLD_NOTICE_DATE")),
                "report_type": report_type_of(end_ts),
                "holder_num": _to_float(row.get("HOLDER_NUM")),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values(["symbol", "ann_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 巨潮: 股本变动 (逐股)
# ---------------------------------------------------------------------------


def get_share_change_rows(
    symbol: str,
    start_date: str = "2014-01-01",
    end_date: str | None = None,
    mode: str | None = None,
) -> pd.DataFrame:
    """股本变动事件 (总股本/流通股本/变动原因/公告日/变动日)。

    取数路径(实测, 2026-09-10):
      1. 巨潮 ``webapi.cninfo.com.cn`` — 需 ``Accept-Enckey`` **JS 签名 token**
         (裸请求返回 401/451 "未经授权的访问")。本仓不自带 JS 引擎签名逻辑,
         故走 **akshare 公共函数** ``stock_share_change_cninfo`` (其内置 py_mini_racer
         生成 token; 本项目依赖 akshare>=1.17.44, py_mini_racer 随其安装)。
      2. 巨潮失败时回退东财 ``stock_zh_a_gbjg_em`` (无公告日, 按变动日近似)。

    单位换算(重要): 巨潮返回的 ``总股本``/``已流通股份`` 单位是**万股**,
    本模块统一 ×10000 转为**股** (实测 600519: 125227.0215 万股 = 1,252,270,215 股,
    与 F10 资产负债表 SHARE_CAPITAL 一致)。
    """
    end = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    code = str(symbol).zfill(6)
    frame = _share_change_via_cninfo(code, start_date, end)
    if frame is None or frame.empty:
        frame = _share_change_via_em(code)
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame.sort_values(["symbol", "ann_date"]).reset_index(drop=True)


def _share_change_via_cninfo(code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    try:
        import akshare as ak

        raw = ak.stock_share_change_cninfo(
            symbol=code,
            start_date=pd.to_datetime(start_date).strftime("%Y%m%d"),
            end_date=pd.to_datetime(end_date).strftime("%Y%m%d"),
        )
    except Exception as err:  # noqa: BLE001 — 单股失败由调用方计入 failures
        print(f"[financial] {code} 巨潮股本变动失败: {type(err).__name__}: {err}")
        return None
    if raw is None or raw.empty:
        return None
    records: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        total_share = _to_float(row.get("总股本"))
        circ_share = _to_float(row.get("已流通股份"))
        records.append(
            {
                "symbol": code,
                "ann_date": _parse_date_value(row.get("公告日期")),
                "effective_date": _parse_date_value(row.get("变动日期")),
                # 万股 → 股
                "total_share": float("nan") if pd.isna(total_share) else total_share * 10000.0,
                "circ_share": float("nan") if pd.isna(circ_share) else circ_share * 10000.0,
                "reason": str(row.get("变动原因") or row.get("变动原因编码") or "").strip(),
            }
        )
    return pd.DataFrame(records)


def _share_change_via_em(code: str) -> pd.DataFrame | None:
    """回退: 东财股本结构 (无公告日 → 以变动日作为 ann_date 近似, 并标注)。"""
    market = "SH" if code.startswith("6") else "SZ"
    try:
        import akshare as ak

        raw = ak.stock_zh_a_gbjg_em(symbol=f"{code}.{market}")
    except Exception as err:  # noqa: BLE001
        print(f"[financial] {code} 东财股本结构回退失败: {type(err).__name__}: {err}")
        return None
    if raw is None or raw.empty:
        return None
    records: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        change_date = _parse_date_value(row.get("变更日期"))
        records.append(
            {
                "symbol": code,
                # 该源无公告日: 用变动日近似, reason 里标注来源以免被误当首次公告日
                "ann_date": change_date,
                "effective_date": change_date,
                "total_share": _to_float(row.get("总股本")),
                "circ_share": _to_float(row.get("已上市流通A股")),
                "reason": f"{str(row.get('变动原因') or '').strip()}[源=东财股本结构,ann_date=变动日近似]",
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 一次性对拍信源 (不入库)
# ---------------------------------------------------------------------------


def get_period_announce_crosscheck_rows(period: str, mode: str | None = None) -> pd.DataFrame:
    """业绩报表 (``RPT_LICO_FN_CPD``) —— **仅用于一次性双源公告日对拍, 不入库**。

    该表的 "最新公告日期" 会随更正公告前移, 不能作为 PIT 的 ann_date 来源。
    """
    report_date = _parse_period(period).strftime("%Y-%m-%d")
    rows = datacenter_rows(
        ANNOUNCE_REPORT_NAME,
        f"(REPORTDATE='{report_date}')",
        sort_columns="UPDATE_DATE,SECURITY_CODE",
        sort_types="-1,-1",
        mode=mode,
    )
    kept = _a_share_rows(rows)
    records = [
        {
            "symbol": str(row.get("SECURITY_CODE")).zfill(6),
            "report_date": _parse_date_value(row.get("REPORTDATE")),
            "ann_date": _parse_date_value(row.get("NOTICE_DATE")),
            "update_date": _parse_date_value(row.get("UPDATE_DATE")),
            "net_profit_attr_p": _to_float(row.get("PARENT_NETPROFIT")),
            "revenue": _to_float(row.get("TOTAL_OPERATE_INCOME")),
            "eps_basic": _to_float(row.get("BASIC_EPS")),
        }
        for row in kept
    ]
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values(["symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def company_type_of(symbol: str, mode: str | None = None) -> str:
    """公开入口: 单只股票的东财 F10 companyType (银行=3/证券=1/保险=2/通用=4)。"""
    return _resolve_company_type(str(symbol).zfill(6), mode=mode)


def available_periods(start_year: int, end_year: int | None = None, today: pd.Timestamp | None = None) -> list[str]:
    """列出应抓取的报告期 (YYYY-MM-DD), 已跳过披露窗口未结束的期。"""
    from data_manager.financial_schema import legal_deadline

    now = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    last_year = int(end_year) if end_year is not None else now.year
    periods: list[str] = []
    for year in range(int(start_year), last_year + 1):
        for mmdd in ("03-31", "06-30", "09-30", "12-31"):
            period = f"{year}-{mmdd}"
            deadline = legal_deadline(period)
            if pd.isna(deadline):
                continue
            if now >= deadline:  # 披露窗口已结束才抓, 避免把空期写成缺失
                periods.append(period)
    return periods


# ---------------------------------------------------------------------------
# 巨潮公告: 定期报告的**首次公告日** (ann_date 的独立仲裁者)
# ---------------------------------------------------------------------------

#: 定期报告标题 → 报告期月日
CNINFO_REPORT_KIND_MMDD: dict[str, str] = {
    "年度报告": "12-31",
    "半年度报告": "06-30",
    "中期报告": "06-30",
    "第一季度报告": "03-31",
    "第三季度报告": "09-31",  # 占位, 实际由下面修正为 09-30
}
CNINFO_REPORT_KIND_MMDD["第三季度报告"] = "09-30"

#: 严格标题匹配: 必须是"报告本身", 不能是"关于…年度报告的审核意见/专项说明"
#: (实测搜"年度报告"会命中监事意见等 20+ 条噪声, 故用锚定正则 + 后缀白名单)
_CNINFO_TITLE_RE = re.compile(
    r"^(?P<year>\d{4})年(?P<kind>年度报告|半年度报告|第一季度报告|第三季度报告|中期报告)"
    r"(?P<suffix>摘要|全文|正文|英文版|（修订版）|\(修订版\)|财务报告)?$"
)
#: 标题黑名单: 更正/更新后/已取消 的版本不是首次公告
_CNINFO_TITLE_BLOCK = ("更新后", "更正", "已取消", "取消", "（修订版）摘要")


def get_periodic_report_announcements(
    symbol: str,
    start_date: str = "2010-01-01",
    end_date: str | None = None,
    mode: str | None = None,
) -> pd.DataFrame:
    """单只股票的定期报告公告 → ``(report_date, ann_date)`` 首次公告日映射。

    这是 `ann_date` 的**独立仲裁者**: 巨潮公告标题自带年份与报告类型
    (如「2016年第一季度报告全文」), 因此能判断"某一天发的是哪一期的报告"。
    东财 F10 对部分报告期返回的 `NOTICE_DATE` 实为**下一期同日历期**的公告日
    (错位一年), 只有本接口能判谁对 (见 docs/financial_data_provenance 相关章节)。

    :return: 列 ``symbol, report_date, ann_date, title``; 无公告返回空帧。
    """
    import akshare as ak

    code = str(symbol).zfill(6)
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce") if end_date else pd.Timestamp.today()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=["symbol", "report_date", "ann_date", "title"])
    raw = ak.stock_zh_a_disclosure_report_cninfo(
        symbol=code,
        market="沪深京",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )
    if raw is None or raw.empty or "公告标题" not in raw.columns:
        return pd.DataFrame(columns=["symbol", "report_date", "ann_date", "title"])
    records: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        title = str(row.get("公告标题") or "").strip()
        if any(block in title for block in _CNINFO_TITLE_BLOCK):
            continue
        match = _CNINFO_TITLE_RE.match(title)
        if not match:
            continue
        mmdd = CNINFO_REPORT_KIND_MMDD.get(match.group("kind"))
        if not mmdd:
            continue
        ann = _parse_date_value(row.get("公告时间"))
        if pd.isna(ann):
            continue
        report_date = pd.Timestamp(f"{match.group('year')}-{mmdd}")
        # **硬护栏**: 定期报告不可能在报告期结束前公告。实测巨潮档案里存在被错误命名的标题
        # (如 2016-08-26 发布「2016年年度报告摘要」、2016-04-26 发布「2016年第三季度报告(修订版)」),
        # 照单全收会把 ann_date 改到报告期之前 —— 这类记录直接丢弃。
        if ann.normalize() < report_date:
            continue
        records.append(
            {"symbol": code, "report_date": report_date, "ann_date": ann, "title": title}
        )
    if not records:
        return pd.DataFrame(columns=["symbol", "report_date", "ann_date", "title"])
    frame = pd.DataFrame(records)
    # 同一报告期可能有"正文/全文/摘要"多条, 取**最早**一次 = 首次公告日
    frame = (
        frame.sort_values("ann_date")
        .drop_duplicates(subset=["symbol", "report_date"], keep="first")
        .reset_index(drop=True)
    )
    return frame
