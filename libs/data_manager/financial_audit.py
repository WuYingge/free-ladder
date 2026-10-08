"""财务数据跨源审计 (cross-source audit)

体检是"单源不变量"(自洽性), 本模块补上**跨源一致性** —— 每一列都要有一个独立仲裁者:

| 审计 | 数据源对 | 成本 | 覆盖 |
|---|---|---|---|
| ``shares`` | `balance_q.total_share`(东财 F10) vs `share_capital.total_share`(巨潮) | 离线 | 全量 |
| ``quotes`` | `balance_q.total_share` vs 行情侧 `daily_basic.float_share` | 离线 | 全量 |
| ``values`` | 本地三表 vs **新浪**三表 (核心字段逐格) | 联网 | 分层抽样 |

为什么这三条: 它们覆盖三个**结构上独立**的链路 —— 东财 F10 报表、巨潮股本、行情衍生指标、
新浪报表。任何一条链路的系统性缺陷都会被另外两条中的一条抓到 (本次的 `ann_date` 错位就是这样
被"三表互不一致 + 巨潮公告原文"抓出来的)。

结果写 ``data/financial/audit_report.json``; 体检项 ``cross_source_audit`` 会读取它并要求:
  * 一致率达标;  * 报告未过期 (``MAX_AUDIT_AGE_DAYS``)。

CLI: ``libs/scripts/audit_financials.py``
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Sequence

import pandas as pd

from config import DataPath
from data_manager.financial_schema import CANONICAL_DATE_FORMAT

#: 相对容差: 两源数值差异在此以内视为一致 (源端小数位/单位换算的常见差异)
DEFAULT_TOLERANCE = 0.005
#: 绝对容差下限 (元): 极小基数下相对容差会误判
ABS_TOLERANCE = 1.0
#: 审计报告有效期 (天): 超过则体检提示需要重跑
MAX_AUDIT_AGE_DAYS = 90
#: 新浪三表 → 本地列映射 (字段名逐源核对; 银行/保险用同一中文列名, 缺失则跳过该字段)
SINA_FIELD_MAP: dict[str, dict[str, str]] = {
    "利润表": {"营业收入": "revenue", "净利润": "net_profit"},
    "资产负债表": {"资产总计": "total_assets", "负债合计": "total_liabilities"},
    "现金流量表": {"经营活动产生的现金流量净额": "ocf_net"},
}
#: 需要判定的字段 (revenue 只记录: 东财 TOTAL_OPERATE_INCOME=营业总收入 与新浪"营业收入"口径可能不同)
GATED_VALUE_FIELDS: tuple[str, ...] = ("net_profit", "total_assets", "total_liabilities", "ocf_net")


def audit_fp() -> str:
    from data_manager.financial_manager import report_fp

    return report_fp("audit_report.json")


# ---------------------------------------------------------------------------
# 离线: 股本双源 / 行情侧
# ---------------------------------------------------------------------------


def _load(table: str) -> pd.DataFrame:
    from data_manager.financial_manager import load_table

    frame, _ = load_table(table)
    if not frame.empty and "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    return frame


def audit_share_capital(tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """`balance_q.total_share`(东财) vs `share_capital.total_share`(巨潮), 按生效日 asof 对齐。

    两个**不同信源**给出的总股本必须一致 (股本变动按生效日生效); 偏差超容差即说明其中
    一条链路有问题 (实测正常值 0.000%, 期内发生股本变动时会出现 <1% 的时点差)。
    """
    bal = _load("balance_q")
    shares = _load("share_capital")
    result: dict[str, Any] = {"source_pair": "em_f10.balance_q vs cninfo.share_capital", "tolerance": tolerance}
    if bal.empty or shares.empty:
        result["status"] = "skip"
        result["reason"] = "缺 balance_q 或 share_capital"
        return result
    left = bal.dropna(subset=["total_share", "report_date"])[["symbol", "report_date", "total_share"]]
    left = left.rename(columns={"total_share": "balance_share"}).sort_values("report_date")
    right = shares.dropna(subset=["total_share", "effective_date"])[["symbol", "effective_date", "total_share"]]
    right = right.rename(columns={"total_share": "cninfo_share"}).sort_values("effective_date")
    merged = pd.merge_asof(
        left, right, left_on="report_date", right_on="effective_date", by="symbol", direction="backward"
    ).dropna(subset=["cninfo_share"])
    if merged.empty:
        result["status"] = "skip"
        result["reason"] = "无可比行"
        return result
    merged["deviation"] = (merged["cninfo_share"] - merged["balance_share"]).abs() / merged["balance_share"].abs()
    mismatch = merged[merged["deviation"] > tolerance]
    result.update(
        {
            "status": "ok",
            "compared_rows": int(len(merged)),
            "mismatch_rows": int(len(mismatch)),
            "agreement": round(1.0 - len(mismatch) / len(merged), 6),
            "deviation_p50": float(merged["deviation"].median()),
            "deviation_p99": float(merged["deviation"].quantile(0.99)),
            "examples": mismatch.head(5)[["symbol", "report_date", "balance_share", "cninfo_share", "deviation"]]
            .assign(report_date=lambda d: d["report_date"].dt.strftime(CANONICAL_DATE_FORMAT))
            .to_dict("records"),
        }
    )
    return result


def audit_quote_float_share(tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """行情侧 `daily_basic.float_share`(流通股本) 与 `balance_q.total_share`(总股本) 的关系。

    判据两条: ① **流通股本不得超过总股本** (硬不变量, 违反即数据错);
    ② 全流通股 (float==total) 的占比 —— 数值不参与判定, 只作记录 (两者**不必然**相等)。
    """
    bal = _load("balance_q")
    result: dict[str, Any] = {"source_pair": "daily_basic.float_share vs balance_q.total_share", "tolerance": tolerance}
    if bal.empty:
        result["status"] = "skip"
        result["reason"] = "缺 balance_q"
        return result
    codes = sorted(bal["symbol"].unique())
    rows: list[dict[str, Any]] = []
    basic_dir = DataPath.DAILY_BASIC_PATH
    for code in codes:
        fp = os.path.join(basic_dir, f"{code}.csv")
        if not os.path.exists(fp):
            continue
        sub = bal[bal["symbol"] == code].dropna(subset=["total_share", "report_date"]).sort_values("report_date")
        if sub.empty:
            continue
        try:
            basic = pd.read_csv(fp, usecols=lambda c: c in ("date", "float_share"), parse_dates=["date"])
        except Exception:  # noqa: BLE001 — 单文件损坏不影响整体审计
            continue
        if basic.empty or "float_share" not in basic.columns:
            continue
        merged = pd.merge_asof(
            sub[["report_date", "total_share"]].rename(columns={"total_share": "balance_share"}),
            basic.dropna(subset=["float_share"]).rename(columns={"date": "report_date", "float_share": "quote_float"}),
            on="report_date",
            direction="backward",
        ).dropna(subset=["quote_float"])
        for _, row in merged.iterrows():
            rows.append(
                {
                    "symbol": code,
                    "report_date": row["report_date"],
                    "balance_share": row["balance_share"],
                    "quote_float": row["quote_float"],
                }
            )
    if not rows:
        result["status"] = "skip"
        result["reason"] = "无可比行 (缺 daily_basic)"
        return result
    frame = pd.DataFrame(rows)
    frame["deviation"] = (frame["quote_float"] - frame["balance_share"]).abs() / frame["balance_share"].abs()
    exceed = frame[frame["quote_float"] > frame["balance_share"] * (1 + tolerance)]
    fully_floated = frame[frame["deviation"] <= tolerance]
    result.update(
        {
            "status": "ok",
            "compared_rows": int(len(frame)),
            "float_exceeds_total": int(len(exceed)),
            "float_exceeds_total_share": round(len(exceed) / len(frame), 6),
            "fully_floated_share": round(len(fully_floated) / len(frame), 6),
            "examples": exceed.head(5)
            .assign(report_date=lambda d: d["report_date"].dt.strftime(CANONICAL_DATE_FORMAT))
            .to_dict("records"),
        }
    )
    return result


# ---------------------------------------------------------------------------
# 分层抽样
# ---------------------------------------------------------------------------


def _board_of(symbol: str) -> str:
    if symbol.startswith("688"):
        return "科创板"
    if symbol.startswith("6"):
        return "沪主板"
    if symbol.startswith("300") or symbol.startswith("301"):
        return "创业板"
    return "深主板"


def _era_of(list_date: str) -> str:
    year = pd.to_datetime(list_date, errors="coerce")
    if pd.isna(year):
        return "未知"
    value = int(year.year)
    if value < 2005:
        return "2004前"
    if value < 2015:
        return "2005-2014"
    if value < 2020:
        return "2015-2019"
    return "2020后"


def sample_symbols(n: int = 100, seed: int = 7) -> pd.DataFrame:
    """分层抽样: 板块 × 上市年代; 返回 ``symbol/board/era/list_date``。

    分层而不是简单随机: 老股票的数据年代与我们关心的问题 (历史公告日、退市、股本变动)
    强相关, 按年代分层才能保证抽到足够的老样本。
    """
    from data_manager.providers.stock_list_provider import STOCK_LIST

    records: list[dict[str, Any]] = []
    for symbol in sorted(STOCK_LIST.get_all_symbol()):
        code = str(symbol).zfill(6)
        if not code.startswith(("0", "3", "6")):
            continue
        list_date = str(STOCK_LIST.get_list_date(code) or "")
        records.append(
            {"symbol": code, "board": _board_of(code), "era": _era_of(list_date), "list_date": list_date}
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame = frame.assign(_row=range(len(frame)))
    sampled: list[pd.DataFrame] = []
    groups = list(frame.groupby(["board", "era"], sort=True))
    # 按最大余额法分配配额: 层数可能多于 n, 若"每层至少 1"会导致抽样数超过请求量 (实测踩过)
    base = n // max(len(groups), 1)
    remainder = n % max(len(groups), 1)
    for index, (_, group) in enumerate(groups):
        quota = base + (1 if index < remainder else 0)
        if quota <= 0:
            continue
        sampled.append(group.sample(n=min(len(group), quota), random_state=seed))
    out = pd.concat(sampled, ignore_index=True) if sampled else frame.head(0)
    if len(out) < n:  # 层内配额因样本不足没用满时, 从剩余股票补齐
        rest = frame[~frame["symbol"].isin(out["symbol"])]
        if len(rest):
            out = pd.concat(
                [out, rest.sample(n=min(n - len(out), len(rest)), random_state=seed)], ignore_index=True
            )
    return out.drop(columns=["_row"]).sort_values("symbol").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 联网: 新浪三表数值对拍
# ---------------------------------------------------------------------------


def _sina_symbol(code: str) -> str | None:
    if code.startswith("6"):
        return f"sh{code}"
    if code.startswith(("0", "3")):
        return f"sz{code}"
    return None  # 北交所新浪无对应接口


def _normalize_report_date(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.replace("-", "", regex=False).str.strip()
    return pd.to_datetime(text, errors="coerce", format="%Y%m%d")


#: 新浪审计的并发上限: 实测 8 线程连打 300+ 请求会被限速 (返回非 JSON → JSONDecodeError),
#: 因此该套件单独限速: 线程更少 + 每次请求间隔 + 失败退避重试一次。
SINA_AUDIT_MAX_THREADS = 3
SINA_AUDIT_INTERVAL_SECONDS = 0.4


def _fetch_sina(symbol: str) -> dict[str, pd.DataFrame]:
    import time

    import akshare as ak

    from utils.interval_utils import intervals

    out: dict[str, pd.DataFrame] = {}
    sina_code = _sina_symbol(symbol)
    if sina_code is None:
        return out
    for statement in SINA_FIELD_MAP:
        frame = None
        for attempt in (1, 2):
            try:
                frame = ak.stock_financial_report_sina(stock=sina_code, symbol=statement)
                break
            except Exception as err:  # noqa: BLE001 — 限速时退避重试一次
                if attempt == 2:
                    raise
                intervals(SINA_AUDIT_INTERVAL_SECONDS * 5)
        if frame is None or frame.empty or "报告日" not in frame.columns:
            intervals(SINA_AUDIT_INTERVAL_SECONDS)
            continue
        frame = frame.copy()
        frame["report_date"] = _normalize_report_date(frame["报告日"])
        out[statement] = frame.dropna(subset=["report_date"])
        intervals(SINA_AUDIT_INTERVAL_SECONDS)
    return out


def _compare_values(symbol: str, sina: dict[str, pd.DataFrame], local: dict[str, pd.DataFrame], tolerance: float):
    """逐格比对: 返回 ``[{field, table, report_date, local, external, agree}]``。"""
    cells: list[dict[str, Any]] = []
    table_of = {"利润表": "income_q", "资产负债表": "balance_q", "现金流量表": "cashflow_q"}
    for statement, mapping in SINA_FIELD_MAP.items():
        frame = sina.get(statement)
        if frame is None:
            continue
        table = table_of[statement]
        have = local.get(table)
        if have is None or have.empty:
            continue
        for external_col, local_col in mapping.items():
            if external_col not in frame.columns or local_col not in have.columns:
                continue
            ext = frame[["report_date", external_col]].rename(columns={external_col: "external"})
            own = have[["report_date", local_col]].rename(columns={local_col: "local"})
            merged = own.merge(ext, on="report_date", how="inner").dropna(subset=["external"])
            for _, row in merged.iterrows():
                local_value = row["local"]
                external_value = row["external"]
                if pd.isna(local_value):
                    continue
                agree = abs(local_value - external_value) <= max(
                    ABS_TOLERANCE, abs(external_value) * tolerance
                )
                cells.append(
                    {
                        "symbol": symbol,
                        "table": table,
                        "field": local_col,
                        "report_date": row["report_date"].strftime(CANONICAL_DATE_FORMAT),
                        "local": float(local_value),
                        "external": float(external_value),
                        "agree": bool(agree),
                    }
                )
    return cells


def audit_values(
    sample: Sequence[str] | None = None,
    *,
    n: int = 100,
    seed: int = 7,
    threads: int | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_periods_per_symbol: int = 60,
) -> dict[str, Any]:
    """分层抽样 + 新浪三表逐格对拍 (联网)。

    只审计**本地有值**的格子 (源端缺值为空不算不一致); 差异超容差即记为该字段不一致,
    并在结果里给出实例 (symbol/report_date/两源数值), 便于人工复核到底是哪边错。
    """
    from fetcher.financial import FINANCIAL_FETCH_THREADS

    # 新浪侧限速: 线程数被硬上限压住, 避免把审计目标打成限速 (实测踩过)
    pool = min(max(int(threads or FINANCIAL_FETCH_THREADS), 1), SINA_AUDIT_MAX_THREADS)
    symbols = list(sample) if sample else sample_symbols(n=n, seed=seed)["symbol"].tolist()
    local = {table: _load(table) for table in ("income_q", "balance_q", "cashflow_q")}
    result: dict[str, Any] = {
        "source_pair": "local vs sina (stock_financial_report_sina)",
        "tolerance": tolerance,
        "sampled_symbols": len(symbols),
        "seed": seed,
        "status": "ok",
    }
    all_cells: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def _one(code: str) -> tuple[str, list[dict[str, Any]], str | None]:
        try:
            sina = _fetch_sina(code)
        except Exception as err:  # noqa: BLE001 — 单股失败计入 failures
            return code, [], f"{type(err).__name__}: {err}"
        if not sina:
            return code, [], "新浪无数据"
        cells = _compare_values(
            code,
            sina,
            {table: frame[frame["symbol"] == code] for table, frame in local.items()},
            tolerance,
        )
        return code, cells, None

    with ThreadPoolExecutor(max_workers=pool) as executor:
        for code, cells, reason in executor.map(_one, symbols):
            if reason:
                failures.append({"symbol": code, "reason": reason})
            all_cells.extend(cells)

    if not all_cells:
        result["status"] = "skip"
        result["reason"] = "无可用对拍格子"
        result["failures"] = failures[:10]
        return result
    frame = pd.DataFrame(all_cells)
    by_field: dict[str, Any] = {}
    for field, group in frame.groupby("field"):
        mismatch = group[~group["agree"]]
        by_field[field] = {
            "compared": int(len(group)),
            "mismatch": int(len(mismatch)),
            "agreement": round(1.0 - len(mismatch) / len(group), 6),
            "gated": field in GATED_VALUE_FIELDS,
            "examples": mismatch.head(3)[["symbol", "report_date", "local", "external"]].to_dict("records"),
        }
    result["fields"] = by_field
    result["failures"] = failures[:20]
    result["failure_count"] = len(failures)
    return result


# ---------------------------------------------------------------------------
# 汇总与落盘
# ---------------------------------------------------------------------------


def run_audit(
    suites: Iterable[str] = ("shares", "quotes"),
    *,
    n: int = 100,
    seed: int = 7,
    threads: int | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    sample: Sequence[str] | None = None,
) -> dict[str, Any]:
    """按 suite 跑审计并返回报告 payload (不落盘)。"""
    selected = [item.strip() for item in suites if item and item.strip()]
    payload: dict[str, Any] = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tolerance": tolerance,
        "suites": {},
    }
    if "shares" in selected:
        payload["suites"]["shares"] = audit_share_capital(tolerance)
    if "quotes" in selected:
        payload["suites"]["quotes"] = audit_quote_float_share(tolerance)
    if "values" in selected:
        payload["suites"]["values"] = audit_values(
            sample, n=n, seed=seed, threads=threads, tolerance=tolerance
        )
    return payload


def write_audit_report(payload: dict[str, Any]) -> str:
    fp = audit_fp()
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return fp


def load_audit_report() -> dict[str, Any]:
    fp = audit_fp()
    if not os.path.exists(fp):
        return {}
    try:
        with open(fp, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001 — 报告损坏按"缺失"处理
        traceback.print_exc()
        return {}


def audit_age_days(payload: dict[str, Any] | None = None) -> float | None:
    payload = payload if payload is not None else load_audit_report()
    stamp = payload.get("generated_at")
    if not stamp:
        return None
    generated = pd.to_datetime(stamp, errors="coerce")
    if pd.isna(generated):
        return None
    return float((pd.Timestamp.now() - generated).total_seconds() / 86400.0)


__all__ = [
    "ABS_TOLERANCE",
    "DEFAULT_TOLERANCE",
    "GATED_VALUE_FIELDS",
    "MAX_AUDIT_AGE_DAYS",
    "SINA_AUDIT_INTERVAL_SECONDS",
    "SINA_AUDIT_MAX_THREADS",
    "SINA_FIELD_MAP",
    "audit_age_days",
    "audit_fp",
    "audit_quote_float_share",
    "audit_share_capital",
    "audit_values",
    "load_audit_report",
    "run_audit",
    "sample_symbols",
    "write_audit_report",
]
