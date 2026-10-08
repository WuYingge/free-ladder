"""财报/财务数据落盘、加载与日更编排 (Financial Data Manager)

职责:
  * 抓取编排: 逐股表 (F10 三张报表 / 巨潮股本变动) 与逐期表 (预告/分红/股东户数);
  * 落盘: ``data/financial/<表>.csv`` 宽表, 去重 keep="first" (首次公告版本优先,
    更正行不覆盖首版), 原子写 (临时文件 + os.replace), 重跑幂等 (日更必需);
  * 加载: ``load_table`` / ``load_financial_events`` 带行级门禁 ——
    ann_date 缺失、``ann_date < report_date``、超出该股行情区间的行一律排除并计数,
    绝不静默进入回测 (这是"两条时间轴准确性"的最后一道闸);
  * notebook 接口: ``batch_check_financials_updated`` / ``update_financials``
    (与 ``daily_basic_manager`` 同款形状, 供 ``notebooks/dailyUpdate.ipynb`` 直接调用)。

时间口径:
  * ``ann_date`` = 首次公告日 (已知时刻), 当天即可用于交易决策;
  * ``effective_trade_date`` = 公告日映射到的首个交易日 (实测约 20% 公告发布在
    非交易日, 属正常); T+1 成交由策略层再加一个交易日。
"""

from __future__ import annotations

import datetime
import json
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from config import DataPath
from data_manager.financial_schema import (
    BASIS_FIRST_REPORTED,
    BASIS_LATEST_VERSION,
    BASIS_MIXED,
    BASIS_REVISED,
    CANONICAL_DATE_FORMAT,
    FINANCIAL_TABLES,
    FinancialTableSpec,
    TableQuality,
    check_column_order,
    fiscal_year_start,
    is_canonical_date_text,
    normalize_table,
    parse_date_column,
)
from fetcher.financial import (
    FINANCIAL_FETCH_THREADS,
    available_periods,
    fetch_statement_symbols,
    get_period_dividend_rows,
    get_period_forecast_rows,
    get_period_holder_num_rows,
    get_share_change_rows,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 逐股抓取时的进程内线程数 (HTTP 密集, 默认与 fetcher 一致)
FINANCIAL_FETCH_POOL_SIZE = FINANCIAL_FETCH_THREADS
#: 一张表被视为"当日已更新"的新鲜度阈值 (小时)
FRESH_HOURS = 20.0
#: 一份财报在披露窗口结束后仍未入表的告警阈值 (天)
STALE_DAYS = 45
#: 披露高峰月 (年报/一季报 4 月, 半年报 8 月, 三季报 10 月)
DISCLOSURE_PEAK_MONTHS = (4, 8, 10)
#: 逐股表: 默认取最近 N 个报告期 (日更), None = 全历史 (回填)
DEFAULT_STATEMENT_PERIODS = 2
#: 逐期表默认回溯年数 (holder_num 自 2013 起; 预告自 2008 起, 取 2014 与行情对齐)
DEFAULT_PERIOD_START_YEAR = 2014
#: 表 key → 抓取粒度
SYMBOL_GRANULARITY_TABLES = ("income_q", "balance_q", "cashflow_q", "share_capital")
PERIOD_GRANULARITY_TABLES = ("forecast", "dividend", "holder_num")

STATEMENT_TABLE_TO_NAME = {
    "income_q": "income",
    "balance_q": "balance",
    "cashflow_q": "cashflow",
}


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------


def get_financial_dir() -> str:
    os.makedirs(DataPath.FINANCIAL_DIR, exist_ok=True)
    return DataPath.FINANCIAL_DIR


def table_fp(table: str) -> str:
    spec = get_spec(table)
    return os.path.join(get_financial_dir(), spec.filename)


def report_fp(name: str) -> str:
    return os.path.join(get_financial_dir(), name)


def get_spec(table: str) -> FinancialTableSpec:
    spec = FINANCIAL_TABLES.get(table)
    if spec is None:
        raise ValueError(f"未知财务表: {table!r}; 可用: {sorted(FINANCIAL_TABLES)}")
    return spec


def list_tables() -> list[str]:
    return list(FINANCIAL_TABLES)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UpsertStat:
    table: str
    rows_before: int = 0
    rows_after: int = 0
    rows_added: int = 0
    rows_dropped_duplicate: int = 0
    quality: TableQuality | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_added": self.rows_added,
            "rows_dropped_duplicate": self.rows_dropped_duplicate,
        }


@dataclass(slots=True)
class DropLog:
    """加载期被门禁排除的行 (绝不静默丢弃)。"""

    table: str
    missing_ann_date: int = 0
    ann_before_report: int = 0
    out_of_quote_range: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.missing_ann_date + self.ann_before_report + self.out_of_quote_range

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "missing_ann_date": self.missing_ann_date,
            "ann_before_report": self.ann_before_report,
            "out_of_quote_range": self.out_of_quote_range,
            "total": self.total,
            "examples": self.examples[:10],
        }


# ---------------------------------------------------------------------------
# 原始帧 → 落盘帧
# ---------------------------------------------------------------------------


def _apply_basis(frame: pd.DataFrame, spec: FinancialTableSpec) -> pd.DataFrame:
    """补 update_date/basis 两个方法论文档列。

    basis=first_reported 表示该行数值即首次公告版本; revised 表示数值在首次公告后
    被追溯调整过 (无法还原首版值, 必须显式标注); latest_version 表示该表无版本概念。
    逐行判定, 不做股票级聚合 —— 同一股票内混装 (部分报告期被修订) 由健康报告的
    "混装股票数" 指标单独披露。
    """
    out = frame.copy()
    source_update = (
        out["_source_update_date"]
        if "_source_update_date" in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    out["update_date"] = pd.to_datetime(source_update, errors="coerce")
    if spec.versioned:
        notice = pd.to_datetime(out["ann_date"], errors="coerce")
        updated = out["update_date"]
        revision_known = updated.notna() & notice.notna()
        revised = revision_known & (updated > notice)
        basis = pd.Series(BASIS_FIRST_REPORTED, index=out.index, dtype="string")
        basis.loc[revised] = BASIS_REVISED
        out["basis"] = basis
    else:
        out["basis"] = BASIS_LATEST_VERSION
    out["update_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


def mixed_basis_symbols(table: str) -> list[str]:
    """同一股票内既有 first_reported 又有 revised 行的股票 (数值链不纯)。"""
    frame, _ = load_table(table)
    if frame.empty or "basis" not in frame.columns:
        return []
    counts = frame.groupby(frame["symbol"].astype(str))["basis"].nunique()
    return sorted(counts[counts > 1].index.tolist())


def prepare_for_storage(raw: pd.DataFrame, table: str) -> tuple[pd.DataFrame, TableQuality]:
    """信源原始帧 → (落盘帧, 质量报告)。含 basis/update_time 补列与 schema 规范化。"""
    spec = get_spec(table)
    if raw is None or raw.empty:
        empty = pd.DataFrame(columns=list(spec.columns))
        return empty, TableQuality(table=table, filename=spec.filename, n_rows=0)
    decorated = _apply_basis(raw, spec)
    normalized, quality = normalize_table(decorated, spec)
    return normalized, quality


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------


def _existing_frame(table: str) -> tuple[pd.DataFrame, TableQuality]:
    spec = get_spec(table)
    fp = table_fp(table)
    if not os.path.exists(fp):
        return pd.DataFrame(columns=list(spec.columns)), TableQuality(
            table=table, filename=spec.filename, n_rows=0
        )
    raw = pd.read_csv(fp, dtype={"symbol": str})
    order_ok, missing, extra = check_column_order(list(raw.columns), spec)
    quality = TableQuality(
        table=table,
        filename=spec.filename,
        n_rows=int(len(raw)),
        schema_ok=not missing,
        missing_columns=missing,
        unexpected_columns=extra,
        column_order_ok=order_ok,
    )
    if missing:
        raise ValueError(
            f"{spec.filename}: 缺少必需列 {missing}; 期望列序 {list(spec.columns)}"
        )
    return raw[list(spec.columns)], quality


def _coerce_dates(frame: pd.DataFrame, spec: FinancialTableSpec) -> pd.DataFrame:
    """把日期列统一为 datetime64, 保证新旧行拼接后 dedup 键可比较。

    ``strict=True``: 非空但不可解析的日期直接抛错。原因是"解析成 NaT"等价于把该行的
    公告日清空, PIT 门禁随后会整行剔除 —— 静默丢数据的代价远大于让更新任务失败一次。
    """
    out = frame.copy()
    for col in spec.date_columns:
        if col in out.columns:
            out[col] = parse_date_column(
                out[col], strict=True, context=f"{spec.key}.{col}"
            )
    return out


def _canonical_for_write(frame: pd.DataFrame, spec: FinancialTableSpec) -> pd.DataFrame:
    """写盘前把日期列钉成 ``datetime64`` —— **别删这个函数**。

    ``to_csv`` 只在列 dtype 为 datetime64 时才应用 ``date_format``; 一旦列退化成 object
    (``pd.concat`` 把"字符串列"与"datetime64 列"合并时必然如此), pandas 会写
    ``str(Timestamp)`` 即 ``2021-04-20 00:00:00``, 落盘格式即被污染; 而下一次读取若用朴素
    ``to_datetime`` 解析这种混格式列又会静默判成 NaT (实测 12.7 万行公告日因此变 NaT)。
    两道防线合起来才闭环: 这里保证写出的永远是规范格式, 读取端 ``parse_date_column``
    保证混格式也能正确解析。
    """
    out = _coerce_dates(frame, spec)
    for col in spec.date_columns:
        if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
            raise ValueError(
                f"{spec.table}.{col}: 写盘前未能规范为 datetime64 (dtype={out[col].dtype})"
            )
    return out


def _reapply_corrections(frame: pd.DataFrame, spec: FinancialTableSpec) -> pd.DataFrame:
    """写盘前**重放溯源修正** —— 保护已仲裁的单元格不被重抓覆盖。

    **为什么必须有**: 去重规则是"字段填充数多者优先 → 同分取新抓行"。若某格已被人工仲裁修正
    (如用巨潮公告原文改掉东财"错位一年"的 `ann_date`), 之后跑一次全量重抓, 新抓行与修正行
    填充数相同 → **新抓行胜出, 仲裁结果被静默冲掉**。因此修正流水里的值在写盘前重新覆盖一次;
    要撤销修正必须显式调用 ``revert_corrections`` (它同样留痕)。
    """
    from data_manager.financial_provenance import load_corrections

    log = load_corrections(spec.key)
    if log.empty:
        return frame
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    keys = [col for col in spec.dedup_key if col in out.columns]
    if keys:
        out = out.set_index(keys, drop=False)
    replayed = 0
    latest = log.drop_duplicates(subset=["symbol", "report_date", "forecast_indicator_cn", "column"], keep="last")
    for _, row in latest.iterrows():
        key = (row["symbol"], pd.Timestamp(row["report_date"]))
        if "forecast_indicator_cn" in keys:
            key = (*key, str(row.get("forecast_indicator_cn") or ""))
        if key not in out.index:
            continue
        column = row["column"]
        if column not in out.columns:
            continue
        raw = row["new_value"]
        if raw == "" or raw is None:
            continue
        current = out.loc[key, column]
        if isinstance(current, pd.Series):
            current = current.iloc[0]
        value = _coerce_like_static(current, raw)
        if column in spec.date_columns:
            value = parse_date_column(pd.Series([value]), context=f"{spec.key}.{column}").iloc[0]
        out.loc[key, column] = value
        replayed += 1
    if replayed and keys:
        out = out.reset_index(drop=True)
    elif keys:
        out = out.reset_index(drop=True)
    return out


def _coerce_like_static(old_value: Any, new_value: Any) -> Any:
    """把流水里的字符串还原成主表该列的取值类型 (日期 → Timestamp; 数值 → float)。"""
    if isinstance(old_value, pd.Timestamp):
        return pd.to_datetime(new_value, errors="coerce", format="mixed")
    if isinstance(old_value, (int, float)):
        return pd.to_numeric(new_value, errors="coerce")
    return str(new_value)


def _fetched_with_data(frame: pd.DataFrame) -> set[str]:
    """本次抓取中确实产生行的股票代码 (用于 replace 时避免抹掉抓取失败的股票)。"""
    if frame.empty or "symbol" not in frame.columns:
        return set()
    return set(frame["symbol"].astype(str).str.zfill(6))


def upsert_table(table: str, new_frame: pd.DataFrame, replace: bool = False) -> UpsertStat:
    """把新抓取的行并入落盘 CSV (幂等, 原子写).

    去重语义 (**踩过的坑, 别改回去**):
      * 源端对每个 ``(symbol, report_date)`` **只提供一行版本** (实测验证: 更正没有独立
        版本行), 因此合并天然以报告期为键; ``ann_date`` **不进键** —— 否则同一报告期的
        "缺公告日的旧行"与"带公告日的新行"会被当成两条并存, 而 ``keep="first"`` 会让
        **陈旧的不完整行胜出** (实测: 33k 行 ann_date 缺失本可自愈却没被覆盖)。
      * 采纳顺序: 字段填充数多者优先 → 同分时取**后写入**的新抓行 (让重跑能修复旧数据);
        ``forecast`` 额外按 ``forecast_indicator_cn`` 区分归母/扣非两种并列口径。
    最后写入恒为最新的抓取结果, 因此重跑幂等且具备自愈能力。

    :param replace: True = **整表替换** (只保留本次抓取结果), 用于全量重抓。
        False = 与既有文件合并 (日更增量)。
        **为什么需要 replace**: 合并时"字段更全的陈旧行"可能赢过新行 —— 实测
        按全部列统计填充数时, 缺 ``ann_date`` 的历史行因其他列更全而胜出, 导致
        33k 行 ``ann_date`` 缺失无法自愈。全量重抓的正确做法是整表替换而非合并。
    """
    spec = get_spec(table)
    old, quality = _existing_frame(table)
    stat = UpsertStat(table=table, rows_before=int(len(old)), quality=quality)
    fp = table_fp(table)

    if new_frame is None or new_frame.empty:
        if not os.path.exists(fp):
            # 首次抓取为空 → 仍落一个只有表头的文件, 便于后续判断"已尝试"
            pd.DataFrame(columns=list(spec.columns)).to_csv(fp, index=False, encoding="utf-8-sig")
        stat.rows_after = stat.rows_before
        return stat

    new_part = _coerce_dates(new_frame[list(spec.columns)], spec)
    old_part = _coerce_dates(old, spec)
    if replace:
        # **按股票子集替换** (不是整表替换): 丢弃"本次确实抓到的股票"的旧行, 其余股票原样保留。
        # 原因: ① 增量运行时若整表替换, 未抓取的股票会被抹掉 (实测踩过: --replace --symbols 只传
        # 3 只股票时整表从 318,694 行塌成 246 行); ② 抓取失败的股票不应丢失既有数据。
        fetched_symbols = _fetched_with_data(new_part)
        kept_old = old_part[
            ~old_part["symbol"].astype(str).str.zfill(6).isin(fetched_symbols)
        ]
        combined = pd.concat([kept_old, new_part], ignore_index=True)
        combined["_is_new"] = [0] * len(kept_old) + [1] * len(new_part)
    else:
        combined = pd.concat([old_part, new_part], ignore_index=True)
        # 来源标记: 新抓行 = 1 (同分优先), 旧行 = 0
        combined["_is_new"] = [0] * len(old_part) + [1] * len(new_part)

    key = [col for col in spec.dedup_key if col in combined.columns]
    if key:
        combined["_fill_score"] = combined.notna().sum(axis=1)
        before = len(combined)
        combined = combined.sort_values(
            ["_fill_score", "_is_new"], ascending=[False, False], kind="stable"
        )
        combined = combined.drop_duplicates(subset=key, keep="first")
        stat.rows_dropped_duplicate = int(before - len(combined))
        combined = combined.drop(columns=["_fill_score"])

    combined = combined.drop(columns=["_is_new"])
    sort_columns = [col for col in ("symbol", "ann_date", "report_date") if col in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="stable")
    combined = combined.reset_index(drop=True)
    stat.rows_after = int(len(combined))
    stat.rows_added = int(stat.rows_after - stat.rows_before)

    tmp_fp = fp + ".tmp"
    # 写盘前再规范化一次: concat 可能把 datetime64 列与对象列混成 object,
    # 那样 date_format 失效、会写出 "2021-04-20 00:00:00" 污染下一次解析 (见 _canonical_for_write)
    combined = _canonical_for_write(combined, spec)
    # 溯源修正优先: 防止全量重抓把已仲裁的单元格冲掉 (见 _reapply_corrections)
    combined = _reapply_corrections(combined, spec)
    combined.to_csv(
        tmp_fp, index=False, encoding="utf-8-sig", date_format=CANONICAL_DATE_FORMAT
    )
    os.replace(tmp_fp, fp)
    return stat


def load_table(table: str) -> tuple[pd.DataFrame, TableQuality]:
    """读取落盘表; 缺文件返回空帧 (不抛错), 列不合法抛 ValueError。"""
    frame, quality = _existing_frame(table)
    spec = get_spec(table)
    for col in spec.date_columns:
        if col in frame.columns:
            # 必须走 parse_date_column (混格式列用朴素 to_datetime 会静默变 NaT)
            frame[col] = parse_date_column(frame[col], context=f"{table}.{col}")
    return frame, quality


def raw_date_format_issues(table: str, limit: int = 5) -> dict[str, Any]:
    """只读检查落盘 CSV 的日期列文本是否为规范 ``YYYY-MM-DD`` (不回写)。

    "格式被污染"必须能被体检独立发现: 落盘一旦出现 ``2021-04-20 00:00:00``, 说明写入
    路径曾发生 dtype 退化 (object + ``str(Timestamp)``); 而朴素解析器读这种列会把它当
    NaT, 从而静默丢公告日。
    """
    spec = get_spec(table)
    fp = table_fp(table)
    result: dict[str, Any] = {"table": table, "columns": {}, "total": 0, "examples": []}
    if not os.path.exists(fp):
        return result
    cols = [col for col in spec.date_columns]
    if not cols:
        return result
    raw = pd.read_csv(fp, usecols=cols, dtype="string")
    for col in cols:
        series = raw[col]
        bad = ~series.map(is_canonical_date_text)
        count = int(bad.sum())
        if not count:
            continue
        result["columns"][col] = count
        result["total"] += count
        for value in series[bad].head(limit).tolist():
            result["examples"].append(f"{col}={value}")
    result["examples"] = result["examples"][:limit]
    return result


def load_all_tables() -> dict[str, tuple[pd.DataFrame, TableQuality]]:
    return {table: load_table(table) for table in FINANCIAL_TABLES}


# ---------------------------------------------------------------------------
# 事件帧加载 (带行级门禁)
# ---------------------------------------------------------------------------


def _quote_bounds(symbol: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """该股行情首末日 (停牌/退市即末值靠前), 用于排除越界公告。"""
    fp = os.path.join(DataPath.STOCK_PATH, f"{str(symbol).zfill(6)}.csv")
    if not os.path.exists(fp):
        return None, None
    try:
        dates = pd.read_csv(fp, usecols=["date"], parse_dates=["date"])["date"]
    except Exception:  # noqa: BLE001 — 行情文件损坏时不做区间门禁
        return None, None
    if dates.empty:
        return None, None
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def load_financial_events(
    symbol: str,
    basis: str | None = None,
    tables: Sequence[str] | None = None,
    apply_quote_bounds: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, DropLog]]:
    """单只股票的财务事件帧 (index = 事件日, 升序)。

    :param basis: 口径过滤; **默认 None = 全部行** (PIT 正确性由 ``ann_date`` 门禁保证)。
        ``first_reported`` 是**可选严格模式**: 只保留首次公告版本 —— 实测 2010-2024 年报告期
        只有 3%~13% 的行是 ``first_reported`` (源端 ``UPDATE_DATE`` 普遍晚于 ``NOTICE_DATE``),
        若把它当默认会让加载路径**几乎取不到历史财务数据** (实测 2020-06 区间 0/20 天有值,
        2015 年区间整段为空), 故改为显式选择。
    :param apply_quote_bounds: 排除公告日早于行情首日 / 晚于行情末日的行
    :return: ``({table: DataFrame}, {table: DropLog})``
    """
    code = str(symbol).zfill(6)
    selected = list(tables) if tables is not None else list(FINANCIAL_TABLES)
    first_quote, last_quote = _quote_bounds(code) if apply_quote_bounds else (None, None)

    frames: dict[str, pd.DataFrame] = {}
    drops: dict[str, DropLog] = {}
    for table in selected:
        spec = get_spec(table)
        frame, _ = load_table(table)
        log = DropLog(table=table)
        if frame.empty or "symbol" not in frame.columns:
            frames[table] = pd.DataFrame(columns=list(spec.columns))
            drops[table] = log
            continue
        subset = frame[frame["symbol"].astype(str).str.zfill(6) == code].copy()
        if basis is not None and "basis" in subset.columns:
            subset = subset[subset["basis"].astype(str).str.startswith(basis)]
        if subset.empty:
            frames[table] = pd.DataFrame(columns=list(spec.columns))
            drops[table] = log
            continue

        event_col = spec.event_date_column or "ann_date"
        event_ts = pd.to_datetime(subset.get(event_col), errors="coerce") if event_col in subset.columns else pd.Series(pd.NaT, index=subset.index)
        missing = event_ts.isna()
        log.missing_ann_date = int(missing.sum())
        if log.missing_ann_date and len(log.examples) < 10:
            log.examples += [f"{code} {event_col} 缺失 @row {idx}" for idx in subset.index[missing][:10].tolist()]

        valid = subset[~missing].copy()
        event_ts = event_ts[~missing]

        if "report_date" in valid.columns:
            report_ts = pd.to_datetime(valid["report_date"], errors="coerce")
            # 正式财报: ann_date 不得早于报告期末;
            # 业绩预告/快报: 允许期间结束前发布, 下界 = 会计年度起始日
            lower_bound = (
                report_ts.map(fiscal_year_start) if spec.pre_period_allowed else report_ts
            )
            early = event_ts < lower_bound
            log.ann_before_report = int(early.sum())
            if log.ann_before_report and len(log.examples) < 10:
                log.examples += [
                    f"{code} ann_date 早于可见下界: {valid.loc[idx, event_col]} < "
                    f"{lower_bound.loc[idx]}"
                    for idx in valid.index[early][:10].tolist()
                ]
            valid = valid[~early]
            event_ts = event_ts[~early]

        if apply_quote_bounds and (first_quote is not None or last_quote is not None):
            outside = pd.Series(False, index=valid.index)
            if first_quote is not None:
                outside |= event_ts < first_quote
            if last_quote is not None:
                outside |= event_ts > last_quote
            log.out_of_quote_range = int(outside.sum())
            if log.out_of_quote_range and len(log.examples) < 10:
                log.examples += [
                    f"{code} 公告日越界: {valid.loc[idx, event_col]}" for idx in valid.index[outside][:10].tolist()
                ]
            valid = valid[~outside]
            event_ts = event_ts[~outside]

        valid = valid.drop(columns=["symbol"], errors="ignore")
        valid.index = pd.DatetimeIndex(event_ts.values, name="ann_date")
        valid = valid.sort_index()
        # 同一事件日多行 (如预告多指标) 保留末行, 与字典注册表的 merge_asof 一对一要求一致
        valid = valid[~valid.index.duplicated(keep="last")]
        frames[table] = valid
        drops[table] = log
    return frames, drops


def load_symbol_events(symbol: str, table: str, basis: str | None = BASIS_FIRST_REPORTED) -> pd.DataFrame:
    """单表单股事件帧 (便捷入口)。"""
    frames, _ = load_financial_events(symbol, basis=basis, tables=[table])
    return frames.get(table, pd.DataFrame())


# ---------------------------------------------------------------------------
# 交易日历
# ---------------------------------------------------------------------------


_CALENDAR_CACHE: list[pd.Timestamp] | None = None


def trading_calendar() -> list[pd.Timestamp]:
    """交易日历 (data/const/calandar_df.csv); 读取失败返回空列表。"""
    global _CALENDAR_CACHE
    if _CALENDAR_CACHE is None:
        try:
            frame = pd.read_csv(DataPath.CALANDAR_DF, index_col=0, parse_dates=["trade_date"])
            _CALENDAR_CACHE = sorted(pd.Timestamp(x) for x in frame["trade_date"].tolist())
        except Exception:  # noqa: BLE001
            _CALENDAR_CACHE = []
    return _CALENDAR_CACHE


def effective_trade_date(
    ann_date: object,
    calendar: Sequence[pd.Timestamp] | None = None,
) -> pd.Timestamp:
    """公告日 → 首个 ``>= ann_date`` 的交易日 (非交易日公告顺延到下一交易日)。"""
    ts = pd.Timestamp(pd.to_datetime(ann_date, errors="coerce"))
    if pd.isna(ts):
        return pd.NaT
    days = list(calendar) if calendar is not None else trading_calendar()
    if not days:
        return ts
    index = pd.Series(days).searchsorted(ts, side="left")
    if index >= len(days):
        return pd.NaT
    return pd.Timestamp(days[index])


# ---------------------------------------------------------------------------
# 抓取编排
# ---------------------------------------------------------------------------


def resolve_symbols(
    symbols: Sequence[str] | None = None,
    include_delisted: bool = True,
) -> list[str]:
    """默认代码集 = data/stock_data 文件集 (含退市股历史)。

    :param include_delisted: False 时仅保留 ``STOCK_LIST`` 里在市 (无退市日) 的代码
    """
    from data_manager.daily_basic_manager import list_stock_data_symbols

    if symbols is not None:
        return sorted({str(code).zfill(6) for code in symbols})
    base = {str(code).zfill(6) for code in list_stock_data_symbols()}
    if include_delisted:
        return sorted(base)
    from data_manager.providers.stock_list_provider import STOCK_LIST

    active = {str(code).zfill(6) for code in STOCK_LIST.get_all_symbol()}
    return sorted(base & active)


def fetch_symbol_table(
    table: str,
    symbols: Sequence[str],
    start_year: int = DEFAULT_PERIOD_START_YEAR,
    periods: int | None = DEFAULT_STATEMENT_PERIODS,
    threads: int | None = None,
    mode: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """逐股表抓取 (income_q / balance_q / cashflow_q / share_capital)。"""
    if table in STATEMENT_TABLE_TO_NAME:
        merged, failures = fetch_statement_symbols(
            symbols,
            periods=periods,
            statements=(STATEMENT_TABLE_TO_NAME[table],),
            threads=threads,
            mode=mode,
        )
        return merged.get(STATEMENT_TABLE_TO_NAME[table], pd.DataFrame()), failures

    if table == "share_capital":
        pool_size = max(int(threads or FINANCIAL_FETCH_POOL_SIZE), 1)
        frames: list[pd.DataFrame] = []
        failures: list[tuple[str, str]] = []
        from concurrent.futures import ThreadPoolExecutor

        def _one(code: str) -> tuple[str, pd.DataFrame | None, str | None]:
            try:
                frame = get_share_change_rows(
                    code,
                    start_date=f"{int(start_year)}-01-01",
                    end_date=end_date,
                    mode=mode,
                )
            except Exception as err:  # noqa: BLE001
                return code, None, f"{type(err).__name__}: {err}"
            if frame is None or frame.empty:
                return code, None, "取数为空"
            return code, frame, None

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            for code, frame, reason in executor.map(_one, list(symbols)):
                if reason is not None:
                    failures.append((code, reason))
                if frame is not None and not frame.empty:
                    frames.append(frame)
        merged_share = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return merged_share, failures

    raise ValueError(f"{table} 不是逐股表")


def fetch_period_table(
    table: str,
    start_year: int = DEFAULT_PERIOD_START_YEAR,
    end_year: int | None = None,
    mode: str | None = None,
    today: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """逐期表抓取 (forecast / dividend / holder_num)。返回 (帧, 失败报告期列表)。"""
    period_fetcher = {
        "forecast": get_period_forecast_rows,
        "dividend": get_period_dividend_rows,
        "holder_num": get_period_holder_num_rows,
    }[table]
    periods = available_periods(start_year, end_year, today=today)
    frames: list[pd.DataFrame] = []
    failed_periods: list[str] = []
    for period in periods:
        try:
            frame = period_fetcher(period, mode=mode)
        except Exception as err:  # noqa: BLE001 — 单期失败不影响其他期
            print(f"[financial] {table} {period} 取数失败: {type(err).__name__}: {err}")
            failed_periods.append(period)
            continue
        if frame is not None and not frame.empty:
            frames.append(frame)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return merged, failed_periods


# ---------------------------------------------------------------------------
# 日更接口 (notebook)
# ---------------------------------------------------------------------------


def _freshness(
    table: str,
    frame: pd.DataFrame,
    target_date: datetime.date,
) -> tuple[pd.Timestamp | None, str]:
    """返回 (last_update_time, 原因); 原因以 "当日已更新" 开头表示无需更新。

    判定顺序: 空表 → 缺 update_time → 距上次更新 < FRESH_HOURS → 披露缺口检查。
    """
    if frame.empty:
        return None, "表为空 (未抓取或无数据)"
    if "update_time" not in frame.columns:
        return None, "缺少 update_time 列"
    updates = pd.to_datetime(frame["update_time"], errors="coerce")
    last = updates.max()
    if pd.isna(last):
        return None, "update_time 不可解析"
    last_ts = pd.Timestamp(last)
    hours = (pd.Timestamp.now() - last_ts).total_seconds() / 3600.0
    if hours < FRESH_HOURS:
        return last_ts, f"当日已更新 (距今 {hours:.1f}h)"
    # 披露缺口: 最新已到期的报告期在表内完全缺失, 且已超过 STALE_DAYS
    due_periods = available_periods(DEFAULT_PERIOD_START_YEAR, target_date.year)
    if due_periods and "report_date" in frame.columns:
        reports = pd.to_datetime(frame["report_date"], errors="coerce").dropna()
        newest = pd.Timestamp(due_periods[-1])
        if not reports.empty and reports.max() < newest:
            stale_since = newest + pd.Timedelta(days=STALE_DAYS)
            if pd.Timestamp(target_date) >= stale_since:
                return last_ts, f"缺 {newest.date()} 报告期数据 (已过 {STALE_DAYS} 天)"
    return last_ts, f"超过 {FRESH_HOURS:.0f}h 未更新"


def batch_check_financials_updated(
    target_date: str | datetime.date | None = None,
    tables: Sequence[str] | None = None,
) -> pd.DataFrame:
    """财务表新鲜度体检 (不联网; 与 ``batch_check_daily_basic_updated`` 同款形状)。

    :return: DataFrame[table/filename/exists/rows/row_delta/last_ann_date/
             last_period/last_update_time/target_date/is_updated/reason]
    """
    resolved_date = (
        datetime.date.today() if target_date is None else pd.to_datetime(target_date).date()
    )
    selected = list(tables) if tables is not None else list(FINANCIAL_TABLES)
    rows: list[dict[str, object]] = []
    for table in selected:
        spec = get_spec(table)
        fp = table_fp(table)
        exists = os.path.exists(fp)
        frame: pd.DataFrame = pd.DataFrame()
        if exists:
            try:
                frame, _ = load_table(table)
            except Exception as err:  # noqa: BLE001 — 损坏文件按"需更新"处理并给出原因
                frame = pd.DataFrame()
                rows.append(
                    {
                        "table": table,
                        "filename": spec.filename,
                        "exists": True,
                        "rows": 0,
                        "row_delta": 0,
                        "last_ann_date": None,
                        "last_period": None,
                        "last_update_time": None,
                        "target_date": resolved_date,
                        "is_updated": False,
                        "reason": f"读取失败: {err}",
                    }
                )
                continue
        last_update, reason = (
            _freshness(table, frame, resolved_date) if exists else (None, "文件不存在")
        )
        last_ann = None
        last_period = None
        if not frame.empty:
            if "ann_date" in frame.columns:
                ann = pd.to_datetime(frame["ann_date"], errors="coerce").dropna()
                last_ann = ann.max().date() if not ann.empty else None
            if "report_date" in frame.columns:
                rep = pd.to_datetime(frame["report_date"], errors="coerce").dropna()
                last_period = rep.max().date() if not rep.empty else None
        rows.append(
            {
                "table": table,
                "filename": spec.filename,
                "exists": exists,
                "rows": int(len(frame)),
                "row_delta": 0,
                "last_ann_date": last_ann,
                "last_period": last_period,
                "last_update_time": last_update,
                "target_date": resolved_date,
                "is_updated": bool(reason.startswith("当日已更新")),
                "reason": reason,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["is_updated"] = result["is_updated"].astype(bool)
        result = result.sort_values(["is_updated", "table"], ascending=[True, True]).reset_index(drop=True)
    return result


def plan_basis_recompute(table: str, ann_date_map: pd.DataFrame) -> pd.DataFrame:
    """按 ``update_date > ann_date`` 规则重算 ``basis``, 返回需要变更的行 (column='basis')。

    **为什么必须做**: `basis` 的定义是"数值是否在首次公告后被追溯调整过", 判据为
    ``update_date > ann_date``。公告日被仲裁改早后, 原本 first_reported 的行会变成 revised。

    :param ann_date_map: 必须含 ``symbol, report_date, ann_date_new`` —— **用修正后的公告日**
        比对 (踩过的坑: 拿加载时的旧 ann_date 去比, 结果永远是"无需变更")。
    """
    spec = get_spec(table)
    frame, _ = load_table(table)
    if frame.empty or "basis" not in spec.columns:
        return pd.DataFrame(columns=["symbol", "report_date", "column", "old_value", "new_value"])
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["ann_date"] = parse_date_column(frame["ann_date"], context=f"{table}.ann_date")
    frame["report_date"] = parse_date_column(frame["report_date"], context=f"{table}.report_date")
    if "update_date" in frame.columns:
        frame["update_date"] = parse_date_column(frame["update_date"], context=f"{table}.update_date")
    else:
        frame["update_date"] = pd.NaT
    if ann_date_map is None or ann_date_map.empty:
        return pd.DataFrame(columns=["symbol", "report_date", "column", "old_value", "new_value"])
    wanted = ann_date_map.copy()
    wanted["symbol"] = wanted["symbol"].astype(str).str.zfill(6)
    wanted["report_date"] = parse_date_column(wanted["report_date"], context="keys.report_date")
    wanted["ann_date_new"] = parse_date_column(wanted["ann_date_new"], context="keys.ann_date_new")
    key_cols = [col for col in spec.dedup_key if col in wanted.columns and col in frame.columns]
    frame = frame.merge(wanted[key_cols + ["ann_date_new"]].drop_duplicates(), on=key_cols, how="inner")
    expected = pd.Series(BASIS_FIRST_REPORTED, index=frame.index, dtype="string")
    # 用修正后的公告日判定 (frame 里的 ann_date 还是旧值, 不能拿来比)
    effective_ann = frame["ann_date_new"].fillna(frame["ann_date"])
    revised = frame["update_date"].notna() & effective_ann.notna() & (frame["update_date"] > effective_ann)
    expected.loc[revised] = BASIS_REVISED
    changed = frame["basis"].astype("string") != expected
    if not changed.any():
        return pd.DataFrame(columns=["symbol", "report_date", "column", "old_value", "new_value"])
    out = frame.loc[changed, [col for col in spec.dedup_key if col in frame.columns]].copy()
    out["old_value"] = frame.loc[changed, "basis"].astype("string")
    out["new_value"] = expected.loc[changed]
    out["column"] = "basis"
    return out.reset_index(drop=True)


def plan_ann_date_repairs(
    tables: Sequence[str] = ("income_q", "balance_q", "cashflow_q"),
    start_year: int = 2010,
    symbols: Sequence[str] | None = None,
    threads: int | None = None,
    mode: str | None = None,
    min_gap_days: int = 30,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """用**巨潮公告原文**仲裁 `ann_date`, 产出修正计划 (不改数据)。

    背景: 东财 F10 对部分报告期返回的 `NOTICE_DATE` 实为**下一期同日历期**的公告日
    ("错位一年"; 2016-2020 段 income_q 达 7.7~10.7%)。巨潮公告标题自带年份与报告类型,
    是唯一能判断"这一天发的是哪一期报告"的仲裁者。

    **只修"偏晚"的行** (``min_gap_days`` 以上): 落盘值晚于巨潮给出的首次公告日 → 采纳巨潮;
    巨潮日期更晚或一致 → 一律不动 (避免把"真延迟披露"改成更早的假日期, 也避免巨潮档案缺失时误改)。
    """
    from concurrent.futures import ThreadPoolExecutor

    from fetcher.financial import get_periodic_report_announcements

    spec_tables = [table for table in tables if table in STATEMENT_TABLE_TO_NAME]
    codes = list(symbols) if symbols else resolve_symbols(None, include_delisted=True)
    pool = max(int(threads or FINANCIAL_FETCH_POOL_SIZE), 1)
    start = f"{int(start_year)}-01-01"

    def _one(code: str) -> tuple[str, pd.DataFrame | None, str | None]:
        try:
            frame = get_periodic_report_announcements(code, start_date=start, mode=mode)
        except Exception as err:  # noqa: BLE001 — 单股失败不影响整体
            return code, None, f"{type(err).__name__}: {err}"
        return code, frame, None

    announcements: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=pool) as executor:
        for code, frame, reason in executor.map(_one, codes):
            if reason:
                failures.append({"symbol": code, "reason": reason})
            if frame is not None and not frame.empty:
                announcements[code] = frame

    plans: list[pd.DataFrame] = []
    stats: dict[str, object] = {
        "symbols": len(codes),
        "symbols_with_announcements": len(announcements),
        "start_year": int(start_year),
        "min_gap_days": int(min_gap_days),
        "failures": failures[:20],
        "failure_count": len(failures),
        "per_table": {},
    }
    for table in spec_tables:
        frame, _ = load_table(table)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        # 必须按"整天"比对: 巨潮公告时间带时区偏移 (实测 06:30:00), 不取整会把同一天算成 -1 天
        frame["ann_date"] = parse_date_column(frame["ann_date"], context=f"{table}.ann_date").dt.normalize()
        frame["report_date"] = parse_date_column(frame["report_date"], context=f"{table}.report_date")
        subset = frame[frame["report_date"] >= pd.Timestamp(start)]
        pieces: list[pd.DataFrame] = []
        for code, group in subset.groupby("symbol"):
            reference = announcements.get(code)
            if reference is None:
                continue
            reference_dates = reference.assign(
                cninfo_ann=parse_date_column(reference["ann_date"], context="cninfo.ann_date").dt.normalize()
            )
            merged = group[["symbol", "report_date", "ann_date"]].merge(
                reference_dates[["report_date", "cninfo_ann"]],
                on="report_date",
                how="inner",
            ).dropna(subset=["cninfo_ann"])
            if merged.empty:
                continue
            merged["gap_days"] = (merged["ann_date"] - merged["cninfo_ann"]).dt.days
            merged["table"] = table
            # **硬护栏 1**: 候选公告日不得早于报告期末 (glossary: 定期报告只能在期末后编制披露)
            merged = merged[merged["cninfo_ann"] >= merged["report_date"]]
            pieces.append(merged)
        if not pieces:
            stats["per_table"][table] = {"matched_rows": 0, "proposed": 0}
            continue
        matched = pd.concat(pieces, ignore_index=True)
        equal = matched["gap_days"] == 0
        external_later = matched["gap_days"] < 0
        # **硬护栏 2**: 候选日自身不得是"错位一年"签名 (超出法定截止 300 天以上), 否则说明
        # 巨潮那条也可能是被错误命名的档案 (实测踩过), 宁可不错也不引入新的错值
        from data_manager.financial_schema import legal_deadline as _legal_deadline

        limit_days = (matched["report_date"].map(_legal_deadline) - matched["report_date"]).dt.days
        candidate_ok = (matched["cninfo_ann"] - matched["report_date"]).dt.days <= (limit_days + 300)
        rejected_candidates = int((~candidate_ok).sum())
        matched = matched[candidate_ok]
        fixable = matched[matched["gap_days"] > int(min_gap_days)].copy()
        missing_local = matched["ann_date"].isna()
        fixable = pd.concat([fixable, matched[missing_local & (matched["gap_days"].isna())]], ignore_index=True)
        negative = matched.loc[matched["gap_days"] < 0, "gap_days"]
        stats["per_table"][table] = {
            "matched_rows": int(len(matched)),
            "cninfo_later_gap_p50": float(negative.median()) if len(negative) else 0.0,
            "candidate_rejected_out_of_window": rejected_candidates,
            "already_equal": int(equal.sum()),
            "cninfo_later_skipped": int(external_later.sum()),
            # 落盘晚于巨潮但差距在 min_gap_days 内: 两源对同一次公告的日期取法差异, 不动
            "within_tolerance": int(
                ((matched["gap_days"] > 0) & (matched["gap_days"] <= int(min_gap_days))).sum()
            ),
            "proposed": int(len(fixable)),
            "gap_days_p50": float(matched.loc[matched["gap_days"] > 0, "gap_days"].median())
            if (matched["gap_days"] > 0).any()
            else 0.0,
        }
        if len(fixable):
            fixable = fixable.assign(cninfo_ann=fixable["cninfo_ann"].dt.normalize())
            plans.append(
                fixable.rename(columns={"ann_date": "old_value", "cninfo_ann": "new_value"})[
                    ["table", "symbol", "report_date", "old_value", "new_value", "gap_days"]
                ].assign(column="ann_date")
            )
    plan = pd.concat(plans, ignore_index=True) if plans else pd.DataFrame(
        columns=["table", "symbol", "report_date", "old_value", "new_value", "gap_days", "column"]
    )
    stats["plan_rows"] = int(len(plan))
    return plan, stats


def repair_ann_dates_from_cninfo(
    tables: Sequence[str] = ("income_q", "balance_q", "cashflow_q"),
    start_year: int = 2010,
    symbols: Sequence[str] | None = None,
    threads: int | None = None,
    mode: str | None = None,
    min_gap_days: int = 30,
    apply: bool = False,
) -> dict[str, object]:
    """按巨潮仲裁结果重写 `ann_date` (默认 dry-run; ``apply=True`` 才落盘)。

    落盘同时写入 ``_provenance/<table>.corrections.csv`` (保留原值, 可回滚)。
    """
    from data_manager.financial_provenance import apply_corrections

    plan, stats = plan_ann_date_repairs(
        tables=tables, start_year=start_year, symbols=symbols, threads=threads, mode=mode, min_gap_days=min_gap_days
    )
    applied: dict[str, object] = {}
    if apply and not plan.empty:
        for table, group in plan.groupby("table"):
            stat = apply_corrections(
                table,
                group[["symbol", "report_date", "column", "old_value", "new_value"]],
                source="cninfo:disclosure",
                reason="东财 F10 ann_date 错位一年 (巨潮公告原文仲裁)",
                tool="repair_ann_dates_from_cninfo",
                apply=True,
            )
            applied[table] = stat.as_dict()
    stats["applied"] = applied
    stats["applied_total"] = sum(int(item.get("applied", 0)) for item in applied.values())
    stats["apply"] = bool(apply)
    return {"plan": plan, "stats": stats}


def update_financials(
    tables: Sequence[str] | None = None,
    start_year: int = DEFAULT_PERIOD_START_YEAR,
    end_year: int | None = None,
    symbols: Sequence[str] | None = None,
    include_delisted: bool = True,
    recent_periods: int | None = DEFAULT_STATEMENT_PERIODS,
    threads: int | None = None,
    mode: str | None = None,
    dry_run: bool = False,
    replace: bool = False,
) -> pd.DataFrame:
    """抓取并落盘指定财务表 (日更入口)。

    :param recent_periods: 逐股表只取最近 N 个报告期; None = 全历史回填
    :param replace: True = 整表替换 (全量重抓用, 保证不留历史抓取残留); False = 增量合并
    :return: DataFrame[table/rows/rows_added/rows_dropped_dup/failures/status]
    """
    selected = list(tables) if tables is not None else list(FINANCIAL_TABLES)
    for table in selected:
        get_spec(table)

    codes = resolve_symbols(symbols, include_delisted=include_delisted)
    results: list[dict[str, object]] = []
    for table in selected:
        stat: dict[str, object] = {
            "table": table,
            "rows": 0,
            "rows_added": 0,
            "rows_dropped_dup": 0,
            "failures": 0,
            "status": "skipped",
        }
        try:
            if table in SYMBOL_GRANULARITY_TABLES:
                raw, failures = fetch_symbol_table(
                    table,
                    codes,
                    start_year=start_year,
                    periods=recent_periods,
                    threads=threads,
                    mode=mode,
                )
                stat["failures"] = len(failures)
            else:
                raw, failed_periods = fetch_period_table(
                    table, start_year=start_year, end_year=end_year, mode=mode
                )
                stat["failures"] = len(failed_periods)
            prepared, quality = prepare_for_storage(raw, table)
            stat["rows"] = int(len(prepared))
            if quality is not None:
                stat["bad_symbols"] = quality.bad_symbols
                stat["bad_dates"] = quality.bad_dates
            if dry_run:
                stat["status"] = "dry-run"
            else:
                upsert = upsert_table(table, prepared, replace=replace)
                stat["rows_added"] = upsert.rows_added
                stat["rows_dropped_dup"] = upsert.rows_dropped_duplicate
                stat["status"] = "ok"
        except Exception as err:  # noqa: BLE001 — 单表失败不影响其他表
            traceback.print_exc()
            stat["status"] = f"failed: {type(err).__name__}: {err}"
        results.append(stat)
        print(f"[financial] {table}: {stat}")

    summary = pd.DataFrame(results)
    if not dry_run and not summary.empty and (summary["status"] == "ok").any():
        reload_provider()
    if not dry_run:
        write_update_summary(summary)
    return summary


def write_update_summary(summary: pd.DataFrame) -> str:
    fp = report_fp("update_summary.json")
    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tables": summary.to_dict(orient="records"),
    }
    with open(fp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return fp


def reload_provider() -> None:
    """通知 provider 重新加载 (Jupyter 里落盘后无需重启内核)。"""
    try:
        from data_manager.providers.financial_provider import FINANCIAL

        FINANCIAL.reload()
    except Exception:  # noqa: BLE001 — provider 尚未就绪时静默跳过
        pass


# ---------------------------------------------------------------------------
# 汇总统计 (供报告使用)
# ---------------------------------------------------------------------------


def table_stats(table: str) -> dict[str, object]:
    """单表规模统计: 行数/股票数/期数/公告日区间/缺失率。"""
    spec = get_spec(table)
    frame, quality = load_table(table)
    if frame.empty:
        return {
            "table": table,
            "filename": spec.filename,
            "rows": 0,
            "schema_ok": quality.schema_ok,
            "missing_columns": quality.missing_columns,
        }
    stats: dict[str, object] = {
        "table": table,
        "filename": spec.filename,
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
        "schema_ok": quality.schema_ok,
        "missing_columns": quality.missing_columns,
        "unexpected_columns": quality.unexpected_columns,
        "column_order_ok": quality.column_order_ok,
    }
    if "report_date" in frame.columns:
        report = pd.to_datetime(frame["report_date"], errors="coerce")
        stats["report_date_min"] = str(report.min().date()) if report.notna().any() else None
        stats["report_date_max"] = str(report.max().date()) if report.notna().any() else None
        stats["periods"] = int(report.dt.to_period("Q").nunique())
    if "ann_date" in frame.columns:
        ann = pd.to_datetime(frame["ann_date"], errors="coerce")
        stats["ann_date_min"] = str(ann.min().date()) if ann.notna().any() else None
        stats["ann_date_max"] = str(ann.max().date()) if ann.notna().any() else None
        stats["missing_ann_date"] = int(ann.isna().sum())
    if "basis" in frame.columns:
        stats["basis_counts"] = frame["basis"].value_counts(dropna=False).to_dict()
    missing: dict[str, float] = {}
    for col in frame.columns:
        if col in ("symbol", "basis", "update_time", "currency"):
            continue
        missing[col] = round(float(frame[col].isna().mean()), 6)
    stats["missing_rate"] = missing
    return stats


def iter_symbols(tables: Sequence[str] | None = None) -> Iterable[str]:
    """落盘数据里出现过的全部股票代码 (含退市股)。"""
    selected = list(tables) if tables is not None else list(FINANCIAL_TABLES)
    codes: set[str] = set()
    for table in selected:
        frame, _ = load_table(table)
        if not frame.empty and "symbol" in frame.columns:
            codes |= set(frame["symbol"].astype(str).str.zfill(6))
    return sorted(codes)


__all__ = [
    "BASIS_FIRST_REPORTED",
    "BASIS_LATEST_VERSION",
    "BASIS_MIXED",
    "BASIS_REVISED",
    "DropLog",
    "UpsertStat",
    "batch_check_financials_updated",
    "effective_trade_date",
    "fetch_period_table",
    "fetch_symbol_table",
    "get_financial_dir",
    "get_spec",
    "iter_symbols",
    "list_tables",
    "load_all_tables",
    "load_financial_events",
    "load_symbol_events",
    "load_table",
    "plan_ann_date_repairs",
    "plan_basis_recompute",
    "prepare_for_storage",
    "repair_ann_dates_from_cninfo",
    "raw_date_format_issues",
    "reload_provider",
    "report_fp",
    "resolve_symbols",
    "table_fp",
    "table_stats",
    "trading_calendar",
    "update_financials",
    "upsert_table",
]
