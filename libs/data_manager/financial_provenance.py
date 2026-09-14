"""财务数据溯源 (cell-level provenance)

**为什么要有这个模块**: 取数单把七张表的表头锁死了 (前四列固定, 尾部方法论列固定),
不能在主表里加 "这一格来自哪个源" 的列; 而数据里**确实存在人工介入的修正**
(例如用巨潮公告原文仲裁东财 `ann_date` 的"错位一年"缺陷)。没有溯源就会出现两个后果:
  1. 下次有人看到 `ann_date` 与东财不一致, 会误以为落盘被污染而"修回去";
  2. 任何一次批量修正都不可审计、不可回滚。

因此用 **sidecar**(旁路文件) 记录:
  * ``data/financial/_provenance/<table>.meta.json`` —— 逐列**默认来源声明** (常态来自哪个接口);
  * ``data/financial/_provenance/<table>.corrections.csv`` —— **单元格级修正流水**
    (append-only: 保留 old_value, 记录 new_value/来源/原因/时间/工具)。

于是任意一格的来源 = 默认来源 (meta) + 修正流水覆盖 (corrections); 修正可回滚
(``revert_corrections`` 把 old_value 写回并再记一条流水, 全程留痕)。

修正流水列: ``symbol,report_date,forecast_indicator_cn,column,old_value,new_value,source,reason,corrected_at,tool``
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from data_manager.financial_schema import CANONICAL_DATE_FORMAT, FINANCIAL_TABLES

# ---------------------------------------------------------------------------
# 默认来源声明 (每张表的每一列常态来自哪个接口)
# ---------------------------------------------------------------------------

EM_F10 = "em_f10"
EM_DATACENTER = "em_datacenter"

#: 表 → {列名: 来源}; 未列出的列回退到 ``DEFAULT_TABLE_SOURCE[table]``
COLUMN_SOURCES: dict[str, dict[str, str]] = {
    "income_q": {},
    "balance_q": {},
    "cashflow_q": {},
    "forecast": {
        "forecast_indicator_cn": f"{EM_DATACENTER}:RPT_PUBLIC_OP_NEWPREDICT",
        "forecast_net_profit_mid": "computed:(low+high)/2",
    },
    "dividend": {},
    "holder_num": {},
    "share_capital": {
        "ann_date": "cninfo:p_stock2215",
        "effective_date": "cninfo:p_stock2215",
        "total_share": "cninfo:p_stock2215 (万股→股)",
        "circ_share": "cninfo:p_stock2215 (万股→股)",
        "reason": "cninfo:p_stock2215",
    },
}

DEFAULT_TABLE_SOURCE: dict[str, str] = {
    "income_q": EM_F10,
    "balance_q": EM_F10,
    "cashflow_q": EM_F10,
    "forecast": f"{EM_DATACENTER}:RPT_PUBLIC_OP_NEWPREDICT",
    "dividend": f"{EM_DATACENTER}:RPT_SHAREBONUS_DET",
    "holder_num": f"{EM_DATACENTER}:RPT_HOLDERNUM_DET",
    "share_capital": "cninfo:p_stock2215",
}

#: 已知的**逐行**来源差异 (不是缺陷, 但影响"凭什么"的答案) —— 记进 meta 供查阅
ROW_LEVEL_EXCEPTIONS: dict[str, str] = {
    "share_capital": (
        "东财无股本变动记录的股票走回退接口 em:stock_zh_a_gbjg_em, "
        "该回退无公告日 → 以变动日近似 ann_date 并在 reason 列标注 '[源=东财股本结构,ann_date=变动日近似]'"
    ),
    "balance_q": "bank/broker/insurer 用各自 F10 模板 (companyType 1/2/3), 同一接口不同模板",
}

CORRECTION_COLUMNS: tuple[str, ...] = (
    "symbol",
    "report_date",
    "forecast_indicator_cn",
    "column",
    "old_value",
    "new_value",
    "source",
    "reason",
    "corrected_at",
    "tool",
)


def provenance_dir() -> str:
    """sidecar 目录 ``<FINANCIAL_DIR>/_provenance`` (不存在则创建)。"""
    from data_manager.financial_manager import get_financial_dir

    path = os.path.join(get_financial_dir(), "_provenance")
    os.makedirs(path, exist_ok=True)
    return path


def meta_fp(table: str) -> str:
    get_spec_guard(table)
    return os.path.join(provenance_dir(), f"{table}.meta.json")


def corrections_fp(table: str) -> str:
    get_spec_guard(table)
    return os.path.join(provenance_dir(), f"{table}.corrections.csv")


def get_spec_guard(table: str) -> None:
    if table not in FINANCIAL_TABLES:
        raise ValueError(f"未知财务表: {table}")


def default_source(table: str, column: str) -> str:
    """该列在该表的**默认**来源 (未经修正时)。"""
    get_spec_guard(table)
    return COLUMN_SOURCES.get(table, {}).get(column) or DEFAULT_TABLE_SOURCE[table]


def write_meta(table: str, extra: dict[str, Any] | None = None) -> str:
    """幂等写出 ``<table>.meta.json`` (逐列默认来源 + 逐行例外说明)。"""
    spec = FINANCIAL_TABLES[table]
    payload: dict[str, Any] = {
        "table": table,
        "file": spec.filename,
        "default_source": DEFAULT_TABLE_SOURCE[table],
        "columns": {col: default_source(table, col) for col in spec.columns},
        "row_level_exceptions": ROW_LEVEL_EXCEPTIONS.get(table, ""),
        "note": (
            "任意一格的来源 = 本文件的默认来源 + <table>.corrections.csv 里的修正流水覆盖; "
            "修正保留 old_value, 可用 revert_corrections 回滚"
        ),
    }
    if extra:
        payload.update(extra)
    fp = meta_fp(table)
    with open(fp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return fp


def write_all_meta() -> list[str]:
    return [write_meta(table) for table in FINANCIAL_TABLES]


# ---------------------------------------------------------------------------
# 修正流水
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CorrectionStat:
    """一次单元格修正的结果 (供 CLI 打印与体检统计)。"""

    table: str
    source: str = ""
    reason: str = ""
    planned: int = 0
    applied: int = 0
    unchanged: int = 0
    missing_key: int = 0
    logged: int = 0
    columns: dict[str, int] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "source": self.source,
            "reason": self.reason,
            "planned": self.planned,
            "applied": self.applied,
            "unchanged": self.unchanged,
            "missing_key": self.missing_key,
            "logged": self.logged,
            "columns": dict(self.columns),
            "examples": self.examples[:5],
        }


def load_corrections(table: str) -> pd.DataFrame:
    """读取修正流水 (无文件返回空帧, 列齐全)。"""
    fp = corrections_fp(table)
    if not os.path.exists(fp):
        return pd.DataFrame(columns=list(CORRECTION_COLUMNS))
    frame = pd.read_csv(fp, dtype=str, keep_default_na=False)
    for col in CORRECTION_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    return frame[list(CORRECTION_COLUMNS)]


def log_corrections(table: str, rows: Iterable[dict[str, Any]]) -> int:
    """把修正明细追加到流水 (append-only)。"""
    records = [dict(row) for row in rows]
    if not records:
        return 0
    frame = pd.DataFrame(records)
    for col in CORRECTION_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    fp = corrections_fp(table)
    header = not os.path.exists(fp)
    frame[list(CORRECTION_COLUMNS)].to_csv(
        fp, mode="a", header=header, index=False, encoding="utf-8-sig"
    )
    return len(frame)


def _key_columns(table: str) -> list[str]:
    spec = FINANCIAL_TABLES[table]
    return [col for col in spec.dedup_key if col in spec.columns]


def _normalize_keys(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    if "report_date" in out.columns:
        out["report_date"] = pd.to_datetime(
            out["report_date"], errors="coerce", format="mixed"
        ).dt.strftime(CANONICAL_DATE_FORMAT)
    return out


def apply_corrections(
    table: str,
    corrections: pd.DataFrame | Sequence[dict[str, Any]],
    *,
    source: str,
    reason: str,
    tool: str = "",
    apply: bool = True,
) -> CorrectionStat:
    """按单元格修正主表, 并把明细写入溯源流水。

    :param corrections: 需要 ``symbol, report_date[, forecast_indicator_cn], column, new_value``;
        可选 ``old_value``(校验用, 不一致则跳过该行, 避免并发覆盖别人的修改)。
    :param apply: False = 只做计划 (dry-run), 不写主表也不记流水。
    """
    from data_manager.financial_manager import load_table, upsert_table

    spec = FINANCIAL_TABLES[table]
    stat = CorrectionStat(table=table, source=source, reason=reason)
    plan = pd.DataFrame(list(corrections)) if not isinstance(corrections, pd.DataFrame) else corrections.copy()
    if plan.empty:
        return stat
    for required in ("symbol", "report_date", "column", "new_value"):
        if required not in plan.columns:
            raise ValueError(f"{table}: 修正计划缺少必需列 {required}")
    stat.planned = int(len(plan))
    if "forecast_indicator_cn" in plan.columns:
        plan["forecast_indicator_cn"] = plan["forecast_indicator_cn"].fillna("").astype(str)
    plan = _normalize_keys(plan, spec.dedup_key)

    frame, _ = load_table(table)
    if frame.empty:
        stat.missing_key = stat.planned
        return stat
    frame = _normalize_keys(frame, spec.dedup_key)
    keys = _key_columns(table)
    index = frame.set_index(keys, drop=False)
    plan_index = plan.set_index(keys, drop=False)

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    log_rows: list[dict[str, Any]] = []
    new_values = frame.copy()
    new_values = _normalize_keys(new_values, spec.dedup_key)
    new_values = new_values.set_index(keys, drop=False)

    for key, row in plan_index.iterrows():
        column = str(row["column"])
        if column not in spec.columns:
            raise ValueError(f"{table}: 未知列 {column}")
        if key not in index.index:
            stat.missing_key += 1
            continue
        current = index.loc[key]
        if isinstance(current, pd.DataFrame):  # 主表出现重复键 (理论上不会)
            current = current.iloc[0]
        old_value = current[column]
        if "old_value" in plan.columns and pd.notna(row.get("old_value")):
            expected = row["old_value"]
            if not _same_value(old_value, expected):
                stat.unchanged += 1
                continue
        new_value = _coerce_like(old_value, row["new_value"])
        if _same_value(old_value, new_value):
            stat.unchanged += 1
            continue
        new_values.loc[key, column] = new_value
        stat.applied += 1
        stat.columns[column] = stat.columns.get(column, 0) + 1
        if len(stat.examples) < 5:
            stat.examples.append(
                {"key": list(key), "column": column, "old": str(old_value), "new": str(new_value)}
            )
        log_row: dict[str, Any] = {
            "symbol": key[0],
            "report_date": key[1] if len(key) > 1 and "report_date" in keys else "",
            "forecast_indicator_cn": key[keys.index("forecast_indicator_cn")] if "forecast_indicator_cn" in keys else "",
            "column": column,
            "old_value": "" if pd.isna(old_value) else _stringify(old_value),
            "new_value": "" if pd.isna(new_value) else _stringify(new_value),
            "source": source,
            "reason": reason,
            "corrected_at": stamp,
            "tool": tool,
        }
        log_rows.append(log_row)

    if not apply or not stat.applied:
        return stat
    rewritten = new_values.reset_index(drop=True)
    # **顺序很重要**: 先写流水再落盘 —— 落盘路径会重放流水 (_reapply_corrections),
    # 若先落盘后记流水, 回滚时流水里最新的仍是旧修正, 会被重放撤销 (踩过并加测试)
    stat.logged = log_corrections(table, log_rows)
    upsert_table(table, rewritten[list(spec.columns)], replace=True)
    return stat


def revert_corrections(
    table: str,
    *,
    source: str | None = None,
    reason: str = "回滚修正",
    tool: str = "revert_corrections",
    apply: bool = True,
) -> CorrectionStat:
    """把最近一次修正回滚 (写回 old_value 并再记一条流水)。

    只回滚"当前值仍等于当时写入的 new_value"的单元格 —— 若之后又被改过则跳过, 避免误覆盖。
    """
    log = load_corrections(table)
    if log.empty:
        return CorrectionStat(table=table, source=source or "", reason=reason)
    if source:
        log = log[log["source"] == source]
    if log.empty:
        return CorrectionStat(table=table, source=source or "", reason=reason)
    latest = log.drop_duplicates(subset=["symbol", "report_date", "forecast_indicator_cn", "column"], keep="last")
    # 注意: 必须先选出需要的列再改名 —— 否则 new_value 会出现同名列, row["new_value"] 变成 Series
    plan = latest[["symbol", "report_date", "forecast_indicator_cn", "column", "old_value"]].rename(
        columns={"old_value": "new_value"}
    )
    return apply_corrections(
        table,
        plan,
        source=(source or latest["source"].iloc[-1]) + ":revert",
        reason=reason,
        tool=tool,
        apply=apply,
    )


def provenance_summary(tables: Sequence[str] | None = None) -> pd.DataFrame:
    """每张表的溯源概览: 默认来源 + 修正条数/涉及列/最近修正时间。"""
    selected = list(tables) if tables else list(FINANCIAL_TABLES)
    rows: list[dict[str, Any]] = []
    for table in selected:
        log = load_corrections(table)
        rows.append(
            {
                "table": table,
                "default_source": DEFAULT_TABLE_SOURCE[table],
                "corrections": int(len(log)),
                "columns": ", ".join(sorted(set(log["column"]))) if len(log) else "",
                "sources": ", ".join(sorted(set(log["source"]))) if len(log) else "",
                "last_corrected_at": log["corrected_at"].iloc[-1] if len(log) else "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 值比较/转换小工具
# ---------------------------------------------------------------------------


def _stringify(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime(CANONICAL_DATE_FORMAT)
    return str(value)


def _coerce_like(old_value: Any, new_value: Any) -> Any:
    """按主表当前值的类型转换新值 (日期列 → Timestamp; 数值列 → float; 文本 → str)。"""
    if pd.isna(new_value) or (isinstance(new_value, str) and new_value.strip() == ""):
        return pd.NA if isinstance(old_value, str) else float("nan")
    if isinstance(old_value, pd.Timestamp):
        return pd.Timestamp(pd.to_datetime(new_value, errors="coerce", format="mixed"))
    if isinstance(old_value, (int, float)) or old_value is None:
        return pd.to_numeric(new_value, errors="coerce")
    return str(new_value).strip()


def _same_value(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, pd.Timestamp) or isinstance(right, pd.Timestamp):
        return pd.Timestamp(pd.to_datetime(left, errors="coerce", format="mixed")) == pd.Timestamp(
            pd.to_datetime(right, errors="coerce", format="mixed")
        )
    try:
        return bool(abs(float(left) - float(right)) <= 1e-9)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


__all__ = [
    "COLUMN_SOURCES",
    "CORRECTION_COLUMNS",
    "CorrectionStat",
    "DEFAULT_TABLE_SOURCE",
    "apply_corrections",
    "corrections_fp",
    "default_source",
    "load_corrections",
    "log_corrections",
    "meta_fp",
    "provenance_dir",
    "provenance_summary",
    "revert_corrections",
    "write_all_meta",
    "write_meta",
]
