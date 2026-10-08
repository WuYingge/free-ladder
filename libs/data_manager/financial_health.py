"""财务数据体检 (Financial Data Health Check)

对 ``data/financial/*.csv`` 跑机器可判定的质量校验, 产出 ``health_report.json``。
**默认不联网**: 只做单源不变量检查 (一次性双源公告日对拍见
``libs/scripts/crosscheck_financials.py``)。

十三项检查 (顺序见 ``CHECK_ORDER``):
  1. ``schema_and_column_order`` — 列序 (对照取数单表头逐字) + 缺失留空 (拒绝 0/-1/"NULL" 填充)
  2. ``missing_value_convention`` — 哨兵值填充专检
  3. ``update_time_integrity`` — ``update_time`` 可解析且非未来
  4. ``field_missing_rates`` — 逐字段缺失率 (核心字段 >5% 或整体 >2% 且集中在某年 → fail)
  5. ``date_axis_invariants`` — 两条时间轴核心不变量:
     **日期列文本必须是规范 ``YYYY-MM-DD``** (出现 ``2021-04-20 00:00:00`` 说明写入路径
     dtype 退化过, 朴素解析器读它会静默判成 NaT → 丢公告日);
     其余: ``ann_date >= report_date`` (预告/分红类放宽到会计年度起始日)、
     超法定披露期比例、披露密度峰 (4/8/10 月)、``ann_date - report_date`` 分布、
     非交易日公告占比
  6. ``history_depth`` — 逐股历史深度是否落后全局最新期
  7. ``forecast_leadtime`` — 预告提前量与"晚于正式财报"占比
  8. ``revision_quantification`` — ``basis`` 分布、``update_date - ann_date`` 分位、混装股票数
  9. ``unit_dimension_checks`` — 资产=负债+权益; 归母净利 ≤ 合并净利; eps×总股本 ≈ 归母净利;
     逐字段年度中位数跳变
  10. ``coverage_survivorship`` — 按报告期的"有数据股票数 vs 当期在市数"; 退市股抽查
  11. ``extremes_tradability`` — 分位数、资产负债率 >100%、货币资金为负; 公告日在行情日历内
      比例、T+1 可成交覆盖率、孤儿代码占比
  12. ``field_fillability`` — 33 字段可填率 (对照需求清单 §③-A)

任一 hard check 失败即退出码 1 (``--fail-on hard``, 默认)。
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from config import DataPath
from data_manager.financial_manager import (
    effective_trade_date,
    load_table,
    raw_date_format_issues,
    table_stats,
    trading_calendar,
)
from data_manager.financial_schema import (
    BASIS_REVISED,
    FINANCIAL_FIELD_SOURCES,
    FINANCIAL_TABLES,
    PLACEHOLDER_TOKENS,
    field_implemented,
    fiscal_year_start,
    legal_deadline,
    parse_date_column,
)

CORE_FIELDS: tuple[str, ...] = (
    "net_profit_attr_p",
    "total_share",
    "total_assets",
    "total_equity_attr_p",
)
#: 缺失率阈值
CORE_MISSING_MAX = 0.05
OVERALL_MISSING_MAX = 0.02
#: 超法定披露期比例参考阈值 (仅用于报告上下文 / 可选 fail 开关)
DEADLINE_VIOLATION_MAX = 0.005
#: 是否把"逾期披露比例超阈值"判为 fail; 默认 False —— 逾期披露是真实市场现象
#: (实测 2016+ 达 8.6%, 独立信源对拍一致), 判 fail 会把正常数据误报为错误
DEADLINE_VIOLATION_FAIL = False
#: 量纲体检阈值
BALANCE_SHEET_TOLERANCE = 0.01
BALANCE_SHEET_VIOLATION_MAX = 0.02
NET_PROFIT_VIOLATION_MAX = 0.005
EPS_RATIO_BAND = (0.7, 1.3)
#: 孤儿代码占比上限
ORPHAN_MAX = 0.001
#: "非上市前报告期"的 ann_date 缺失占比上限: 这类缺失等于抓取/解析丢公告日。
#: 上市前报告期源端本就没有公告日 (哨兵 1900-01-01), 单独计数、不计入本阈值。
MISSING_ANN_DATE_OTHER_MAX = 0.001
#: 事件表 (预告/分红) 逐期覆盖相对去年同期的下限: 低于此值说明该期漏抓 (非正常稀疏)。
#: 实测正常年份同比比值在 1.0 附近波动 (forecast 2026H1/2025H1 = 1.14), 取 0.5 留足余量。
EVENT_COVERAGE_YOY_MIN = 0.5
#: "公告日错位一年"判定余量: 逾期且间隔超出法定上限这么多天才判为源端错位 (而非真逾期披露)。
#: 真逾期实测多为超期数十天 (000048: 2017 年报超期 123 天), 错位一年则超期 ~365 天, 取 300 天分界。
SHIFTED_ANN_DATE_MARGIN_DAYS = 300
#: 2016+ (研究建议区间) 的错位占比上限: 超过即 fail (源端缺陷会直接毁掉 PIT 可见时点)。
SHIFTED_ANN_DATE_MAX = 0.005
#: **已知源端日期异常逐行登记** (无法用现有信源修正, 且不修正就不能让 100% 硬规则永远 FAIL)。
#: 新增条目必须写明信源与原因; 命中行仍会出现在报告的 exemptions 里 (不是静默忽略),
#: 而**任何未登记的新倒挂行仍然 fail** —— 规则强度不变, 只是把已查清的个案摘出来。
KNOWN_DATE_EXCEPTIONS: tuple[dict[str, str], ...] = (
    {
        "table": "dividend",
        "symbol": "600153",
        "report_date": "2025-12-31",
        "rule": "ann_before_fiscal_year_start",
        "reason": (
            "东财 RPT_SHAREBONUS_DET 该行 report_date 字段错 (plan_ann_date=2024-12-18 属 2024 年度分红), "
            "巨潮分红公告对拍确认公告日无误 → 错的是源端报告期字段"
        ),
    },
    {
        "table": "holder_num",
        "symbol": "603223",
        "report_date": "2015-06-30",
        "rule": "ann_before_report",
        "reason": (
            "招股书口径的股东户数行 (上市日 2015-06-30, 户数 25,473): 源端把招股书内数据日当作公告日, "
            "早于统计截止日 1 天 → 属上市前材料, 不可做 PIT 定位"
        ),
    },
    {
        "table": "holder_num",
        "symbol": "920675",
        "report_date": "2016-03-31",
        "rule": "ann_before_report",
        "reason": "北交所代码不在 STOCK_LIST 快照内, 无法判定上市日; 该行系上市前材料 (户数=2), ann_date 字段不可靠",
    },
    {
        "table": "holder_num",
        "symbol": "920834",
        "report_date": "2015-09-30",
        "rule": "ann_before_report",
        "reason": "同上: 北交所代码, 上市前材料 (户数=15), ann_date 早于统计截止日",
    },
    {
        "table": "forecast",
        "symbol": "601975",
        "report_date": "2019-12-31",
        "rule": "ann_before_fiscal_year_start",
        "reason": (
            "东财 RPT_PUBLIC_OP_NEWPREDICT 该行预告公告日 2018-12-28 早于会计年度起始日, "
            "疑 report_date 应为 2018-12-31 (源端报告期字段错)"
        ),
    },
)

#: 错位判定的"研究使用区间"起点: 2010 起 (实测错位集中在 2000-2009, 而 2010+ 已近零)
SHIFTED_ANN_DATE_MODERN_FROM = "2010-01-01"
#: 跨源审计阈值 (实测基线: 股本双源一致率 97.7%, 行情侧"流通>总股本"占比 2.2%)
CROSS_SOURCE_MIN_AGREEMENT = 0.95      # 低于此值 → fail
CROSS_SOURCE_WARN_AGREEMENT = 0.99     # 低于此值 → warn (当前常态)
#: "流通股本不得超过总股本" 的违反占比上限 (实测基线 2.2%, 超过明显是系统性错)
FLOAT_EXCEEDS_TOTAL_MAX = 0.05
#: 年度中位数跳变判定的最小年样本量 (低于此不判定, 避免单只股票主导中位数)
MIN_ROWS_PER_YEAR_FOR_JUMP = 20
#: 覆盖抽查的退市股
DELISTED_SAMPLE = ("600696", "600193", "605081")


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str  # pass | warn | fail | skip
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "metrics": self.metrics,
            "failures": self.failures[:50],
        }


def _listing_date(symbol: str) -> str:
    """上市日 (来自 STOCK_LIST 快照); 取不到返回空串。

    用于区分"上市前报告期"与"真实超期披露": IPO 材料一次性披露的历史期间不受
    法定披露期约束, 不应计入超期比例。
    """
    try:
        from data_manager.providers.stock_list_provider import STOCK_LIST

        return str(STOCK_LIST.get_list_date(str(symbol).zfill(6)) or "")
    except Exception:  # noqa: BLE001
        return ""


def _result(
    name: str,
    status: str,
    summary: str,
    metrics: dict[str, Any] | None = None,
    failures: Iterable[dict[str, Any]] | None = None,
) -> CheckResult:
    return CheckResult(
        name=name,
        status=status,
        summary=summary,
        metrics=metrics or {},
        failures=list(failures or []),
    )


# ---------------------------------------------------------------------------
# 1. schema / 列序 / 缺失留空 / update_time
# ---------------------------------------------------------------------------


def check_schema(tables: Sequence[str], data_dir: str) -> CheckResult:
    details: dict[str, Any] = {}
    problems: list[dict[str, Any]] = []
    status = "pass"
    for table in tables:
        spec = FINANCIAL_TABLES[table]
        fp = os.path.join(data_dir, spec.filename)
        if not os.path.exists(fp):
            details[table] = {"exists": False, "expected_columns": list(spec.columns)}
            status = "warn" if status == "pass" else status
            continue
        header = pd.read_csv(fp, nrows=0)
        columns = [str(col).strip().lstrip("\ufeff") for col in header.columns]
        expected = list(spec.columns)
        ok = columns == expected
        missing = [col for col in expected if col not in columns]
        extra = [col for col in columns if col not in expected]
        details[table] = {
            "exists": True,
            "columns_match_take_order": ok,
            "missing_columns": missing,
            "unexpected_columns": extra,
        }
        if not ok:
            status = "fail"
            problems.append(
                {
                    "table": table,
                    "issue": "列序或列集合与取数单不符",
                    "missing_columns": missing,
                    "unexpected_columns": extra,
                }
            )
    summary = "各表列序与取数单一致" if status == "pass" else "存在列序/列集合问题或表缺失"
    return _result("schema_and_column_order", status, summary, details, problems)


def check_placeholder_fills(tables: Sequence[str], data_dir: str) -> CheckResult:
    """缺失值必须留空; 出现 "NULL"/"None"/"-1" 等哨兵填充 → 报异常。

    以**原始文本**读入 (``keep_default_na=False``), 否则 pandas 会把字面 "NULL"/"NA"
    直接转成 NaN 而漏检。``0`` 只在该列 >50% 都是 0 时才可疑 (业务上 0 可能是真值,
    如投资收益为 0); ``-1`` 在金额/股本/比率列一律按填充嫌疑处理。
    """
    problems: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for table in tables:
        spec = FINANCIAL_TABLES[table]
        fp = os.path.join(data_dir, spec.filename)
        if not os.path.exists(fp):
            continue
        raw = pd.read_csv(fp, dtype=str, keep_default_na=False)
        if raw.empty:
            continue
        table_detail: dict[str, Any] = {}
        for col in spec.columns:
            if col not in raw.columns or col in ("symbol", "basis", "update_time", "currency"):
                continue
            text = raw[col].astype(str).str.strip()
            lower = text.str.lower().str.strip('"')
            null_hits = int(lower.isin(PLACEHOLDER_TOKENS).sum())
            minus_one = int(text.eq("-1").sum())
            zero = int(text.eq("0").sum())
            n = int(len(raw))
            entry: dict[str, Any] = {}
            if null_hits:
                entry["null_like_fills"] = null_hits
            if minus_one:
                entry["minus_one_fills"] = minus_one
            if n and zero / n > 0.5:
                entry["zero_share"] = round(zero / n, 4)
                entry["note"] = ">50% 为 0, 疑似用 0 填充缺失"
            if entry:
                table_detail[col] = entry
        if table_detail:
            problems.append({"table": table, "columns": table_detail})
        details[table] = table_detail or "clean"
    status = "fail" if problems else "pass"
    summary = "无哨兵值填充" if not problems else f"{len(problems)} 张表存在哨兵值填充嫌疑"
    return _result("missing_value_convention", status, summary, details, problems)


def check_update_time_monotonic(tables: Sequence[str], data_dir: str) -> CheckResult:
    problems: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for table in tables:
        spec = FINANCIAL_TABLES[table]
        if "update_time" not in spec.columns:
            continue
        frame, _ = load_table(table)
        if frame.empty or "update_time" not in frame.columns:
            details[table] = "no data"
            continue
        parsed = pd.to_datetime(frame["update_time"], errors="coerce")
        unparsable = int(parsed.isna().sum())
        future = int((parsed > pd.Timestamp.now() + pd.Timedelta(days=1)).sum())
        details[table] = {
            "unparsable": unparsable,
            "future_dates": future,
            "min": str(parsed.min()) if parsed.notna().any() else None,
            "max": str(parsed.max()) if parsed.notna().any() else None,
        }
        if unparsable or future:
            problems.append({"table": table, "unparsable": unparsable, "future_dates": future})
    status = "fail" if problems else "pass"
    return _result(
        "update_time_integrity",
        status,
        "update_time 可解析且非未来时间" if not problems else "update_time 存在异常值",
        details,
        problems,
    )


# ---------------------------------------------------------------------------
# 2. 缺失率
# ---------------------------------------------------------------------------


def check_missing_rates(tables: Sequence[str], data_dir: str) -> CheckResult:
    details: dict[str, Any] = {}
    problems: list[dict[str, Any]] = []
    status = "pass"
    for table in tables:
        spec = FINANCIAL_TABLES[table]
        stats = table_stats(table)
        if not stats.get("rows"):
            details[table] = {"rows": 0}
            status = "warn" if status == "pass" else status
            continue
        frame, _ = load_table(table)
        missing: dict[str, float] = {}
        for col in spec.columns:
            if col in ("symbol", "basis", "update_time", "currency"):
                continue
            if col in frame.columns:
                missing[col] = round(float(frame[col].isna().mean()), 6)
        details[table] = {
            "rows": int(len(frame)),
            "missing_rate": missing,
            "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
            "structurally_empty_columns": list(spec.structurally_empty_columns),
            "structurally_empty_note": (
                "结构性空列 (信源无该字段, 保有列位以兼容取数单表头): "
                f"{list(spec.structurally_empty_columns)}"
                if spec.structurally_empty_columns
                else ""
            ),
        }
        for col in CORE_FIELDS:
            if col in spec.structurally_empty_columns:
                continue
            rate = missing.get(col)
            if rate is not None and rate > CORE_MISSING_MAX:
                problems.append(
                    {"table": table, "column": col, "missing_rate": rate, "threshold": CORE_MISSING_MAX}
                )
                status = "fail"
        # 整体缺失集中在特定年份 → warn
        if "report_date" in frame.columns:
            year = pd.to_datetime(frame["report_date"], errors="coerce").dt.year
            by_year = frame.assign(_year=year).groupby("_year").apply(
                lambda part: float(part[spec.columns[4]].isna().mean()) if spec.columns[4] in part else 0.0,
                include_groups=False,
            )
            concentrated = by_year[by_year > OVERALL_MISSING_MAX]
            if len(concentrated) and len(concentrated) < len(by_year):
                details[table]["concentrated_years"] = {str(k): round(v, 4) for k, v in concentrated.items()}
                if status == "pass":
                    status = "warn"
    summary = "逐字段缺失率正常" if status == "pass" else "缺失率存在异常"
    return _result("field_missing_rates", status, summary, details, problems)


# ---------------------------------------------------------------------------
# 3. 日期轴 (核心)
# ---------------------------------------------------------------------------


def check_date_axis(tables: Sequence[str], data_dir: str) -> CheckResult:
    """ann_date 与 report_date 的两条时间轴不变量。"""
    calendar = set(trading_calendar())
    details: dict[str, Any] = {}
    problems: list[dict[str, Any]] = []
    violation_total = 0
    rows_total = 0
    status = "pass"

    for table in tables:
        spec = FINANCIAL_TABLES[table]
        if "ann_date" not in spec.columns or "report_date" not in spec.columns:
            continue
        frame, _ = load_table(table)
        # 落盘日期文本格式必须规范: 出现 "2021-04-20 00:00:00" 说明写入路径曾 dtype 退化,
        # 而朴素解析器读这种混格式列会静默判成 NaT (丢公告日) —— 必须独立报出来
        format_issue = raw_date_format_issues(table)
        if format_issue["total"]:
            status = "fail"
            problems.append(
                {
                    "table": table,
                    "issue": "日期列存在非规范文本 (疑似写入路径 dtype 退化, 混格式会被朴素解析器静默判成 NaT)",
                    "columns": format_issue["columns"],
                    "count": format_issue["total"],
                    "examples": format_issue["examples"],
                }
            )
        if frame.empty:
            details[table] = {"rows": 0}
            continue
        ann = parse_date_column(frame["ann_date"], context=f"{table}.ann_date")
        rep = parse_date_column(frame["report_date"], context=f"{table}.report_date")
        n = int(len(frame))
        rows_total += n
        missing_ann = int(ann.isna().sum())
        missing_rep = int(rep.isna().sum())
        # 正式财报: ann_date 不得早于报告期末 (报表只能在期末后编制);
        # 业绩预告/快报/分红预案: 允许期间结束前发布 (前瞻信息), 下界放宽到会计年度起始日
        if spec.pre_period_allowed:
            lower_bound = rep.map(fiscal_year_start)
            early_kind = "ann_date < 会计年度起始日 (预告/分红类允许早于报告期末)"
        else:
            lower_bound = rep
            early_kind = "ann_date < report_date (脏数据)"
        inverted_all = ann < lower_bound
        # 上界: 报告期不得晚于公告日所属会计年度的次年末 —— 拦截源端把报告期张冠李戴
        # (实测 dividend 600153: report_date=2025-12-31 而 ann_date=2024-12-18)
        upper_bound = ann.map(lambda ts: pd.Timestamp(f"{int(ts.year) + 1}-12-31") if pd.notna(ts) else pd.NaT)
        wrong_period = pd.notna(rep) & (rep > upper_bound)
        deadlines = rep.map(legal_deadline)
        late = ann > deadlines
        # 上市前报告期豁免法定披露期判定: 新上市公司在 IPO 材料里一次性披露上市前多个
        # 报告期 (实测 301669: 2025H1 与 2025Q3 同为 2025-12-04 公告), 这些期从未受
        # 4/30、8/31、10/31 约束 —— 否则会产生纯属误报的"超期披露"
        codes = frame["symbol"].astype(str).str.zfill(6)
        listing = pd.to_datetime(codes.map(_listing_date), errors="coerce")
        # 含上市当日: 招股书口径的数据 (如股东户数) 正是在上市日披露
        pre_listing = listing.notna() & (rep <= listing)
        # 公告日缺失分类: 上市前报告期源端本就没有公告日 (哨兵 1900-01-01 已置空),
        # 只有"非上市前报告期还缺公告日"才说明抓取/解析出了问题
        missing_mask = ann.isna()
        missing_pre_listing = int((missing_mask & pre_listing).sum())
        missing_other = int((missing_mask & ~pre_listing).sum())
        # 上市前报告期同样豁免倒挂判定 (招股书行的字段配对不可靠), 逐行登记已知源端异常
        inverted = inverted_all & ~pre_listing
        rule_name = "ann_before_fiscal_year_start" if spec.pre_period_allowed else "ann_before_report"
        known_keys = {
            (item["symbol"], item["report_date"])
            for item in KNOWN_DATE_EXCEPTIONS
            if item["table"] == table and item["rule"] == rule_name
        }
        if known_keys:
            rep_text = rep.dt.strftime("%Y-%m-%d")
            exempt_mask = pd.Series(
                [(sym, text) in known_keys for sym, text in zip(codes, rep_text)], index=frame.index
            )
        else:
            exempt_mask = pd.Series(False, index=frame.index)
        inverted_exempt = int((inverted & exempt_mask).sum())
        inverted = inverted & ~exempt_mask
        exempt_entries = (
            [
                {
                    "symbol": item["symbol"],
                    "report_date": item["report_date"],
                    "rule": item["rule"],
                    "reason": item["reason"],
                }
                for item in KNOWN_DATE_EXCEPTIONS
                if item["table"] == table
            ]
            if int(inverted_exempt)
            else []
        )
        late_checkable = late & ~pre_listing
        late_pre_listing = late & pre_listing
        gaps = (ann - rep).dt.days
        # **"公告日错位一年"签名 (源端缺陷, 2026-09-12 用巨潮公告原文定案)**:
        # 东财 F10 对相当一部分报告期返回的 NOTICE_DATE 实际是"**下一期同日历期**"的公告日
        # (实测 000670 的 2016Q1 写成 2017-04-20 = 其 2017 年一季报公告日; 000685 的 2016 年报
        #  income/balance 都写成 2018-04-21 = 其 2017 年年报公告日)。判据: 逾期且间隔超出法定
        # 上限 300 天以上 —— 与"真逾期披露"(独立信源已证实, 如 000048 的 2017 年报 = 2018-08-31)
        # 严格分开统计, 否则会把源端缺陷混进"真实市场现象"。
        limit_days = (deadlines - rep).dt.days
        shifted_all = late & pd.notna(gaps) & (gaps > limit_days + SHIFTED_ANN_DATE_MARGIN_DAYS)
        # **上市前报告期豁免** (与超期披露判定一致): 招股书会一次性披露上市前多个报告期,
        # 其"公告日"就是招股书日, 天然晚于报告期数年 —— 不豁免会把它们全算成"错位一年"
        # (实测 income_q 因此虚报 27,000 行, 并让 2016+ 段虚高到 4.8%, 实际只有 3 行)
        shifted = shifted_all & ~pre_listing
        genuine_late = late & ~shifted_all
        # 判定只看**研究实际使用区间** (2010+): 2000-2009 段源端公告日大面积不可靠,
        # 且巨潮档案缺失导致无法仲裁, 属已知遗留缺口 (见 docs §8), 仅记录不判 fail
        modern = rep >= pd.Timestamp(SHIFTED_ANN_DATE_MODERN_FROM)
        shifted_recent = int((shifted & modern).sum())
        shifted_recent_share = float(shifted_recent) / float((modern & pd.notna(rep)).sum() or 1)
        shifted_legacy = int((shifted & ~modern).sum())
        non_trading = int(sum(1 for value in ann.dropna() if value.date() not in calendar)) if calendar else 0
        month_hist = ann.dropna().dt.month.value_counts().sort_index().to_dict()
        peak_share = (
            float(sum(count for month, count in month_hist.items() if month in (4, 8, 10)))
            / float(len(ann.dropna()))
            if ann.notna().any()
            else 0.0
        )
        details[table] = {
            "rows": n,
            "non_canonical_date_text": format_issue["total"],
            "missing_ann_date": missing_ann,
            "missing_ann_date_pre_listing": missing_pre_listing,
            "missing_ann_date_other": missing_other,
            "missing_report_date": missing_rep,
            "missing_ann_date_note": (
                "上市前报告期 (report_date < 上市日) 源端本就没有公告日 "
                "(东财返回哨兵 1900-01-01, 已按'未知'置空), 这类行不可做 PIT 定位, "
                "由行级门禁剔除; 非上市前报告期的公告日缺失才是缺陷"
            ),
            "ann_before_lower_bound": int(inverted.sum()),
            "ann_before_lower_bound_exempt": inverted_exempt + int((inverted_all & pre_listing).sum()),
            "exempt_known_source_issues": exempt_entries,
            "lower_bound_rule": early_kind,
            "ann_before_report_raw": int((ann < rep).sum()),
            "pre_period_allowed": spec.pre_period_allowed,
            "report_after_ann_year_bound": int(wrong_period.sum()),
            "beyond_legal_deadline": int(late.sum()),
            "ann_date_shifted_one_year": int(shifted.sum()),
            "ann_date_shifted_one_year_legacy_pre2010": shifted_legacy,
            "ann_date_shifted_one_year_2010plus": shifted_recent,
            "ann_date_shifted_one_year_2010plus_share": round(shifted_recent_share, 6),
            "ann_date_shifted_pre_listing_exempt": int((shifted_all & pre_listing).sum()),
            "genuine_late_filings": int(genuine_late.sum()),
            "shifted_note": (
                "错位一年 = 源端把公告日写成下一期同日历期的公告日 (巨潮公告原文已定案); "
                "与真逾期披露分开统计"
            ),
            "beyond_legal_deadline_checkable": int(late_checkable.sum()),
            "beyond_legal_deadline_pre_listing": int(late_pre_listing.sum()),
            "beyond_deadline_share": round(float(late_checkable.sum()) / n, 6) if n else 0.0,
            "pre_listing_note": (
                "上市前报告期 (IPO 材料一次性披露的历史期间) 不计入超期判定, 仅记录条数"
            ),
            "gap_days": {
                "p50": float(gaps.median()) if gaps.notna().any() else None,
                "p90": float(gaps.quantile(0.9)) if gaps.notna().any() else None,
                "min": float(gaps.min()) if gaps.notna().any() else None,
                "max": float(gaps.max()) if gaps.notna().any() else None,
            },
            "non_trading_day_ann": non_trading,
            "non_trading_day_share": round(non_trading / n, 6) if n else 0.0,
            "ann_month_histogram": {str(k): int(v) for k, v in month_hist.items()},
            "peak_三个月占比_4_8_10": round(peak_share, 6),
        }
        violation_total += int(late_checkable.sum())

        if missing_rep:
            status = "fail"
            problems.append(
                {"table": table, "issue": "report_date 缺失 (报告期是主键, 不可缺)", "count": missing_rep}
            )
        if missing_other and (missing_other / n if n else 0.0) > MISSING_ANN_DATE_OTHER_MAX:
            status = "fail"
            problems.append(
                {
                    "table": table,
                    "issue": "ann_date 缺失 (非上市前报告期, 说明抓取/解析丢公告日)",
                    "count": missing_other,
                    "share": round(missing_other / n, 6) if n else 0.0,
                    "threshold": MISSING_ANN_DATE_OTHER_MAX,
                    "pre_listing_missing": missing_pre_listing,
                    "examples": frame.loc[missing_mask & ~pre_listing, ["symbol", "report_date", "ann_date"]].head(5).to_dict("records"),
                }
            )
        elif missing_other:
            status = "warn" if status == "pass" else status
            problems.append(
                {
                    "table": table,
                    "issue": "少量 ann_date 缺失 (非上市前报告期, 低于阈值, 仅提示)",
                    "count": missing_other,
                    "share": round(missing_other / n, 6) if n else 0.0,
                    "threshold": MISSING_ANN_DATE_OTHER_MAX,
                    "pre_listing_missing": missing_pre_listing,
                }
            )
        if int(inverted.sum()):
            status = "fail"
            problems.append(
                {
                    "table": table,
                    "issue": early_kind,
                    "count": int(inverted.sum()),
                    "pre_listing_exempt": int((inverted_all & pre_listing).sum()),
                    "examples": frame.loc[inverted, ["symbol", "report_date", "ann_date"]].head(5).to_dict("records"),
                }
            )
        if shifted_recent_share > SHIFTED_ANN_DATE_MAX:
            status = "fail"
            problems.append(
                {
                    "table": table,
                    "issue": "ann_date 疑似被源端写成下一期同日历期的公告日 (错位一年, 2010+ 区间)",
                    "count": shifted_recent,
                    "share": round(shifted_recent_share, 6),
                    "threshold": SHIFTED_ANN_DATE_MAX,
                    "legacy_pre2010": shifted_legacy,
                    "examples": frame.loc[shifted & modern, ["symbol", "report_date", "ann_date"]].head(5).to_dict("records"),
                }
            )
        elif int(shifted.sum()):
            if status == "pass":
                status = "warn"
            problems.append(
                {
                    "table": table,
                    "issue": "ann_date 存在疑似错位一年行 (全部在 2010 年前; 巨潮档案缺失, 无法仲裁)",
                    "count": int(shifted.sum()),
                    "modern_2010plus": shifted_recent,
                }
            )
        if int(wrong_period.sum()):
            status = "fail"
            problems.append(
                {
                    "table": table,
                    "issue": "report_date 晚于公告日所属年度的次年末 (报告期张冠李戴)",
                    "count": int(wrong_period.sum()),
                    "examples": frame.loc[wrong_period, ["symbol", "report_date", "ann_date"]].head(5).to_dict("records"),
                }
            )
        # 逾期披露 = 真实市场现象 (实测 2016+ 达 8.6%, 且独立信源 RPT_LICO_FN_CPD 完全一致),
        # 不是日期错误 → 只给量化上下文 + warn, 不判 fail。即使超阈值也保留 fail 的选项,
        # 通过 DEADLINE_VIOLATION_FAIL 开关控制 (默认关闭)。
        if int(late_checkable.sum()) and spec.deadline_check_applicable:
            share_late = float(late_checkable.sum()) / n if n else 0.0
            late_delay = (ann - deadlines).dt.days[late_checkable]
            details[table]["late_share"] = round(share_late, 6)
            details[table]["late_delay_days"] = {
                "p50": float(late_delay.median()) if len(late_delay) else None,
                "p90": float(late_delay.quantile(0.9)) if len(late_delay) else None,
                "max": float(late_delay.max()) if len(late_delay) else None,
            }
            if status == "pass":
                status = "warn"
            problems.append(
                {
                    "table": table,
                    "issue": "存在逾期披露 (真实市场现象, 非日期错误)",
                    "count": int(late_checkable.sum()),
                    "share": round(share_late, 6),
                    "context_threshold": DEADLINE_VIOLATION_MAX,
                    "delay_days_p50": details[table]["late_delay_days"]["p50"],
                    "note": (
                        "此处的 late 已扣除'错位一年'签名 (见 ann_date_shifted_one_year), 剩下的才是真逾期; "
                        "真逾期是真实市场现象, 如 000048 的 2017 年报确为 2018-08-31 披露 (巨潮公告原文核实); "
                        "研究如需回避, 可在因子层按 ann_date - report_date 过滤"
                    ),
                }
            )
            if DEADLINE_VIOLATION_FAIL and share_late > DEADLINE_VIOLATION_MAX:
                status = "fail"
                problems[-1]["issue"] = "逾期披露比例超过设定阈值 (DEADLINE_VIOLATION_FAIL=True)"
        if spec.peak_shape_applicable and peak_share and peak_share < 0.5:
            if status == "pass":
                status = "warn"
            problems.append(
                {
                    "table": table,
                    "issue": "4/8/10 月披露占比 <50%, 公告日分布形状可疑",
                    "peak_share": round(peak_share, 4),
                }
            )

    summary = (
        f"合计 {rows_total} 行; ann_date>=report_date 全部成立"
        if status == "pass"
        else "日期轴存在需要处理的问题"
    )
    metrics = {
        "rows_total": rows_total,
        "beyond_deadline_total": violation_total,
        "beyond_deadline_share": round(violation_total / rows_total, 6) if rows_total else 0.0,
    }
    return _result("date_axis_invariants", status, summary, {**metrics, "tables": details}, problems)


# ---------------------------------------------------------------------------
# 4. 更正量化
# ---------------------------------------------------------------------------


def check_revision_quantification(tables: Sequence[str], data_dir: str) -> CheckResult:
    details: dict[str, Any] = {}
    problems: list[dict[str, Any]] = []
    for table in tables:
        spec = FINANCIAL_TABLES[table]
        if not spec.versioned:
            continue
        frame, _ = load_table(table)
        if frame.empty:
            details[table] = {"rows": 0}
            continue
        basis_counts = frame["basis"].value_counts(dropna=False).to_dict() if "basis" in frame.columns else {}
        revised = (
            frame[frame["basis"].astype(str).str.startswith(BASIS_REVISED)]
            if "basis" in frame.columns
            else frame.iloc[0:0]
        )
        lags = (revised["update_date"] - revised["ann_date"]).dt.days.dropna()
        mixed = 0
        if "basis" in frame.columns and len(frame):
            counts = frame.groupby(frame["symbol"].astype(str))["basis"].nunique()
            mixed = int((counts > 1).sum())
        details[table] = {
            "rows": int(len(frame)),
            "basis_counts": {str(k): int(v) for k, v in basis_counts.items()},
            "revised_rows": int(len(revised)),
            "revised_symbols": int(revised["symbol"].nunique()) if len(revised) else 0,
            "revision_lag_days": {
                "p50": float(lags.median()) if len(lags) else None,
                "p90": float(lags.quantile(0.9)) if len(lags) else None,
                "max": float(lags.max()) if len(lags) else None,
            },
            "mixed_basis_symbols": mixed,
            "note": (
                "revised = 该行数值在首次公告后被追溯调整过, 当前版本无法还原首版值; "
                "研究默认口径只取 first_reported"
            ),
        }
    summary = "更正行已逐行标记 (basis/update_date)"
    return _result("revision_quantification", "pass", summary, details, problems)


# ---------------------------------------------------------------------------
# 5. 量纲
# ---------------------------------------------------------------------------


def check_dimensions(tables: Sequence[str], data_dir: str) -> CheckResult:
    problems: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    status = "pass"

    if "balance_q" in tables:
        balance, _ = load_table("balance_q")
        if not balance.empty:
            assets = pd.to_numeric(balance["total_assets"], errors="coerce")
            liabilities = pd.to_numeric(balance["total_liabilities"], errors="coerce")
            equity = pd.to_numeric(balance["total_equity"], errors="coerce")
            comparable = assets.notna() & liabilities.notna() & equity.notna() & assets.ne(0)
            deviation = ((assets - liabilities - equity).abs() / assets.abs()).where(comparable)
            bad = deviation > BALANCE_SHEET_TOLERANCE
            share = float(bad.sum()) / float(comparable.sum()) if comparable.sum() else 0.0
            details["balance_q_identity"] = {
                "comparable_rows": int(comparable.sum()),
                "violation_rows": int(bad.sum()),
                "violation_share": round(share, 6),
                "median_deviation": float(deviation.median()) if deviation.notna().any() else None,
            }
            if share > BALANCE_SHEET_VIOLATION_MAX:
                status = "fail"
                problems.append(
                    {
                        "table": "balance_q",
                        "issue": "资产 = 负债 + 权益 恒等式偏离 >1% 的行占比过高",
                        "violation_share": round(share, 6),
                        "examples": balance.loc[bad, ["symbol", "report_date", "total_assets", "total_liabilities", "total_equity"]]
                        .head(5)
                        .to_dict("records"),
                    }
                )

    if "income_q" in tables:
        income, _ = load_table("income_q")
        if not income.empty:
            parent = pd.to_numeric(income["net_profit_attr_p"], errors="coerce")
            total = pd.to_numeric(income["net_profit"], errors="coerce")
            both = parent.notna() & total.notna()
            # 注意: 不能用 "net_profit >= net_profit_attr_p" 当判据 —— 当少数股东承担亏损时
            # 合并净利(含少数股东)可以**小于**归母净利 (实测 000002/000607 均为同号小额差)。
            # 正确的会计恒等式是: 合并净利 = 归母净利 + 少数股东损益。
            below = both & (total < parent)
            share = float(below.sum()) / float(both.sum()) if both.sum() else 0.0

            identity_rows = 0
            identity_violation = 0
            if "minority_interest_profit" in income.columns:
                derived = parent + pd.to_numeric(income["minority_interest_profit"], errors="coerce")
                base = total.abs().clip(lower=1.0)
                deviation = ((total - derived).abs() / base).where(both & derived.notna())
                identity_rows = int(deviation.notna().sum())
                identity_violation = int((deviation > BALANCE_SHEET_TOLERANCE).sum())

            # eps_basic 在 income_q, 股本在 balance_q (F10 利润表无股本字段)
            # → 跨表按 (symbol, report_date) 对齐后做"元 vs 元"量纲交叉验证
            median_ratio: float | None = None
            ratio_rows = 0
            if "balance_q" in tables:
                balance, _ = load_table("balance_q")
                if not balance.empty and "total_share" in balance.columns:
                    merged = income[["symbol", "report_date", "eps_basic", "net_profit_attr_p"]].merge(
                        balance[["symbol", "report_date", "total_share"]],
                        on=["symbol", "report_date"],
                        how="inner",
                    )
                    eps = pd.to_numeric(merged["eps_basic"], errors="coerce")
                    shares = pd.to_numeric(merged["total_share"], errors="coerce")
                    parent_m = pd.to_numeric(merged["net_profit_attr_p"], errors="coerce")
                    comparable = (
                        parent_m.notna() & eps.notna() & shares.notna() & parent_m.ne(0) & eps.ne(0)
                    )
                    ratio = (eps * shares / parent_m).where(comparable & (eps > 0))
                    ratio_rows = int(ratio.notna().sum())
                    median_ratio = float(ratio.median()) if ratio.notna().any() else None

            details["income_q_checks"] = {
                "comparable_rows": int(both.sum()),
                "net_profit_below_parent_rows": int(below.sum()),
                "net_profit_below_parent_share": round(share, 6),
                "net_profit_below_parent_note": (
                    "同号小额差属正常 (少数股东承担亏损时合并净利可小于归母净利), 只作记录不作判错"
                ),
                "net_profit_identity_rows": identity_rows,
                "net_profit_identity_violations": identity_violation,
                "net_profit_identity_rule": "合并净利 ≈ 归母净利 + 少数股东损益 (偏离 >1% 计违规)",
                "eps_x_share_over_parent_rows": ratio_rows,
                "eps_times_share_over_parent_median": median_ratio,
                "eps_ratio_band": list(EPS_RATIO_BAND),
                "note": "eps 来自 income_q, 股本来自 balance_q (跨表对齐); 比值偏离 0.7~1.3 提示量纲混装",
            }
            if identity_rows and identity_violation / identity_rows > BALANCE_SHEET_VIOLATION_MAX:
                status = "fail"
                problems.append(
                    {
                        "table": "income_q+balance_q",
                        "issue": "合并净利 = 归母净利 + 少数股东损益 恒等式偏离 >1% 的样本过多",
                        "rows": identity_rows,
                        "violations": identity_violation,
                    }
                )
            if median_ratio is not None and ratio_rows >= 30 and not (EPS_RATIO_BAND[0] <= median_ratio <= EPS_RATIO_BAND[1]):
                status = "fail"
                problems.append(
                    {
                        "table": "income_q+balance_q",
                        "issue": "eps × 总股本 / 归母净利 中位数落在 0.7~1.3 之外 (疑似万元/元或股/万股混装)",
                        "median": median_ratio,
                        "rows": ratio_rows,
                    }
                )

    # 年度中位数 10000 倍跳变 (仅统计样本量足够的年份: 单只股票的年度中位数会被极值主导,
    # 实测 2023 年仅 2 只退市股(600614/600672)即触发假跳变)
    jumps: dict[str, list[str]] = {}
    for table in tables:
        frame, _ = load_table(table)
        if frame.empty or "report_date" not in frame.columns:
            continue
        spec = FINANCIAL_TABLES[table]
        year = pd.to_datetime(frame["report_date"], errors="coerce").dt.year
        year_counts = year.value_counts()
        reliable_years = {
            int(y) for y, count in year_counts.items() if pd.notna(y) and count >= MIN_ROWS_PER_YEAR_FOR_JUMP
        }
        for col in spec.amount_columns + spec.share_columns:
            if col not in frame.columns:
                continue
            series = pd.to_numeric(frame[col], errors="coerce")
            medians = series.groupby(year).median().dropna()
            medians = medians[[int(idx) in reliable_years for idx in medians.index]]
            medians = medians[medians != 0]
            if len(medians) < 2:
                continue
            ratios = (medians / medians.shift(1)).dropna()
            for idx, ratio in ratios.items():
                if ratio > 1000 or ratio < 1 / 1000:
                    jumps.setdefault(f"{table}.{col}", []).append(
                        f"{int(idx)}: 中位数跳变 x{ratio:.1f}"
                    )
    details["year_median_jumps"] = jumps
    details["year_median_jump_rule"] = f"仅统计该年样本数 >= {MIN_ROWS_PER_YEAR_FOR_JUMP} 行的年份"
    if jumps:
        if status == "pass":
            status = "warn"
        problems.append({"issue": "年度中位数出现 1000 倍级跳变 (疑似源口径切换)", "columns": jumps})

    summary = "量纲恒等式与每股口径检查通过" if status == "pass" else "量纲存在异常"
    return _result("unit_dimension_checks", status, summary, details, problems)


def check_forecast_leadtime(tables: Sequence[str], data_dir: str) -> CheckResult:
    """预告 vs 正式财报的先后关系 (预告类真正有意义的检验)。

    业绩预告**没有法定截止期** (实测延误 p50=15.5 天 / max=203 天), 所以不能用
    4/30-8/31-10/31 判据; 真正要守的不变量是: **同一 (symbol, report_date) 的预告
    应当早于该期正式财报的公告日** —— 否则它不是前瞻信息, 而是事后补充/更正,
    用于事件研究会产生错位。本项统计违背比例 (阈值 5%)。
    """
    if "forecast" not in tables or "income_q" not in tables:
        return _result("forecast_leadtime", "skip", "需要 forecast 与 income_q 同时在表内", {})
    forecast, _ = load_table("forecast")
    income, _ = load_table("income_q")
    if forecast.empty or income.empty:
        return _result("forecast_leadtime", "skip", "数据不足", {})

    key = ["symbol", "report_date"]
    actual = (
        income.assign(_report_ann=pd.to_datetime(income["ann_date"], errors="coerce"))
        .groupby(key, dropna=False)["_report_ann"]
        .min()
        .rename("report_ann_date")
        .reset_index()
    )
    merged = forecast.merge(actual, on=key, how="inner")
    merged["forecast_ann"] = pd.to_datetime(merged["ann_date"], errors="coerce")
    comparable = merged["forecast_ann"].notna() & merged["report_ann_date"].notna()
    merged = merged[comparable]
    if merged.empty:
        return _result("forecast_leadtime", "skip", "无可与正式财报对齐的预告样本", {})

    lead = (merged["report_ann_date"] - merged["forecast_ann"]).dt.days
    after = lead < 0
    share = float(after.sum()) / float(len(merged) )
    metrics = {
        "comparable_rows": int(len(merged)),
        "forecast_after_report_rows": int(after.sum()),
        "forecast_after_report_share": round(share, 6),
        "lead_days": {
            "p10": float(lead.quantile(0.1)),
            "p50": float(lead.median()),
            "p90": float(lead.quantile(0.9)),
            "min": float(lead.min()),
        },
        "rule": "预告公告日应早于同 (symbol, report_date) 正式财报公告日 (lead>=0)",
        "examples": merged.loc[after, ["symbol", "report_date", "forecast_ann", "report_ann_date"]]
        .head(5)
        .astype(str)
        .to_dict("records"),
    }
    problems: list[dict[str, Any]] = []
    status = "pass"
    if share > 0.05:
        status = "fail"
        problems.append(
            {
                "issue": "预告晚于正式财报的比例 >5% (预告不再是前瞻信息)",
                "share": round(share, 6),
                "rows": int(after.sum()),
            }
        )
    elif int(after.sum()):
        status = "warn"
        problems.append(
            {"issue": "少数预告晚于正式财报 (多为事后补充/更正公告)", "rows": int(after.sum())}
        )
    summary = f"预告平均提前 {metrics['lead_days']['p50']:.0f} 天 (p50); 晚于正式财报占比 {share:.4%}"
    return _result("forecast_leadtime", status, summary, metrics, problems)


# ---------------------------------------------------------------------------
# 6. 覆盖与幸存者
# ---------------------------------------------------------------------------


def check_history_depth(tables: Sequence[str], data_dir: str) -> CheckResult:
    """历史深度异常检查 (发现"每只股票都只剩最近几期"这类静默截断)。

    动机: 实测 2026-09-11 发现 F10 报表端点**单次请求超过 5 个日期时只返回最新 5 期**,
    导致 `--all-history` 回填实际只落了 5 期。逐字段缺失率/日期轴都检测不到这类问题
    (公告日与报告期本身都合法), 只有"相对全局最新期的深度分布"能暴露它。

    判据: 按 symbol 取该股最新 report_date, 统计落在全局最新期**之前两期以上**的股票占比;
    占比 > 20% 判 fail (全量回填后应接近 0: 最新期总有股票尚未披露, 但不会是大面积)。
    """
    problems: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    status = "pass"
    for table in tables:
        spec = FINANCIAL_TABLES[table]
        if "report_date" not in spec.columns:
            continue
        frame, _ = load_table(table)
        if frame.empty:
            continue
        rep = pd.to_datetime(frame["report_date"], errors="coerce")
        latest_by_symbol = rep.groupby(frame["symbol"].astype(str)).max()
        global_latest = rep.max()
        periods = sorted(rep.dropna().unique())
        details[table] = {
            "symbols": int(len(latest_by_symbol)),
            "global_latest_period": str(pd.Timestamp(global_latest).date()) if pd.notna(global_latest) else None,
            "max_periods_per_symbol": int(rep.groupby(frame["symbol"].astype(str)).size().max()),
            "median_periods_per_symbol": float(rep.groupby(frame["symbol"].astype(str)).size().median()),
        }
        if not spec.peak_shape_applicable:
            # **事件表 (业绩预告/分红预案) 不适用"落后全局最新期"判据**: 股票只在有事可报
            # 时才发预告/分红, 单期覆盖率本就只有三成 (实测 forecast 2026H1 覆盖 3,708/5,202),
            # 用逐股滞后率判会把正常稀疏误报成"回填不完整" (实测误报 35%)。
            # 改用**逐期覆盖的同比稳定性**: 同一报告期相对去年同期的股票数比值,
            # 若某期覆盖塌到去年同期一半以下, 才是抓取漏期的签名。
            per_period = rep.dt.strftime("%Y-%m-%d").value_counts()
            yoy: dict[str, float] = {}
            weak: list[dict[str, Any]] = []
            for period, count in per_period.items():
                previous = f"{int(period[:4]) - 1}{period[4:]}"
                base = int(per_period.get(previous, 0))
                if base < 50:  # 去年同期样本太少, 比值不可信
                    continue
                ratio = float(count) / float(base)
                yoy[period] = round(ratio, 4)
                if ratio < EVENT_COVERAGE_YOY_MIN:
                    weak.append(
                        {
                            "period": period,
                            "symbols": int(count),
                            "same_quarter_last_year": base,
                            "ratio": round(ratio, 4),
                        }
                    )
            details[table]["judgement"] = "事件表: 逐期覆盖同比稳定性 (不判逐股滞后)"
            details[table]["coverage_yoy"] = {k: v for k, v in sorted(yoy.items())[-12:]}
            details[table]["weak_periods"] = weak
            if weak:
                status = "fail"
                problems.append(
                    {
                        "table": table,
                        "issue": f"逐期覆盖相对去年同期塌陷 (比值 < {EVENT_COVERAGE_YOY_MIN}, 疑似漏期/抓取截断)",
                        "periods": weak[:5],
                    }
                )
            continue
        # 用全局期次序列定位"两期以前"
        if len(periods) < 3 or latest_by_symbol.empty:
            continue
        cutoff = pd.Timestamp(periods[-3])
        behind = latest_by_symbol[latest_by_symbol < cutoff]
        share = float(len(behind)) / float(len(latest_by_symbol))
        details[table].update(
            {
                "cutoff_period": str(cutoff.date()),
                "symbols_behind_two_periods": int(len(behind)),
                "share": round(share, 6),
                "examples": behind.head(5).index.tolist(),
            }
        )
        if len(latest_by_symbol) >= 100 and share > 0.2:
            status = "fail"
            problems.append(
                {
                    "table": table,
                    "issue": "超过 20% 的股票历史深度落后全局最新期 2 期以上 (疑似抓取截断/回填不完整)",
                    "share": round(share, 6),
                    "cutoff_period": str(cutoff.date()),
                }
            )
    summary = (
        "各表历史深度分布正常"
        if status == "pass"
        else "存在大面积历史深度偏浅 (疑似抓取被截断)"
    )
    return _result("history_depth", status, summary, details, problems)


def check_coverage_and_survivorship(tables: Sequence[str], data_dir: str) -> CheckResult:
    from data_manager.providers.stock_list_provider import STOCK_LIST

    problems: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    status = "pass"
    table = "income_q" if "income_q" in tables else (tables[0] if tables else "income_q")
    frame, _ = load_table(table)
    if frame.empty:
        return _result("coverage_survivorship", "skip", "无数据, 跳过覆盖检查", {})

    rep = pd.to_datetime(frame["report_date"], errors="coerce")
    coverage: dict[str, Any] = {}
    for year in sorted(rep.dt.year.dropna().unique()):
        year_rows = frame[rep.dt.year == year]
        symbols = set(year_rows["symbol"].astype(str).str.zfill(6))
        expected = 0
        for code in STOCK_LIST.get_all_symbol():
            list_date = STOCK_LIST.get_list_date(code)
            delist_date = STOCK_LIST.get_delist_date(code)
            ref = f"{int(year)}-12-31"
            if list_date and str(list_date)[:10] > ref:
                continue
            if delist_date and str(delist_date)[:10] < ref:
                continue
            expected += 1
        coverage[str(int(year))] = {
            "symbols_with_data": len(symbols),
            "symbols_listed_ref": expected,
            "coverage": round(len(symbols) / expected, 4) if expected else None,
        }
    details["coverage_by_year"] = coverage

    low = {year: value for year, value in coverage.items() if year >= "2016" and (value["coverage"] or 0) < 0.95}
    if low:
        status = "fail"
        problems.append({"issue": "2016 年后年度覆盖率 <95%", "years": low})

    # 幸存者偏差: 覆盖率不得随年份单调上升 (退市股必须保留在退市前的报告期)
    values = [value["coverage"] for year, value in sorted(coverage.items()) if value["coverage"]]
    if len(values) >= 3 and all(b >= a - 1e-9 for a, b in zip(values, values[1:])):
        if status == "pass":
            status = "warn"
        problems.append(
            {"issue": "覆盖率随年份单调上升, 疑似只取了当前在市股票 (幸存者偏差)", "values": values}
        )

    delisted = {}
    for code in DELISTED_SAMPLE:
        subset = frame[frame["symbol"].astype(str).str.zfill(6) == code]
        delist_date = STOCK_LIST.get_delist_date(code)
        before = subset[
            pd.to_datetime(subset["report_date"], errors="coerce")
            < (pd.Timestamp(delist_date) if delist_date else pd.Timestamp("2100-01-01"))
        ]
        delisted[code] = {
            "delist_date": delist_date or "",
            "rows": int(len(subset)),
            "rows_before_delist": int(len(before)),
        }
    details["delisted_sample"] = delisted
    missing_delisted = [code for code, info in delisted.items() if info["rows"] == 0]
    if missing_delisted:
        if status == "pass":
            status = "warn"
        problems.append({"issue": "退市股抽查无数据 (幸存者偏差风险)", "symbols": missing_delisted})

    summary = "覆盖与退市样本正常" if status == "pass" else "覆盖/幸存者存在风险"
    return _result("coverage_survivorship", status, summary, details, problems)


# ---------------------------------------------------------------------------
# 7. 极值与可交易性
# ---------------------------------------------------------------------------


def check_extremes_and_tradability(tables: Sequence[str], data_dir: str) -> CheckResult:
    problems: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    status = "pass"

    for table in tables:
        frame, _ = load_table(table)
        if frame.empty:
            continue
        spec = FINANCIAL_TABLES[table]
        quantiles: dict[str, Any] = {}
        for col in spec.amount_columns + spec.share_columns + spec.rate_columns:
            if col not in frame.columns:
                continue
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            if series.empty:
                continue
            quantiles[col] = {
                "p01": float(series.quantile(0.01)),
                "p50": float(series.quantile(0.5)),
                "p99": float(series.quantile(0.99)),
                "min": float(series.min()),
                "max": float(series.max()),
                "negatives": int((series < 0).sum()),
            }
        details[f"{table}_quantiles"] = quantiles

    balance, _ = load_table("balance_q")
    if not balance.empty:
        assets = pd.to_numeric(balance["total_assets"], errors="coerce")
        liabilities = pd.to_numeric(balance["total_liabilities"], errors="coerce")
        equity = pd.to_numeric(balance["total_equity_attr_p"], errors="coerce")
        ratio = (liabilities / equity.replace(0, pd.NA)).where(equity > 0)
        high_leverage = ratio > 1.0
        cash = pd.to_numeric(balance["monetary_funds"], errors="coerce")
        negative_cash = cash < 0
        details["balance_q_extremes"] = {
            "leverage_gt_1_rows": int(high_leverage.sum()),
            "leverage_gt_1_symbols": int(balance.loc[high_leverage, "symbol"].nunique()),
            "negative_monetary_funds": int(negative_cash.sum()),
        }
        if int(negative_cash.sum()):
            status = "warn"
            problems.append({"table": "balance_q", "issue": "货币资金为负", "count": int(negative_cash.sum())})

    # 公告日 → 行情日历 / T+1 可成交
    calendar = trading_calendar()
    tradability: dict[str, Any] = {}
    table = "income_q" if "income_q" in tables else (tables[0] if tables else "")
    if table and calendar:
        frame, _ = load_table(table)
        if not frame.empty:
            sample = frame.dropna(subset=["ann_date"]).head(2000)
            non_trading = 0
            next_day_missing = 0
            for _, row in sample.iterrows():
                ann = pd.Timestamp(row["ann_date"])
                if ann.date() not in set(calendar):
                    non_trading += 1
                effective = effective_trade_date(ann, calendar)
                if pd.isna(effective):
                    next_day_missing += 1
            tradability = {
                "sampled": int(len(sample)),
                "ann_on_non_trading_day": non_trading,
                "no_effective_trade_date": next_day_missing,
                "note": "公告日常在非交易日 (实测约 20%), 属正常; 生效交易日 = 首个 >= 公告日的交易日",
            }
    details["tradability"] = tradability if tradability else "no calendar or no data"

    # 孤儿代码: 财务表出现但行情侧与退市名单都没有 (只统计 A 股口径 6 位代码)
    from data_manager.financial_manager import iter_symbols

    stock_dir = DataPath.STOCK_PATH
    quote_codes = (
        {name[:6] for name in os.listdir(stock_dir) if name.endswith(".csv") and name[:6].isdigit()}
        if os.path.isdir(stock_dir)
        else set()
    )
    from data_manager.providers.stock_list_provider import STOCK_LIST

    known = quote_codes | set(STOCK_LIST.get_all_symbol())
    fin_codes = {code for code in iter_symbols(tables) if code.startswith(("0", "3", "6"))}
    # 结构性外池代码不算孤儿 (B 股 200/900、北交所 4xx/8xx/92x: 客户范围外但源端会给)
    extras = {code for code in iter_symbols(tables) if code.startswith(("2", "4", "8", "9"))}
    orphans = sorted(fin_codes - known)
    share = len(orphans) / len(fin_codes) if fin_codes else 0.0
    details["orphan_symbols"] = {
        "financial_symbols_a_share": len(fin_codes),
        "structurally_out_of_scope": {
            "count": len(extras),
            "note": "B 股(2xx/9xx)与北交所(4xx/8xx/92x)属客户范围外, 不计入孤儿率",
            "examples": sorted(extras)[:10],
        },
        "orphans": len(orphans),
        "share": round(share, 6),
        "examples": orphans[:20],
        "note": (
            "孤儿 = 财务数据里 6 位 A 股代码在 data/stock_data 与 stock_name_list 均不存在; "
            "阈值 0.1% 按全量样本 (5000+ 只) 设定, 小样本冒烟时会偏高"
        ),
    }
    # 判据只对"逐股抓取"的报表表生效: 报表是**按我们的股票池逐股**抓的, 出现池外代码
    # 说明股票池漂移 (真缺陷)。预告/分红/股东户数是**按报告期全市场**抓的, 天然会带入
    # 快照外代码 (新股未入快照、退市股、无行情文件), 这不是财务取数越界 —— 这类代码
    # 没有行情文件, 合并时天然不可见, 只作记录。
    stmt_orphans = sorted(
        {
            code
            for code in iter_symbols([t for t in ("income_q", "balance_q", "cashflow_q") if t in FINANCIAL_TABLES])
            if code.startswith(("0", "3", "6"))
        }
        - known
    )
    event_only_orphans = sorted(set(orphans) - set(stmt_orphans))
    details["orphan_symbols"]["statement_orphans"] = {
        "count": len(stmt_orphans),
        "share": round(len(stmt_orphans) / len(fin_codes), 6) if fin_codes else 0.0,
        "examples": stmt_orphans[:10],
        "note": "报表表逐股抓取, 出现池外代码 = 股票池漂移 (阈值 0.1%)",
    }
    details["orphan_symbols"]["event_only_orphans"] = {
        "count": len(event_only_orphans),
        "examples": event_only_orphans[:15],
        "note": (
            "仅出现在逐期全市场表 (预告/分红/股东户数): 新股未入 STOCK_LIST 快照或退市股, "
            "无行情文件 → 研究合并时不可见; 属行情侧快照滞后, 非财务取数缺陷"
        ),
    }
    stmt_share = len(stmt_orphans) / len(fin_codes) if fin_codes else 0.0
    if stmt_share > ORPHAN_MAX:
        status = "fail" if status == "pass" else status
        problems.append(
            {
                "issue": "报表表孤儿代码占比超过 0.1% (逐股抓取却出现池外代码 → 股票池漂移)",
                "share": round(stmt_share, 6),
                "examples": stmt_orphans[:10],
            }
        )
    if event_only_orphans:
        if status == "pass":
            status = "warn"
        problems.append(
            {
                "issue": "逐期全市场表带入快照外代码 (无行情文件, 合并时不可见; 行情侧快照滞后)",
                "count": len(event_only_orphans),
                "examples": event_only_orphans[:10],
            }
        )

    summary = "极值与可交易性检查完成" if status == "pass" else "极值/可交易性存在告警"
    return _result("extremes_tradability", status, summary, details, problems)


# ---------------------------------------------------------------------------
# 7.5 跨源审计 (离线部分; 联网抽样审计结果从 audit_report.json 读取)
# ---------------------------------------------------------------------------


def check_cross_source_audit(tables: Sequence[str], data_dir: str) -> CheckResult:
    """跨源一致性: 股本双源、行情侧流通股本、联网抽样审计的新鲜度与一致率。

    体检其余各项都是**单源不变量**(自洽性), 抓不到"源端整体偏移"这类错误 ——
    本次 `ann_date` 错位就是靠跨源比对发现的, 故把跨源审计固化成常驻项。
    """
    from data_manager import financial_audit as audit

    metrics: dict[str, Any] = {}
    problems: list[dict[str, Any]] = []
    status = "pass"

    shares = audit.audit_share_capital()
    metrics["share_capital_two_sources"] = {k: v for k, v in shares.items() if k != "examples"}
    if shares.get("status") == "ok":
        agreement = float(shares.get("agreement") or 0.0)
        if agreement < CROSS_SOURCE_MIN_AGREEMENT:
            status = "fail"
            problems.append(
                {
                    "issue": "股本双源一致率低于下限 (东财 F10 vs 巨潮)",
                    "agreement": agreement,
                    "threshold": CROSS_SOURCE_MIN_AGREEMENT,
                    "examples": shares.get("examples", [])[:5],
                }
            )
        elif agreement < CROSS_SOURCE_WARN_AGREEMENT:
            if status == "pass":
                status = "warn"
            problems.append(
                {
                    "issue": "股本双源存在个股级差异 (东财 F10 的 total_share 偏旧, 巨潮/行情侧更接近实际)",
                    "agreement": agreement,
                    "mismatch_rows": shares.get("mismatch_rows"),
                    "examples": shares.get("examples", [])[:5],
                }
            )

    quotes = audit.audit_quote_float_share()
    metrics["quote_float_share"] = {k: v for k, v in quotes.items() if k != "examples"}
    if quotes.get("status") == "ok":
        bad_share = float(quotes.get("float_exceeds_total_share") or 0.0)
        if bad_share > FLOAT_EXCEEDS_TOTAL_MAX:
            status = "fail"
            problems.append(
                {
                    "issue": "流通股本超过总股本的占比过高 (行情侧 float_share vs 财务侧 total_share)",
                    "share": bad_share,
                    "threshold": FLOAT_EXCEEDS_TOTAL_MAX,
                    "examples": quotes.get("examples", [])[:5],
                }
            )
        elif quotes.get("float_exceeds_total"):
            if status == "pass":
                status = "warn"
            problems.append(
                {
                    "issue": "存在流通股本 > 总股本的行 (两源对股本的生效时点不一致, 量化到具体个股)",
                    "rows": quotes.get("float_exceeds_total"),
                    "share": bad_share,
                    "examples": quotes.get("examples", [])[:3],
                }
            )

    # 联网抽样审计: 由 libs/scripts/audit_financials.py 写 audit_report.json, 这里只读不联
    report = audit.load_audit_report()
    age = audit.audit_age_days(report)
    values = (report.get("suites") or {}).get("values") or {}
    metrics["network_audit"] = {
        "generated_at": report.get("generated_at", ""),
        "age_days": round(age, 2) if age is not None else None,
        "sampled_symbols": values.get("sampled_symbols"),
        "fields": {
            field: {k: v for k, v in info.items() if k != "examples"}
            for field, info in (values.get("fields") or {}).items()
        },
    }
    if not report:
        if status == "pass":
            status = "warn"
        problems.append(
            {
                "issue": "尚无联网抽样审计报告 (运行 libs/scripts/audit_financials.py --suites values 生成)",
                "expect": str(audit.audit_fp()),
            }
        )
    elif age is not None and age > audit.MAX_AUDIT_AGE_DAYS:
        if status == "pass":
            status = "warn"
        problems.append(
            {"issue": "联网抽样审计报告已过期, 需重跑", "age_days": round(age, 1), "max_age_days": audit.MAX_AUDIT_AGE_DAYS}
        )
    else:
        for field, info in (values.get("fields") or {}).items():
            if not info.get("gated"):
                continue
            if float(info.get("agreement") or 0.0) < CROSS_SOURCE_MIN_AGREEMENT:
                status = "fail"
                problems.append(
                    {
                        "issue": f"字段 {field} 与新浪逐格一致率低于下限",
                        "agreement": info.get("agreement"),
                        "compared": info.get("compared"),
                        "examples": info.get("examples", [])[:3],
                    }
                )

    summary = "跨源审计通过" if status == "pass" else "跨源一致性需要关注"
    return _result("cross_source_audit", status, summary, metrics, problems)


# ---------------------------------------------------------------------------
# 8. 字段可填率
# ---------------------------------------------------------------------------


def check_field_fillability(tables: Sequence[str], data_dir: str) -> CheckResult:
    rows: list[dict[str, Any]] = []
    for no in sorted(FINANCIAL_FIELD_SOURCES):
        name, table, source = FINANCIAL_FIELD_SOURCES[no]
        implemented = field_implemented(no)
        filled: float | None = None
        if table and table in FINANCIAL_TABLES:
            frame, _ = load_table(table)
            if not frame.empty and name in frame.columns:
                filled = round(float(frame[name].notna().mean()), 6)
        rows.append(
            {
                "no": no,
                "field": name,
                "table": table,
                "source": source,
                "implemented": implemented,
                "filled_rate": filled,
            }
        )
    not_supported = [row for row in rows if not row["implemented"] and row["no"] != 29]
    summary = (
        f"33 字段中 {sum(1 for r in rows if r['implemented'])} 项有落地表, "
        f"{len(not_supported)} 项无 PIT 源 (已声明)"
    )
    return _result(
        "field_fillability",
        "warn" if not_supported else "pass",
        summary,
        {"fields": rows, "not_supported": [row["field"] for row in not_supported]},
        [],
    )


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

CHECK_ORDER = (
    "schema_and_column_order",
    "missing_value_convention",
    "update_time_integrity",
    "field_missing_rates",
    "date_axis_invariants",
    "history_depth",
    "forecast_leadtime",
    "revision_quantification",
    "unit_dimension_checks",
    "coverage_survivorship",
    "extremes_tradability",
    "cross_source_audit",
    "field_fillability",
)


def build_report(
    tables: Sequence[str] | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """跑全部体检项, 返回可 JSON 序列化的报告 dict。"""
    resolved_dir = data_dir or DataPath.FINANCIAL_DIR
    selected = list(tables) if tables else list(FINANCIAL_TABLES)
    checks = [
        check_schema(selected, resolved_dir),
        check_placeholder_fills(selected, resolved_dir),
        check_update_time_monotonic(selected, resolved_dir),
        check_missing_rates(selected, resolved_dir),
        check_date_axis(selected, resolved_dir),
        check_history_depth(selected, resolved_dir),
        check_forecast_leadtime(selected, resolved_dir),
        check_revision_quantification(selected, resolved_dir),
        check_dimensions(selected, resolved_dir),
        check_coverage_and_survivorship(selected, resolved_dir),
        check_extremes_and_tradability(selected, resolved_dir),
        check_cross_source_audit(selected, resolved_dir),
        check_field_fillability(selected, resolved_dir),
    ]
    order = {name: index for index, name in enumerate(CHECK_ORDER)}
    checks.sort(key=lambda item: order.get(item.name, 99))
    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    for item in checks:
        counts[item.status] = counts.get(item.status, 0) + 1
    return {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": resolved_dir,
        "tables": selected,
        "summary": counts,
        "verdict": "fail" if counts.get("fail") else ("warn" if counts.get("warn") else "pass"),
        "checks": [item.as_dict() for item in checks],
        "table_stats": {table: table_stats(table) for table in selected},
    }


def write_report(report: dict[str, Any], path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    return path


def print_report(report: dict[str, Any]) -> None:
    icons = {"pass": "✅", "warn": "⚠️", "fail": "❌", "skip": "➖"}
    print(f"\n财务数据体检 · {report['generated_at']} · 目录 {report['data_dir']}")
    print(f"判定: {report['verdict'].upper()}  {report['summary']}\n")
    for check in report["checks"]:
        print(f"{icons.get(check['status'], '?')} [{check['status']:4}] {check['name']}: {check['summary']}")
        for failure in check["failures"][:3]:
            print(f"        - {failure}")


__all__ = [
    "CHECK_ORDER",
    "CheckResult",
    "build_report",
    "check_coverage_and_survivorship",
    "check_date_axis",
    "check_dimensions",
    "check_extremes_and_tradability",
    "check_field_fillability",
    "check_forecast_leadtime",
    "check_history_depth",
    "check_missing_rates",
    "check_placeholder_fills",
    "check_revision_quantification",
    "check_schema",
    "check_update_time_monotonic",
    "print_report",
    "write_report",
]
