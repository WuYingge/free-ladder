"""财报/财务数据集 schema 与口径纯函数 (Financial Dataset Schema)

本模块只做"规格"与"纯函数", 不做网络与落盘, 供
``libs/fetcher/financial.py`` / ``libs/data_manager/financial_manager.py`` /
``libs/data_manager/providers/financial_provider.py`` 共用。

数据落地约定 (对应公司 2026-09-10 取数单与需求清单 §8.2/§8.3):
  * 目录 ``data/financial/``, 每张表一个宽表 CSV (UTF-8 with BOM, 表头 = 首行);
  * 列序: 前四列固定 ``symbol, report_date, ann_date, report_type``;
    取数单点名的 ``net_profit_attr_p, total_share, revenue`` 紧随为第 5~7 列
    (income_q 表头与取数单逐字一致); 其余财务列按需求清单 §8.2 编号顺序;
    末尾三个方法论列 ``currency, update_date, basis, update_time``;
  * 金额一律**元**(原始小数保留), 股本一律**股**, 比率一律**小数**;
    缺失一律**留空**(NaN), 禁止 0 / -1 / "NULL" 填充 (见 ``PLACEHOLDER_TOKENS``);
  * 金额字段一律**累计口径**(三季报 = 前三季合计), 单季由 ``quarter_diff`` 差分;
  * ``ann_date`` = 该财报**首次对外公布日**; ``report_date`` = 会计截止日;
    两者共同构成 PIT (point-in-time) 两条时间轴, 缺一不可。

方法论文档列 (这是"日期与数值是否可信"的机器可读答案):
  * ``update_date`` — 数据源标注的最后更正日 (东财 F10 的 ``UPDATE_DATE``);
    无此概念的表留空。
  * ``basis``      — ``first_reported`` 表示 notice_date == update_date,
    当前数值即首次公告版本; ``revised`` 表示该行数值在首次公告后被人为改写过
    (数值可能是追溯调整后的版本, 不能还原首次公告值); ``latest_version``
    表示该表无版本概念 (分红预案、股东户数); ``_mixed`` 表示同一股票内混装。
    研究与因子默认口径: **只取 first_reported** (见 provider 的 basis 参数)。
  * ``update_time`` — 本仓落盘时间, 供 ``batch_check_financials_updated`` 判新鲜度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# 口径常量
# ---------------------------------------------------------------------------

CURRENCY_CNY = "CNY"
DATE_FORMAT = "%Y-%m-%d"

#: 报告期类型 (由 report_date 的月日决定)
REPORT_TYPE_BY_MMDD: dict[str, str] = {
    "03-31": "Q1",
    "06-30": "H1",
    "09-30": "Q3",
    "12-31": "ANNUAL",
}

#: 报告类型 → 法定披露截止 (年报/一季报 4/30, 半年报 8/31, 三季报 10/31;
#: 年报的 4/30 在**次年**, 由 legal_deadline 处理跨年)
DISCLOSURE_DEADLINE_MMDD: dict[str, str] = {
    "Q1": "04-30",
    "H1": "08-31",
    "Q3": "10-31",
    "ANNUAL": "04-30",
}

#: 禁止用作缺失填充的哨兵值 (字符串原始值层面比对, 早于数值转换)
PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {"null", "none", "nan", "n/a", "na", "-", "--", "nil", "undefined", "\\n"}
)

#: basis 取值
BASIS_FIRST_REPORTED = "first_reported"
BASIS_REVISED = "revised"
BASIS_LATEST_VERSION = "latest_version"
BASIS_MIXED = "_mixed"

#: 三张报表的具体口径说明 (写入文档与报告)
STATEMENT_BASIS_NOTE = (
    "income_q/balance_q/cashflow_q 的金额为口径累计值 (资产负债表为期末时点值); "
    "basis=revised 表示该行数值在其首次公告后被追溯调整过。"
)

# ---------------------------------------------------------------------------
# 表规格
# ---------------------------------------------------------------------------

Granularity = Literal["symbol", "period"]

#: 三张财务报表: 逐股抓取 (一次拿全历史)
INCOME_COLUMNS: tuple[str, ...] = (
    "symbol",
    "report_date",
    "ann_date",
    "report_type",
    "net_profit_attr_p",
    "total_share",
    "revenue",
    "eps_basic",
    "operating_profit",
    "net_profit",
    "net_profit_deducted",
    "minority_interest_profit",
    "rd_expense",
    "sell_admin_expense",
    "currency",
    "update_date",
    "basis",
    "update_time",
)

BALANCE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "report_date",
    "ann_date",
    "report_type",
    "total_assets",
    "total_equity_attr_p",
    "total_liabilities",
    "total_equity",
    "total_share",
    "monetary_funds",
    "accounts_receivable",
    "inventory",
    "goodwill",
    "minority_interest_equity",
    "currency",
    "update_date",
    "basis",
    "update_time",
)

CASHFLOW_COLUMNS: tuple[str, ...] = (
    "symbol",
    "report_date",
    "ann_date",
    "report_type",
    "ocf_net",
    "currency",
    "update_date",
    "basis",
    "update_time",
)

#: 事件表 (需求清单 §8.4)
FORECAST_COLUMNS: tuple[str, ...] = (
    "symbol",
    "report_date",
    "ann_date",
    "report_type",
    "forecast_ann_date",
    "forecast_indicator_cn",
    "announce_type",
    "announce_type_en",
    "forecast_net_profit_low",
    "forecast_net_profit_high",
    "forecast_net_profit_mid",
    "yoy_low",
    "yoy_high",
    "update_date",
    "basis",
    "update_time",
)

SHARE_CAPITAL_COLUMNS: tuple[str, ...] = (
    "symbol",
    "ann_date",
    "effective_date",
    "total_share",
    "circ_share",
    "reason",
    "update_time",
)

DIVIDEND_COLUMNS: tuple[str, ...] = (
    "symbol",
    "report_date",
    "ann_date",
    "report_type",
    "plan_ann_date",
    "implement_date",
    "cash_div_per_share",
    "bonus_ratio",
    "transfer_ratio",
    "update_time",
)

HOLDER_NUM_COLUMNS: tuple[str, ...] = (
    "symbol",
    "report_date",
    "ann_date",
    "report_type",
    "holder_num",
    "update_time",
)

_METHOD_COLUMNS: tuple[str, ...] = ("update_date", "basis")

_DATE_ROLE_COLUMNS: tuple[str, ...] = (
    "report_date",
    "ann_date",
    "update_date",
    "effective_date",
    "plan_ann_date",
    "implement_date",
    "forecast_ann_date",
)


#: 取数单 (inputs/2026-09-10_financial_data_order.md) 明确要求的 income_q 表头前 7 列
INCOME_ORDER_HEAD: list[str] = [
    "symbol",
    "report_date",
    "ann_date",
    "report_type",
    "net_profit_attr_p",
    "total_share",
    "revenue",
]

#: 给人工/报错信息用的列序说明
FIELD_ORDER_HELP = (
    "列序约定: symbol, report_date, ann_date, report_type 固定在前四列; "
    "income_q 第 5~7 列为 net_profit_attr_p, total_share, revenue; "
    "其余财务列按需求清单 §8.2 编号顺序; 末尾为 currency, update_date, basis, update_time。"
)


@dataclass(frozen=True, slots=True)
class FinancialTableSpec:
    """单张财务表的 schema 规格。"""

    key: str
    filename: str
    columns: tuple[str, ...]
    #: 事件索引列 (事件帧 index); None 表示该表无单一时点列
    event_date_column: str | None
    #: 逐股表 / 逐期(全市场)表
    granularity: Granularity
    #: 金额列 (元, 累计口径)
    amount_columns: tuple[str, ...] = ()
    #: 股本列 (股)
    share_columns: tuple[str, ...] = ()
    #: 比率列 (小数)
    rate_columns: tuple[str, ...] = ()
    #: 枚举/文本列
    text_columns: tuple[str, ...] = ()
    #: 该表是否承载 PIT 财务时序 (除股本事件表外均为 True)
    pit_required: bool = True
    #: 是否具备版本更正概念 (决定 basis 取 first_reported/revised 还是 latest_version)
    versioned: bool = False
    #: True 表示预告/快报类: ann_date 允许早于 report_date (期间结束前的前瞻信息),
    #: 下界放宽到会计年度起始日; False 表示正式财报: ann_date 不得早于报告期末
    pre_period_allowed: bool = False
    #: 结构性空列: 该列按信源能力**注定为空**, 不作缺失率告警 (如 F10 利润表无股本字段)
    structurally_empty_columns: tuple[str, ...] = ()
    #: 月度披露峰检查是否适用: 仅"某报告期的正式财报集中披露"适用 (4/8/10 峰);
    #: 预告/分红等事件表在单一报告期内天然分散, 不做峰形判定
    peak_shape_applicable: bool = True
    #: 法定披露截止期检查是否适用: **仅正式财报**适用 (4/30、8/31、10/31)。
    #: 实测: 业绩预告没有法定截止期 (可在任意时点发布/更正, 延误天数实测 p50=15.5 / max=203 天),
    #: 分红预案同样不受财报披露期约束 (中期分红可晚于报告期数月公告) → 二者均置 False。
    deadline_check_applicable: bool = True
    #: 说明 (写入报告)
    note: str = ""

    def __post_init__(self) -> None:
        missing = [
            col
            for col in (*self.amount_columns, *self.share_columns, *self.rate_columns)
            if col not in self.columns
        ]
        if missing:
            raise ValueError(f"{self.key}: 类型列不在列序中: {missing}")
        if "symbol" not in self.columns:
            raise ValueError(f"{self.key}: 缺少 symbol 列")

    @property
    def date_columns(self) -> tuple[str, ...]:
        return tuple(col for col in self.columns if col in _DATE_ROLE_COLUMNS)

    @property
    def method_columns(self) -> tuple[str, ...]:
        return tuple(col for col in self.columns if col in _METHOD_COLUMNS)

    @property
    def dedup_key(self) -> tuple[str, ...]:
        """落盘去重键 = 事实主键 (**不含 ann_date**)。

        实测结论: 源端对每个报告期**只提供一行** (更正没有独立版本行, 见
        ``RPT_LICO_FN_CPD`` 每股票每期唯一), 所以 ``(symbol, report_date)`` 就是事实主键。
        把 ``ann_date`` 放进键会让"缺公告日的旧行"与"带公告日的新行"并存, 而重跑时
        陈旧的不完整行会胜出 (实测踩过: 33k 行 ann_date 缺失本可自愈却没被覆盖)。

        ``forecast`` 额外带 ``forecast_indicator_cn``: 同一股票同一公告日会同时给出
        "归母净利润" 与 "扣非净利润" 两行, 它们是并列事实而非版本, 不能互相覆盖。
        """
        base: list[str] = ["symbol"]
        if "report_date" in self.columns:
            base.append("report_date")
        if self.key == "forecast":
            base.append("forecast_indicator_cn")
        return tuple(base)

    @property
    def prefix_columns(self) -> tuple[str, ...]:
        """公司要求固定的共同前缀列。"""
        return tuple(
            col
            for col in ("symbol", "report_date", "ann_date", "report_type")
            if col in self.columns
        )


FINANCIAL_TABLES: dict[str, FinancialTableSpec] = {
    "income_q": FinancialTableSpec(
        key="income_q",
        filename="income_q.csv",
        columns=INCOME_COLUMNS,
        event_date_column="ann_date",
        granularity="symbol",
        amount_columns=(
            "net_profit_attr_p",
            "revenue",
            "operating_profit",
            "net_profit",
            "net_profit_deducted",
            "minority_interest_profit",
            "rd_expense",
            "sell_admin_expense",
        ),
        share_columns=(),
        rate_columns=("eps_basic",),
        versioned=True,
        structurally_empty_columns=("total_share",),
        note=(
            "利润表 (累计口径, 元)。注意: 东财 F10 利润表**没有股本字段**, "
            "本表 total_share 保留列位但恒为空 (与取数单表头兼容), "
            "总股本请取 balance_q.total_share 或 share_capital 表。"
            "恒等式: net_profit = net_profit_attr_p + minority_interest_profit。"
        ),
    ),
    "balance_q": FinancialTableSpec(
        key="balance_q",
        filename="balance_q.csv",
        columns=BALANCE_COLUMNS,
        event_date_column="ann_date",
        granularity="symbol",
        amount_columns=(
            "total_assets",
            "total_equity_attr_p",
            "total_liabilities",
            "total_equity",
            "monetary_funds",
            "accounts_receivable",
            "inventory",
            "goodwill",
            "minority_interest_equity",
        ),
        share_columns=("total_share",),
        versioned=True,
        note=(
            "资产负债表 (期末时点值, 元)。恒等式: 资产 = 负债 + 所有者权益; "
            "total_share = 期末股本 (股), 是 income_q 缺失股本字段的替代来源; "
            "minority_interest_equity = 少数股东**权益** (净资产科目)。"
            "注意与利润表的少数股东**损益** (income_q.minority_interest_profit) 区分, 二者量纲相近但语义不同。"
        ),
    ),
    "cashflow_q": FinancialTableSpec(
        key="cashflow_q",
        filename="cashflow_q.csv",
        columns=CASHFLOW_COLUMNS,
        event_date_column="ann_date",
        granularity="symbol",
        amount_columns=("ocf_net",),
        versioned=True,
        note="现金流量表 (累计口径, 元)。ocf_net = 经营活动产生的现金流量净额。",
    ),
    "forecast": FinancialTableSpec(
        key="forecast",
        filename="forecast.csv",
        columns=FORECAST_COLUMNS,
        event_date_column="ann_date",
        granularity="period",
        amount_columns=(
            "forecast_net_profit_low",
            "forecast_net_profit_high",
            "forecast_net_profit_mid",
        ),
        rate_columns=("yoy_low", "yoy_high"),
        text_columns=("announce_type", "announce_type_en", "forecast_indicator_cn"),
        versioned=True,
        pre_period_allowed=True,
        peak_shape_applicable=False,
        deadline_check_applicable=False,
        note=(
            "业绩预告 (事件表, **期间结束前的前瞻信息**)。ann_date = 预告披露日 "
            "(与 forecast_ann_date 同值, 兼容需求清单的两种列名); 金额为归母净利预测"
            "区间 (元); forecast_indicator_cn 区分归母/扣非两种口径 (同一公告日两行); "
            "因此 ann_date 允许早于 report_date (下界 = 会计年度起始日)。"
        ),
    ),
    "share_capital": FinancialTableSpec(
        key="share_capital",
        filename="share_capital.csv",
        columns=SHARE_CAPITAL_COLUMNS,
        event_date_column="ann_date",
        granularity="symbol",
        share_columns=("total_share", "circ_share"),
        text_columns=("reason",),
        pit_required=True,
        note=(
            "股本变动事件表 (股)。ann_date = 公告日, effective_date = 变动生效/变动日; "
            "**总股本/流通股本以本表为准** (income_q.total_share 仅为报表口径快照)。"
        ),
    ),
    "dividend": FinancialTableSpec(
        key="dividend",
        filename="dividend.csv",
        columns=DIVIDEND_COLUMNS,
        event_date_column="ann_date",
        granularity="period",
        amount_columns=("cash_div_per_share",),
        rate_columns=("bonus_ratio", "transfer_ratio"),
        pre_period_allowed=True,
        peak_shape_applicable=False,
        deadline_check_applicable=False,
        note=(
            "分红送转 (事件表)。ann_date = plan_ann_date = 预案公告日; "
            "cash_div_per_share 单位元/股; bonus_ratio/transfer_ratio 为每 10 股比例。"
            "中期分红预案可能在报告期结束前公告 (实测 000157 在 2026-03-31 公告 "
            "2026-06-30 报告期的分红) → 允许 ann_date 早于 report_date, 下界 = 会计年度起始日。"
        ),
    ),
    "holder_num": FinancialTableSpec(
        key="holder_num",
        filename="holder_num.csv",
        columns=HOLDER_NUM_COLUMNS,
        event_date_column="ann_date",
        granularity="period",
        text_columns=(),
        note="股东户数 (股东户数统计截止日 → report_date, 公告日 → ann_date)。",
    ),
}

#: 表 key → 需求清单 §③-A 字段号的覆盖说明 (供报告反查)
TABLE_FIELD_COVERAGE: dict[str, tuple[int, ...]] = {
    "income_q": (3, 4, 5, 6, 7, 8, 17, 31, 33),
    "balance_q": (10, 11, 12, 13, 14, 15, 16, 30, 32),
    "cashflow_q": (9,),
    "forecast": (18, 19, 20, 21),
    "share_capital": (17, 25),
    "dividend": (27,),
    "holder_num": (26,),
}

# ---------------------------------------------------------------------------
# §③-A 33 字段 → 实现状态
# ---------------------------------------------------------------------------

#: 无 PIT 数据源可得 (不得静默留空, 必须在报告中声明)
NOT_SUPPORTED_FIELDS: dict[int, str] = {
    22: "st_status_hist — 无 PIT 源; stock_name_list.csv 仅当前快照无时间戳, 禁止反推历史",
    23: "suspend_hist — 无 PIT 源; 现仅能由行情日历缺行隐含",
    24: "limit_hist — 无 PIT 源; 需 ST/板块规则表才能反推涨跌停带",
    28: "index_membership — 无 PIT 源; data/index 只有指数行情, 无成分股历史",
}

#: 33 字段机读表: 字段号 → (字段名, 落地表 key 或 None, 数据源/说明)
FINANCIAL_FIELD_SOURCES: dict[int, tuple[str, str | None, str]] = {
    1: ("ann_date", None, "*_q/forecast 各表共同列; 东财 F10 NOTICE_DATE / 东财预告 NOTICE_DATE"),
    2: ("report_date", None, "*_q/forecast 各表共同列; 东财 F10 REPORT_DATE"),
    3: ("net_profit_attr_p", "income_q", "东财 F10 lrb PARENT_NETPROFIT"),
    4: ("eps_basic", "income_q", "东财 F10 lrb BASIC_EPS"),
    5: ("revenue", "income_q", "东财 F10 lrb TOTAL_OPERATE_INCOME"),
    6: ("operating_profit", "income_q", "东财 F10 lrb OPERATE_PROFIT"),
    7: ("net_profit", "income_q", "东财 F10 lrb NETPROFIT (含少数股东)"),
    8: ("net_profit_deducted", "income_q", "东财 F10 lrb DEDUCT_PARENT_NETPROFIT"),
    9: ("ocf_net", "cashflow_q", "东财 F10 xjllb NETCASH_OPERATE"),
    10: ("total_assets", "balance_q", "东财 F10 zcfzb TOTAL_ASSETS"),
    11: ("total_equity_attr_p", "balance_q", "东财 F10 zcfzb TOTAL_PARENT_EQUITY"),
    12: ("total_liabilities", "balance_q", "东财 F10 zcfzb TOTAL_LIABILITIES"),
    13: ("total_equity", "balance_q", "东财 F10 zcfzb TOTAL_EQUITY"),
    14: ("monetary_funds", "balance_q", "东财 F10 zcfzb MONETARYFUNDS"),
    15: ("accounts_receivable", "balance_q", "东财 F10 zcfzb ACCOUNTS_RECE"),
    16: ("inventory", "balance_q", "东财 F10 zcfzb INVENTORY"),
    17: ("total_share", "balance_q", "东财 F10 zcfzb SHARE_CAPITAL (期末股本, 股); 建议以 share_capital 表为准"),
    18: ("announce_type", "forecast", "东财 RPT_PUBLIC_OP_NEWPREDICT PREDICT_TYPE (+ FORECAST_STATE 英文枚举)"),
    19: ("forecast_net_profit_low", "forecast", "东财 RPT_PUBLIC_OP_NEWPREDICT PREDICT_AMT_LOWER"),
    20: ("forecast_net_profit_high", "forecast", "东财 RPT_PUBLIC_OP_NEWPREDICT PREDICT_AMT_UPPER"),
    21: ("forecast_ann_date", "forecast", "东财 RPT_PUBLIC_OP_NEWPREDICT NOTICE_DATE"),
    22: ("st_status_hist", None, NOT_SUPPORTED_FIELDS[22]),
    23: ("suspend_hist", None, NOT_SUPPORTED_FIELDS[23]),
    24: ("limit_hist", None, NOT_SUPPORTED_FIELDS[24]),
    25: ("circ_share", "share_capital", "巨潮 stock_share_change_cninfo 已流通股份"),
    26: ("holder_num", "holder_num", "东财 RPT_HOLDERNUM_DET 股东户数-本次"),
    27: ("divid_pre_plan_ann_date", "dividend", "东财 RPT_SHAREBONUS_DET 预案公告日"),
    28: ("index_membership", None, NOT_SUPPORTED_FIELDS[28]),
    29: ("sw_industry_pit", None, "已在位: data/const/stock_sw_industry_clf.csv (with_industry)"),
    30: ("goodwill", "balance_q", "东财 F10 zcfzb GOODWILL"),
    31: ("rd_expense", "income_q", "东财 F10 lrb RESEARCH_EXPENSE"),
    32: ("minority_interest", "income_q", "东财 F10 lrb MINORITY_INTEREST = 少数股东损益; balance_q.minority_interest_equity 为少数股东权益"),
    33: ("sell_admin_expense", "income_q", "东财 F10 lrb SALE_EXPENSE + MANAGE_EXPENSE"),
}


def field_implemented(field_no: int) -> bool:
    """该字段是否有落地表 (29 为既有 with_industry, 视为已在位)。"""
    if field_no == 29:
        return True
    return FINANCIAL_FIELD_SOURCES.get(field_no, (None, None, ""))[1] is not None


# ---------------------------------------------------------------------------
# 质量报告
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TableQuality:
    """单表规范化质量报告 (不抛错, 只记账)。"""

    table: str
    filename: str
    n_rows: int = 0
    schema_ok: bool = True
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    column_order_ok: bool = True
    bad_symbols: int = 0
    bad_symbol_examples: list[str] = field(default_factory=list)
    bad_dates: int = 0
    bad_date_examples: list[str] = field(default_factory=list)
    non_numeric: int = 0
    non_numeric_examples: list[str] = field(default_factory=list)
    placeholder_fills: int = 0
    placeholder_examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "filename": self.filename,
            "n_rows": self.n_rows,
            "schema_ok": self.schema_ok,
            "column_order_ok": self.column_order_ok,
            "missing_columns": self.missing_columns,
            "unexpected_columns": self.unexpected_columns,
            "bad_symbols": self.bad_symbols,
            "bad_symbol_examples": self.bad_symbol_examples,
            "bad_dates": self.bad_dates,
            "bad_date_examples": self.bad_date_examples,
            "non_numeric": self.non_numeric,
            "non_numeric_examples": self.non_numeric_examples,
            "placeholder_fills": self.placeholder_fills,
            "placeholder_examples": self.placeholder_examples,
        }


# ---------------------------------------------------------------------------
# 纯函数: 口径与日期
# ---------------------------------------------------------------------------

CANONICAL_DATE_FORMAT = "%Y-%m-%d"

#: 源端"无此日期"的哨兵值: 实测东财 F10 对**上市前报告期**(招股书一次性披露的历史期间)
#: 返回 ``1900-01-01 00:05:43`` (Excel 纪元残留)。它**不是**真实公告日:
#: 若当值使用, ``ann_date=1900`` 会让该期数据在整条行情序列上恒可见 → 前视偏差。
#: 故一律按"源端声明未知"置空; 置空后该行会被 PIT 行级门禁剔除 (缺公告日不可定位时点)。
SENTINEL_DATES: tuple[pd.Timestamp, ...] = (pd.Timestamp("1900-01-01"),)


def parse_date_column(
    values: object,
    *,
    strict: bool = False,
    context: str = "",
) -> pd.Series:
    """把任意日期表示解析成 ``datetime64``, **显式兼容混格式**。

    **踩过的坑 (别退回朴素解析)**: pandas 3 的 ``pd.to_datetime`` 在**混格式**列上按首个
    非空值推断单一格式, 把少数派格式**静默**判成 NaT (无警告)。实测一列里同时存在
    ``2021-04-20`` 与 ``2021-04-20 00:00:00`` 时, 后者全变 NaT —— 三张报表合计 12.7 万行
    ``ann_date`` 因此变成 NaT, 随后被 PIT 行级门禁整行剔除 (表面症状是"缺公告日",
    实为解析丢数据)。故一律走 ``format="mixed"``。

    :param strict: True 时"非空但不可解析"直接抛 ``ValueError``。宁可让更新任务失败,
        也不允许静默把公告日清空 (清空即等于该行退出 PIT 可见集)。
    :param context: 出错信息里附带的定位串 (如 ``income_q.ann_date``)。
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.Series(pd.to_datetime(series, errors="coerce"), index=series.index)
        blank = series.isna()
    else:
        text = series.astype("string").str.strip()
        blank = text.isna() | text.eq("")
        parsed = pd.Series(
            pd.to_datetime(series, errors="coerce", format="mixed"), index=series.index
        )
    # 哨兵值 (1900-01-01) = 源端声明"无此日期" → 置空; 不计入"不可解析"(不是脏值)
    sentinel = parsed.notna() & parsed.dt.normalize().isin(list(SENTINEL_DATES))
    unparsed = parsed.isna() & ~blank.reindex(series.index, fill_value=True) & ~sentinel
    if bool(unparsed.any()):
        examples = [str(value) for value in series[unparsed].head(5).tolist()]
        if strict:
            raise ValueError(
                f"{context or 'date column'}: {int(unparsed.sum())} 行非空但不可解析的日期 "
                f"{examples}; 拒绝静默清空 (如需排查请直接检查落盘 CSV 该列)"
            )
    return parsed.mask(sentinel)


def is_canonical_date_text(value: object) -> bool:
    """判断落盘文本是否为规范 ``YYYY-MM-DD`` (空值视为规范)。"""
    if value is None:
        return True
    try:
        if pd.isna(value):  # 含 pandas <NA> / NaN / NaT
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "nat", "none", "<na>"}:
        return True
    if len(text) != 10 or text[4] != "-" or text[7] != "-":
        return False
    return text[:4].isdigit() and text[5:7].isdigit() and text[8:10].isdigit()


def report_type_of(value: object) -> str:
    """由日期/月日推断报告期类型; 非标准报告期返回空串 (Q1/H1/Q3/ANNUAL)。"""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return REPORT_TYPE_BY_MMDD.get(f"{ts.month:02d}-{ts.day:02d}", "")


def period_key(value: object) -> str:
    """报告期归一为 YYYY-MM-DD 字符串; 解析失败返回空串。"""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime(DATE_FORMAT)


def fiscal_year_start(report_date: object) -> pd.Timestamp | pd.NaT:
    """报告期所属会计年度起始日 (2024-12-31 → 2024-01-01)。"""
    ts = pd.to_datetime(report_date, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(f"{ts.year}-01-01")


def report_window_start(report_date: object) -> pd.Timestamp | pd.NaT:
    """该报告期"最早可能被公告"的日期。

    正式财报: 不早于报告期末 (报表只能在期末之后编制);
    业绩预告 / 快报: **天然是期间结束前的前瞻信息** —— 公司会在报告期结束前就发布
    业绩预告 (实测 000703 在 2026-06-26 预告 2026-06-30 报告期, 提前 4 天;
    000792 提前 43 天), 因此下界放宽到**所在会计年度起始日**。
    """
    start = fiscal_year_start(report_date)
    return start


def earliest_announcement(report_date: object, pre_period_allowed: bool = False) -> pd.Timestamp | pd.NaT:
    """按表类型给出 ann_date 的下界。

    :param pre_period_allowed: True 表示预告/快报类(允许期间结束前发布),
                               False 表示正式财报(不得早于报告期末)
    """
    ts = pd.to_datetime(report_date, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return report_window_start(ts) if pre_period_allowed else ts


def legal_deadline(report_date: object) -> pd.Timestamp | pd.NaT:
    """法定披露截止日: 年报/一季报次年(当年)4-30, 半年报 8-31, 三季报 10-31。"""
    ts = pd.to_datetime(report_date, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    rtype = report_type_of(ts)
    mmdd = DISCLOSURE_DEADLINE_MMDD.get(rtype)
    if mmdd is None:
        return pd.NaT
    year = ts.year + 1 if rtype == "ANNUAL" else ts.year
    return pd.Timestamp(f"{year}-{mmdd}")


def quarter_diff(cumulative: pd.Series, report_date: pd.Series) -> pd.Series:
    """累计口径 → 单季口径 (Q1 自身, H1−Q1, Q3−H1, ANNUAL−Q3)。

    :param cumulative: 累计金额 (元)
    :param report_date: 对应报告期 (与 cumulative 同索引)
    :return: 单季金额; 缺失上一期累计值时返回 NaN (不填 0)
    """
    if len(cumulative) == 0:
        return pd.Series([], dtype="float64")
    frame = pd.DataFrame(
        {
            "cumulative": pd.to_numeric(cumulative, errors="coerce"),
            "report_date": pd.to_datetime(report_date, errors="coerce"),
        },
        index=cumulative.index,
    )
    frame["report_type"] = frame["report_date"].map(report_type_of)
    frame["year"] = frame["report_date"].dt.year
    order = {"Q1": 1, "H1": 2, "Q3": 3, "ANNUAL": 4}
    frame["seq"] = frame["report_type"].map(order)
    frame = frame.sort_values(["year", "seq"], kind="stable")
    prev_seq = frame["seq"].shift(1)
    prev_year = frame["year"].shift(1)
    prev_cum = frame["cumulative"].shift(1)
    diffable = (frame["seq"] > 1) & (prev_seq == frame["seq"] - 1) & (prev_year == frame["year"])
    single = pd.Series(float("nan"), index=frame.index, dtype="float64")
    # Q1 (年内首期) 单季 = 累计自身
    single.loc[frame["seq"] == 1] = frame.loc[frame["seq"] == 1, "cumulative"]
    # 同年前一期存在的期次做累计差分; 跳期 (缺上一期) 保持 NaN, 不造假值
    single.loc[diffable] = frame.loc[diffable, "cumulative"] - prev_cum.loc[diffable]
    return single.reindex(cumulative.index)


# ---------------------------------------------------------------------------
# 纯函数: 规范化
# ---------------------------------------------------------------------------


def _clean_symbol(value: object) -> str:
    """6 位数字代码; 带交易所后缀/前缀的脏值原样返回空串 (由调用方记账)。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().strip('"').upper()
    if not text:
        return ""
    if len(text) == 6 and text.isdigit():
        return text
    # 000001.SZ / SH600519 / 000001.XSHE 等一律视为脏值
    return ""


def _placeholder_hits(raw: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    """字符串层面的缺失哨兵命中样例 (0/-1 不在此列: 它们可能是真值)。"""
    examples: list[str] = []
    for col in columns:
        if col not in raw.columns:
            continue
        text = raw[col].astype(str).str.strip().str.lower()
        hit = text.isin(PLACEHOLDER_TOKENS)
        for value in text[hit].head(5):
            examples.append(f"{col}={value}")
    return examples


def normalize_table(
    raw: pd.DataFrame,
    spec: FinancialTableSpec,
) -> tuple[pd.DataFrame, TableQuality]:
    """把信源原始行规范化为落盘 schema (不抛错, 质量问题记入 TableQuality)。

    处理: 去 BOM/剥空白 → symbol 校验(zfill/拒后缀) → 日期解析 → 数值解析 →
    按 spec 列序裁剪补列 → 留空校验。返回的 DataFrame 已含 spec.columns 全部列。
    """
    quality = TableQuality(table=spec.key, filename=spec.filename)
    quality.n_rows = int(len(raw))
    frame = raw.copy()
    frame.columns = [str(col).strip().lstrip("\ufeff") for col in frame.columns]

    missing = [col for col in spec.columns if col not in frame.columns]
    quality.unexpected_columns = [col for col in frame.columns if col not in spec.columns]
    quality.missing_columns = missing
    quality.schema_ok = not missing
    if "symbol" not in frame.columns:
        # 无 symbol 列时无法继续, 返回全空帧 + schema_ok=False
        empty = pd.DataFrame(columns=list(spec.columns))
        return empty, quality

    # symbol
    cleaned = frame["symbol"].map(_clean_symbol)
    bad_mask = cleaned.eq("") & frame["symbol"].notna()
    quality.bad_symbols = int(bad_mask.sum())
    quality.bad_symbol_examples = [
        str(v) for v in frame.loc[bad_mask, "symbol"].head(5).tolist()
    ]
    out = pd.DataFrame(index=frame.index)
    out["symbol"] = cleaned

    # 文本列原样 (剥空白)
    for col in spec.columns:
        if col == "symbol":
            continue
        if col in spec.date_columns:
            continue
        if col in set(spec.amount_columns) | set(spec.share_columns) | set(spec.rate_columns):
            continue
        if col in frame.columns:
            out[col] = frame[col].astype("string").str.strip()
        else:
            out[col] = pd.NA

    # 日期列 (必须走 parse_date_column: 混格式列用朴素 to_datetime 会静默变 NaT)
    for col in spec.date_columns:
        if col not in frame.columns:
            out[col] = pd.NaT
            continue
        parsed = parse_date_column(frame[col], context=f"{spec.key}.{col}")
        invalid = parsed.isna() & frame[col].notna()
        quality.bad_dates += int(invalid.sum())
        if len(quality.bad_date_examples) < 5:
            quality.bad_date_examples += [
                f"{col}={value}"
                for value in frame.loc[invalid, col].head(5).astype(str).tolist()
            ][: 5 - len(quality.bad_date_examples)]
        out[col] = parsed

    # 数值列
    numeric_columns = set(spec.amount_columns) | set(spec.share_columns) | set(spec.rate_columns)
    for col in spec.columns:
        if col not in numeric_columns:
            continue
        if col not in frame.columns:
            out[col] = float("nan")
            continue
        parsed = pd.to_numeric(frame[col], errors="coerce")
        invalid = parsed.isna() & frame[col].notna()
        quality.non_numeric += int(invalid.sum())
        if len(quality.non_numeric_examples) < 5:
            quality.non_numeric_examples += [
                f"{col}={value}"
                for value in frame.loc[invalid, col].head(5).astype(str).tolist()
            ][: 5 - len(quality.non_numeric_examples)]
        out[col] = parsed.astype("float64")

    examples = _placeholder_hits(frame, spec.columns)
    quality.placeholder_fills = len(examples)
    quality.placeholder_examples = examples[:10]

    out = out[list(spec.columns)]
    quality.column_order_ok = list(out.columns) == list(spec.columns)
    return out, quality


def check_column_order(columns: list[str] | tuple[str, ...], spec: FinancialTableSpec) -> tuple[bool, list[str], list[str]]:
    """校验已落盘 CSV 的列序; 返回 (是否通过, 缺失列, 多余列)。"""
    current = [str(col).strip().lstrip("\ufeff") for col in columns]
    expected = list(spec.columns)
    missing = [col for col in expected if col not in current]
    extra = [col for col in current if col not in expected]
    return (current == expected), missing, extra
