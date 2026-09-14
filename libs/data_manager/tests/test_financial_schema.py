"""财务 schema 与口径纯函数单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_financial_schema.py -v

覆盖: 列序与取数单表头一致性、报告期类型/法定截止日、累计→单季差分、
规范化质量记账 (坏代码/坏日期/缺列/占位符)、33 字段实现状态。
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_manager.financial_schema import (
    FINANCIAL_FIELD_SOURCES,
    FINANCIAL_TABLES,
    FIELD_ORDER_HELP,
    INCOME_ORDER_HEAD,
    check_column_order,
    field_implemented,
    legal_deadline,
    normalize_table,
    parse_date_column,
    period_key,
    quarter_diff,
    report_type_of,
)


def test_take_order_header_matches_client_spec():
    """取数单要求 income_q 表头逐字为 symbol,report_date,ann_date,report_type,
    net_profit_attr_p,total_share,revenue。"""
    spec = FINANCIAL_TABLES["income_q"]
    assert list(spec.columns[:7]) == [
        "symbol",
        "report_date",
        "ann_date",
        "report_type",
        "net_profit_attr_p",
        "total_share",
        "revenue",
    ]
    assert list(spec.columns[:7]) == INCOME_ORDER_HEAD


def test_all_tables_share_required_prefix():
    """除股本事件表外, 各表前四列均为 symbol/report_date/ann_date/report_type。"""
    for key, spec in FINANCIAL_TABLES.items():
        if key == "share_capital":
            assert list(spec.columns[:2]) == ["symbol", "ann_date"]
            continue
        assert list(spec.columns[:4]) == ["symbol", "report_date", "ann_date", "report_type"]


def test_field_coverage_33_fields():
    """§③-A 33 字段全部登记; 27 项有落地表, 6 项为已知不可得(含既有行业表)。"""
    assert set(FINANCIAL_FIELD_SOURCES) == set(range(1, 34))
    implemented = [no for no in range(1, 34) if field_implemented(no)]
    assert len(implemented) == 27
    for unsupported in (22, 23, 24, 28):
        assert not field_implemented(unsupported)
    # 29 申万行业为"已在位"(with_industry), 视为有落地
    assert field_implemented(29)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-03-31", "Q1"),
        ("2024-06-30", "H1"),
        ("2024-09-30", "Q3"),
        ("2024-12-31", "ANNUAL"),
        ("2024-05-15", ""),
        (pd.NaT, ""),
    ],
)
def test_report_type_of(value, expected):
    assert report_type_of(value) == expected
    if expected:
        assert period_key(value) == pd.Timestamp(value).strftime("%Y-%m-%d")


@pytest.mark.parametrize(
    ("report_date", "deadline"),
    [
        ("2024-03-31", "2024-04-30"),
        ("2024-06-30", "2024-08-31"),
        ("2024-09-30", "2024-10-31"),
        ("2024-12-31", "2025-04-30"),  # 年报跨年到次年 4/30
    ],
)
def test_legal_deadline(report_date, deadline):
    assert legal_deadline(report_date) == pd.Timestamp(deadline)


def test_quarter_diff_closes_cumulative_chain():
    report = pd.to_datetime(
        ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"]
    )
    cumulative = pd.Series([100.0, 250.0, 400.0, 600.0, 120.0], index=range(5))
    single = quarter_diff(cumulative, pd.Series(report, index=range(5)))
    assert single.tolist() == [100.0, 150.0, 150.0, 200.0, 120.0]
    # 累计 = 年内单季累加
    assert single[:4].sum() == pytest.approx(600.0)


def test_quarter_diff_handles_missing_previous():
    """跳期 (缺 H1) 时不做差分, 返回 NaN 而非造假值。"""
    report = pd.to_datetime(["2024-03-31", "2024-09-30"])
    cumulative = pd.Series([100.0, 400.0], index=[0, 1])
    single = quarter_diff(cumulative, pd.Series(report, index=[0, 1]))
    assert single.iloc[0] == pytest.approx(100.0)
    assert pd.isna(single.iloc[1])


def test_normalize_table_orders_and_flags_quality():
    raw = pd.DataFrame(
        {
            "revenue": ["2e9", None],
            "symbol": ["600519", "600519.SH"],  # 第二行带后缀 → 坏代码
            "report_date": ["2024-12-31", "坏日期"],
            "ann_date": ["2025-04-03", "2025-04-03"],
            "net_profit_attr_p": ["1.5e9", "-1"],
            "total_share": [1250000000, 1250000000],
            "report_type": ["ANNUAL", "ANNUAL"],
            "basis": ["NULL", "first_reported"],
        }
    )
    frame, quality = normalize_table(raw, FINANCIAL_TABLES["income_q"])
    assert list(frame.columns) == list(FINANCIAL_TABLES["income_q"].columns)
    assert quality.bad_symbols == 1
    assert quality.bad_symbol_examples == ["600519.SH"]
    assert quality.bad_dates == 1
    assert quality.placeholder_fills >= 1
    # -1 是数值而非缺失, 原样保留 (不当作缺失哨兵)
    assert frame.loc[1, "net_profit_attr_p"] == pytest.approx(-1.0)
    assert frame.loc[0, "net_profit_attr_p"] == pytest.approx(1.5e9)
    # 未提供的列补空, 不报错
    assert "rd_expense" in frame.columns


def test_normalize_table_reports_missing_columns():
    raw = pd.DataFrame({"symbol": ["600519"]})
    _, quality = normalize_table(raw, FINANCIAL_TABLES["cashflow_q"])
    assert quality.schema_ok is False
    assert "report_date" in quality.missing_columns
    assert "ocf_net" in quality.missing_columns


def test_check_column_order_detects_swap():
    spec = FINANCIAL_TABLES["income_q"]
    ok, missing, extra = check_column_order(list(spec.columns), spec)
    assert ok and not missing and not extra
    swapped = list(spec.columns)
    swapped[4], swapped[5] = swapped[5], swapped[4]
    ok, missing, extra = check_column_order(swapped, spec)
    assert ok is False
    assert missing == [] and extra == []


def test_field_order_help_is_actionable():
    assert "symbol, report_date, ann_date" in FIELD_ORDER_HELP
    assert FINANCIAL_TABLES["income_q"].filename == "income_q.csv"


# ---------------------------------------------------------------------------
# 源端哨兵日期 (1900-01-01) 与混格式解析
# ---------------------------------------------------------------------------


def test_parse_date_column_handles_mixed_formats():
    """混格式列必须全部解析 (朴素 to_datetime 会把少数派格式静默判成 NaT)。"""
    raw = pd.Series(["2021-04-20", "2021-04-21 00:00:00", "", None], dtype="string")
    parsed = parse_date_column(raw, context="t.ann_date")
    assert parsed.notna().sum() == 2
    assert set(parsed.dropna()) == {pd.Timestamp("2021-04-20"), pd.Timestamp("2021-04-21")}


def test_parse_date_column_strict_raises_on_garbage_only():
    """非空不可解析 → 抛错; 空值与哨兵值不算错。"""
    ok = pd.Series(["2021-04-20", "", None, "1900-01-01 00:05:43"], dtype="string")
    parsed = parse_date_column(ok, strict=True, context="t.ann_date")
    assert parsed.notna().sum() == 1, "哨兵值必须置空且不抛错"
    with pytest.raises(ValueError, match="不可解析"):
        parse_date_column(pd.Series(["不是日期"], dtype="string"), strict=True, context="t.ann_date")


def test_sentinel_dates_become_missing_not_1900():
    """1900-01-01 是源端'无此日期'哨兵: 当真实公告日会构成前视 (恒可见), 必须置空。"""
    raw = pd.DataFrame(
        [{"symbol": "002500", "report_date": pd.Timestamp("2001-12-31"),
          "ann_date": pd.Timestamp("1900-01-01 00:05:43"), "report_type": "ANNUAL"}]
    )
    normalized, quality = normalize_table(raw, FINANCIAL_TABLES["balance_q"])
    assert pd.isna(normalized.loc[0, "ann_date"])
    assert quality.bad_dates == 1  # 计入"非有效日期", 便于质量报告留痕
