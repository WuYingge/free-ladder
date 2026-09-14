"""财务数据体检项单元测试 (合成 CSV 驱动, 不联网、不依赖真实 data/)。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_check_financials.py -v

覆盖: 列序错位、哨兵值填充 (0/-1/"NULL")、ann_date < report_date、
超法定披露期、孤儿代码、量纲 10000 倍错、退市股缺数据的告警/失败判定。
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from config import DataPath
from data_manager import financial_health as health
from data_manager.financial_schema import FINANCIAL_TABLES


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(DataPath, "FINANCIAL_DIR", str(tmp_path / "financial"))
    monkeypatch.setattr(DataPath, "STOCK_PATH", str(tmp_path / "stock_data"))
    monkeypatch.setattr(DataPath, "CALANDAR_DF", str(tmp_path / "calendar.csv"))
    os.makedirs(DataPath.FINANCIAL_DIR, exist_ok=True)
    os.makedirs(DataPath.STOCK_PATH, exist_ok=True)
    # 交易日历: 与 data/const/calandar_df.csv 同结构 (trade_date 为列, 含索引列)
    pd.DataFrame(
        {"trade_date": pd.to_datetime(["2025-01-02", "2025-04-03", "2025-04-25", "2025-08-13", "2025-08-14"])}
    ).to_csv(DataPath.CALANDAR_DF, index=True)
    # 行情侧代码
    for code in ("600519", "000001"):
        pd.DataFrame({"date": ["2025-01-02", "2025-12-31"], "open": 1.0, "close": 1.0,
                      "high": 1.0, "low": 1.0, "volume": 1.0, "value": 1.0,
                      "range": 1.0, "gain": 0.0, "change": 0.0, "turnOver": 1.0}).to_csv(
            os.path.join(DataPath.STOCK_PATH, f"{code}.csv"), index=False, encoding="utf-8-sig"
        )
    return tmp_path


def _write(table: str, rows: list[dict], spec_columns: list[str] | None = None) -> None:
    spec = FINANCIAL_TABLES[table]
    frame = pd.DataFrame(rows)
    if spec_columns is not None:
        frame = frame[spec_columns]
    frame.to_csv(os.path.join(DataPath.FINANCIAL_DIR, spec.filename), index=False, encoding="utf-8-sig")


def _income_row(symbol="600519", report="2024-12-31", ann="2025-04-03", profit=1.5e9, **overrides):
    row = {
        "symbol": symbol,
        "report_date": report,
        "ann_date": ann,
        "report_type": "ANNUAL",
        "net_profit_attr_p": profit,
        "total_share": 1.256e9,
        "revenue": profit * 2,
        "eps_basic": profit / 1.025e9,
        "operating_profit": profit * 1.1,
        "net_profit": profit * 1.02,
        "net_profit_deducted": profit * 0.95,
        "minority_interest_profit": profit * 0.02,  # 恒等式: net = parent + minority
        "rd_expense": 1.0,
        "sell_admin_expense": 2.0,
        "currency": "CNY",
        "update_date": ann,
        "basis": "first_reported",
        "update_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    row.update(overrides)
    return row


def _balance_row(symbol="600519", report="2024-12-31", ann="2025-04-03"):
    return {
        "symbol": symbol, "report_date": report, "ann_date": ann, "report_type": "ANNUAL",
        "total_assets": 3e11, "total_equity_attr_p": 2e11, "total_liabilities": 1e11,
        "total_equity": 2e11, "total_share": 1.256e9, "monetary_funds": 5e10,
        "accounts_receivable": 1e9, "inventory": 2e9, "goodwill": 0.0,
        "minority_interest_equity": 1e6,
        "currency": "CNY", "update_date": ann, "basis": "first_reported",
        "update_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def test_schema_check_detects_wrong_column_order(tmp_paths):
    spec = FINANCIAL_TABLES["income_q"]
    swapped = list(spec.columns)
    swapped[4], swapped[5] = swapped[5], swapped[4]
    _write("income_q", [_income_row()], spec_columns=swapped)
    result = health.check_schema(["income_q"], DataPath.FINANCIAL_DIR)
    assert result.status == "fail"
    assert result.metrics["income_q"]["columns_match_take_order"] is False


def test_schema_check_passes_on_correct_header(tmp_paths):
    _write("income_q", [_income_row()])
    result = health.check_schema(["income_q"], DataPath.FINANCIAL_DIR)
    assert result.status == "pass"
    assert result.metrics["income_q"]["columns_match_take_order"] is True


def test_placeholder_fill_detection(tmp_paths):
    _write("income_q", [_income_row(), _income_row(symbol="000001", net_profit_attr_p="NULL")])
    result = health.check_placeholder_fills(["income_q"], DataPath.FINANCIAL_DIR)
    assert result.status == "fail"
    assert any("net_profit_attr_p" in str(item) for item in result.failures)


def test_date_axis_detects_ann_before_report(tmp_paths):
    _write("income_q", [_income_row(ann="2024-06-01")])  # 公告早于报告期 → 脏数据
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    assert result.status == "fail"
    metrics = result.metrics["tables"]["income_q"]
    assert metrics["ann_before_lower_bound"] == 1
    assert metrics["pre_period_allowed"] is False


def test_date_axis_allows_pre_period_forecast(tmp_paths):
    """业绩预告是期间结束前的前瞻信息 → ann_date 早于 report_date 属合法。

    参照实测: 000703 在 2026-06-26 预告 2026-06-30 报告期。
    """
    _write(
        "forecast",
        [
            {
                "symbol": "000703", "report_date": "2026-06-30", "ann_date": "2026-06-26",
                "report_type": "H1", "forecast_ann_date": "2026-06-26",
                "forecast_indicator_cn": "归属于上市公司股东的净利润",
                "announce_type": "预增", "announce_type_en": "increase",
                "forecast_net_profit_low": 5.5e9, "forecast_net_profit_high": 5.5e9,
                "forecast_net_profit_mid": 5.5e9, "yoy_low": 0.5, "yoy_high": 0.6,
                "update_date": "2026-06-26", "basis": "first_reported",
                "update_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ],
    )
    result = health.check_date_axis(["forecast"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["tables"]["forecast"]
    assert metrics["pre_period_allowed"] is True
    assert metrics["ann_before_report_raw"] == 1  # 记录原始事实
    assert metrics["ann_before_lower_bound"] == 0  # 但不判为脏数据
    assert result.status != "fail"


def test_date_axis_flags_forecast_ann_before_fiscal_year(tmp_paths):
    """预告早于会计年度起始日 → 仍是脏数据 (预告最早只能在当年年初之后披露)。"""
    _write(
        "forecast",
        [
            {
                "symbol": "000792", "report_date": "2025-06-30", "ann_date": "2024-05-01",
                "report_type": "H1", "forecast_ann_date": "2024-05-01",
                "forecast_indicator_cn": "归属于上市公司股东的净利润",
                "announce_type": "略增", "announce_type_en": "slightly_increase",
                "forecast_net_profit_low": 1e9, "forecast_net_profit_high": 1e9,
                "forecast_net_profit_mid": 1e9, "yoy_low": 0.1, "yoy_high": 0.2,
                "update_date": "2024-05-01", "basis": "first_reported",
                "update_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ],
    )
    result = health.check_date_axis(["forecast"], DataPath.FINANCIAL_DIR)
    assert result.metrics["tables"]["forecast"]["ann_before_lower_bound"] == 1
    assert result.status == "fail"


def test_date_axis_reports_late_filings_as_warning_with_context(tmp_paths):
    """逾期披露是真实市场现象 → 给量化上下文 + warn, 默认不判 fail。

    实测: 2016+ 全市场逾期披露 8.6%, 且独立信源 RPT_LICO_FN_CPD 与 F10 完全一致
    (000048 的 2017 年报确为 2018-08-31 披露) —— 判 fail 会把正常数据误报为错误。
    """
    rows = [_income_row(symbol=f"60000{i}") for i in range(9)]
    rows.append(_income_row(symbol="600099", ann="2025-05-20"))  # 超 4/30 披露期 (10%)
    _write("income_q", rows)
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["tables"]["income_q"]
    assert metrics["beyond_deadline_share"] == pytest.approx(0.1)
    assert metrics["late_delay_days"]["p50"] == pytest.approx(20.0)
    assert result.status == "warn"  # 记录 + 告警, 但不是 fail


def test_date_axis_late_filing_can_be_hard_gated(tmp_paths, monkeypatch):
    """需要把逾期披露当硬门禁时, 打开 DEADLINE_VIOLATION_FAIL 即可恢复 fail。"""
    monkeypatch.setattr(health, "DEADLINE_VIOLATION_FAIL", True)
    rows = [_income_row(symbol=f"60000{i}") for i in range(9)]
    rows.append(_income_row(symbol="600099", ann="2025-05-20"))
    _write("income_q", rows)
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    assert result.status == "fail"


def test_date_axis_counts_non_trading_day_but_does_not_fail(tmp_paths):
    _write("income_q", [_income_row(ann="2025-04-05")])  # 周六, 非交易日
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    table_metrics = result.metrics["tables"]["income_q"]
    assert table_metrics["non_trading_day_ann"] == 1
    # 非交易日公告不是错误 (实测约 20%)
    assert result.status != "fail"


def test_dimension_check_catches_unit_scale_error(tmp_paths):
    # 资产负债表恒等式被打破 (负债 = 资产 + 权益, 明显量纲/口径错)
    bad = _balance_row()
    bad["total_liabilities"] = 9e11
    _write("balance_q", [bad])
    result = health.check_dimensions(["balance_q"], DataPath.FINANCIAL_DIR)
    assert result.status == "fail"
    assert result.metrics["balance_q_identity"]["violation_share"] == pytest.approx(1.0)


def test_dimension_check_passes_on_consistent_data(tmp_paths):
    _write("balance_q", [_balance_row()])
    _write("income_q", [_income_row()])
    result = health.check_dimensions(["balance_q", "income_q"], DataPath.FINANCIAL_DIR)
    assert result.status == "pass"
    assert result.metrics["income_q_checks"]["eps_times_share_over_parent_median"] == pytest.approx(
        1.256e9 / 1.025e9, rel=1e-6
    )


def test_orphan_symbols_flagged(tmp_paths):
    """A 股口径孤儿代码 → fail; B 股/北交所等范围外代码只记录不计入。"""
    _write("income_q", [_income_row(symbol="000004"), _income_row(symbol="200011", ann="2025-04-03")])
    result = health.check_extremes_and_tradability(["income_q"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["orphan_symbols"]
    assert metrics["orphans"] == 1  # 000004 不在行情侧也不在退市名单
    assert metrics["share"] == pytest.approx(1.0)
    assert metrics["structurally_out_of_scope"]["count"] == 1  # 200011 B 股
    assert result.status == "fail"


def test_coverage_reports_delisted_gap(tmp_paths):
    _write("income_q", [_income_row()])
    result = health.check_coverage_and_survivorship(["income_q"], DataPath.FINANCIAL_DIR)
    assert "delisted_sample" in result.metrics
    assert "600696" in result.metrics["delisted_sample"]


def test_forecast_leadtime_flags_forecast_after_actual_report(tmp_paths):
    """预告应早于正式财报; 晚于正式财报的预告占比 >5% 时判 fail。"""
    income_rows = [_income_row(symbol=f"60000{i}", report="2024-12-31", ann="2025-04-03") for i in range(20)]
    _write("income_q", income_rows)
    forecast_rows = []
    for i in range(20):
        forecast_rows.append(
            {
                "symbol": f"60000{i}", "report_date": "2024-12-31",
                # 前两个晚于正式财报公告日 (4/3), 其余提前
                "ann_date": "2025-04-20" if i < 2 else "2025-01-20",
                "report_type": "ANNUAL", "forecast_ann_date": "2025-04-20" if i < 2 else "2025-01-20",
                "forecast_indicator_cn": "归属于上市公司股东的净利润",
                "announce_type": "预增", "announce_type_en": "increase",
                "forecast_net_profit_low": 1e9, "forecast_net_profit_high": 1.2e9,
                "forecast_net_profit_mid": 1.1e9, "yoy_low": 0.1, "yoy_high": 0.2,
                "update_date": "2025-01-20", "basis": "first_reported",
                "update_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    _write("forecast", forecast_rows)
    result = health.check_forecast_leadtime(["forecast", "income_q"], DataPath.FINANCIAL_DIR)
    assert result.metrics["forecast_after_report_rows"] == 2
    assert result.metrics["forecast_after_report_share"] == pytest.approx(0.1)
    assert result.status == "fail"


def test_date_axis_deadline_not_applied_to_forecast(tmp_paths):
    """业绩预告无 4/30 之类的法定截止期 → 晚公告不算超期 (实测 max 延误 203 天)。"""
    _write(
        "forecast",
        [
            {
                "symbol": "001215", "report_date": "2020-09-30", "ann_date": "2020-12-29",
                "report_type": "Q3", "forecast_ann_date": "2020-12-29",
                "forecast_indicator_cn": "归属于上市公司股东的净利润",
                "announce_type": "略增", "announce_type_en": "slightly_increase",
                "forecast_net_profit_low": 1e8, "forecast_net_profit_high": 1e8,
                "forecast_net_profit_mid": 1e8, "yoy_low": 0.1, "yoy_high": 0.1,
                "update_date": "2020-12-29", "basis": "first_reported",
                "update_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ],
    )
    result = health.check_date_axis(["forecast"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["tables"]["forecast"]
    assert metrics["beyond_legal_deadline"] == 1  # 事实仍记录
    assert result.status != "fail"  # 但不判错 (该表 deadline_check_applicable=False)


def test_date_axis_exempts_pre_listing_periods(tmp_paths, monkeypatch):
    """IPO 材料一次性披露的上市前报告期不受法定披露期约束 → 不计入超期判定。

    参照实测 301669: 2025H1 与 2025Q3 同为 2025-12-04 公告 (远晚于 8/31、10/31)。
    """
    from data_manager.providers.stock_list_provider import STOCK_LIST

    monkeypatch.setattr(STOCK_LIST, "get_list_date", lambda code: "2025-12-04" if code == "301669" else "")
    _write(
        "income_q",
        [
            _income_row(symbol="301669", report="2025-06-30", ann="2025-12-04"),  # 上市前
            _income_row(symbol="301669", report="2025-09-30", ann="2025-12-04"),  # 上市前
            _income_row(symbol="600519", report="2024-12-31", ann="2025-09-30"),  # 真超期
        ],
    )
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["tables"]["income_q"]
    assert metrics["beyond_legal_deadline"] == 3  # 事实全部记录
    assert metrics["beyond_legal_deadline_pre_listing"] == 2  # 两条为上市前
    assert metrics["beyond_legal_deadline_checkable"] == 1  # 只有 600519 计入判定


def test_build_report_and_write(tmp_paths):
    _write("income_q", [_income_row()])
    _write("balance_q", [_balance_row()])
    report = health.build_report(tables=["income_q", "balance_q"])
    assert report["verdict"] in {"pass", "warn", "fail"}
    assert [check["name"] for check in report["checks"]] == list(health.CHECK_ORDER)
    out = health.write_report(report, os.path.join(DataPath.FINANCIAL_DIR, "health_report.json"))
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as handle:
        assert json.load(handle)["tables"] == ["income_q", "balance_q"]


def test_field_fillability_declares_unsupported_fields(tmp_paths):
    _write("income_q", [_income_row()])
    result = health.check_field_fillability(["income_q"], DataPath.FINANCIAL_DIR)
    assert result.status == "warn"
    unsupported = result.metrics["not_supported"]
    assert "st_status_hist" in unsupported
    assert "index_membership" in unsupported
    assert len(result.metrics["fields"]) == 33


def test_date_axis_flags_non_canonical_date_text(tmp_paths):
    """混格式落盘必须被独立点名 (写入路径 dtype 退化 → 朴素解析会静默丢公告日)。"""
    _write(
        "income_q",
        [
            _income_row(),
            # 只污染 ann_date, update_date 保持规范 → 断言能定位到具体列
            _income_row(symbol="000001", ann="2025-04-03 00:00:00", update_date="2025-04-03"),
        ],
    )
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    assert result.status == "fail"
    issue = next(p for p in result.failures if "非规范" in p["issue"])
    assert issue["columns"] == {"ann_date": 1}
    assert issue["count"] == 1
    # 关键: 该行不得被算成 "ann_date 缺失" (格式问题 ≠ 缺数据)
    assert result.metrics["tables"]["income_q"]["missing_ann_date"] == 0
    assert result.metrics["tables"]["income_q"]["non_canonical_date_text"] == 1


def test_date_axis_passes_on_canonical_text(tmp_paths):
    _write("income_q", [_income_row(), _income_row(symbol="000001")])
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    assert result.metrics["tables"]["income_q"]["non_canonical_date_text"] == 0
    assert not [p for p in result.failures if "非规范" in p["issue"]]


def test_date_axis_classifies_missing_ann_date(tmp_paths, monkeypatch):
    """公告日缺失必须区分"上市前报告期(源端本来就没有)"与"抓取/解析丢失"。"""
    from data_manager.providers.stock_list_provider import STOCK_LIST

    monkeypatch.setattr(
        STOCK_LIST, "get_list_date", lambda code: "2025-09-01" if code == "600519" else ""
    )
    # 600519 上市前报告期 (2025-06-30 < 上市日) 缺公告日 → 豁免
    _write("income_q", [_income_row(symbol="600519", report="2025-06-30", ann=None)])
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["tables"]["income_q"]
    assert metrics["missing_ann_date_pre_listing"] == 1
    assert metrics["missing_ann_date_other"] == 0
    assert not [p for p in result.failures if "非上市前报告期" in p["issue"]]


def test_date_axis_fails_on_missing_ann_date_after_listing(tmp_paths, monkeypatch):
    """上市后报告期缺公告日 = 真缺陷 → fail (阈值 0.1%)。"""
    from data_manager.providers.stock_list_provider import STOCK_LIST

    monkeypatch.setattr(
        STOCK_LIST, "get_list_date", lambda code: "2020-01-01" if code == "600519" else ""
    )
    _write("income_q", [_income_row(symbol="600519", report="2025-06-30", ann=None)])
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    issue = next(p for p in result.failures if "非上市前报告期" in p["issue"])
    assert issue["count"] == 1
    assert result.status == "fail"


def test_date_axis_masks_sentinel_1900_date(tmp_paths):
    """哨兵 1900-01-01 必须在解析层置空, 不能作为真实公告日进入数据。"""
    _write("income_q", [_income_row(symbol="600519", report="2001-12-31", ann="1900-01-01 00:05:43")])
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["tables"]["income_q"]
    assert metrics["missing_ann_date"] == 1
    assert metrics["gap_days"]["min"] is None  # 没有任何 1900 年差值参与统计
    assert metrics["ann_before_lower_bound"] == 0  # 哨兵不会被当成"早于报告期"


def _forecast_row(symbol: str, report: str, ann: str) -> dict:
    return {
        "symbol": symbol, "report_date": report, "ann_date": ann,
        "report_type": "ANNUAL", "forecast_ann_date": ann,
        "forecast_indicator_cn": "归属于上市公司股东的净利润",
        "announce_type": "预增", "announce_type_en": "increase",
        "forecast_net_profit_low": 1e8, "forecast_net_profit_high": 2e8,
        "forecast_net_profit_mid": 1.5e8, "yoy_low": 0.1, "yoy_high": 0.2,
        "update_date": ann, "basis": "first_reported",
        "update_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def test_history_depth_event_table_not_flagged_for_sparse_coverage(tmp_paths):
    """事件表 (预告) 的稀疏是常态: 不得用"逐股落后全局最新期"判成截断。

    实测 forecast 单期覆盖仅 ~3,700/5,200 只, 若按逐股滞后率判会误报 35%。
    """
    rows = []
    # 2024 年度: 200 只; 2025 年度: 210 只; 2026 年度: 205 只 → 同比稳定
    for i in range(200):
        rows.append(_forecast_row(f"{i:06d}", "2024-12-31", "2025-01-20"))
    for i in range(210):
        rows.append(_forecast_row(f"{i:06d}", "2025-12-31", "2026-01-20"))
    for i in range(205):
        rows.append(_forecast_row(f"{i:06d}", "2026-12-31", "2026-10-20"))
    _write("forecast", rows)
    result = health.check_history_depth(["forecast"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["forecast"]
    assert "事件表" in metrics["judgement"]
    assert metrics["weak_periods"] == []
    assert not [p for p in result.failures if "塌陷" in p["issue"]]


def test_history_depth_event_table_flags_collapsed_period(tmp_paths):
    """某期覆盖相对去年同期塌陷过半 → 漏期/截断, 必须 fail。"""
    rows = [_forecast_row(f"{i:06d}", "2025-12-31", "2026-01-20") for i in range(200)]
    rows += [_forecast_row(f"{i:06d}", "2026-12-31", "2026-10-20") for i in range(40)]
    _write("forecast", rows)
    result = health.check_history_depth(["forecast"], DataPath.FINANCIAL_DIR)
    issue = next(p for p in result.failures if "塌陷" in p["issue"])
    assert issue["periods"][0]["period"] == "2026-12-31"
    assert issue["periods"][0]["ratio"] == pytest.approx(0.2)


def test_orphan_in_statement_table_fails(tmp_paths, monkeypatch):
    """报表表逐股抓取, 出现池外代码 = 股票池漂移 → fail。"""
    from data_manager.providers.stock_list_provider import STOCK_LIST

    monkeypatch.setattr(STOCK_LIST, "get_all_symbol", lambda: ["600519", "000001"])
    _write("income_q", [_income_row(symbol="000002")])
    result = health.check_extremes_and_tradability(["income_q"], DataPath.FINANCIAL_DIR)
    assert result.metrics["orphan_symbols"]["statement_orphans"]["count"] == 1
    assert any("股票池漂移" in p["issue"] for p in result.failures)


def test_orphan_only_in_event_table_is_warn_not_fail(tmp_paths, monkeypatch):
    """逐期全市场表带入的快照外代码 (新股/退市) 不是取数缺陷 → 只 warn。"""
    from data_manager.providers.stock_list_provider import STOCK_LIST

    monkeypatch.setattr(STOCK_LIST, "get_all_symbol", lambda: ["600519", "000001"])
    _write("forecast", [_forecast_row("000002", "2025-12-31", "2026-01-20")])
    result = health.check_extremes_and_tradability(["forecast"], DataPath.FINANCIAL_DIR)
    orphans = result.metrics["orphan_symbols"]
    assert orphans["statement_orphans"]["count"] == 0
    assert orphans["event_only_orphans"]["count"] == 1
    assert not [p for p in result.failures if "股票池漂移" in p["issue"]]
    assert any("快照外代码" in p["issue"] for p in result.failures)


def test_date_axis_detects_one_year_shifted_ann_date(tmp_paths):
    """源端把公告日写成"下一期同日历期"的公告日 → 必须独立报出, 且不与真逾期混淆。

    实测 000670 的 2016Q1 写成 2017-04-20 (= 其 2017 一季报公告日, 巨潮公告原文定案);
    真逾期如 000048 的 2017 年报 = 2018-08-31 (超期 123 天) 则不得被判为错位。
    """
    _write(
        "income_q",
        [
            _income_row(symbol="000670", report="2016-03-31", ann="2017-04-20", report_type="Q1"),  # 错位一年
            _income_row(symbol="000048", report="2017-12-31", ann="2018-08-31"),                     # 真逾期
        ],
    )
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["tables"]["income_q"]
    assert metrics["ann_date_shifted_one_year"] == 1
    assert metrics["genuine_late_filings"] == 1
    issue = next(p for p in result.failures if "错位一年" in p["issue"])
    assert issue["count"] == 1
    assert issue["examples"][0]["symbol"] == "000670"


def test_date_axis_shifted_detector_ignores_real_late_filings(tmp_paths):
    """只有真逾期、没有错位时, 不得触发错位判定。"""
    _write("income_q", [_income_row(symbol="000048", report="2017-12-31", ann="2018-08-31")])
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["tables"]["income_q"]
    assert metrics["ann_date_shifted_one_year"] == 0
    assert not [p for p in result.failures if "错位一年" in p["issue"]]


def test_date_axis_shifted_detector_exempts_pre_listing_periods(tmp_paths, monkeypatch):
    """招股书一次性披露的上市前报告期, 公告日天然晚于报告期数年 → 不得算成"错位一年"。

    实测: 不豁免时 income_q 虚报 27,000 行, 并把 2016+ 段虚高到 4.8% (实际仅 3 行)。
    """
    from data_manager.providers.stock_list_provider import STOCK_LIST

    monkeypatch.setattr(STOCK_LIST, "get_list_date", lambda code: "2021-04-28" if code == "001201" else "")
    _write(
        "income_q",
        [
            # 上市前报告期 (2017 年报在 2021 招股书里一次性披露) → 豁免
            _income_row(symbol="001201", report="2017-12-31", ann="2021-04-30"),
        ],
    )
    result = health.check_date_axis(["income_q"], DataPath.FINANCIAL_DIR)
    metrics = result.metrics["tables"]["income_q"]
    assert metrics["ann_date_shifted_one_year"] == 0
    assert metrics["ann_date_shifted_pre_listing_exempt"] == 1
    assert not [p for p in result.failures if "错位一年" in p["issue"]]
