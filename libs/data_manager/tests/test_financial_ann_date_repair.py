"""ann_date 巨潮仲裁修复的单元测试 (不联网, 公告源以 monkeypatch 注入)。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_financial_ann_date_repair.py -v

覆盖: 错位行被提出修正、巨潮更晚/一致的行不动、真延迟行保留、
dry-run 不落盘、apply 落盘并写溯源、报告期过滤、按股票子集。
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from config import DataPath
from data_manager import financial_manager as fm
from data_manager import financial_provenance as prov
from fetcher import financial as ff


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(DataPath, "FINANCIAL_DIR", str(tmp_path / "financial"))
    os.makedirs(DataPath.FINANCIAL_DIR, exist_ok=True)
    return tmp_path


def _seed(table: str, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    fm.upsert_table(table, fm.prepare_for_storage(frame, table)[0])


def _income(symbol: str, report: str, ann: str) -> dict:
    return {
        "symbol": symbol,
        "report_date": pd.Timestamp(report),
        "ann_date": pd.Timestamp(ann),
        "report_type": "Q1" if report.endswith("03-31") else "ANNUAL",
        "net_profit_attr_p": 1e8,
        "revenue": 2e8,
        "currency": "CNY",
        # 必须带 _source_update_date: manager 的 _apply_basis 是从它推导 update_date/basis 的,
        # 只写 update_date 会被覆盖成 NaT (踩过的坑)
        "_source_update_date": pd.Timestamp(ann),
        "update_date": pd.Timestamp(ann),
    }


@pytest.fixture
def fake_announcements(monkeypatch):
    """公告源: 000776 的 2016Q1 真值是 2016-04-29; 000048 的 2017 年报确实是 2018-08-31。"""
    data = {
        "000776": pd.DataFrame(
            [
                {"symbol": "000776", "report_date": pd.Timestamp("2016-03-31"), "ann_date": pd.Timestamp("2016-04-29"), "title": "2016年第一季度报告正文"},
                {"symbol": "000776", "report_date": pd.Timestamp("2016-06-30"), "ann_date": pd.Timestamp("2016-08-27"), "title": "2016年半年度报告"},
            ]
        ),
        "000048": pd.DataFrame(
            [
                {"symbol": "000048", "report_date": pd.Timestamp("2017-12-31"), "ann_date": pd.Timestamp("2018-08-31"), "title": "2017年年度报告摘要"},
            ]
        ),
        "600519": pd.DataFrame(
            [
                {"symbol": "600519", "report_date": pd.Timestamp("2016-03-31"), "ann_date": pd.Timestamp("2016-05-10"), "title": "2016年第一季度报告"},
            ]
        ),
    }

    def _fake(symbol: str, start_date: str = "2010-01-01", end_date=None, mode=None):
        return data.get(str(symbol).zfill(6), pd.DataFrame(columns=["symbol", "report_date", "ann_date", "title"]))

    monkeypatch.setattr(ff, "get_periodic_report_announcements", _fake)
    return data


def test_plan_flags_shifted_row_and_keeps_genuine_delay(tmp_paths, fake_announcements):
    _seed(
        "income_q",
        [
            _income("000776", "2016-03-31", "2017-04-28"),  # 错位一年 → 应修
            _income("000048", "2017-12-31", "2018-08-31"),  # 真延迟 (巨潮同日) → 不动
            _income("600519", "2016-03-31", "2016-04-21"),  # 巨潮更晚 (2016-05-10) → 不动
        ],
    )
    plan, stats = fm.plan_ann_date_repairs(symbols=["000776", "000048", "600519"], threads=1)
    assert list(plan["symbol"]) == ["000776"]
    assert plan.iloc[0]["new_value"] == pd.Timestamp("2016-04-29")
    assert int(plan.iloc[0]["gap_days"]) == 364
    per_table = stats["per_table"]["income_q"]
    assert per_table["matched_rows"] == 3  # 本地只有 3 行能匹配到公告 (000776 2016Q1 + 000048 年报 + 600519 2016Q1)
    assert per_table["already_equal"] == 1
    assert per_table["cninfo_later_skipped"] == 1
    assert per_table["proposed"] == 1


def test_dry_run_leaves_table_untouched(tmp_paths, fake_announcements):
    _seed("income_q", [_income("000776", "2016-03-31", "2017-04-28")])
    result = fm.repair_ann_dates_from_cninfo(symbols=["000776"], threads=1, apply=False)
    assert result["stats"]["plan_rows"] == 1
    assert result["stats"]["applied_total"] == 0
    frame, _ = fm.load_table("income_q")
    assert frame.loc[0, "ann_date"] == pd.Timestamp("2017-04-28")
    assert prov.load_corrections("income_q").empty


def test_apply_rewrites_and_records_provenance(tmp_paths, fake_announcements):
    _seed("income_q", [_income("000776", "2016-03-31", "2017-04-28")])
    result = fm.repair_ann_dates_from_cninfo(symbols=["000776"], threads=1, apply=True)
    assert result["stats"]["applied_total"] == 1
    frame, _ = fm.load_table("income_q")
    assert frame.loc[0, "ann_date"] == pd.Timestamp("2016-04-29")
    log = prov.load_corrections("income_q")
    assert len(log) == 1
    assert log.loc[0, "source"] == "cninfo:disclosure"
    assert log.loc[0, "old_value"] == "2017-04-28"
    assert log.loc[0, "new_value"] == "2016-04-29"
    # 回滚可用
    prov.revert_corrections("income_q", source="cninfo:disclosure")
    frame2, _ = fm.load_table("income_q")
    assert frame2.loc[0, "ann_date"] == pd.Timestamp("2017-04-28")


def test_min_gap_days_filters_small_differences(tmp_paths, monkeypatch):
    """小于 min_gap_days 的天数差不动 (源端对同一次公告的日期取法可能有 1~2 天差异)。"""
    def _fake(symbol: str, start_date: str = "2010-01-01", end_date=None, mode=None):
        return pd.DataFrame(
            [
                {"symbol": "000776", "report_date": pd.Timestamp("2016-03-31"), "ann_date": pd.Timestamp("2016-04-27"), "title": "2016年第一季度报告"}
            ]
        )

    monkeypatch.setattr(ff, "get_periodic_report_announcements", _fake)
    _seed("income_q", [_income("000776", "2016-03-31", "2016-04-29")])
    plan, stats = fm.plan_ann_date_repairs(symbols=["000776"], threads=1, min_gap_days=30)
    assert plan.empty
    assert stats["per_table"]["income_q"]["within_tolerance"] == 1  # 差 2 天 → 容差内不动


def test_start_year_filters_old_periods(tmp_paths, fake_announcements):
    _seed("income_q", [_income("000776", "2016-03-31", "2017-04-28")])
    plan, stats = fm.plan_ann_date_repairs(symbols=["000776"], threads=1, start_year=2020)
    assert plan.empty
    assert stats["per_table"]["income_q"]["matched_rows"] == 0


def test_announcement_fetcher_parses_titles_and_skips_cancelled(monkeypatch):
    """标题解析: 只认报告本身 (排除监事意见等噪声与已取消/更新后版本)。"""
    raw = pd.DataFrame(
        {
            "公告标题": [
                "2016年第一季度报告全文",
                "2016年第一季度报告正文",
                "监事会关于公司2016年年度报告的审核意见",
                "2017年年度报告（更新后）",
                "2018年半年度报告（已取消）",
                "2019年年度报告摘要",
                "2020年半年度报告",
            ],
            "公告时间": [
                "2016-04-29", "2016-04-29", "2017-04-21",
                "2018-04-21", "2018-08-31", "2020-04-30", "2020-08-28",
            ],
        }
    )
    monkeypatch.setattr("akshare.stock_zh_a_disclosure_report_cninfo", lambda **kwargs: raw)
    frame = ff.get_periodic_report_announcements("000776", "2016-01-01", "2021-01-01")
    got = {(str(r.report_date)[:10], str(r.ann_date)[:10]) for r in frame.itertuples()}
    assert ("2016-03-31", "2016-04-29") in got
    assert ("2019-12-31", "2020-04-30") in got
    assert ("2020-06-30", "2020-08-28") in got
    assert all("更新后" not in title and "已取消" not in title for title in frame["title"])
    assert all("意见" not in title for title in frame["title"])
    # 同一报告期的 正文/全文 去重为一条, 取最早
    assert len(frame[frame["report_date"] == pd.Timestamp("2016-03-31")]) == 1


def test_plan_basis_recompute_uses_corrected_ann_date(tmp_paths):
    """公告日改早后, basis 必须按新日期重算 (否则 first_reported 标记自相矛盾)。

    踩过的坑: 拿加载时的旧 ann_date 比对 update_date, 会得出"无需变更"的错误结论。
    """
    _seed("income_q", [_income("000776", "2016-03-31", "2017-04-28")])  # update_date == 旧 ann_date
    frame, _ = fm.load_table("income_q")
    assert frame.loc[0, "basis"] == "first_reported"
    plan = fm.plan_basis_recompute(
        "income_q",
        pd.DataFrame([{"symbol": "000776", "report_date": "2016-03-31", "ann_date_new": "2016-04-29"}]),
    )
    assert len(plan) == 1
    assert plan.loc[0, "old_value"] == "first_reported"
    assert plan.loc[0, "new_value"] == "revised"
    assert plan.loc[0, "column"] == "basis"


def test_plan_basis_recompute_skips_when_update_date_missing(tmp_paths):
    """update_date 缺失时无从判断是否被追溯调整 → 保持 first_reported, 不产生变更。"""
    _seed(
        "income_q",
        [
            {
                "symbol": "000776",
                "report_date": pd.Timestamp("2016-03-31"),
                "ann_date": pd.Timestamp("2017-04-28"),
                "report_type": "Q1",
                "net_profit_attr_p": 1e8,
                "revenue": 2e8,
                "currency": "CNY",
            }
        ],
    )
    plan = fm.plan_basis_recompute(
        "income_q",
        pd.DataFrame([{"symbol": "000776", "report_date": "2016-03-31", "ann_date_new": "2016-04-29"}]),
    )
    assert plan.empty


def test_announcement_fetcher_drops_titles_dated_before_period_end(monkeypatch):
    """被错误命名的标题 (公告日早于报告期末) 必须丢弃 —— 实测巨潮档案里存在。

    例: 2016-08-26 发布「2016年年度报告摘要」、2016-04-26 发布「2016年第三季度报告(修订版)」。
    照单全收会把 ann_date 改到报告期之前 (体检的 ann_date >= report_date 会立刻报错)。
    """
    raw = pd.DataFrame(
        {
            "公告标题": ["2016年年度报告摘要", "2016年第三季度报告(修订版)", "2017年年度报告摘要"],
            "公告时间": ["2016-08-26", "2016-04-26", "2018-04-20"],
        }
    )
    monkeypatch.setattr("akshare.stock_zh_a_disclosure_report_cninfo", lambda **kwargs: raw)
    frame = ff.get_periodic_report_announcements("600010", "2015-01-01", "2020-01-01")
    assert list(frame["title"]) == ["2017年年度报告摘要"]
    assert str(frame.iloc[0]["ann_date"])[:10] == "2018-04-20"


def test_plan_rejects_candidate_before_period_end(tmp_paths, monkeypatch):
    """候选日早于报告期末 → 不采纳 (护栏在计划层也要有, 不只依赖抓取层)。"""
    def _fake(symbol: str, start_date: str = "2010-01-01", end_date=None, mode=None):
        return pd.DataFrame(
            [
                {"symbol": "600010", "report_date": pd.Timestamp("2016-12-31"), "ann_date": pd.Timestamp("2016-08-26"), "title": "2016年年度报告摘要"}
            ]
        )

    monkeypatch.setattr(ff, "get_periodic_report_announcements", _fake)
    _seed("income_q", [_income("600010", "2016-12-31", "2018-03-08")])
    plan, stats = fm.plan_ann_date_repairs(symbols=["600010"], threads=1)
    assert plan.empty
    assert stats["per_table"]["income_q"]["matched_rows"] == 0  # 该候选被护栏剔除
