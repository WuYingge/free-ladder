"""财务溯源 sidecar 单元测试 (不联网)。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_financial_provenance.py -v

覆盖: meta 逐列来源声明、修正流水保留原值、dry-run 不落盘、
按 old_value 校验跳过并发修改、回滚写回并留痕、汇总。
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from config import DataPath
from data_manager import financial_manager as fm
from data_manager import financial_provenance as prov


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(DataPath, "FINANCIAL_DIR", str(tmp_path / "financial"))
    os.makedirs(DataPath.FINANCIAL_DIR, exist_ok=True)
    return tmp_path


def _seed_income(symbol: str, report: str, ann: str, profit: float = 1e9) -> None:
    raw = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "report_date": pd.Timestamp(report),
                "ann_date": pd.Timestamp(ann),
                "report_type": "ANNUAL",
                "net_profit_attr_p": profit,
                "revenue": profit * 2,
                "eps_basic": 1.0,
                "currency": "CNY",
                "update_date": pd.Timestamp(ann),
            }
        ]
    )
    fm.upsert_table("income_q", fm.prepare_for_storage(raw, "income_q")[0])


def test_meta_declares_source_per_column(tmp_paths):
    fp = prov.write_meta("income_q")
    payload = json.loads(open(fp, encoding="utf-8").read())
    assert payload["default_source"] == prov.EM_F10
    assert payload["columns"]["ann_date"] == prov.EM_F10
    assert payload["columns"]["symbol"] == prov.EM_F10
    # 巨潮来源的表要逐列写清
    fp2 = prov.write_meta("share_capital")
    payload2 = json.loads(open(fp2, encoding="utf-8").read())
    assert payload2["columns"]["total_share"].startswith("cninfo")
    assert payload2["row_level_exceptions"]


def test_apply_corrections_writes_table_and_log(tmp_paths):
    _seed_income("000776", "2016-03-31", "2017-04-28")
    stat = prov.apply_corrections(
        "income_q",
        pd.DataFrame(
            [{"symbol": "000776", "report_date": "2016-03-31", "column": "ann_date", "new_value": "2016-04-29"}]
        ),
        source="cninfo:disclosure",
        reason="东财错位一年",
        tool="test",
    )
    assert stat.applied == 1 and stat.logged == 1
    frame, _ = fm.load_table("income_q")
    assert frame.loc[0, "ann_date"] == pd.Timestamp("2016-04-29")
    log = prov.load_corrections("income_q")
    assert list(log["old_value"]) == ["2017-04-28"]
    assert list(log["new_value"]) == ["2016-04-29"]
    assert log.loc[0, "source"] == "cninfo:disclosure"


def test_dry_run_does_not_touch_table_or_log(tmp_paths):
    _seed_income("000776", "2016-03-31", "2017-04-28")
    stat = prov.apply_corrections(
        "income_q",
        pd.DataFrame(
            [{"symbol": "000776", "report_date": "2016-03-31", "column": "ann_date", "new_value": "2016-04-29"}]
        ),
        source="cninfo:disclosure",
        reason="dry-run",
        apply=False,
    )
    assert stat.applied == 1 and stat.logged == 0
    frame, _ = fm.load_table("income_q")
    assert frame.loc[0, "ann_date"] == pd.Timestamp("2017-04-28")
    assert prov.load_corrections("income_q").empty


def test_old_value_mismatch_skips_row(tmp_paths):
    """并发修改保护: 计划里的 old_value 与主表不一致时跳过, 不覆盖别人的修改。"""
    _seed_income("000776", "2016-03-31", "2017-04-28")
    stat = prov.apply_corrections(
        "income_q",
        pd.DataFrame(
            [
                {
                    "symbol": "000776",
                    "report_date": "2016-03-31",
                    "column": "ann_date",
                    "old_value": "1999-01-01",
                    "new_value": "2016-04-29",
                }
            ]
        ),
        source="cninfo:disclosure",
        reason="并发保护",
    )
    assert stat.applied == 0 and stat.unchanged == 1
    frame, _ = fm.load_table("income_q")
    assert frame.loc[0, "ann_date"] == pd.Timestamp("2017-04-28")


def test_missing_key_counted(tmp_paths):
    _seed_income("000776", "2016-03-31", "2017-04-28")
    stat = prov.apply_corrections(
        "income_q",
        pd.DataFrame(
            [{"symbol": "999999", "report_date": "2016-03-31", "column": "ann_date", "new_value": "2016-04-29"}]
        ),
        source="cninfo:disclosure",
        reason="不存在的主键",
    )
    assert stat.missing_key == 1 and stat.applied == 0


def test_revert_restores_old_value_and_logs(tmp_paths):
    _seed_income("000776", "2016-03-31", "2017-04-28")
    prov.apply_corrections(
        "income_q",
        pd.DataFrame(
            [{"symbol": "000776", "report_date": "2016-03-31", "column": "ann_date", "new_value": "2016-04-29"}]
        ),
        source="cninfo:disclosure",
        reason="东财错位一年",
    )
    stat = prov.revert_corrections("income_q", source="cninfo:disclosure", apply=True)
    assert stat.applied == 1
    frame, _ = fm.load_table("income_q")
    assert frame.loc[0, "ann_date"] == pd.Timestamp("2017-04-28")
    log = prov.load_corrections("income_q")
    assert len(log) == 2  # 原修正 + 回滚各一条, 全程留痕
    assert log.iloc[-1]["source"].endswith(":revert")


def test_numeric_correction_keeps_dtype(tmp_paths):
    _seed_income("000776", "2016-03-31", "2017-04-28", profit=1e9)
    stat = prov.apply_corrections(
        "income_q",
        pd.DataFrame(
            [{"symbol": "000776", "report_date": "2016-03-31", "column": "revenue", "new_value": 12345.0}]
        ),
        source="sina:income",
        reason="数值仲裁",
    )
    assert stat.applied == 1
    frame, _ = fm.load_table("income_q")
    assert frame.loc[0, "revenue"] == pytest.approx(12345.0)


def test_provenance_summary_lists_corrections(tmp_paths):
    _seed_income("000776", "2016-03-31", "2017-04-28")
    prov.apply_corrections(
        "income_q",
        pd.DataFrame(
            [{"symbol": "000776", "report_date": "2016-03-31", "column": "ann_date", "new_value": "2016-04-29"}]
        ),
        source="cninfo:disclosure",
        reason="东财错位一年",
    )
    summary = prov.provenance_summary(["income_q", "balance_q"])
    row = summary[summary["table"] == "income_q"].iloc[0]
    assert row["corrections"] == 1
    assert row["columns"] == "ann_date"
    assert row["sources"] == "cninfo:disclosure"
    assert summary[summary["table"] == "balance_q"].iloc[0]["corrections"] == 0


def test_full_refetch_cannot_overwrite_corrected_cell(tmp_paths):
    """全量重抓不得冲掉已仲裁的修正 —— 修正流水在写盘前重放。

    踩过的坑: 去重规则是"字段填充数多者优先 → 同分取新抓行"; 用巨潮仲裁改过 ann_date 后,
    再跑一次 --all-history --replace 会抓到东财原值, 填充数相同 → 新行胜出、修正被静默冲掉。
    """
    _seed_income("000776", "2016-03-31", "2017-04-28")
    prov.apply_corrections(
        "income_q",
        pd.DataFrame(
            [{"symbol": "000776", "report_date": "2016-03-31", "column": "ann_date", "new_value": "2016-04-29"}]
        ),
        source="cninfo:disclosure",
        reason="东财错位一年",
    )
    # 模拟全量重抓: 东财又返回那个错位日期
    _seed_income("000776", "2016-03-31", "2017-04-28", profit=1.1e9)
    frame, _ = fm.load_table("income_q")
    assert frame.loc[0, "ann_date"] == pd.Timestamp("2016-04-29"), "重抓后修正必须仍然生效"
    assert frame.loc[0, "net_profit_attr_p"] == pytest.approx(1.1e9), "非修正列的数值应更新为重抓值"
    # 撤销修正后, 东财原值恢复可用 (显式回滚才生效)
    prov.revert_corrections("income_q", source="cninfo:disclosure")
    frame2, _ = fm.load_table("income_q")
    assert frame2.loc[0, "ann_date"] == pd.Timestamp("2017-04-28")
