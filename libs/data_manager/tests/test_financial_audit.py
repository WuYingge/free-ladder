"""跨源审计单元测试 (离线; 联网部分以合成报告驱动)。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_financial_audit.py -v
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from config import DataPath
from data_manager import financial_audit as audit
from data_manager import financial_health as health
from data_manager import financial_manager as fm


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(DataPath, "FINANCIAL_DIR", str(tmp_path / "financial"))
    monkeypatch.setattr(DataPath, "STOCK_PATH", str(tmp_path / "stock_data"))
    monkeypatch.setattr(DataPath, "DAILY_BASIC_PATH", str(tmp_path / "daily_basic"))
    os.makedirs(DataPath.FINANCIAL_DIR, exist_ok=True)
    os.makedirs(DataPath.STOCK_PATH, exist_ok=True)
    os.makedirs(DataPath.DAILY_BASIC_PATH, exist_ok=True)
    return tmp_path


def _seed_balance(symbol: str, report: str, share: float) -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "report_date": pd.Timestamp(report),
                "ann_date": pd.Timestamp(report) + pd.Timedelta(days=100),
                "report_type": "ANNUAL",
                "total_assets": 1e10,
                "total_equity_attr_p": 5e9,
                "total_liabilities": 5e9,
                "total_equity": 5e9,
                "minority_interest_equity": 0.0,
                "total_share": share,
                "currency": "CNY",
                "update_date": pd.Timestamp(report) + pd.Timedelta(days=100),
            }
        ]
    )
    fm.upsert_table("balance_q", fm.prepare_for_storage(frame, "balance_q")[0])


def _seed_share_capital(symbol: str, effective: str, share: float) -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "ann_date": pd.Timestamp(effective),
                "effective_date": pd.Timestamp(effective),
                "total_share": share,
                "circ_share": share,
                "reason": "测试",
            }
        ]
    )
    fm.upsert_table("share_capital", fm.prepare_for_storage(frame, "share_capital")[0])


def test_share_capital_audit_reports_agreement(tmp_paths):
    _seed_balance("600519", "2025-12-31", 1.25e9)
    _seed_balance("000001", "2025-12-31", 1.94e10)
    _seed_share_capital("600519", "2025-12-31", 1.25e9)   # 一致
    _seed_share_capital("000001", "2025-12-31", 1.00e10)  # 不一致 (差 48%)
    result = audit.audit_share_capital()
    assert result["status"] == "ok"
    assert result["compared_rows"] == 2
    assert result["mismatch_rows"] == 1
    assert result["agreement"] == pytest.approx(0.5)
    assert result["examples"][0]["symbol"] == "000001"


def test_share_capital_audit_asof_uses_effective_date(tmp_paths):
    """股本变动按生效日生效: 报告期之前的最新一次变动才是可比基准。"""
    _seed_balance("600519", "2025-06-30", 2.0e9)
    _seed_share_capital("600519", "2024-01-01", 1.0e9)
    _seed_share_capital("600519", "2025-06-01", 2.0e9)  # 期内生效 → 应当取这条
    result = audit.audit_share_capital()
    assert result["mismatch_rows"] == 0


def test_quote_float_audit_flags_exceeding_rows(tmp_paths):
    _seed_balance("600519", "2025-12-31", 1.25e9)
    pd.DataFrame(
        {"date": pd.to_datetime(["2025-12-31"]), "float_share": [1.25e9], "circ_mv": [1.0], "total_mv": [1.0]}
    ).to_csv(os.path.join(DataPath.DAILY_BASIC_PATH, "600519.csv"), index=False)
    ok = audit.audit_quote_float_share()
    assert ok["float_exceeds_total"] == 0
    assert ok["fully_floated_share"] == pytest.approx(1.0)

    pd.DataFrame(
        {"date": pd.to_datetime(["2025-12-31"]), "float_share": [2.0e9], "circ_mv": [1.0], "total_mv": [1.0]}
    ).to_csv(os.path.join(DataPath.DAILY_BASIC_PATH, "600519.csv"), index=False)
    bad = audit.audit_quote_float_share()
    assert bad["float_exceeds_total"] == 1
    assert bad["examples"][0]["symbol"] == "600519"


def test_sample_symbols_is_stratified_and_deterministic(tmp_paths, monkeypatch):
    from data_manager.providers.stock_list_provider import STOCK_LIST

    codes = ["600000", "600519", "000001", "000002", "300750", "301111", "688111"]
    dates = {code: "2019-01-01" for code in codes}
    dates.update({"600000": "1999-11-10", "000001": "1991-04-03", "301111": "2021-03-01", "688111": "2021-06-01"})
    monkeypatch.setattr(STOCK_LIST, "get_all_symbol", lambda: codes)
    monkeypatch.setattr(STOCK_LIST, "get_list_date", lambda code: dates.get(code, ""))
    first = audit.sample_symbols(n=6, seed=7)
    second = audit.sample_symbols(n=6, seed=7)
    assert first["symbol"].tolist() == second["symbol"].tolist()  # 同种子可复现
    assert len(first) == 6, "抽样数必须等于请求量 (层数多于 n 时也不能超抽)"
    assert set(first["board"]) >= {"沪主板", "深主板"}
    assert set(first["era"]) >= {"2004前", "2020后"}
    assert audit.sample_symbols(n=100, seed=7).shape[0] <= len(codes)  # 宇宙小于 n 时不报错


def test_audit_report_roundtrip_and_age(tmp_paths):
    assert audit.load_audit_report() == {}
    assert audit.audit_age_days() is None
    payload = {"generated_at": "2020-01-01 00:00:00", "suites": {"values": {"sampled_symbols": 1}}}
    fp = audit.write_audit_report(payload)
    assert json.loads(open(fp, encoding="utf-8").read())["suites"]["values"]["sampled_symbols"] == 1
    assert audit.audit_age_days() > 1


def test_health_cross_source_audit_warns_without_network_report(tmp_paths):
    _seed_balance("600519", "2025-12-31", 1.25e9)
    result = health.check_cross_source_audit(["balance_q"], DataPath.FINANCIAL_DIR)
    assert result.metrics["share_capital_two_sources"]["status"] == "skip"
    assert any("联网抽样审计报告" in p["issue"] for p in result.failures)
    assert result.status in {"warn", "fail"}


def test_health_cross_source_audit_reads_latest_report(tmp_paths):
    _seed_balance("600519", "2025-12-31", 1.25e9)
    _seed_share_capital("600519", "2025-12-31", 1.25e9)
    audit.write_audit_report(
        {
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "suites": {
                "values": {
                    "sampled_symbols": 100,
                    "fields": {"net_profit": {"compared": 300, "mismatch": 0, "agreement": 1.0, "gated": True}},
                }
            },
        }
    )
    result = health.check_cross_source_audit(["balance_q"], DataPath.FINANCIAL_DIR)
    assert result.metrics["network_audit"]["sampled_symbols"] == 100
    assert not [p for p in result.failures if "一致率低于下限" in p["issue"]]
    assert not [p for p in result.failures if "尚无联网抽样审计报告" in p["issue"]]


def test_health_cross_source_audit_fails_on_low_field_agreement(tmp_paths):
    _seed_balance("600519", "2025-12-31", 1.25e9)
    audit.write_audit_report(
        {
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "suites": {
                "values": {
                    "sampled_symbols": 10,
                    "fields": {"net_profit": {"compared": 100, "mismatch": 40, "agreement": 0.6, "gated": True}},
                }
            },
        }
    )
    result = health.check_cross_source_audit(["balance_q"], DataPath.FINANCIAL_DIR)
    assert any("net_profit" in p["issue"] for p in result.failures)
