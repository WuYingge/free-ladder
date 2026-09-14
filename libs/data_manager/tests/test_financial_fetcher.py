"""财务取数层单元测试 (companyType 解析 / 预告指标过滤 / 单位换算)。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_financial_fetcher.py -v

不联网: 用假 response 替换 `financial_get`, 断言请求参数正确。

覆盖的关键坑位 (均来自 2026-09-10 实测):
  * companyType 必须逐股解析 —— 硬编码 4 会让银行(3)/证券(1)/保险(2)的
    资产负债表与现金流量表整表为空;
  * 业绩预告的过滤字段是 REPORT_DATE (写 REPORTDATE 会静默拿到业绩报表字段);
  * 巨潮股本变动单位是万股, 落盘统一转股。
"""

from __future__ import annotations

import pandas as pd
import pytest

from fetcher import financial as ff


class _FakeResponse:
    def __init__(self, payload: dict | str):
        self._payload = payload
        self.text = payload if isinstance(payload, str) else ""
        self.status_code = 200

    def json(self) -> dict:
        assert isinstance(self._payload, dict)
        return self._payload

    def raise_for_status(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _clear_cache():
    ff._COMPANY_TYPE_CACHE.clear()
    yield
    ff._COMPANY_TYPE_CACHE.clear()


def test_company_type_is_resolved_per_symbol(monkeypatch):
    """银行/证券/保险必须用各自的 companyType, 不能硬编码 4。"""
    seen: list[tuple[str, dict]] = []

    def fake_get(url, params=None, timeout=20, mode=None):
        seen.append((url, dict(params or {})))
        if url.endswith("/Index"):
            ctype = {"SZ000001": "3", "SZ000776": "1", "SH601318": "2", "SH600519": "4"}[params["code"].upper()]
            return _FakeResponse(f'<input id="hidctype" value="{ctype}" type="hidden">')
        return _FakeResponse({"data": [{"REPORT_DATE": "2026-06-30 00:00:00"}]})

    monkeypatch.setattr(ff, "financial_get", fake_get)

    assert ff.company_type_of("000001") == "3"
    assert ff.company_type_of("000776") == "1"
    assert ff.company_type_of("601318") == "2"
    assert ff.company_type_of("600519") == "4"

    # 资产负债表请求必须带对应 companyType
    ff._statement_dates("000001", "balance")
    balance_call = [p for u, p in seen if u.endswith("/zcfzbDateAjaxNew")][-1]
    assert balance_call["companyType"] == "3"
    assert balance_call["code"] == "SZ000001"


def test_company_type_is_cached(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=20, mode=None):
        calls["n"] += 1
        if url.endswith("/Index"):
            return _FakeResponse('<input id="hidctype" value="3" type="hidden">')
        return _FakeResponse({"data": []})

    monkeypatch.setattr(ff, "financial_get", fake_get)
    ff.company_type_of("000001")
    ff.company_type_of("000001")
    ff.company_type_of("000001")
    assert calls["n"] == 1  # 解析一次后走缓存


def test_forecast_filter_field_is_report_date():
    """预告 filter 字段必须是 REPORT_DATE (REPORTDATE 会静默返回业绩报表数据集)。"""
    assert ff.DATACENTER_DATE_FIELD_BY_REPORT["RPT_PUBLIC_OP_NEWPREDICT"] == "REPORT_DATE"
    assert ff.FORECAST_REPORT_NAME == "RPT_PUBLIC_OP_NEWPREDICT"
    assert ff.DATACENTER_HOST_BY_REPORT[ff.FORECAST_REPORT_NAME] == ff.DATACENTER_ALT_URL


def test_forecast_rejects_wrong_dataset(monkeypatch):
    """接口返回非预告字段时抛错, 而不是静默产出空表。"""
    monkeypatch.setattr(
        ff, "datacenter_rows",
        lambda *a, **k: [{"SECURITY_CODE": "000001", "PARENT_NETPROFIT": 1.0}],
    )
    with pytest.raises(ff.FinancialFetchError):
        ff.get_period_forecast_rows("2025-12-31")


def test_forecast_indicator_filter_excludes_non_amount_metrics():
    assert ff._is_net_profit_indicator("归属于上市公司股东的净利润")
    assert ff._is_net_profit_indicator("扣除非经常性损益后的净利润")
    # 每股收益(元/股) 与 营业收入(元) 量纲不同 → 必须排除
    assert not ff._is_net_profit_indicator("每股收益")
    assert not ff._is_net_profit_indicator("营业收入")
    assert not ff._is_net_profit_indicator("扣除后营业收入")
    assert not ff._is_net_profit_indicator("")


def test_market_symbol_resolution():
    assert ff._resolve_market_symbol("600519") == "SH600519"
    assert ff._resolve_market_symbol("1") == "SZ000001"
    assert ff._resolve_market_symbol("920025") == "BJ920025"
    assert ff._resolve_market_symbol("830799") == "BJ830799"


def test_available_periods_skips_open_disclosure_windows():
    """披露窗口未结束的报告期不抓 (避免把空期写成数据缺失)。"""
    periods = ff.available_periods(2025, today=pd.Timestamp("2026-09-10"))
    assert "2025-12-31" in periods
    assert "2026-06-30" in periods  # 8/31 截止已过
    assert "2026-09-30" not in periods  # 10/31 截止未到
    assert "2026-12-31" not in periods
    # 2026-04-01 时点: 2025 年报(4/30 截止)与 2026Q1 都还没到期
    early = ff.available_periods(2025, today=pd.Timestamp("2026-04-01"))
    assert "2025-12-31" not in early
    assert "2025-09-30" in early


def test_parse_plan_profile_units():
    """分红方案文本: 每 10 股 → 每股现金; 送/转比例保留每 10 股口径。"""
    cash, transfer, bonus = ff._parse_plan_profile("10派1.16元(含税,扣税后1.044元)")
    assert cash == pytest.approx(0.116)
    assert pd.isna(transfer) and pd.isna(bonus)

    cash, transfer, bonus = ff._parse_plan_profile("10转4.50股")
    assert pd.isna(cash)
    assert transfer == pytest.approx(4.5)

    cash, transfer, bonus = ff._parse_plan_profile("10送2转3派1.00元")
    assert cash == pytest.approx(0.1)
    assert bonus == pytest.approx(2.0)
    assert transfer == pytest.approx(3.0)
