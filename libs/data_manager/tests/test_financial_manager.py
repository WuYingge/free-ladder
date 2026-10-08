"""财务落盘/加载/门禁/日更体检单元测试 (不联网, 全用 tmp_path 合成数据)。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_financial_manager.py -v

覆盖: upsert 幂等、更正行不覆盖首版、原子写、列序校验、行级门禁
(ann_date 缺失 / ann_date<report_date / 越出行情区间)、basis 口径、
交易日历换算、batch_check_financials_updated 三态。
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from config import DataPath
from data_manager import financial_manager as fm
from data_manager.financial_schema import BASIS_FIRST_REPORTED, BASIS_REVISED, FINANCIAL_TABLES


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(DataPath, "FINANCIAL_DIR", str(tmp_path / "financial"))
    monkeypatch.setattr(DataPath, "STOCK_PATH", str(tmp_path / "stock_data"))
    os.makedirs(DataPath.FINANCIAL_DIR, exist_ok=True)
    os.makedirs(DataPath.STOCK_PATH, exist_ok=True)
    return tmp_path


def _raw_income(symbol: str, report_date: str, ann_date: str, profit: float, update: str | None = None):
    """构造 fetcher 输出的原始行 (含 _source_update_date 与未使用列)。"""
    return {
        "symbol": symbol,
        "report_date": pd.Timestamp(report_date),
        "ann_date": pd.Timestamp(ann_date),
        "report_type": "ANNUAL",
        "net_profit_attr_p": profit,
        "revenue": profit * 2,
        "eps_basic": 1.0,
        "total_share": 1_000_000_000.0,
        "operating_profit": profit * 1.1,
        "net_profit": profit * 1.02,
        "net_profit_deducted": profit * 0.95,
        "minority_interest_profit": profit * 0.02,
        "rd_expense": 1.0,
        "sell_admin_expense": 2.0,
        "currency": "CNY",
        "_source_update_date": pd.Timestamp(update) if update else pd.NaT,
        "unexpected_column": "should_be_dropped",
    }


def _write_stock_csv(code: str, dates: list[str]) -> None:
    pd.DataFrame({"date": dates, "open": 1.0, "close": 1.0, "high": 1.0, "low": 1.0,
                  "volume": 1.0, "value": 1.0, "range": 1.0, "gain": 0.0,
                  "change": 0.0, "turnOver": 1.0}).to_csv(
        os.path.join(DataPath.STOCK_PATH, f"{code}.csv"), index=False, encoding="utf-8-sig"
    )


def test_upsert_is_idempotent(tmp_paths):
    raw = pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)])
    prepared, quality = fm.prepare_for_storage(raw, "income_q")
    assert quality.schema_ok
    assert list(prepared.columns) == list(FINANCIAL_TABLES["income_q"].columns)
    first = fm.upsert_table("income_q", prepared)
    second = fm.upsert_table("income_q", prepared)
    assert first.rows_added == 1
    assert second.rows_after == 1
    assert second.rows_added == 0
    assert os.path.exists(fm.table_fp("income_q"))
    assert not os.path.exists(fm.table_fp("income_q") + ".tmp")


def test_reupsert_replaces_incomplete_row(tmp_paths):
    """重跑必须能自愈: 缺公告日的旧行应被新抓的完整行覆盖 (按 (symbol, report_date) 合并)。

    背景: 曾用 (symbol, report_date, ann_date) 当去重键 + keep="first", 结果"缺 ann_date
    的旧行"与"带 ann_date 的新行"并存, 陈旧的不完整行胜出 → 实测 33k 行 ann_date 缺失
    无法自愈。现语义: 字段填充数多者优先, 同分取新抓行。
    """
    incomplete = _raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)
    incomplete["ann_date"] = pd.NaT          # 旧数据缺公告日
    incomplete["eps_basic"] = float("nan")
    fm.upsert_table("income_q", fm.prepare_for_storage(pd.DataFrame([incomplete]), "income_q")[0])

    complete = pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)])
    stat = fm.upsert_table("income_q", fm.prepare_for_storage(complete, "income_q")[0])

    assert stat.rows_dropped_duplicate == 1  # 同一事实键 → 合并为 1 行
    frame, _ = fm.load_table("income_q")
    assert len(frame) == 1
    assert frame.iloc[0]["ann_date"] == pd.Timestamp("2025-04-03")  # 公告日被补上
    assert frame.iloc[0]["eps_basic"] == pytest.approx(1.0)
    # 事件帧里能正常出现 (ann_date 缺失的行会被门禁剔除)
    events, drops = fm.load_financial_events("600519", tables=["income_q"])
    assert drops["income_q"].missing_ann_date == 0
    assert len(events["income_q"]) == 1


def test_reupsert_later_value_wins_when_equally_filled(tmp_paths):
    """字段同等完整时以**最新抓取**为准 (重跑幂等 + 可修复旧值)。"""
    first = pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)])
    fm.upsert_table("income_q", fm.prepare_for_storage(first, "income_q")[0])
    second = pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 9.9e9)])
    stat = fm.upsert_table("income_q", fm.prepare_for_storage(second, "income_q")[0])
    assert stat.rows_dropped_duplicate == 1
    frame, _ = fm.load_table("income_q")
    assert len(frame) == 1
    assert frame.iloc[0]["net_profit_attr_p"] == pytest.approx(9.9e9)


def test_replace_is_per_symbol_not_whole_table(tmp_paths):
    """replace=True 只替换"本次抓到的股票", 不能抹掉未抓取的股票。

    实测踩过: `--replace --symbols 3只` 时整表从 318,694 行塌成 246 行 (把未抓取股票全删了)。
    """
    fm.upsert_table(
        "income_q",
        fm.prepare_for_storage(
            pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)]), "income_q"
        )[0],
    )
    fm.upsert_table(
        "income_q",
        fm.prepare_for_storage(
            pd.DataFrame([_raw_income("000001", "2024-12-31", "2025-03-15", 4.4e10)]), "income_q"
        )[0],
    )
    # 只对 600519 做 replace → 000001 必须留下
    stat = fm.upsert_table(
        "income_q",
        fm.prepare_for_storage(
            pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 9.9e9)]), "income_q"
        )[0],
        replace=True,
    )
    frame, _ = fm.load_table("income_q")
    assert set(frame["symbol"]) == {"600519", "000001"}
    assert len(frame) == 2
    assert frame.loc[frame["symbol"] == "600519", "net_profit_attr_p"].iloc[0] == pytest.approx(9.9e9)
    assert stat.rows_after == 2


def test_replace_discards_stale_rows_for_fetched_symbol(tmp_paths):
    """replace 应丢弃被替换股票的**陈旧行**（缺 ann_date 的历史抓取残留）。"""
    incomplete = _raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)
    incomplete["ann_date"] = pd.NaT
    fm.upsert_table("income_q", fm.prepare_for_storage(pd.DataFrame([incomplete]), "income_q")[0])
    fm.upsert_table(
        "income_q",
        fm.prepare_for_storage(
            pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)]), "income_q"
        )[0],
        replace=True,
    )
    frame, _ = fm.load_table("income_q")
    assert len(frame) == 1
    assert pd.notna(frame.iloc[0]["ann_date"])


def test_forecast_records_two_indicators_per_period(tmp_paths):
    """预告的归母/扣非是并列事实, 同一 (symbol, report_date) 必须保留两行。"""
    rows = []
    for indicator in ("归属于上市公司股东的净利润", "扣除非经常性损益后的净利润"):
        rows.append(
            {
                "symbol": "600519", "report_date": pd.Timestamp("2024-12-31"),
                "ann_date": pd.Timestamp("2025-01-20"), "report_type": "ANNUAL",
                "forecast_ann_date": pd.Timestamp("2025-01-20"),
                "forecast_indicator_cn": indicator,
                "announce_type": "预增", "announce_type_en": "increase",
                "forecast_net_profit_low": 1.0e9, "forecast_net_profit_high": 1.2e9,
                "forecast_net_profit_mid": 1.1e9, "yoy_low": 0.1, "yoy_high": 0.2,
                "update_date": pd.Timestamp("2025-01-20"), "basis": "first_reported",
            }
        )
    stat = fm.upsert_table("forecast", fm.prepare_for_storage(pd.DataFrame(rows), "forecast")[0])
    assert stat.rows_after == 2
    again = fm.upsert_table("forecast", fm.prepare_for_storage(pd.DataFrame(rows), "forecast")[0])
    assert again.rows_after == 2  # 幂等


def test_revised_basis_when_source_update_date_is_later(tmp_paths):
    raw = pd.DataFrame(
        [
            _raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9, update="2025-06-01"),
            _raw_income("000001", "2024-12-31", "2025-03-15", 4.4e10, update="2025-03-15"),
        ]
    )
    prepared, _ = fm.prepare_for_storage(raw, "income_q")
    basis = prepared.set_index("symbol")["basis"].to_dict()
    assert basis["600519"] == BASIS_REVISED
    assert basis["000001"] == BASIS_FIRST_REPORTED


def test_load_table_missing_file_and_bad_columns(tmp_paths):
    frame, quality = fm.load_table("income_q")
    assert frame.empty
    assert quality.schema_ok  # 缺文件不是 schema 错
    # 列序不符 → ValueError (防止下游静默错位)
    pd.DataFrame({"symbol": ["600519"], "revenue": [1.0]}).to_csv(
        fm.table_fp("income_q"), index=False, encoding="utf-8-sig"
    )
    with pytest.raises(ValueError):
        fm.load_table("income_q")


def test_load_financial_events_gates_bad_rows(tmp_paths):
    _write_stock_csv("600519", ["2025-01-02", "2025-04-03", "2025-06-30"])
    raw = pd.DataFrame(
        [
            _raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9),        # 正常
            _raw_income("600519", "2023-12-31", "2023-06-01", 1.0e9),        # ann < report → 剔除
            _raw_income("600519", "2022-12-31", "2026-09-01", 0.5e9),        # 晚于行情末日 → 剔除
            _raw_income("600519", "2021-12-31", "2022-04-01", 0.4e9),        # 早于行情首日 → 剔除
        ]
    )
    frame, quality = fm.prepare_for_storage(raw, "income_q")
    fm.upsert_table("income_q", frame)
    events, drops = fm.load_financial_events("600519", tables=["income_q"])
    log = drops["income_q"]
    assert log.ann_before_report == 1
    assert log.out_of_quote_range == 2
    kept = events["income_q"]
    assert len(kept) == 1
    assert kept.index[0] == pd.Timestamp("2025-04-03")
    assert kept.iloc[0]["net_profit_attr_p"] == pytest.approx(1.5e9)


def test_load_financial_events_drops_missing_ann_date(tmp_paths):
    _write_stock_csv("600519", ["2025-01-02", "2025-12-31"])
    raw = pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)])
    raw.loc[0, "ann_date"] = pd.NaT
    prepared, _ = fm.prepare_for_storage(raw, "income_q")
    fm.upsert_table("income_q", prepared)
    events, drops = fm.load_financial_events("600519", tables=["income_q"])
    assert drops["income_q"].missing_ann_date == 1
    assert events["income_q"].empty


def test_basis_filter_excludes_revised_rows(tmp_paths):
    _write_stock_csv("600519", ["2025-01-02", "2025-12-31"])
    raw = pd.DataFrame(
        [
            _raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9, update="2025-06-01"),
            _raw_income("600519", "2025-03-31", "2025-04-25", 0.5e9, update="2025-04-25"),
        ]
    )
    prepared, _ = fm.prepare_for_storage(raw, "income_q")
    fm.upsert_table("income_q", prepared)
    first_only, _ = fm.load_financial_events("600519", basis=BASIS_FIRST_REPORTED, tables=["income_q"])
    everything, _ = fm.load_financial_events("600519", basis=None, tables=["income_q"])
    assert len(first_only["income_q"]) == 1
    assert len(everything["income_q"]) == 2


def test_effective_trade_date_rolls_forward():
    days = [pd.Timestamp(d) for d in ("2025-08-28", "2025-08-29", "2025-09-01")]
    assert fm.effective_trade_date("2025-08-29", days) == pd.Timestamp("2025-08-29")
    # 周六公告 → 顺延到下一交易日
    assert fm.effective_trade_date("2025-08-30", days) == pd.Timestamp("2025-09-01")
    # 超出日历末值 → NaT (不可成交)
    assert pd.isna(fm.effective_trade_date("2025-12-31", days))


def test_batch_check_three_states(tmp_paths):
    status = fm.batch_check_financials_updated(tables=["income_q"]).iloc[0]
    assert not status["exists"]
    assert not bool(status["is_updated"])
    assert "不存在" in status["reason"]

    raw = pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)])
    fm.upsert_table("income_q", fm.prepare_for_storage(raw, "income_q")[0])
    fresh = fm.batch_check_financials_updated(tables=["income_q"]).iloc[0]
    assert bool(fresh["is_updated"])
    assert fresh["rows"] == 1
    assert fresh["last_ann_date"] == pd.Timestamp("2025-04-03").date()
    assert fresh["last_period"] == pd.Timestamp("2024-12-31").date()

    # 把 update_time 改老 → 需更新 (原因不再是"当日已更新")
    frame, _ = fm.load_table("income_q")
    frame["update_time"] = "2020-01-01 00:00:00"
    frame.to_csv(fm.table_fp("income_q"), index=False, encoding="utf-8-sig")
    stale = fm.batch_check_financials_updated(tables=["income_q"]).iloc[0]
    assert not bool(stale["is_updated"])
    assert "未更新" in stale["reason"] or "缺" in stale["reason"]


def test_normalize_drops_unknown_columns(tmp_paths):
    raw = pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)])
    prepared, quality = fm.prepare_for_storage(raw, "income_q")
    assert "unexpected_column" not in prepared.columns
    assert "unexpected_column" in quality.unexpected_columns


def test_update_financials_dry_run_does_not_write(tmp_paths, monkeypatch):
    def _fake_fetch(table, symbols, **kwargs):
        return pd.DataFrame([_raw_income("600519", "2024-12-31", "2025-04-03", 1.5e9)]), []

    monkeypatch.setattr(fm, "fetch_symbol_table", _fake_fetch)
    monkeypatch.setattr(fm, "resolve_symbols", lambda symbols=None, include_delisted=True: ["600519"])
    summary = fm.update_financials(tables=["income_q"], symbols=["600519"], dry_run=True)
    assert summary.iloc[0]["status"] == "dry-run"
    assert not os.path.exists(fm.table_fp("income_q"))


# ---------------------------------------------------------------------------
# 日期格式污染 (2026-09 事故回归): 混格式落盘 → 朴素解析静默变 NaT → 公告日丢失
# ---------------------------------------------------------------------------


def test_mixed_date_formats_in_existing_file_still_load(tmp_paths):
    """库存文件混装 '2021-04-20' 与 '2021-04-20 00:00:00' 时, 读取不得丢公告日。

    pandas 3 的朴素 to_datetime 对混格式列按首值推断格式, 会把少数派格式静默判成 NaT
    (实测三表 12.7 万行); 这里钉死"读取端必须兼容混格式"。
    """
    raw = pd.DataFrame(
        [
            _raw_income("600519", "2020-12-31", "2021-04-01", 1e9),
            _raw_income("000001", "2020-12-31", "2021-04-02", 2e9),
        ]
    )
    fm.upsert_table("income_q", fm.prepare_for_storage(raw, "income_q")[0])
    # 手工把其中一行改写成带时间后缀的旧格式 (模拟历史污染)
    fp = fm.table_fp("income_q")
    text = open(fp, encoding="utf-8-sig").read()
    text = text.replace("2021-04-02", "2021-04-02 00:00:00")
    open(fp, "w", encoding="utf-8-sig").write(text)

    frame, _ = fm.load_table("income_q")
    assert frame["ann_date"].notna().all(), "混格式列不得被解析成 NaT"
    assert pd.Timestamp("2021-04-02") in set(frame["ann_date"])


def test_upsert_writes_canonical_dates_only(tmp_paths):
    """写盘必须只产生规范 YYYY-MM-DD, 即便传入帧的日期列已被污染成混格式。"""
    raw = pd.DataFrame(
        [
            _raw_income("600519", "2020-12-31", "2021-04-01", 1e9),
            _raw_income("000001", "2020-12-31", "2021-04-02", 2e9),
        ]
    )
    prepared, _ = fm.prepare_for_storage(raw, "income_q")
    # 把一行的 ann_date 换成字符串/时间戳混装, 制造 object dtype
    prepared["ann_date"] = pd.Series(
        [pd.Timestamp("2021-04-01"), "2021-04-02 00:00:00"], dtype=object
    )
    fm.upsert_table("income_q", prepared)

    issue = fm.raw_date_format_issues("income_q")
    assert issue["total"] == 0, issue
    raw_text = open(fm.table_fp("income_q"), encoding="utf-8-sig").read()
    assert "00:00:00" not in raw_text
    frame, _ = fm.load_table("income_q")
    assert frame["ann_date"].notna().all()


def test_upsert_rejects_unparseable_date_instead_of_blanking(tmp_paths):
    """非空但不可解析的日期必须抛错 —— 静默清空公告日等于丢数据。"""
    raw = pd.DataFrame(
        [_raw_income("600519", "2020-12-31", "2021-04-01", 1e9)],
    )
    prepared, _ = fm.prepare_for_storage(raw, "income_q")
    prepared["ann_date"] = pd.Series(["不是日期"], dtype="string")
    with pytest.raises(ValueError, match="不可解析"):
        fm.upsert_table("income_q", prepared)


def test_raw_date_format_issues_reports_contamination(tmp_paths):
    """体检用的格式探针: 污染文件必须被点名, 且给出列级计数。"""
    raw = pd.DataFrame([_raw_income("600519", "2020-12-31", "2021-04-01", 1e9)])
    fm.upsert_table("income_q", fm.prepare_for_storage(raw, "income_q")[0])
    assert fm.raw_date_format_issues("income_q")["total"] == 0

    fp = fm.table_fp("income_q")
    text = open(fp, encoding="utf-8-sig").read().replace("2021-04-01", "2021-04-01 00:00:00")
    open(fp, "w", encoding="utf-8-sig").write(text)
    issue = fm.raw_date_format_issues("income_q")
    assert issue["total"] == 1
    assert issue["columns"]["ann_date"] == 1
    assert "ann_date=2021-04-01 00:00:00" in issue["examples"]
