"""财务数据按公告日时点合并 (with_financial) 单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_financial_merge.py -v

核心断言 (两条时间轴准确性):
  * 公告日之前 → 该期财务数据不可见 (NaN), 公告日当天起才有值 (无前视);
  * 报告期与公告日各自保留, 报告期不因公告日被覆盖;
  * 同一公告日多行 → 只保留首版 (更正行不覆盖);
  * with_financial=False 时列集合与旧版完全一致 (回归护栏)。
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from config import DataPath
from data_manager import financial_manager as fm
from data_manager.datasets import FINANCIAL_DATASET_KEYS, resolve_enabled_datasets
from data_manager.stock_data_manager import get_stock_data_by_symbol, get_stock_data_by_symbols

QUOTE_DATES = [
    "2025-04-01",
    "2025-04-02",
    "2025-04-03",
    "2025-04-04",
    "2025-08-01",
    "2025-08-13",
    "2025-08-14",
]


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(DataPath, "FINANCIAL_DIR", str(tmp_path / "financial"))
    monkeypatch.setattr(DataPath, "STOCK_PATH", str(tmp_path / "stock_data"))
    os.makedirs(DataPath.FINANCIAL_DIR, exist_ok=True)
    os.makedirs(DataPath.STOCK_PATH, exist_ok=True)
    _write_stock_csv("600519", QUOTE_DATES)
    yield tmp_path
    # provider 是单例, 测试后重置缓存避免污染其他用例
    from data_manager.providers.financial_provider import FINANCIAL

    FINANCIAL.reload()


def _write_stock_csv(code: str, dates: list[str]) -> None:
    pd.DataFrame(
        {
            "date": dates,
            "open": 10.0, "close": 10.0, "high": 11.0, "low": 9.0,
            "volume": 100.0, "value": 1e6, "range": 1.0,
            "gain": 0.0, "change": 0.0, "turnOver": 1.0,
        }
    ).to_csv(os.path.join(DataPath.STOCK_PATH, f"{code}.csv"), index=False, encoding="utf-8-sig")


def _income_rows() -> pd.DataFrame:
    """两期利润表: 2025Q1 (4/25 公告) 与 2025H1 (8/13 公告)。"""
    return pd.DataFrame(
        [
            {
                "symbol": "600519",
                "report_date": pd.Timestamp("2025-03-31"),
                "ann_date": pd.Timestamp("2025-04-25"),
                "report_type": "Q1",
                "net_profit_attr_p": 2.5e10,
                "revenue": 5.0e10,
                "eps_basic": 19.9,
                "total_share": 1.256e9,
                "currency": "CNY",
                "_source_update_date": pd.Timestamp("2025-04-25"),
            },
            {
                "symbol": "600519",
                "report_date": pd.Timestamp("2025-06-30"),
                "ann_date": pd.Timestamp("2025-08-13"),
                "report_type": "H1",
                "net_profit_attr_p": 4.54e10,
                "revenue": 9.11e10,
                "eps_basic": 36.18,
                "total_share": 1.256e9,
                "currency": "CNY",
                "_source_update_date": pd.Timestamp("2025-08-13"),
            },
        ]
    )


def _seed_tables() -> None:
    for table, raw in (
        ("income_q", _income_rows()),
    ):
        prepared, _ = fm.prepare_for_storage(raw, table)
        fm.upsert_table(table, prepared)
    from data_manager.providers.financial_provider import FINANCIAL

    FINANCIAL.reload()


def test_no_lookahead_before_announcement_date(tmp_paths, monkeypatch):
    """公告日之前拿不到该期数据 (前视偏差的直接防线)。"""
    _seed_tables()
    obj = get_stock_data_by_symbol("600519", with_income=True, with_ochl=False)
    data = obj.data
    q1_ann = pd.Timestamp("2025-04-25")
    h1_ann = pd.Timestamp("2025-08-13")
    # 公告日之前: 只有当日之前的记录; 该股在 4/25 之前没有任何已公告报告期 → 全 NaN
    assert data.loc[data.index < q1_ann, "net_profit_attr_p"].isna().all()
    # 8/01: Q1 已公告 (2.5e10), H1 尚未公告 → 值仍为 Q1
    assert data.loc[pd.Timestamp("2025-08-01"), "net_profit_attr_p"] == pytest.approx(2.5e10)
    # 8/14 (H1 公告次日): 取到 H1 值
    assert data.loc[pd.Timestamp("2025-08-14"), "net_profit_attr_p"] == pytest.approx(4.54e10)


def test_two_timelines_are_preserved(tmp_paths):
    """report_date 与 ann_date 同时保留, 报告期不被公告日覆盖。"""
    _seed_tables()
    events = fm.load_symbol_events("600519", "income_q")
    assert list(events.index) == [pd.Timestamp("2025-04-25"), pd.Timestamp("2025-08-13")]
    assert events.loc[pd.Timestamp("2025-08-13"), "report_date"] == pd.Timestamp("2025-06-30")
    assert events.loc[pd.Timestamp("2025-08-13"), "report_type"] == "H1"


def test_same_ann_date_keeps_first_version(tmp_paths):
    """同一公告日两行 (首版 + 更正) → 只保留首版, 更正行不覆盖。"""
    raw = pd.concat(
        [
            _income_rows().head(1),
            _income_rows().head(1).assign(net_profit_attr_p=9.9e10,
                                          _source_update_date=pd.Timestamp("2025-06-01")),
        ],
        ignore_index=True,
    )
    prepared, _ = fm.prepare_for_storage(raw, "income_q")
    fm.upsert_table("income_q", prepared)
    frame, _ = fm.load_financial_events("600519", basis=None, tables=["income_q"])
    assert len(frame["income_q"]) == 1
    assert frame["income_q"].iloc[0]["net_profit_attr_p"] == pytest.approx(2.5e10)
    # 仅取首版口径时, 同为 4/25 的"更正行"不会出现
    first_only, _ = fm.load_financial_events("600519", tables=["income_q"])
    assert len(first_only["income_q"]) == 1


def test_with_financial_false_keeps_legacy_columns(tmp_paths):
    """回归护栏: 不带开关时列集合与旧版完全一致。"""
    _seed_tables()
    obj = get_stock_data_by_symbol("600519")
    assert set(obj.data.columns) == {
        "open", "close", "high", "low", "volume", "value",
        "range", "gain", "change", "turnOver",
    }
    assert obj.metadata["datasets"]["with_income"] is False


def test_blanket_switch_enables_all_financial_tables(tmp_paths):
    _seed_tables()
    obj = get_stock_data_by_symbol("600519", with_financial=True, with_ochl=False)
    data = obj.data
    # 七表列全部就位 (缺文件的表补 NaN 列, 不抛错)
    for col in ("net_profit_attr_p", "revenue", "total_assets", "ocf_net",
                "announce_type", "circ_share", "cash_div_per_share", "holder_num"):
        assert col in data.columns, col
    assert data.loc[pd.Timestamp("2025-08-14"), "net_profit_attr_p"] == pytest.approx(4.54e10)
    # 缺数据的表 → NaN, 不是 0
    assert data["holder_num"].isna().all()
    assert obj.metadata["datasets"]["with_forecast"] is True


def test_missing_financial_dir_is_silent(tmp_paths, monkeypatch):
    """财务目录/文件完全缺失时, getter 不抛错, 只补 NaN 列。"""
    monkeypatch.setattr(DataPath, "FINANCIAL_DIR", str(tmp_paths / "nope"))
    from data_manager.providers.financial_provider import FINANCIAL

    FINANCIAL.reload()
    obj = get_stock_data_by_symbol("600519", with_financial=True, with_ochl=False)
    assert "net_profit_attr_p" in obj.data.columns
    assert obj.data["net_profit_attr_p"].isna().all()


def test_batch_getter_forwards_financial_flags(tmp_paths):
    _write_stock_csv("000001", QUOTE_DATES)
    _seed_tables()
    objs = get_stock_data_by_symbols(["600519", "000001"], with_income=True, with_ochl=False)
    assert len(objs) == 2
    assert objs[0].data.loc[pd.Timestamp("2025-08-14"), "net_profit_attr_p"] == pytest.approx(4.54e10)
    # 无财务数据的股票 → NaN 列, 不抛错
    assert objs[1].data["net_profit_attr_p"].isna().all()


def test_resolve_enabled_datasets_matrix():
    assert resolve_enabled_datasets() == set()
    assert resolve_enabled_datasets(with_financial=True) == set(FINANCIAL_DATASET_KEYS)
    assert resolve_enabled_datasets(with_income=True) == {"income_q"}
    assert resolve_enabled_datasets(with_financial=True, with_income=False, with_balance=False) == set(
        FINANCIAL_DATASET_KEYS
    )
    assert resolve_enabled_datasets(with_basic=True, with_industry=True) == {"basic", "industry"}


def test_default_basis_keeps_revised_rows_available(tmp_paths, monkeypatch):
    """默认口径必须**包含 revised 行** —— 否则历史财务数据会被整体挡掉。

    实测教训: 源端 `UPDATE_DATE` 普遍晚于 `NOTICE_DATE`, 2010-2024 年报告期只有 3%~13% 的行是
    `first_reported`; 若把 `first_reported` 当默认, `with_financial=True` 在 2020-06 区间
    只有 0/20 天有值、2015 年整段为空。故默认 = 全部行, 严格模式需显式传 basis。
    """
    rows = _income_rows()
    # 把 2025H1 行改成"更正版": 源端更新日晚于首次公告日 → basis=revised
    # (行情日期为 04-01~04-04 与 08-01/08-13/08-14, 故用 H1 行断言可见性)
    rows.loc[1, "_source_update_date"] = pd.Timestamp("2025-09-01")
    fm.upsert_table("income_q", fm.prepare_for_storage(rows, "income_q")[0])
    from data_manager.providers.financial_provider import FINANCIAL

    FINANCIAL.reload()
    assert "revised" in set(FINANCIAL.to_dataframe("income_q")["basis"])

    default_events = FINANCIAL.get_events("600519", "income_q")
    assert pd.Timestamp("2025-06-30") in set(default_events["report_date"]), "默认口径必须保留 revised 行"

    strict_events = FINANCIAL.get_events("600519", "income_q", basis="first_reported")
    assert pd.Timestamp("2025-06-30") not in set(strict_events["report_date"]), "严格模式才排除 revised 行"
    assert pd.Timestamp("2025-03-31") in set(strict_events["report_date"])

    data = get_stock_data_by_symbol("600519", start_date="2025-08-13", end_date="2025-08-14", with_income=True)
    assert data.data["net_profit_attr_p"].notna().all(), "加载路径默认不应把更正版行滤成 NaN"
