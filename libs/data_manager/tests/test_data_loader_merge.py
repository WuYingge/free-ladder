"""统一 getter 合并 (with_basic/with_ochl/区间) 单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_data_loader_merge.py -v
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from config import DataPath
from data_manager.daily_basic_manager import DAILY_BASIC_COLUMNS, load_daily_basic
from data_manager.stock_data_manager import get_stock_data_by_symbol, get_stock_data_by_symbols


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(DataPath, "STOCK_PATH", str(tmp_path / "stock_data"))
    monkeypatch.setattr(DataPath, "DAILY_BASIC_PATH", str(tmp_path / "daily_basic"))
    os.makedirs(DataPath.STOCK_PATH, exist_ok=True)
    os.makedirs(DataPath.DAILY_BASIC_PATH, exist_ok=True)
    return tmp_path


def _write_stock_csv(tmp_path, code: str, dates: list[str]) -> None:
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
            "volume": 10000.0, "value": 1e8, "range": 2.0,
            "gain": 1.0, "change": 1.0, "turnOver": 1.0,
        }
    )
    df.to_csv(DataPath.STOCK_PATH + f"/{code}.csv", index=False, encoding="utf-8-sig")


def _write_basic_csv(code: str, rows: list[tuple[str, float, float, float]]) -> None:
    idx = pd.DatetimeIndex(pd.to_datetime([r[0] for r in rows]), name="date")
    df = pd.DataFrame(
        {"circ_mv": [r[1] for r in rows], "total_mv": [r[2] for r in rows],
         "float_share": [r[3] for r in rows]},
        index=idx,
    )
    df.to_csv(DataPath.DAILY_BASIC_PATH + f"/{code}.csv", encoding="utf-8-sig", index=True)


def test_default_call_unchanged(tmp_paths):
    _write_stock_csv(tmp_paths, "000001", ["2026-09-01", "2026-09-02"])
    obj = get_stock_data_by_symbol("000001")
    # 默认不附加市值列, 且索引统一为日期索引 (消费方 normalize 兼容)
    expected_cols = {"open", "high", "low", "close", "volume", "value", "range",
                     "gain", "change", "turnOver"}
    assert set(obj.data.columns) == expected_cols
    assert isinstance(obj.data.index, pd.DatetimeIndex)


def test_with_basic_merges_and_left_joins(tmp_paths):
    _write_stock_csv(tmp_paths, "000001", ["2026-09-01", "2026-09-02", "2026-09-03"])
    _write_basic_csv(
        "000001",
        [("2026-09-01", 1e10, 1.2e10, 1e8), ("2026-09-03", 1.1e10, 1.3e10, 1.1e8)],
    )
    obj = get_stock_data_by_symbol("000001", with_basic=True)
    assert DAILY_BASIC_COLUMNS[0] in obj.data.columns
    # 2026-09-02 无市值行 → NaN (left join, 不强填)
    assert pd.isna(obj.data.loc[pd.Timestamp("2026-09-02"), "circ_mv"])
    assert obj.data.loc[pd.Timestamp("2026-09-01"), "circ_mv"] == pytest.approx(1e10)
    assert obj.metadata["datasets"]["with_basic"] is True


def test_with_basic_missing_file_adds_nan_columns(tmp_paths):
    _write_stock_csv(tmp_paths, "000001", ["2026-09-01"])
    obj = get_stock_data_by_symbol("000001", with_basic=True)
    # 文件缺失不报错: 附加列以 NaN 存在
    for col in DAILY_BASIC_COLUMNS:
        assert col in obj.data.columns
        assert obj.data[col].isna().all()


def test_with_ochl_false_placeholder(tmp_paths):
    _write_stock_csv(tmp_paths, "000001", ["2026-09-01"])
    _write_basic_csv("000001", [("2026-09-01", 1e10, 1.2e10, 1e8)])
    obj = get_stock_data_by_symbol("000001", with_ochl=False, with_basic=True)
    # 占位构造通过校验, OHLCV 全为 NaN, 市值列保留
    assert obj.data["open"].isna().all()
    assert obj.data["circ_mv"].iloc[0] == pytest.approx(1e10)
    assert obj.metadata["datasets"]["ochl"] is False


def test_start_end_slice(tmp_paths):
    _write_stock_csv(tmp_paths, "000001", ["2026-08-31", "2026-09-01", "2026-09-02"])
    _write_basic_csv(
        "000001",
        [("2026-08-31", 1e10, 1.2e10, 1e8), ("2026-09-01", 1.1e10, 1.3e10, 1.1e8),
         ("2026-09-02", 1.2e10, 1.4e10, 1.2e8)],
    )
    obj = get_stock_data_by_symbol("000001", start_date="2026-09-01", with_basic=True)
    assert len(obj.data) == 2
    assert obj.data.index.min() == pd.Timestamp("2026-09-01")
    assert obj.data.loc[pd.Timestamp("2026-09-01"), "circ_mv"] == pytest.approx(1.1e10)


def test_batch_getter_forward_params(tmp_paths):
    _write_stock_csv(tmp_paths, "000001", ["2026-09-01"])
    _write_stock_csv(tmp_paths, "000002", ["2026-09-01"])
    _write_basic_csv("000001", [("2026-09-01", 1e10, 1.2e10, 1e8)])
    objs = get_stock_data_by_symbols(["000001", "000002"], with_basic=True)
    assert len(objs) == 2
    assert [o.symbol for o in objs] == ["000001", "000002"]
    assert "circ_mv" in objs[0].data.columns
    assert "circ_mv" in objs[1].data.columns  # 缺文件 → NaN 列


def test_load_daily_basic_roundtrip(tmp_paths):
    _write_basic_csv("000001", [("2026-09-01", 1e10, 1.2e10, 1e8)])
    df = load_daily_basic("000001")
    assert list(df.columns) == DAILY_BASIC_COLUMNS
    assert df.index.name == "date"
    assert load_daily_basic("999999").empty
