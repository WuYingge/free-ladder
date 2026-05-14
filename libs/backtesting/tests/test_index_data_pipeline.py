"""Focused regression tests for the local index data pipeline.

Run from project root:
    PYTHONPATH=libs pytest libs/backtesting/tests/test_index_data_pipeline.py -v
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest

from config import DataPath
from core.models.index_daily_data import IndexDailyData
import data_manager.index_data_manager as index_manager
from data_manager.providers.index_list_provider import _IndexListProvider
from fetcher.index import STANDARD_HISTORY_COLUMNS, normalize_index_history_frame


def _make_raw_history(
    dates: list[pd.Timestamp],
    *,
    close_start: float = 100.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, date in enumerate(dates):
        close = close_start + offset
        rows.append(
            {
                "日期": date.strftime("%Y-%m-%d"),
                "开盘": close - 1.0,
                "收盘": close,
                "最高": close + 1.0,
                "最低": close - 2.0,
                "成交量": 100_000 + offset,
                "成交额": 1_000_000.0 + offset,
                "振幅": 1.2 + offset,
                "涨跌幅": 0.5 + offset,
                "涨跌额": 0.3 + offset,
                "换手率": 0.1 + offset,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def index_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = tmp_path / "index_name_list.csv"
    manifest.write_text(
        "symbol,name,source,category,enabled\n"
        "000300,沪深300,index_zh_a_hist,core,true\n"
        "000905,中证500,stock_zh_index_hist_csindex,core,false\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "index"
    data_dir.mkdir()

    original_manifest = DataPath.INDEX_NAME_LIST_DF
    original_index_path = DataPath.INDEX_PATH

    monkeypatch.setattr(DataPath, "INDEX_NAME_LIST_DF", str(manifest))
    monkeypatch.setattr(DataPath, "INDEX_PATH", str(data_dir))

    provider = _IndexListProvider.get_instance()
    provider.init()

    yield provider, data_dir

    monkeypatch.setattr(DataPath, "INDEX_NAME_LIST_DF", original_manifest)
    monkeypatch.setattr(DataPath, "INDEX_PATH", original_index_path)
    provider.init()


def test_index_list_provider_reads_manifest_metadata(index_test_env):
    provider, _ = index_test_env

    assert provider.get_name("000300") == "沪深300"
    assert provider.get_source("000300") == "index_zh_a_hist"
    assert provider.get_category("000300") == "core"
    assert provider.get_all_symbol() == ["000300"]
    assert provider.get_all_symbol(enabled_only=False) == ["000300", "000905"]


def test_normalize_csindex_history_converts_units_and_schema():
    raw = pd.DataFrame(
        [
            {
                "日期": "20260514",
                "开盘": "5027.64",
                "收盘": "4914.60",
                "最高": "5030.52",
                "最低": "4913.60",
                "成交量": "28069.54",
                "成交金额": "8574.46",
                "涨跌": "-83.74",
                "涨跌幅": "-1.68",
            }
        ]
    )

    normalized = normalize_index_history_frame(
        raw,
        source="stock_zh_index_hist_csindex",
    )

    assert normalized.columns.tolist() == STANDARD_HISTORY_COLUMNS
    row = normalized.iloc[0]
    assert row["日期"] == "20260514"
    assert row["成交量"] == pytest.approx(280_695_400.0)
    assert row["成交额"] == pytest.approx(857_446_000_000.0)
    assert pd.isna(row["振幅"])
    assert pd.isna(row["换手率"])


def test_update_single_index_data_saves_local_csv(index_test_env, monkeypatch: pytest.MonkeyPatch):
    _, data_dir = index_test_env
    target_date = pd.Timestamp("2026-05-14")
    raw_history = _make_raw_history([target_date])

    monkeypatch.setattr(index_manager, "_fetch_full_index_history", lambda symbol: raw_history)

    symbol, result = index_manager.update_single_index_data("000300")
    status = index_manager.batch_check_index_data_updated(["000300"], target_date=target_date)
    local_data = index_manager.get_index_data_by_symbol("000300")

    assert (symbol, result) == ("000300", True)
    assert (data_dir / "000300.csv").exists()
    assert bool(status.iloc[0]["is_updated"]) is True
    assert local_data.name == "沪深300"
    assert local_data.columns.tolist() == [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "value",
        "range",
        "gain",
        "change",
        "turnOver",
    ]


def test_fetch_index_history_in_slices_chunks_and_deduplicates(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    responses = {
        ("2026-01-01", "2026-01-02"): _make_raw_history(
            [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
            close_start=10.0,
        ),
        ("2026-01-03", "2026-01-04"): _make_raw_history(
            [
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-01-03"),
                pd.Timestamp("2026-01-04"),
            ],
            close_start=20.0,
        ),
        ("2026-01-05", "2026-01-05"): _make_raw_history(
            [pd.Timestamp("2026-01-05")],
            close_start=30.0,
        ),
    }

    def fake_fetch(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        del symbol
        key = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        calls.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
        return responses[key]

    monkeypatch.setattr(index_manager, "_fetch_index_history", fake_fetch)

    combined = index_manager._fetch_index_history_in_slices(
        symbol="000300",
        start_date=datetime.datetime(2026, 1, 1),
        end_date=datetime.datetime(2026, 1, 5),
        max_slice_days=2,
    )

    assert calls == [
        (pd.Timestamp("2026-01-01 00:00:00"), pd.Timestamp("2026-01-02 00:00:00")),
        (pd.Timestamp("2026-01-03 00:00:00"), pd.Timestamp("2026-01-04 00:00:00")),
        (pd.Timestamp("2026-01-05 00:00:00"), pd.Timestamp("2026-01-05 00:00:00")),
    ]
    assert combined["日期"].tolist() == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
    ]


def test_update_existing_index_file_merges_new_rows(index_test_env, monkeypatch: pytest.MonkeyPatch):
    _, data_dir = index_test_env
    today = pd.Timestamp(datetime.date.today())
    old_date = today - pd.Timedelta(days=2)
    new_date = today - pd.Timedelta(days=1)

    index_manager.save_index_data("000300", _make_raw_history([old_date], close_start=100.0))
    monkeypatch.setattr(
        index_manager,
        "_fetch_index_history",
        lambda symbol, start_date, end_date: _make_raw_history([new_date], close_start=200.0),
    )

    assert index_manager.update("000300") is True

    saved = pd.read_csv(data_dir / "000300.csv")
    assert saved["date"].tolist() == [old_date.strftime("%Y-%m-%d"), new_date.strftime("%Y-%m-%d")]
    assert saved["close"].tolist() == [100.0, 200.0]


def test_index_daily_data_slice_date_range_preserves_identity():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [1_000, 1_100, 1_200, 1_300],
            "value": [10_000, 11_000, 12_000, 13_000],
            "turnOver": [0.1, 0.2, 0.3, 0.4],
            "gain": [0.0, 0.1, 0.2, 0.3],
            "change": [0.0, 1.0, 1.0, 1.0],
        }
    )

    data = IndexDailyData(frame, symbol="000300", name="沪深300")
    sliced = data.slice_date_range("2026-01-02", "2026-01-03")

    assert data.validate_data() is True
    assert sliced.symbol == "000300"
    assert sliced.name == "沪深300"
    assert sliced.data["close"].tolist() == [11.5, 12.5]
    assert sliced.data["date"].astype(str).tolist() == ["2026-01-02", "2026-01-03"]