"""Regression tests for ETF update resilience: pending-slice resume + backoff.

Run from project root:
    PYTHONPATH=libs pytest libs/backtesting/tests/test_etf_data_manager_resilience.py -v
"""

from __future__ import annotations

import datetime
import json

import pandas as pd
import pytest

import data_manager.etf_data_manager as etf_data_manager
from config import DataPath


EM_COLUMNS = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]


def _em_df(dates: list[str]) -> pd.DataFrame:
    """Build a fake EastMoney-style kline frame for the given dates."""
    rows = [[d, 1.0, 1.0, 1.0, 1.0, 100, 100.0, 0.0, 0.0, 0.0, 0.0] for d in dates]
    return pd.DataFrame(rows, columns=EM_COLUMNS)


@pytest.fixture
def data_dir(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point DEFAULT_PATH at a temp dir so no real data/etf_data is touched."""
    path = str(tmp_path / "etf_data")
    monkeypatch.setattr(DataPath, "DEFAULT_PATH", path)
    return path


def test_fetch_slices_with_retry_collects_failed_slices(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str]] = []

    def fake_fetch(code, s, e):
        calls.append((s, e))
        if s == "20260220":
            raise RuntimeError("boom")
        return _em_df([s[:8]])

    monkeypatch.setattr(etf_data_manager, "get_etf_certain_date_data", fake_fetch)
    monkeypatch.setattr(etf_data_manager, "intervals", lambda seconds=1.0: None)

    df, failed = etf_data_manager._fetch_slices_with_retry(
        "123456", [("20260101", "20260219"), ("20260220", "20260409"), ("20260410", "20260529")]
    )

    assert len(df) == 2
    assert failed == [("20260220", "20260409")]


def test_fetch_slices_retry_uses_exponential_backoff(monkeypatch: pytest.MonkeyPatch):
    sleeps: list[float] = []

    def fake_fetch(code, s, e):
        raise RuntimeError("always fails")

    monkeypatch.setattr(etf_data_manager, "get_etf_certain_date_data", fake_fetch)
    monkeypatch.setattr(etf_data_manager, "intervals", lambda seconds=1.0: sleeps.append(seconds))

    df, failed = etf_data_manager._fetch_slices_with_retry("123456", [("20260101", "20260219")])

    assert df.empty
    assert failed == [("20260101", "20260219")]
    # 5 retries -> 4 sleeps between them, exponential 0.5/1/2/4 (capped at 8)
    assert sleeps == [0.5, 1.0, 2.0, 4.0]


def test_get_with_retry_writes_pending_on_partial_failure(data_dir: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(etf_data_manager, "_is_not_listed_yet", lambda code: False)
    monkeypatch.setattr(
        etf_data_manager,
        "_fetch_slices_with_retry",
        lambda code, slices: (_em_df(["20260101"]), [("20260220", "20260409")]),
    )

    df = etf_data_manager.get_with_retry("123456", 100)

    assert df is not None and len(df) == 1
    pending = etf_data_manager._load_pending_slices("123456")
    assert pending == [("20260220", "20260409")]


def test_get_with_retry_resumes_from_pending_slices(data_dir: str, monkeypatch: pytest.MonkeyPatch):
    etf_data_manager._save_pending_slices("123456", [("20260220", "20260409")])
    seen_slices: list[list[tuple[str, str]]] = []
    monkeypatch.setattr(etf_data_manager, "_is_not_listed_yet", lambda code: False)
    monkeypatch.setattr(
        etf_data_manager,
        "_fetch_slices_with_retry",
        lambda code, slices: seen_slices.append(slices) or (_em_df(["20260221"]), []),
    )

    df = etf_data_manager.get_with_retry("123456", 4000)

    # Must retry only the pending slice, not the full 4000-day slice list.
    assert seen_slices == [[("20260220", "20260409")]]
    assert df is not None and len(df) == 1
    assert etf_data_manager._load_pending_slices("123456") is None


def test_get_with_retry_clears_pending_on_full_success(data_dir: str, monkeypatch: pytest.MonkeyPatch):
    etf_data_manager._save_pending_slices("123456", [("20260220", "20260409")])
    monkeypatch.setattr(etf_data_manager, "_is_not_listed_yet", lambda code: False)
    monkeypatch.setattr(
        etf_data_manager,
        "_fetch_slices_with_retry",
        lambda code, slices: (_em_df(["20260221"]), []),
    )

    etf_data_manager.get_with_retry("123456", 100)

    assert not etf_data_manager._has_pending_slices("123456")


def test_get_with_retry_returns_none_for_empty_data(data_dir: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(etf_data_manager, "_is_not_listed_yet", lambda code: False)
    monkeypatch.setattr(
        etf_data_manager,
        "_fetch_slices_with_retry",
        lambda code, slices: (pd.DataFrame(), []),
    )

    assert etf_data_manager.get_with_retry("123456", 100) is None


def test_update_single_etf_data_partial_success_saves_and_marks_failed(
    data_dir: str, monkeypatch: pytest.MonkeyPatch
):
    def fake_get_with_retry(code, last_n_days):
        etf_data_manager._save_pending_slices(code, [("20260220", "20260409")])
        return _em_df(["20260101"])

    monkeypatch.setattr(etf_data_manager, "get_with_retry", fake_get_with_retry)

    code, ok = etf_data_manager.update_single_etf_data("123456")

    assert ok is False
    assert etf_data_manager.get_symbol_fp("123456") and __import__("os").path.exists(
        etf_data_manager.get_symbol_fp("123456")
    )
    # Second run: file exists + pending -> goes through resume path.
    monkeypatch.setattr(etf_data_manager, "_complete_pending_slices", lambda code: True)
    assert etf_data_manager.update_single_etf_data("123456") == ("123456", True)


def test_update_single_etf_data_completes_pending_when_file_exists(
    data_dir: str, monkeypatch: pytest.MonkeyPatch
):
    etf_data_manager._save_pending_slices("123456", [("20260220", "20260409")])
    completed: list[str] = []
    monkeypatch.setattr(
        etf_data_manager,
        "_complete_pending_slices",
        lambda code: completed.append(code) or True,
    )

    assert etf_data_manager.update_single_etf_data("123456") == ("123456", True)
    assert completed == ["123456"]


def test_complete_pending_slices_merges_gap(data_dir: str, monkeypatch: pytest.MonkeyPatch):
    # Existing CSV already has data before and after the missing window.
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-03-01", "2026-03-02"]),
            "open": [1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0, 1.0],
            "volume": [100, 100, 100, 100],
            "value": [100.0, 100.0, 100.0, 100.0],
            "range": [0.0, 0.0, 0.0, 0.0],
            "gain": [0.0, 0.0, 0.0, 0.0],
            "change": [0.0, 0.0, 0.0, 0.0],
            "turnOver": [0.0, 0.0, 0.0, 0.0],
        }
    ).set_index("date")
    existing.to_csv(etf_data_manager.get_symbol_fp("123456"), encoding="utf-8-sig", index=True)

    etf_data_manager._save_pending_slices("123456", [("20260103", "20260228")])
    monkeypatch.setattr(
        etf_data_manager,
        "_fetch_slices_with_retry",
        lambda code, slices: (_em_df(["20260103", "20260105", "20260228"]), []),
    )

    assert etf_data_manager._complete_pending_slices("123456") is True

    merged = pd.read_csv(etf_data_manager.get_symbol_fp("123456"), parse_dates=True, index_col=0)
    assert merged.index.is_unique
    assert pd.Timestamp("2026-01-03") in merged.index
    assert pd.Timestamp("2026-02-28") in merged.index
    assert len(merged) == 7  # 4 existing + 3 gap rows, no duplicates


def test_complete_pending_slices_keeps_pending_on_further_failure(
    data_dir: str, monkeypatch: pytest.MonkeyPatch
):
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "open": [1.0], "close": [1.0], "high": [1.0], "low": [1.0],
            "volume": [100], "value": [100.0], "range": [0.0],
            "gain": [0.0], "change": [0.0], "turnOver": [0.0],
        }
    ).set_index("date")
    existing.to_csv(etf_data_manager.get_symbol_fp("123456"), encoding="utf-8-sig", index=True)

    etf_data_manager._save_pending_slices("123456", [("20260103", "20260228")])
    monkeypatch.setattr(
        etf_data_manager,
        "_fetch_slices_with_retry",
        lambda code, slices: (_em_df(["20260103"]), [("20260105", "20260228")]),
    )

    assert etf_data_manager._complete_pending_slices("123456") is False
    assert etf_data_manager._load_pending_slices("123456") == [("20260105", "20260228")]
    # Successful slice data must have been merged already.
    merged = pd.read_csv(etf_data_manager.get_symbol_fp("123456"), parse_dates=True, index_col=0)
    assert pd.Timestamp("2026-01-03") in merged.index


def test_update_etf_data_uses_configured_pool_size(monkeypatch: pytest.MonkeyPatch):
    class FakePool:
        instances: list[FakePool] = []

        def __init__(self, size, initializer=None):
            self.size = size
            FakePool.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, fn, items):
            return [(code, True) for code in items]

    monkeypatch.setattr(etf_data_manager, "Pool", FakePool)
    monkeypatch.setattr(etf_data_manager, "ETF_UPDATE_POOL_SIZE", 3)

    etf_data_manager.update_etf_data(["001234", "002345", "003456"])

    assert FakePool.instances[-1].size == 3


def test_pending_slices_survive_json_roundtrip(data_dir: str):
    etf_data_manager._save_pending_slices("123456", [("20260101", "20260219"), ("20260301", "20260419")])
    assert etf_data_manager._load_pending_slices("123456") == [
        ("20260101", "20260219"),
        ("20260301", "20260419"),
    ]
    assert etf_data_manager._has_pending_slices("123456")
    etf_data_manager._clear_pending_slices("123456")
    assert not etf_data_manager._has_pending_slices("123456")
