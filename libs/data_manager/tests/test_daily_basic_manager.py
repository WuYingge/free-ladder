"""daily_basic_manager 单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_daily_basic_manager.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from config import DataPath
from data_manager import daily_basic_manager as dbm
from data_manager.daily_basic_manager import (
    DAILY_BASIC_COLUMNS,
    backfill_daily_basic_symbol,
    batch_check_daily_basic_updated,
    estimate_from_stock_history,
    update_daily_basic_symbol,
)


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(DataPath, "STOCK_PATH", str(tmp_path / "stock_data"))
    monkeypatch.setattr(DataPath, "DAILY_BASIC_PATH", str(tmp_path / "daily_basic"))
    os.makedirs(DataPath.STOCK_PATH, exist_ok=True)
    os.makedirs(DataPath.DAILY_BASIC_PATH, exist_ok=True)
    return tmp_path


@pytest.fixture
def no_delist(monkeypatch):
    monkeypatch.setattr(dbm.STOCK_LIST, "get_delist_date", lambda symbol: "")


def _write_stock_csv(tmp_path, code: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(
        DataPath.STOCK_PATH + f"/{code}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _stock_row(d: str, *, volume=10000.0, value=1e8, turn_over=1.0) -> dict:
    return {
        "date": d,
        "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
        "volume": volume, "value": value, "range": 2.0,
        "gain": 1.0, "change": 1.0, "turnOver": turn_over,
    }


# ---------------------------------------------------------------------------
# 估算公式
# ---------------------------------------------------------------------------

def test_estimate_formula_exact():
    idx = pd.to_datetime(["2017-01-03", "2017-01-04"])
    df = pd.DataFrame(
        {
            "volume": [10000.0, 5000.0],   # 手
            "value": [1e8, 2e8],           # 元
            "turnOver": [1.0, 2.0],        # %
        },
        index=idx,
    )
    res = estimate_from_stock_history(df, ratio=1.5)
    # circ = 1e8 / 0.01 = 1e10; float = 10000*10000/1.0 = 1e8
    assert res.loc[idx[0], "circ_mv"] == pytest.approx(1e10)
    assert res.loc[idx[0], "float_share"] == pytest.approx(1e8)
    assert res.loc[idx[0], "total_mv"] == pytest.approx(1.5e10)
    # 第二行: circ = 2e8/0.02 = 1e10; float = 5000*10000/2.0 = 2.5e7
    assert res.loc[idx[1], "circ_mv"] == pytest.approx(1e10)
    assert res.loc[idx[1], "float_share"] == pytest.approx(2.5e7)
    assert list(res.columns) == DAILY_BASIC_COLUMNS


def test_estimate_drops_non_positive_turnover():
    idx = pd.to_datetime(["2017-01-03", "2017-01-04", "2017-01-05"])
    df = pd.DataFrame(
        {
            "volume": [10000.0, 0.0, 10000.0],
            "value": [1e8, 0.0, 1e8],
            "turnOver": [1.0, 0.0, -1.0],
        },
        index=idx,
    )
    res = estimate_from_stock_history(df, ratio=None)
    assert len(res) == 1
    assert np.isnan(res["total_mv"].iloc[0])


def test_estimate_requires_columns():
    df = pd.DataFrame({"volume": [1.0], "value": [1.0]})
    with pytest.raises(ValueError):
        estimate_from_stock_history(df)


# ---------------------------------------------------------------------------
# backfill: 真值 + 估算段拼接
# ---------------------------------------------------------------------------

def _make_truth(dates: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    return pd.DataFrame(
        {"circ_mv": 2e10, "total_mv": 3e10, "float_share": 2e8},
        index=idx,
    )


def test_backfill_merges_estimate_and_truth(tmp_paths, no_delist, monkeypatch):
    _write_stock_csv(
        tmp_paths, "000001",
        [_stock_row("2016-12-30"), _stock_row("2017-01-03"), _stock_row("2018-01-02")],
    )
    truth = _make_truth(["2018-01-02", "2018-01-03"])
    monkeypatch.setattr(dbm, "get_stock_daily_basic_em", lambda symbol, start_date="", end_date="": truth)

    assert backfill_daily_basic_symbol("000001") is True

    df = dbm.load_daily_basic("000001")
    assert len(df) == 4
    # 前两行估算 (2016-2017), 第三行开始真值
    est_rows = df.loc[df.index < pd.Timestamp("2018-01-02")]
    assert len(est_rows) == 2
    # ratio = 3e10/2e10 = 1.5 → total_mv = circ * 1.5
    assert est_rows["total_mv"].iloc[0] == pytest.approx(est_rows["circ_mv"].iloc[0] * 1.5)
    # 真值原样保留
    assert df.loc[pd.Timestamp("2018-01-02"), "total_mv"] == pytest.approx(3e10)


def test_backfill_truth_only_for_new_stock(tmp_paths, no_delist, monkeypatch):
    # 2018 年后上市的股票: 无估算段
    _write_stock_csv(tmp_paths, "000002", [_stock_row("2020-01-02")])
    truth = _make_truth(["2020-01-02"])
    monkeypatch.setattr(dbm, "get_stock_daily_basic_em", lambda symbol, start_date="", end_date="": truth)

    assert backfill_daily_basic_symbol("000002") is True
    df = dbm.load_daily_basic("000002")
    assert len(df) == 1
    assert list(df.columns) == DAILY_BASIC_COLUMNS


def test_backfill_delisted_estimate_only(tmp_paths, no_delist, monkeypatch):
    _write_stock_csv(
        tmp_paths, "601558",
        [_stock_row("2016-05-01"), _stock_row("2017-01-01")],
    )
    empty = pd.DataFrame(
        columns=DAILY_BASIC_COLUMNS, index=pd.DatetimeIndex([], name="date")
    )
    monkeypatch.setattr(dbm, "get_stock_daily_basic_em", lambda symbol, start_date="", end_date="": empty)

    assert backfill_daily_basic_symbol("601558") is True
    df = dbm.load_daily_basic("601558")
    assert len(df) == 2
    assert df["total_mv"].isna().all()  # 无真值 ratio → NaN
    assert df["circ_mv"].notna().all()


def test_backfill_skips_when_no_source(tmp_paths, no_delist, monkeypatch):
    empty = pd.DataFrame(
        columns=DAILY_BASIC_COLUMNS, index=pd.DatetimeIndex([], name="date")
    )
    monkeypatch.setattr(dbm, "get_stock_daily_basic_em", lambda symbol, start_date="", end_date="": empty)
    assert backfill_daily_basic_symbol("999999") is False


# ---------------------------------------------------------------------------
# 增量更新
# ---------------------------------------------------------------------------

def test_update_incremental_idempotent(tmp_paths, no_delist, monkeypatch):
    # 本地止于 2026-09-01, 接口返回 2026-08-31..2026-09-08
    truth = _make_truth(["2026-08-31", "2026-09-01", "2026-09-08"])
    monkeypatch.setattr(dbm, "get_stock_daily_basic_em", lambda symbol, start_date="", end_date="": truth)

    # 先回填到 2026-09-01
    _write_stock_csv(tmp_paths, "000001", [_stock_row("2026-09-01")])
    truth_seed = _make_truth(["2026-09-01"])
    monkeypatch.setattr(dbm, "get_stock_daily_basic_em", lambda symbol, start_date="", end_date="": truth_seed)
    assert backfill_daily_basic_symbol("000001") is True
    assert len(dbm.load_daily_basic("000001")) == 1

    # 切换接口内容
    monkeypatch.setattr(dbm, "get_stock_daily_basic_em", lambda symbol, start_date="", end_date="": truth)
    code, ok = update_daily_basic_symbol("000001")
    assert ok and code == "000001"
    df = dbm.load_daily_basic("000001")
    assert df.index.max() == pd.Timestamp("2026-09-08")
    n_rows = len(df)

    # 再跑一次: 幂等 (本地已达最新 → 不重复拉取/行数不变)
    code, ok = update_daily_basic_symbol("000001")
    assert ok
    assert len(dbm.load_daily_basic("000001")) == n_rows


def test_update_skips_delisted(tmp_paths, monkeypatch):
    monkeypatch.setattr(dbm.STOCK_LIST, "get_delist_date", lambda symbol: "2020-06-23")
    # 无文件 → backfill; 有文件 → 直接跳过
    truth = _make_truth(["2026-01-01"])
    monkeypatch.setattr(dbm, "get_stock_daily_basic_em", lambda symbol, start_date="", end_date="": truth)
    _write_stock_csv(tmp_paths, "601558", [_stock_row("2026-01-01")])
    backfill_daily_basic_symbol("601558")

    code, ok = update_daily_basic_symbol("601558")
    assert ok
    # 不应新增行
    assert len(dbm.load_daily_basic("601558")) == 1


# ---------------------------------------------------------------------------
# batch_check
# ---------------------------------------------------------------------------

def test_batch_check_daily_basic_updated(tmp_paths, monkeypatch):
    monkeypatch.setattr(dbm.STOCK_LIST, "get_delist_date", lambda s: ("2020-06-23" if s == "601558" else ""))
    target = "2026-09-07"
    # 000001: 无文件 → 未更新
    df = batch_check_daily_basic_updated(["000001", "601558"], target_date=target)
    row_map = df.set_index("symbol")
    assert not bool(row_map.loc["000001", "exists"])
    assert not bool(row_map.loc["000001", "is_updated"])
    # 退市股: 无文件也恒为已更新
    assert bool(row_map.loc["601558", "is_updated"])

    # 000001: 有文件且到 target → 已更新
    truth = _make_truth(["2026-09-08"])
    monkeypatch.setattr(dbm, "get_stock_daily_basic_em", lambda symbol, start_date="", end_date="": truth)
    _write_stock_csv(tmp_paths, "000001", [_stock_row("2026-09-08")])
    assert backfill_daily_basic_symbol("000001") is True
    df = batch_check_daily_basic_updated(["000001"], target_date=target)
    assert bool(df.iloc[0]["is_updated"])
