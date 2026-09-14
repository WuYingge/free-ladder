"""复权因子数据层单测：构造 / vwap / 体检 / registry 集成。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_adj_factor.py -v
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from config import DataPath
from data_manager.adj_factor_manager import (
    VOLUME_LOT_SIZE,
    build_adj_factor_frame,
    build_health_report,
    check_vwap_bounds,
    compute_vwap,
    load_adj_factor,
)
from data_manager.providers.adj_factor_provider import ADJ_FACTOR
from data_manager.stock_data_manager import get_stock_data_by_symbol, get_stock_data_by_symbols


def _series(values: list[float], dates: list[str]) -> pd.Series:
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype="float64")


class TestBuildAdjFactorFrame:
    """纯函数：不复权/后复权收盘 → (close_raw, adj_factor)。"""

    def test_factor_is_hfq_over_raw(self) -> None:
        dates = ["2024-01-02", "2024-01-03"]
        frame = build_adj_factor_frame(
            _series([10.0, 10.5], dates), _series([15.0, 15.75], dates)
        )
        assert list(frame.columns) == ["close_raw", "adj_factor"]
        assert frame["adj_factor"].tolist() == pytest.approx([1.5, 1.5])
        assert frame.index.name == "date"

    def test_keeps_only_dates_present_on_both_sides(self) -> None:
        frame = build_adj_factor_frame(
            _series([10.0, 11.0], ["2024-01-02", "2024-01-03"]),
            _series([15.0], ["2024-01-03"]),
        )
        assert frame.index == pd.DatetimeIndex(["2024-01-03"])

    def test_drops_non_positive_and_nan(self) -> None:
        frame = build_adj_factor_frame(
            _series([10.0, 0.0, float("nan")], ["2024-01-02", "2024-01-03", "2024-01-04"]),
            _series([15.0, 16.0, 17.0], ["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        assert frame.index == pd.DatetimeIndex(["2024-01-02"])

    def test_unsorted_and_duplicate_dates(self) -> None:
        frame = build_adj_factor_frame(
            _series([10.0, 11.0, 12.0], ["2024-01-03", "2024-01-02", "2024-01-02"]),
            _series([16.5, 15.0, 15.0], ["2024-01-03", "2024-01-02", "2024-01-02"]),
        )
        assert frame.index.is_monotonic_increasing
        assert frame.loc["2024-01-02", "close_raw"] == pytest.approx(12.0)

    def test_empty_alignment_returns_standard_empty_frame(self) -> None:
        frame = build_adj_factor_frame(
            _series([10.0], ["2024-01-02"]), _series([15.0], ["2024-01-05"])
        )
        assert frame.empty
        assert list(frame.columns) == ["close_raw", "adj_factor"]


class TestComputeVwap:
    """vwap = 成交额 / (成交量[手] x 100) x 复权因子。"""

    def test_lot_size_and_factor_applied(self) -> None:
        dates = ["2024-01-02", "2024-01-03"]
        value = _series([1e8, 2e8], dates)          # 元
        volume = _series([1e4, 2e4], dates)         # 手 (=100 股)
        factor = _series([2.0, 2.0], dates)
        vwap = compute_vwap(value, volume, factor)
        assert vwap.iloc[0] == pytest.approx(1e8 / (1e4 * VOLUME_LOT_SIZE) * 2.0)
        assert vwap.iloc[1] == pytest.approx(2e8 / (2e4 * VOLUME_LOT_SIZE) * 2.0)

    def test_missing_factor_day_is_nan_not_wrong_scale(self) -> None:
        value = _series([1e8, 1e8, 1e8], ["2024-01-02", "2024-01-03", "2024-01-04"])
        volume = _series([1e4] * 3, ["2024-01-02", "2024-01-03", "2024-01-04"])
        factor = _series([2.0, 2.0], ["2024-01-02", "2024-01-03"])
        vwap = compute_vwap(value, volume, factor)
        assert pd.isna(vwap.loc["2024-01-04"])

    def test_dataframe_matrix_elementwise(self) -> None:
        idx = pd.DatetimeIndex(["2024-01-02"], name="date")
        value = pd.DataFrame({"a": [1e8], "b": [2e8]}, index=idx)
        volume = pd.DataFrame({"a": [1e4], "b": [2e4]}, index=idx)
        factor = pd.DataFrame({"a": [1.0], "b": [3.0]}, index=idx)
        vwap = compute_vwap(value, volume, factor)
        assert vwap.loc["2024-01-02", "a"] == pytest.approx(100.0)
        assert vwap.loc["2024-01-02", "b"] == pytest.approx(300.0)


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """隔离 stock_data / adj_factor 目录; 结束后恢复真实 provider 缓存。"""
    monkeypatch.setattr(DataPath, "STOCK_PATH", str(tmp_path / "stock_data"))
    monkeypatch.setattr(DataPath, "ADJ_FACTOR_PATH", str(tmp_path / "adj_factor"))
    os.makedirs(DataPath.STOCK_PATH, exist_ok=True)
    os.makedirs(DataPath.ADJ_FACTOR_PATH, exist_ok=True)
    yield tmp_path
    monkeypatch.undo()
    ADJ_FACTOR.reload()


def _write_stock_csv(code: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(
        os.path.join(DataPath.STOCK_PATH, f"{code}.csv"), index=False, encoding="utf-8-sig"
    )


def _quote_rows(dates: list[str], close: float = 4.6, factor: float = 1.0) -> list[dict]:
    """后复权行情：low/high 取 close±0.5%, 成交额/成交量 => 均价 == close/factor。"""
    low, high = close * 0.995, close * 1.005
    rows = []
    for date in dates:
        volume = 1e4                       # 手
        # 让不复权均价 = close/factor (即乘因子后正好等于 close)
        value = close / factor * volume * VOLUME_LOT_SIZE
        rows.append(
            {
                "date": date, "open": close, "high": high, "low": low, "close": close,
                "volume": volume, "value": value, "range": 1.0, "gain": 0.0,
                "change": 0.0, "turnOver": 1.0,
            }
        )
    return rows


def _write_adj_factor_csv(code: str, dates: list[str], factor: float) -> None:
    idx = pd.DatetimeIndex(dates, name="date")
    pd.DataFrame({"close_raw": 4.6 / factor, "adj_factor": factor}, index=idx).to_csv(
        os.path.join(DataPath.ADJ_FACTOR_PATH, f"{code}.csv"), encoding="utf-8-sig", index=True
    )


class TestLoadAdjFactor:
    def test_missing_file_returns_empty_standard_frame(self, tmp_env) -> None:
        frame = load_adj_factor("000001")
        assert frame.empty
        assert list(frame.columns) == ["close_raw", "adj_factor"]

    def test_roundtrip(self, tmp_env) -> None:
        _write_adj_factor_csv("000001", ["2024-01-02", "2024-01-03"], factor=1.2)
        frame = load_adj_factor("000001")
        assert frame["adj_factor"].tolist() == pytest.approx([1.2, 1.2])
        assert frame.index.name == "date"

    def test_missing_column_raises(self, tmp_env) -> None:
        fp = os.path.join(DataPath.ADJ_FACTOR_PATH, "000001.csv")
        pd.DataFrame({"date": ["2024-01-02"], "close_raw": [4.0]}).to_csv(
            fp, index=False, encoding="utf-8-sig"
        )
        with pytest.raises(ValueError, match="缺少列"):
            load_adj_factor("000001")


class TestVwapBoundsCheck:
    """硬不变量: low <= vwap <= high。"""

    def test_ok_when_factor_consistent(self, tmp_env) -> None:
        dates = ["2024-01-02", "2024-01-03"]
        _write_stock_csv("000001", _quote_rows(dates, close=4.6, factor=1.2))
        _write_adj_factor_csv("000001", dates, factor=1.2)
        ADJ_FACTOR.reload("000001")
        row = check_vwap_bounds("000001")
        assert row["status"] == "ok"
        assert row["rows"] == 2 and row["violations"] == 0

    def test_detects_wrong_factor(self, tmp_env) -> None:
        dates = ["2024-01-02", "2024-01-03"]
        _write_stock_csv("000001", _quote_rows(dates, close=4.6, factor=1.2))
        # 因子配错 (1.2 vs 1.0) → vwap 明显越界
        _write_adj_factor_csv("000001", dates, factor=1.0)
        ADJ_FACTOR.reload("000001")
        row = check_vwap_bounds("000001")
        assert row["status"] == "violation"
        assert row["violations"] == 2

    def test_detects_missing_lot_conversion(self, tmp_env) -> None:
        """忘记 x100 的经典错误：vwap 会被放大 100 倍 → 必然越界。"""
        dates = ["2024-01-02"]
        rows = _quote_rows(dates, close=4.6, factor=1.0)
        rows[0]["value"] = rows[0]["value"] * 100  # 模拟口径写错
        _write_stock_csv("000001", rows)
        _write_adj_factor_csv("000001", dates, factor=1.0)
        ADJ_FACTOR.reload("000001")
        assert check_vwap_bounds("000001")["status"] == "violation"

    def test_missing_factor_reported_not_counted(self, tmp_env) -> None:
        _write_stock_csv("000001", _quote_rows(["2024-01-02"], close=4.6, factor=1.0))
        ADJ_FACTOR.reload("000001")
        row = check_vwap_bounds("000001")
        assert row["status"] == "missing_data"
        assert row["rows"] == 0

    def test_health_report_aggregates(self, tmp_env) -> None:
        dates = ["2024-01-02"]
        _write_stock_csv("000001", _quote_rows(dates, close=4.6, factor=1.2))
        _write_adj_factor_csv("000001", dates, factor=1.2)
        _write_stock_csv("000002", _quote_rows(dates, close=10.0, factor=1.0))
        ADJ_FACTOR.reload()
        report = build_health_report(["000001", "000002"], "stock")
        assert report["n_symbols"] == 2
        assert report["n_missing_factor"] == 1
        assert report["n_violation_rows"] == 0
        assert report["n_checked_rows"] == 1


class TestWithAdjFactorGetter:
    def test_default_call_has_no_adj_factor_cols(self, tmp_env) -> None:
        _write_stock_csv("000001", _quote_rows(["2024-01-02"]))
        obj = get_stock_data_by_symbol("000001")
        assert "adj_factor" not in obj.data.columns
        assert obj.metadata["datasets"]["with_adj_factor"] is False

    def test_with_adj_factor_merges_daily(self, tmp_env) -> None:
        dates = ["2024-01-02", "2024-01-03"]
        _write_stock_csv("000001", _quote_rows(dates))
        _write_adj_factor_csv("000001", dates, factor=1.5)
        ADJ_FACTOR.reload("000001")
        obj = get_stock_data_by_symbol("000001", with_adj_factor=True)
        df = obj.data
        assert df.loc[pd.Timestamp("2024-01-02"), "adj_factor"] == pytest.approx(1.5)
        assert df.loc[pd.Timestamp("2024-01-02"), "close_raw"] == pytest.approx(4.6 / 1.5)
        assert obj.metadata["datasets"]["with_adj_factor"] is True

    def test_symbol_without_factor_gets_nan_columns(self, tmp_env) -> None:
        _write_stock_csv("999999", _quote_rows(["2024-01-02"]))
        ADJ_FACTOR.reload("999999")
        obj = get_stock_data_by_symbol("999999", with_adj_factor=True)
        assert pd.isna(obj.data["adj_factor"].iloc[0])

    def test_batch_getter_forwards_flag(self, tmp_env) -> None:
        dates = ["2024-01-02"]
        _write_stock_csv("000001", _quote_rows(dates))
        _write_stock_csv("000002", _quote_rows(dates))
        _write_adj_factor_csv("000001", dates, factor=1.5)
        ADJ_FACTOR.reload()
        objs = get_stock_data_by_symbols(["000001", "000002"], with_adj_factor=True)
        assert objs[0].data["adj_factor"].iloc[0] == pytest.approx(1.5)
        assert pd.isna(objs[1].data["adj_factor"].iloc[0])
