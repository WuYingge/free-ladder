"""with_industry 数据集单测：时点(asof)合并逻辑 + registry 集成。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_sw_industry_merge.py -v
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from config import DataPath
from data_manager.datasets import merge_point_in_time
from data_manager.providers.sw_industry_provider import SW_INDUSTRY
from data_manager.stock_data_manager import get_stock_data_by_symbol, get_stock_data_by_symbols

INDUSTRY_COLS = ["industry_code", "level1_name", "level2_name", "level3_name"]


def _events_df() -> pd.DataFrame:
    """000001 三次分类变动 (2021 行含名称; 更早时代无名称)。"""
    return pd.DataFrame(
        {
            "start_date": ["2021-07-30", "1991-04-03", "2014-02-21"],
            "industry_code": ["480301", "440101", "480101"],
            "level1_name": ["银行", "", ""],
            "level2_name": ["股份制银行Ⅱ", "", ""],
            "level3_name": ["股份制银行Ⅲ", "", ""],
        }
    ).assign(start_date=lambda d: pd.to_datetime(d["start_date"]))


class TestMergePointInTime:
    """纯逻辑: 事件帧按日向后合并。"""

    def _base(self, dates: list[str]) -> pd.DataFrame:
        df = pd.DataFrame({"close": range(len(dates))}, index=pd.to_datetime(dates))
        df.index.name = "date"
        return df

    def test_backward_picks_effective_event(self) -> None:
        base = self._base(
            ["1990-12-01", "1991-04-03", "2014-02-21", "2021-07-30", "2026-01-05"]
        )
        out = merge_point_in_time(base, _events_df(), INDUSTRY_COLS)
        codes = out["industry_code"].tolist()
        # 首事件前无归属 → NaN (fill 由 merge_extra_datasets 按 fill_value 完成)
        assert pd.isna(codes[0])
        assert codes[1:] == ["440101", "480101", "480301", "480301"]
        # 名称: 生效日 >= 2021-07-30 才非空 (2021 版代码), 历史行空串
        assert out.loc[pd.Timestamp("2026-01-05"), "level1_name"] == "银行"
        assert out.loc[pd.Timestamp("2014-02-21"), "level1_name"] == ""

    def test_before_first_event_is_nan(self) -> None:
        base = self._base(["1990-01-01"])
        out = merge_point_in_time(base, _events_df(), INDUSTRY_COLS)
        assert pd.isna(out["industry_code"].iloc[0])

    def test_unsorted_events_and_duplicate_start_kept_last(self) -> None:
        ev = pd.concat(
            [
                _events_df(),
                pd.DataFrame(
                    {
                        "start_date": [pd.Timestamp("2014-02-21")],
                        "industry_code": ["999999"],
                        "level1_name": [""],
                        "level2_name": [""],
                        "level3_name": [""],
                    }
                ),
            ]
        )
        base = self._base(["2014-03-01"])
        out = merge_point_in_time(base, ev, INDUSTRY_COLS)
        assert out["industry_code"].iloc[0] == "999999"

    def test_empty_events_returns_base_unchanged(self) -> None:
        empty = _events_df().iloc[0:0]
        base = self._base(["2020-01-01"])
        out = merge_point_in_time(base, empty, INDUSTRY_COLS)
        assert list(out.columns) == ["close"]


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """隔离目录 + 注入 SW_INDUSTRY 缓存的数据路径; 结束后恢复真实文件缓存。"""
    monkeypatch.setattr(DataPath, "STOCK_PATH", str(tmp_path / "stock_data"))
    monkeypatch.setattr(DataPath, "DAILY_BASIC_PATH", str(tmp_path / "daily_basic"))
    monkeypatch.setattr(
        DataPath, "STOCK_SW_INDUSTRY_CLF_CSV", str(tmp_path / "sw_industry_clf.csv")
    )
    os.makedirs(DataPath.STOCK_PATH, exist_ok=True)
    yield tmp_path
    monkeypatch.undo()
    SW_INDUSTRY.reload()  # 恢复为真实 data/const 文件, 避免污染同会话后续测试


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


def _write_clf_csv(tmp_path, symbol: str = "000001") -> None:
    rows = [
        (symbol, "1991-04-03", "440101", "", "", ""),
        (symbol, "2014-02-21", "480101", "", "", ""),
        (symbol, "2021-07-30", "480301", "银行", "股份制银行Ⅱ", "股份制银行Ⅲ"),
    ]
    df = pd.DataFrame(
        rows, columns=["symbol", "start_date", "industry_code",
                       "level1_name", "level2_name", "level3_name"]
    )
    df.to_csv(
        DataPath.STOCK_SW_INDUSTRY_CLF_CSV, index=False, encoding="utf-8-sig"
    )
    SW_INDUSTRY.reload()


class TestWithIndustryGetter:
    def test_default_call_has_no_industry_cols(self, tmp_env) -> None:
        _write_stock_csv(tmp_env, "000001", ["2026-09-01"])
        obj = get_stock_data_by_symbol("000001")
        assert not (set(INDUSTRY_COLS) & set(obj.data.columns))
        assert obj.metadata["datasets"]["with_industry"] is False

    def test_with_industry_point_in_time(self, tmp_env) -> None:
        _write_stock_csv(
            tmp_env, "000001",
            ["2014-03-01", "2021-07-30", "2021-07-29", "2026-09-01"],
        )
        _write_clf_csv(tmp_env)
        obj = get_stock_data_by_symbol("000001", with_industry=True)
        df = obj.data
        assert set(INDUSTRY_COLS) <= set(df.columns)
        # 生效日当天按新分类 (含边界)
        assert df.loc[pd.Timestamp("2021-07-30"), "industry_code"] == "480301"
        # 生效日前一天仍为旧分类 (时点语义)
        assert df.loc[pd.Timestamp("2021-07-29"), "industry_code"] == "480101"
        assert df.loc[pd.Timestamp("2014-03-01"), "industry_code"] == "480101"
        # 2021 版代码 → 中文名; 旧版代码 → 空串
        assert df.loc[pd.Timestamp("2026-09-01"), "level1_name"] == "银行"
        assert df.loc[pd.Timestamp("2021-07-29"), "level1_name"] == ""
        assert obj.metadata["datasets"]["with_industry"] is True

    def test_unknown_symbol_fills_empty(self, tmp_env) -> None:
        _write_stock_csv(tmp_env, "999999", ["2026-09-01"])
        _write_clf_csv(tmp_env)
        obj = get_stock_data_by_symbol("999999", with_industry=True)
        assert obj.data["industry_code"].iloc[0] == ""
        assert obj.data["level1_name"].iloc[0] == ""

    def test_with_ochl_false_keeps_industry(self, tmp_env) -> None:
        _write_stock_csv(tmp_env, "000001", ["2026-09-01"])
        _write_clf_csv(tmp_env)
        obj = get_stock_data_by_symbol(
            "000001", with_ochl=False, with_industry=True
        )
        assert obj.data["open"].isna().all()
        assert obj.data["industry_code"].iloc[0] == "480301"
        assert obj.metadata["datasets"]["ochl"] is False

    def test_slice_applied_after_merge(self, tmp_env) -> None:
        _write_stock_csv(tmp_env, "000001", ["2021-07-29", "2021-07-30"])
        _write_clf_csv(tmp_env)
        obj = get_stock_data_by_symbol(
            "000001", start_date="2021-07-30", with_industry=True
        )
        assert len(obj.data) == 1
        assert obj.data["industry_code"].iloc[0] == "480301"

    def test_batch_getter_forward_params(self, tmp_env) -> None:
        _write_stock_csv(tmp_env, "000001", ["2026-09-01"])
        _write_stock_csv(tmp_env, "000002", ["2026-09-01"])
        _write_clf_csv(tmp_env, symbol="000001")
        objs = get_stock_data_by_symbols(["000001", "000002"], with_industry=True)
        assert "industry_code" in objs[0].data.columns
        assert "industry_code" in objs[1].data.columns  # 无事件 → 空串列
        assert objs[1].data["industry_code"].iloc[0] == ""

    def test_combined_with_basic(self, tmp_env) -> None:
        _write_stock_csv(tmp_env, "000001", ["2026-09-01"])
        _write_clf_csv(tmp_env)
        os.makedirs(DataPath.DAILY_BASIC_PATH, exist_ok=True)
        idx = pd.DatetimeIndex([pd.Timestamp("2026-09-01")], name="date")
        pd.DataFrame(
            {"circ_mv": [1e10], "total_mv": [1.2e10], "float_share": [1e8]},
            index=idx,
        ).to_csv(DataPath.DAILY_BASIC_PATH + "/000001.csv", encoding="utf-8-sig", index=True)
        obj = get_stock_data_by_symbol(
            "000001", with_basic=True, with_industry=True
        )
        assert obj.data.loc[pd.Timestamp("2026-09-01"), "circ_mv"] == pytest.approx(1e10)
        assert obj.data.loc[pd.Timestamp("2026-09-01"), "industry_code"] == "480301"
