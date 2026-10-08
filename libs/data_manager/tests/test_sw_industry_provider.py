"""sw_industry_provider 单测：时点归属解析、档位前缀、真实 CSV 集成不变量。"""

from __future__ import annotations

import pandas as pd
import pytest

from data_manager.providers.sw_industry_provider import (
    SW_INDUSTRY,
    effective_industry_mapping,
)


def _hist_df() -> pd.DataFrame:
    """构造含多次分类变动的样本历史（故意乱序，验证排序鲁棒性）。"""
    return pd.DataFrame(
        {
            "symbol": ["000002", "000001", "000001", "000001", "000003", "000002"],
            "start_date": [
                "2014-02-21",
                "2014-02-21",
                "1991-04-03",
                "2021-07-30",
                "1991-04-14",
                "1991-01-29",
            ],
            "industry_code": ["430101", "480101", "440101", "480301", "510101", "429999"],
        }
    )


class TestEffectiveIndustryMapping:
    def test_latest_without_asof(self) -> None:
        out = effective_industry_mapping(_hist_df())
        by_symbol = dict(zip(out["symbol"], out["industry_code"]))
        assert by_symbol == {"000001": "480301", "000002": "430101", "000003": "510101"}

    def test_asof_picks_effective_row(self) -> None:
        out = effective_industry_mapping(_hist_df(), asof="2020-01-01")
        by_symbol = dict(zip(out["symbol"], out["industry_code"]))
        # 000001 生效行为 2014-02-21 行（2021 行未到生效日）
        assert by_symbol == {"000001": "480101", "000002": "430101", "000003": "510101"}

    def test_asof_boundary_inclusive(self) -> None:
        out = effective_industry_mapping(_hist_df(), asof="2021-07-30")
        assert dict(zip(out["symbol"], out["industry_code"]))["000001"] == "480301"

    def test_asof_before_first_row_excludes_symbol(self) -> None:
        out = effective_industry_mapping(_hist_df(), asof="1990-12-31")
        assert out.empty

    def test_asof_mid_history_keeps_structure(self) -> None:
        out = effective_industry_mapping(_hist_df(), asof="2010-06-01")
        # 000001 此时生效行为 1991 行；000002 只有 1991 行（2014 行未生效）
        assert dict(zip(out["symbol"], out["industry_code"]))["000001"] == "440101"
        assert set(out["symbol"]) == {"000001", "000002", "000003"}

    def test_rows_sorted_by_symbol(self) -> None:
        out = effective_industry_mapping(_hist_df())
        assert out["symbol"].tolist() == sorted(out["symbol"].tolist())


class TestProviderWithRealFile:
    """依赖 data/const/stock_sw_industry_clf.csv 的集成校验（文件缺失时跳过）。"""

    @pytest.fixture(scope="class")
    def mapping(self) -> pd.DataFrame:
        m = SW_INDUSTRY.get_mapping()
        if m.empty:
            pytest.skip("stock_sw_industry_clf.csv 尚未生成")
        return m

    def test_latest_covers_whole_market(self, mapping: pd.DataFrame) -> None:
        assert len(mapping) > 5000
        assert mapping["symbol"].is_unique

    def test_industry_code_always_6_digits(self, mapping: pd.DataFrame) -> None:
        lengths = mapping["industry_code"].str.len().unique()
        assert lengths.tolist() == [6]

    def test_known_symbols_resolved(self, mapping: pd.DataFrame) -> None:
        by_symbol = dict(zip(mapping["symbol"], mapping["industry_code"]))
        assert by_symbol.get("000001", "").startswith("48")  # 平安银行 → 银行(48)
        assert "600519" in by_symbol
        assert "000003" in by_symbol  # 含退市股（PT 金田）

    def test_point_in_time_differs_across_reclass(self) -> None:
        """000001 至少存在一次分类变动：不同 asof 的归属应不同。"""
        hist = SW_INDUSTRY.to_dataframe()
        row = hist[hist["symbol"] == "000001"]
        if row["start_date"].nunique() < 2:
            pytest.skip("样本行无分类变动")
        change_day = row["start_date"].sort_values().iloc[-2]
        before = SW_INDUSTRY.get_industry("000001", asof=change_day - pd.Timedelta(days=1))
        after = SW_INDUSTRY.get_industry("000001", asof=change_day)
        assert before and after and before != after

    def test_group_series_levels(self) -> None:
        s1 = SW_INDUSTRY.get_group_series(asof="2024-12-31", level=1)
        s2 = SW_INDUSTRY.get_group_series(asof="2024-12-31", level=2)
        s3 = SW_INDUSTRY.get_group_series(asof="2024-12-31", level=3)
        assert s1.str.len().eq(2).all()
        assert s2.str.len().eq(4).all()
        assert s3.str.len().eq(6).all()
        assert s1.name == "sw_level1"

    def test_group_series_filters_symbols(self) -> None:
        s = SW_INDUSTRY.get_group_series(symbols=["000001", "600519"], level=1)
        assert sorted(s.index) == ["000001", "600519"]
        assert s.loc["000001"] == "48"

    def test_group_series_rejects_bad_level(self) -> None:
        with pytest.raises(ValueError):
            SW_INDUSTRY.get_group_series(level=4)
