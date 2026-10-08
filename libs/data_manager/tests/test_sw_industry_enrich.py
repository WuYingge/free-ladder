"""sw_industry_manager 时代化附名 (enrich_industry_names) 单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_sw_industry_enrich.py -v
"""

from __future__ import annotations

import pandas as pd

from data_manager.sw_industry_manager import enrich_industry_names


def _std14() -> pd.DataFrame:
    """2014 版小表: 480101 为 2014 独有链; 480301 与 2021 同名不同义。"""
    return pd.DataFrame(
        {
            "industry_code": ["480101", "480301", "440101"],
            "level1_name": ["银行", "银行", "金融保险"],
            "level2_name": ["银行", "银行", "银行"],
            "level3_name": ["银行", "2014细分", "旧银行"],
        }
    )


def _std21() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "industry_code": ["480301", "999999"],
            "level1_name": ["银行", "测试业"],
            "level2_name": ["股份制银行Ⅱ", "测试二级"],
            "level3_name": ["股份制银行Ⅲ", "测试三级"],
        }
    )


def _clf_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E", "F"],
            # 2021 时代 / 2014 时代 / 2014时代且2021表也收录 / 更早时代 / 未知代码 / 日期缺失
            "start_date": [
                "2022-01-01",
                "2015-01-01",
                "2016-01-01",
                "2010-01-01",
                "2015-01-01",
                "invalid",
            ],
            "industry_code": [
                "480301",  # 应取 2021 名 (股份制银行Ⅲ)
                "480101",  # 应取 2014 名 (银行/银行/银行)
                "480301",  # 2014 时代 → 优先 2014 名 ("2014细分")
                "480301",  # 更早时代 → 先 2014 后 2021 → "2014细分"
                "123456",  # 两表皆无 → 空串
                "999999",  # 日期缺失 → legacy 链 (2014 无 → 2021 名)
            ],
        }
    )


def test_era_aware_naming() -> None:
    out = enrich_industry_names(_clf_rows(), standard_2021=_std21(), standard_2014=_std14())
    expected = [
        ("银行", "股份制银行Ⅱ", "股份制银行Ⅲ"),   # 2021 时代 → 2021
        ("银行", "银行", "银行"),                  # 2014 时代 2014 独有码 → 2014
        ("银行", "银行", "2014细分"),              # 2014 时代双版本码 → 2014 优先
        ("银行", "银行", "2014细分"),              # 更早时代 → 2014 优先
        ("", "", ""),                              # 未知代码
        ("测试业", "测试二级", "测试三级"),        # 日期缺失 → 2014→2021 回退
    ]
    assert [tuple(out.iloc[i][["level1_name", "level2_name", "level3_name"]]) for i in range(len(out))] == expected


def test_missing_tables_returns_empty_names() -> None:
    out = enrich_industry_names(
        _clf_rows(), standard_2021=pd.DataFrame(), standard_2014=pd.DataFrame()
    )
    assert out["level1_name"].eq("").all()


def test_only_2021_table_available() -> None:
    out = enrich_industry_names(
        _clf_rows(), standard_2021=_std21(), standard_2014=pd.DataFrame()
    )
    # 480301 各行与 999999 行得 2021 名; 其余空
    l3 = dict(zip(out["industry_code"], out["level3_name"]))
    assert l3["480301"] == "股份制银行Ⅲ"
    assert l3["999999"] == "测试三级"
    assert l3["480101"] == ""
