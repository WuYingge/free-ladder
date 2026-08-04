"""dedupe_etf_name_list 单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_etf_name_dedupe.py -v
"""

from __future__ import annotations

import pandas as pd

from data_manager.utils import dedupe_etf_name_list


def test_dedupe_prefers_name_with_etf():
    """全称（含 ETF）与简称并存时，保留全称行。"""
    df = pd.DataFrame(
        {
            "symbol": ["513170", "513170", "588690", "588690", "000001"],
            "name": [
                "恒生央企ETF鹏华",
                "鹏华恒生中国央企(QDII)",
                "科创增强ETF银华",
                "科综指增",
                "沪深300ETF易方达",
            ],
            "type": ["指数型-海外股票"] * 5,
        }
    )
    out = dedupe_etf_name_list(df)
    assert len(out) == 3
    assert out.loc[out["symbol"] == "513170", "name"].iloc[0] == "恒生央企ETF鹏华"
    assert out.loc[out["symbol"] == "588690", "name"].iloc[0] == "科创增强ETF银华"


def test_dedupe_keeps_longer_when_no_etf():
    """都不含 ETF 时保留名称更长的一行。"""
    df = pd.DataFrame(
        {
            "symbol": ["111111", "111111"],
            "name": ["短名", "较长名称变体"],
        }
    )
    out = dedupe_etf_name_list(df)
    assert len(out) == 1
    assert out.iloc[0]["name"] == "较长名称变体"


def test_dedupe_noop_when_symbol_unique():
    """symbol 无重复时原样返回（行数不变）。"""
    df = pd.DataFrame({"symbol": ["000001", "000002"], "name": ["A", "B"]})
    out = dedupe_etf_name_list(df)
    assert len(out) == 2


def test_dedupe_missing_columns_returns_unchanged():
    """缺少 name 列时不改动输入。"""
    df = pd.DataFrame({"symbol": ["000001"]})
    out = dedupe_etf_name_list(df)
    assert len(out) == 1
