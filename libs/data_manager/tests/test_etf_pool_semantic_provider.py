"""etf_pool_semantic_provider 单元测试。

Run from project root:
    PYTHONPATH=libs pytest libs/data_manager/tests/test_etf_pool_semantic_provider.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import DataPath
from data_manager.providers.etf_pool_semantic_provider import ETF_POOL_SEMANTIC


@pytest.fixture
def pool_df() -> pd.DataFrame:
    """读取当前精选池 CSV。"""
    return pd.read_csv(DataPath.ETF_POOL_SEMANTIC_CSV, dtype=str, encoding="utf-8-sig")


def test_provider_loaded_106_symbols():
    """精选池应有 106 个精选 ETF。"""
    symbols = ETF_POOL_SEMANTIC.get_all_symbols()
    assert len(symbols) == 106
    assert len(set(symbols)) == 106  # 无重复 symbol


def test_all_symbols_in_index_map():
    """与 etf_index_map 对齐：每个精选 symbol 都应在 etf_index_map 中。"""
    from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP

    map_symbols = set(ETF_INDEX_MAP.get_all_symbols())
    pool_symbols = set(ETF_POOL_SEMANTIC.get_all_symbols())
    assert pool_symbols <= map_symbols
    assert len(pool_symbols) == 106


def test_all_tracked_indices_in_index_map():
    """与 etf_index_map 对齐：每个精选 tracked_index 都应在 etf_index_map 中。"""
    from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP

    map_tis = set(ETF_INDEX_MAP.get_all_tracked_indices())
    pool_tis = set(ETF_POOL_SEMANTIC.get_all_tracked_indices())
    assert pool_tis <= map_tis
    assert len(pool_tis) == 106


def test_group_columns_present(pool_df: pd.DataFrame):
    """CSV 应包含 group 列，且无缺失值。"""
    assert "group" in pool_df.columns
    assert pool_df["group"].notna().all()
    assert (pool_df["group"].str.strip() != "").all()


def test_symbol_and_name_consistent(pool_df: pd.DataFrame):
    """provider 的 symbol/name 与 CSV 一致。"""
    for _, row in pool_df.iterrows():
        ti = str(row["tracked_index"]).strip()
        assert ETF_POOL_SEMANTIC.get_symbol(ti) == str(row["selected_symbol"]).strip()
        assert ETF_POOL_SEMANTIC.get_name(ti) == str(row["selected_name"]).strip()


def test_group_mapping(pool_df: pd.DataFrame):
    """get_group / get_symbols_for_group / get_groups 与 CSV 一致。"""
    for _, row in pool_df.iterrows():
        ti = str(row["tracked_index"]).strip()
        group = str(row["group"]).strip()
        assert ETF_POOL_SEMANTIC.get_group(ti) == group

    groups = ETF_POOL_SEMANTIC.get_groups()
    assert len(groups) == pool_df["group"].nunique()
    for group, symbols in groups.items():
        # 每个分组内 symbol 可反向查到该组
        for sym in symbols:
            tis = [ti for ti in ETF_POOL_SEMANTIC.get_all_tracked_indices()
                   if ETF_POOL_SEMANTIC.get_symbol(ti) == sym]
            assert tis, f"group {group} 的 symbol {sym} 反向查不到"
            assert all(ETF_POOL_SEMANTIC.get_group(ti) == group for ti in tis)


def test_candidates_recovered(pool_df: pd.DataFrame):
    """candidates 应与 etf_index_map 完全一致（Excel 千分位已修复）。"""
    from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP

    for _, row in pool_df.iterrows():
        ti = str(row["tracked_index"]).strip()
        assert ETF_POOL_SEMANTIC.get_candidates(ti) == ETF_INDEX_MAP.get_candidates(ti)


def test_symbols_for_group():
    """get_symbols_for_group 应返回排序后的列表。"""
    groups = ETF_POOL_SEMANTIC.get_all_groups()
    assert len(groups) >= 1
    g = groups[0]
    syms = ETF_POOL_SEMANTIC.get_symbols_for_group(g)
    assert syms == sorted(syms)
    assert syms == ETF_POOL_SEMANTIC.get_groups()[g]
