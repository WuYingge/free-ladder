from __future__ import annotations
from typing import override
from typing_extensions import Self
import pandas as pd
from config import DataPath
from data_manager.providers.base_provider import BaseProvider


class _ETFIndexMapProvider(BaseProvider):
    """跟踪指数 → 最优 ETF 映射 Provider。

    从 data/const/etf_index_map.csv 加载，提供：
    - get_symbol(tracked_index): 获取选中的 ETF symbol
    - get_name(tracked_index): 获取选中的 ETF 名称
    - get_candidates(tracked_index): 获取该跟踪指数的所有候选 symbol 列表
    - get_all_symbols(): 获取所有选中 ETF symbol 列表
    - get_all_tracked_indices(): 获取所有跟踪指数列表
    - get_symbols_for_tracked_indices(tracked_indices): 批量获取 symbol
    - mapping(): 返回 {tracked_index: selected_symbol} 完整映射
    """

    @override
    def init(self) -> None:
        self._symbol_map: dict[str, str] = {}
        self._name_map: dict[str, str] = {}
        self._candidates_map: dict[str, list[str]] = {}
        self._tracked_indices: list[str] = []
        self._initialize()

    @override
    @classmethod
    def get_instance(cls) -> Self:
        return cls()

    def _initialize(self) -> None:
        path = DataPath.ETF_INDEX_MAP_CSV
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except FileNotFoundError:
            # 文件尚未生成时返回空映射
            return
        except Exception:
            return

        if df.empty:
            return

        # 标准化列名
        df.columns = df.columns.str.strip().str.lstrip("\ufeff")
        required_cols = {"tracked_index", "selected_symbol"}
        if not required_cols.issubset(set(df.columns)):
            return

        for _, row in df.iterrows():
            ti = str(row["tracked_index"]).strip()
            if not ti:
                continue
            symbol = str(row["selected_symbol"]).strip()
            self._symbol_map[ti] = symbol

            if "selected_name" in df.columns:
                self._name_map[ti] = str(row["selected_name"])

            if "candidates" in df.columns:
                raw = str(row["candidates"])
                self._candidates_map[ti] = [s.strip() for s in raw.split(",") if s.strip()]

        self._tracked_indices = sorted(self._symbol_map.keys())

    def get_symbol(self, tracked_index: str) -> str:
        """获取跟踪指数对应的最优 ETF symbol。"""
        return self._symbol_map.get(tracked_index, "")

    def get_name(self, tracked_index: str) -> str:
        """获取跟踪指数对应的最优 ETF 名称。"""
        return self._name_map.get(tracked_index, "")

    def get_candidates(self, tracked_index: str) -> list[str]:
        """获取跟踪指数对应的所有候选 ETF symbol 列表。"""
        return self._candidates_map.get(tracked_index, [])

    def get_all_symbols(self) -> list[str]:
        """获取所有选中 ETF symbol 列表（去重，按跟踪指数字母序）。"""
        return [self._symbol_map[ti] for ti in self._tracked_indices]

    def get_all_tracked_indices(self) -> list[str]:
        """获取所有跟踪指数列表（按字母序）。"""
        return list(self._tracked_indices)

    def get_symbols_for_tracked_indices(self, tracked_indices: list[str]) -> list[str]:
        """批量获取多个跟踪指数对应的 symbol 列表，保持输入顺序。"""
        return [self._symbol_map.get(ti, "") for ti in tracked_indices]

    def mapping(self) -> dict[str, str]:
        """返回 {tracked_index: selected_symbol} 完整映射。"""
        return self._symbol_map.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """返回完整映射表 DataFrame。"""
        path = DataPath.ETF_INDEX_MAP_CSV
        try:
            return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()


ETF_INDEX_MAP = _ETFIndexMapProvider.get_instance()
