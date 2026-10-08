from __future__ import annotations
from typing import override
from typing_extensions import Self
import pandas as pd
from config import DataPath
from data_manager.providers.base_provider import BaseProvider


class _ETFPoolSemanticProvider(BaseProvider):
    """语义精选 ETF 池 Provider。

    从 data/const/etf_pool_semantic_*.csv 加载（106 个语义精选 ETF，含 group 分组），
    与 etf_index_map 对齐。提供：
    - get_symbol(tracked_index): 获取精选池中该跟踪指数的 ETF symbol
    - get_name(tracked_index): 获取该跟踪指数的 ETF 名称
    - get_group(tracked_index): 获取该跟踪指数的语义分组（如 宽基-沪深300）
    - get_candidates(tracked_index): 获取该跟踪指数的所有候选 symbol 列表
    - get_all_symbols(): 获取精选池所有 symbol 列表
    - get_all_tracked_indices(): 获取精选池所有跟踪指数列表
    - get_symbols_for_group(group): 获取指定分组的所有 symbol 列表
    - get_groups(): 获取全部 group -> [symbols] 映射
    - get_symbols_for_tracked_indices(tracked_indices): 批量获取 symbol
    - mapping(): 返回 {tracked_index: selected_symbol} 完整映射
    """

    @override
    def init(self) -> None:
        self._symbol_map: dict[str, str] = {}
        self._name_map: dict[str, str] = {}
        self._candidates_map: dict[str, list[str]] = {}
        self._group_map: dict[str, str] = {}  # tracked_index -> group
        self._group_symbols: dict[str, list[str]] = {}  # group -> [symbols]
        self._tracked_indices: list[str] = []
        self._initialize()

    @override
    @classmethod
    def get_instance(cls) -> Self:
        return cls()

    def _initialize(self) -> None:
        path = DataPath.ETF_POOL_SEMANTIC_CSV
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

            if "group" in df.columns:
                group = str(row["group"]).strip()
                if group:
                    self._group_map[ti] = group
                    self._group_symbols.setdefault(group, []).append(symbol)

            if "candidates" in df.columns:
                raw = str(row["candidates"])
                self._candidates_map[ti] = [s.strip() for s in raw.split(",") if s.strip()]

        self._tracked_indices = sorted(self._symbol_map.keys())

    def get_symbol(self, tracked_index: str) -> str:
        """获取跟踪指数对应的精选 ETF symbol。"""
        return self._symbol_map.get(tracked_index, "")

    def get_name(self, tracked_index: str) -> str:
        """获取跟踪指数对应的 ETF 名称。"""
        return self._name_map.get(tracked_index, "")

    def get_group(self, tracked_index: str) -> str:
        """获取跟踪指数对应的语义分组。"""
        return self._group_map.get(tracked_index, "")

    def get_candidates(self, tracked_index: str) -> list[str]:
        """获取跟踪指数对应的所有候选 symbol 列表。"""
        return self._candidates_map.get(tracked_index, [])

    def get_all_symbols(self) -> list[str]:
        """获取精选池所有 symbol 列表（按跟踪指数字母序）。"""
        return [self._symbol_map[ti] for ti in self._tracked_indices]

    def get_all_tracked_indices(self) -> list[str]:
        """获取精选池所有跟踪指数列表（按字母序）。"""
        return list(self._tracked_indices)

    def get_symbols_for_group(self, group: str) -> list[str]:
        """获取指定语义分组的全部 symbol 列表（按跟踪指数字母序）。"""
        return sorted(self._group_symbols.get(group, []))

    def get_groups(self) -> dict[str, list[str]]:
        """返回 {group: [symbols]} 完整分组映射。"""
        return {g: sorted(ss) for g, ss in self._group_symbols.items()}

    def get_all_groups(self) -> list[str]:
        """获取全部语义分组名称（按字母序）。"""
        return sorted(self._group_symbols.keys())

    def get_symbols_for_tracked_indices(self, tracked_indices: list[str]) -> list[str]:
        """批量获取多个跟踪指数对应的 symbol 列表，保持输入顺序。"""
        return [self._symbol_map.get(ti, "") for ti in tracked_indices]

    def mapping(self) -> dict[str, str]:
        """返回 {tracked_index: selected_symbol} 完整映射。"""
        return self._symbol_map.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """返回完整映射表 DataFrame。"""
        path = DataPath.ETF_POOL_SEMANTIC_CSV
        try:
            return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()


ETF_POOL_SEMANTIC = _ETFPoolSemanticProvider.get_instance()
