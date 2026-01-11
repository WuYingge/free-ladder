from __future__ import annotations
from typing import override
from typing_extensions import Self
import pandas as pd
from config import DataPath
from data_manager.providers.base_provider import BaseProvider

class _ETFListProvider(BaseProvider):
    
    @override
    def init(self) -> None:
        self._type: dict[str, str] = {}
        self._name: dict[str, str] = {}
        self._initialize_etf_list()
    
    @override
    @classmethod
    def get_instance(cls) -> Self:
        return cls()
        
    def _initialize_etf_list(self) -> None:
        df = pd.read_excel(DataPath.ETF_NAME_LIST_DF, dtype=str)
        df.index = df['symbol'].astype(str)
        self._type.update(df['type'].to_dict())
        self._name.update(df['name'].to_dict())
        
    def get_type(self, symbol: str) -> str:
        return self._type.get(symbol, "")
    
    def get_name(self, symbol: str) -> str:
        return self._name.get(symbol, "")
    
    def name_dict(self) -> dict[str, str]:
        return self._name.copy()
    
    def get_all_symbol(self) -> list[str]:
        return list(self._name.keys())
    
ETF_LIST = _ETFListProvider.get_instance()
