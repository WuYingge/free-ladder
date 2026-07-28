from __future__ import annotations

from pathlib import Path
from typing import override

import pandas as pd
from typing_extensions import Self

from config import DataPath
from data_manager.providers.base_provider import BaseProvider


class _StockListProvider(BaseProvider):

    @override
    def init(self) -> None:
        self._name: dict[str, str] = {}
        self._market: dict[str, str] = {}
        self._list_date: dict[str, str] = {}
        self._delist_date: dict[str, str] = {}
        self._all_symbols: list[str] = []
        self._initialize_stock_list()

    @override
    @classmethod
    def get_instance(cls) -> Self:
        return cls()

    def _initialize_stock_list(self) -> None:
        csv_path = str(DataPath.STOCK_NAME_LIST_DF)
        if not Path(csv_path).exists():
            return

        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
        if df.empty:
            return

        df.columns = df.columns.str.strip().str.lstrip("\ufeff")
        if "symbol" not in df.columns:
            return

        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        df = df.set_index("symbol")

        self._name.update(df["name"].dropna().astype(str).to_dict())
        if "market" in df.columns:
            self._market.update(df["market"].dropna().astype(str).to_dict())
        if "list_date" in df.columns:
            self._list_date.update(df["list_date"].dropna().astype(str).to_dict())
        if "delist_date" in df.columns:
            self._delist_date.update(df["delist_date"].dropna().astype(str).to_dict())

        self._all_symbols = list(self._name.keys())

    # ---- query methods ----

    def get_name(self, symbol: str) -> str:
        return self._name.get(str(symbol).zfill(6), "")

    def get_market(self, symbol: str) -> str:
        return self._market.get(str(symbol).zfill(6), "")

    def get_list_date(self, symbol: str) -> str:
        return self._list_date.get(str(symbol).zfill(6), "")

    def get_delist_date(self, symbol: str) -> str:
        return self._delist_date.get(str(symbol).zfill(6), "")

    def is_active(self, symbol: str, date: str | None = None) -> bool:
        """Check if the symbol was active on *date* (or is active now)."""
        delist = self.get_delist_date(symbol)
        if not delist:
            return True  # no delist date → still active
        if date is None:
            return False  # has delist date → not active
        return date < delist

    def get_all_symbol(self) -> list[str]:
        return list(self._all_symbols)

    def name_dict(self) -> dict[str, str]:
        return self._name.copy()


STOCK_LIST = _StockListProvider.get_instance()
