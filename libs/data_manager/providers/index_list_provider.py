from __future__ import annotations

from pathlib import Path
from typing import override

import pandas as pd
from typing_extensions import Self

from config import DataPath
from data_manager.providers.base_provider import BaseProvider


TRUTHY_VALUES = {"1", "true", "yes", "y"}


def _read_index_manifest(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    raise ValueError(f"Unsupported index manifest format: {path}")


def _parse_enabled(value: object) -> bool:
    return str(value).strip().lower() in TRUTHY_VALUES


class _IndexListProvider(BaseProvider):

    @override
    def init(self) -> None:
        self._name: dict[str, str] = {}
        self._source: dict[str, str] = {}
        self._category: dict[str, str] = {}
        self._enabled_symbols: list[str] = []
        self._initialize_index_list()

    @override
    @classmethod
    def get_instance(cls) -> Self:
        return cls()

    def _initialize_index_list(self) -> None:
        df = _read_index_manifest(DataPath.INDEX_NAME_LIST_DF)
        if df.empty:
            return

        required_columns = {"symbol", "name", "source", "enabled"}
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(
                f"Index manifest missing required columns: {missing_columns}"
            )

        frame = df.copy()
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        frame["enabled"] = frame["enabled"].apply(_parse_enabled)
        if "category" not in frame.columns:
            frame["category"] = ""

        self._name.update(frame.set_index("symbol")["name"].astype(str).to_dict())
        self._source.update(frame.set_index("symbol")["source"].astype(str).to_dict())
        self._category.update(frame.set_index("symbol")["category"].astype(str).to_dict())
        self._enabled_symbols = frame.loc[frame["enabled"], "symbol"].astype(str).tolist()

    def get_name(self, symbol: str) -> str:
        return self._name.get(str(symbol).zfill(6), "")

    def get_source(self, symbol: str) -> str:
        return self._source.get(str(symbol).zfill(6), "")

    def get_category(self, symbol: str) -> str:
        return self._category.get(str(symbol).zfill(6), "")

    def name_dict(self) -> dict[str, str]:
        return self._name.copy()

    def get_all_symbol(self, enabled_only: bool = True) -> list[str]:
        if enabled_only:
            return list(self._enabled_symbols)
        return list(self._name.keys())


INDEX_LIST = _IndexListProvider.get_instance()
