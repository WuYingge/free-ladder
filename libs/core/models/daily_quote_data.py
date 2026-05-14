from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

import pandas as pd
from typing_extensions import Self

from core.models.data_base import FinancialData
from data_manager.utils import get_symbol_name_from_fp


class DailyQuoteData(FinancialData):
    """Shared daily OHLCV-style data model for asset classes stored as local CSV."""

    REQUIRED_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "turnOver",
        "gain",
        "change",
    ]

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        name: str = "",
    ):
        super().__init__(data, metadata)
        self.symbol = symbol
        self.name = name

        if not self.validate_data():
            warnings.warn(
                "Data validation failed. Some operations may not work correctly."
            )

    def validate_data(self) -> bool:
        return all(col in self._data.columns for col in self.REQUIRED_COLUMNS)

    @classmethod
    def from_csv(
        cls,
        fp: str,
        *,
        symbol: Optional[str] = None,
        name: str = "",
    ) -> Self:
        df = pd.read_csv(fp)
        resolved_symbol = symbol or get_symbol_name_from_fp(fp)
        return cls(df, symbol=resolved_symbol, name=name)

    def output_with_factors_to(self, path: str) -> None:
        df = self.output_with_factors()
        symbol = self.symbol or get_symbol_name_from_fp(path)
        df.to_csv(
            os.path.join(path, f"{symbol}.csv"),
            index=False,
            header=True,
            encoding="utf-8-sig",
        )

    def _date_series(self) -> pd.Series:
        if "date" in self._data.columns:
            return pd.Series(
                pd.to_datetime(self._data["date"], errors="coerce"),
                index=self._data.index,
            )
        return pd.Series(
            pd.to_datetime(self._data.index, errors="coerce"),
            index=self._data.index,
        )

    def slice_date_range(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Self:
        start_ts = pd.to_datetime(start_date) if start_date else None
        end_ts = pd.to_datetime(end_date) if end_date else None

        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValueError("start_date must be earlier than or equal to end_date")

        date_series = self._date_series()
        mask = pd.Series(True, index=self._data.index)
        if start_date:
            assert start_ts is not None
            mask &= date_series >= start_ts
        if end_date:
            assert end_ts is not None
            mask &= date_series <= end_ts

        data_copy = self._data.loc[mask].copy()
        sliced = self.__class__(
            data_copy,
            metadata=self._metadata.copy(),
            symbol=self.symbol,
            name=self.name,
        )
        sliced.factors = list(self.factors)

        for factor, series in self.factor_results.items():
            if series.index.equals(self._data.index):
                series_copy = series.loc[mask].copy()
            else:
                series_dates = pd.Series(
                    pd.to_datetime(series.index, errors="coerce"),
                    index=series.index,
                )
                series_mask = pd.Series(True, index=series.index)
                if start_date:
                    assert start_ts is not None
                    series_mask &= series_dates >= start_ts
                if end_date:
                    assert end_ts is not None
                    series_mask &= series_dates <= end_ts
                series_copy = series.loc[series_mask].copy()
            sliced.factor_results[factor] = series_copy

        return sliced