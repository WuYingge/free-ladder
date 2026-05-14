from __future__ import annotations

import pandas as pd

from core.models.daily_quote_data import DailyQuoteData
from data_manager.providers.index_list_provider import INDEX_LIST
from data_manager.utils import get_symbol_name_from_fp


class IndexDailyData(DailyQuoteData):
    """Index daily quote model backed by the local index csv storage format."""

    @classmethod
    def from_csv(cls, fp: str) -> IndexDailyData:
        df = pd.read_csv(fp)
        symbol = get_symbol_name_from_fp(fp)
        name = INDEX_LIST.get_name(symbol=symbol)
        return cls(df, symbol=symbol, name=name)