from __future__ import annotations

import pandas as pd

from core.models.daily_quote_data import DailyQuoteData
from data_manager.providers.stock_list_provider import STOCK_LIST
from data_manager.utils import get_symbol_name_from_fp


class StockDailyData(DailyQuoteData):
    """A-share stock daily quote model backed by the local stock csv storage format."""

    @classmethod
    def from_csv(cls, fp: str) -> StockDailyData:
        df = pd.read_csv(fp)
        symbol = get_symbol_name_from_fp(fp)
        name = STOCK_LIST.get_name(symbol=symbol)
        return cls(df, symbol=symbol, name=name)
