from __future__ import annotations
import pandas as pd

from core.models.daily_quote_data import DailyQuoteData
from data_manager.utils import get_symbol_name_from_fp
from data_manager.providers.etf_list_provider import ETF_LIST

class EtfData(DailyQuoteData):
    """ETF daily quote model backed by the local ETF csv storage format."""

    @classmethod
    def from_csv(cls, fp: str) -> EtfData:
        df = pd.read_csv(fp)
        symbol = get_symbol_name_from_fp(fp)
        name = ETF_LIST.get_name(symbol=symbol)
        return cls(df, symbol=symbol, name=name)
