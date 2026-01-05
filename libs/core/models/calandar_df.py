from __future__ import annotations

import os
import pandas as pd
import akshare as ak
tool_trade_date_hist_sina_df = ak.tool_trade_date_hist_sina()
from config import DataPath

class CalandarDF:
    CALANDAR_DF_PATH: str = DataPath.CALANDAR_DF
    
    def __init__(self):
        self._calandar_df = None
        
    @property
    def calandar_df(self) -> pd.DataFrame:
        if (self._calandar_df is None):
            self._calandar_df = pd.read_csv(self.CALANDAR_DF_PATH, index_col=0, parse_dates=['trade_date'])
        return self._calandar_df
    
    def get_last_n_trading_days(self, n:int) -> pd.Index:
        calandar_df = self.calandar_df
        last_n_days = calandar_df['trade_date'].sort_values(ascending=False).head(n)
        return pd.Index(last_n_days)
    
    def get_last_n_trading_days_from(self, end_date: pd.Timestamp, n:int) -> pd.Index:
        calandar_df = self.calandar_df
        filtered_days = calandar_df[calandar_df['trade_date'] <= end_date]
        last_n_days = filtered_days['trade_date'].sort_values(ascending=False).head(n)
        return pd.Index(last_n_days)
    
    def get_trading_days_between(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Index:
        calandar_df = self.calandar_df
        mask = (calandar_df['trade_date'] >= start_date) & (calandar_df['trade_date'] <= end_date)
        trading_days = calandar_df.loc[mask, 'trade_date'].sort_values()
        return pd.Index(trading_days)
    
    def is_trading_day(self, date: pd.Timestamp) -> bool:
        calandar_df = self.calandar_df
        return not calandar_df[calandar_df['trade_date'] == date].empty
    
    def update_calandar_df(self):
        ak.tool_trade_date_hist_sina().to_csv(self.CALANDAR_DF_PATH, index=True, encoding="utf-8-sig")
        self._calandar_df = None
        
CALANDAR = CalandarDF()
