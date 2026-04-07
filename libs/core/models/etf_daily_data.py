from __future__ import annotations
import warnings
from typing import Optional, Dict, Any, Self
import os
import pandas as pd
import numpy as np
from core.models.data_base import FinancialData
from data_manager.utils import get_symbol_name_from_fp
from data_manager.providers.etf_list_provider import ETF_LIST

class EtfData(FinancialData):
    """股票数据类"""
    
    # 必需列的定义
    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "value", "turnOver", "gain", "change"]
    DTYPES = [pd.Float64Dtype, pd.Float64Dtype, pd.Float64Dtype, 
              pd.Float64Dtype, pd.Int64Dtype, pd.Int64Dtype, 
              pd.Float64Dtype, pd.Float64Dtype, pd.Float64Dtype]
    
    def __init__(self, data: pd.DataFrame, 
                 metadata: Optional[Dict[str, Any]] = None,
                 symbol: Optional[str] = None,
                 name: str = ""):
        """
        初始化股票数据
        
        Parameters:
        data: 包含OHLCV数据的DataFrame
        metadata: 元数据
        symbol: 股票代码
        """
        super().__init__(data, metadata)
        
        # todo move to base class?
        self.symbol = symbol
        self.name = name
        
        # 验证数据
        if not self.validate_data():
            warnings.warn("Data validation failed. Some operations may not work correctly.")
    
    def validate_data(self) -> bool:
        """验证数据是否包含必需的列"""
        return all(col in self._data.columns for col in self.REQUIRED_COLUMNS)
    
    @classmethod
    def from_csv(cls: type[EtfData], fp: str) -> EtfData:
        df = pd.read_csv(fp)
        symbol = get_symbol_name_from_fp(fp)
        name = ETF_LIST.get_name(symbol=symbol)
        # todo make date as index
        return cls(df, symbol=symbol, name=name)
    
    def output_with_factors_to(self, path) -> None:
        """将包含因子结果的数据保存到CSV文件"""
        df = self.output_with_factors()
        df.to_csv(os.path.join(path, f"{self.symbol}.csv"), index=False, header=True, encoding="utf-8-sig")

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
        """按日期范围截取数据，并保留 symbol/name/因子结果。"""
        start_ts = pd.to_datetime(start_date) if start_date else None
        end_ts = pd.to_datetime(end_date) if end_date else None

        if start_ts is not None and end_ts is not None:
            if start_ts > end_ts:
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
