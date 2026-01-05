from __future__ import annotations
import warnings
from typing import Optional, Dict, Any, Self
import os
import pandas as pd
import numpy as np
from core.models.data_base import FinancialData
from data_manager.utils import get_symbol_name_from_fp

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
        # todo make date as index
        return cls(df, symbol=symbol)
    
    def output_with_factors_to(self, path) -> None:
        """将包含因子结果的数据保存到CSV文件"""
        df = self.output_with_factors()
        df.to_csv(os.path.join(path, f"{self.symbol}.csv"), index=False, header=True, encoding="utf-8-sig")
