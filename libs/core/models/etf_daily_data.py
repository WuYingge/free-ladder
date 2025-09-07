from __future__ import annotations
import warnings
from typing import Optional, Dict, Any, Self
import pandas as pd
import numpy as np
from data_base import FinancialData


class EtfData(FinancialData):
    """股票数据类"""
    
    # 必需列的定义
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
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
        self.symbol = symbol
        self.name = name
        
        # 验证数据
        if not self.validate_data():
            warnings.warn("Data validation failed. Some operations may not work correctly.")
    
    def validate_data(self) -> bool:
        """验证数据是否包含必需的列"""
        return all(col in self._data.columns for col in self.REQUIRED_COLUMNS)
    
    def calculate_returns(self, period: int = 1, 
                         return_type: str = 'simple') -> Self:
        """
        计算收益率
        
        Parameters:
        period: 计算周期
        return_type: 收益率类型 ('simple' 或 'log')
        
        Returns:
        包含收益率的新EtfData对象
        """
        def _calc_returns(data):
            close_prices = data['close']
            if return_type == 'simple':
                returns = close_prices.pct_change(periods=period)
            elif return_type == 'log':
                returns = np.log(close_prices / close_prices.shift(period))
            else:
                raise ValueError("return_type must be 'simple' or 'log'")
            
            # 添加收益率列而不修改原始数据
            result = data.copy()
            result[f'return_{period}_{return_type}'] = returns
            return result
        
        return self._apply_operation(_calc_returns)
    
    def calculate_moving_average(self, window: int, 
                                column: str = 'close') -> EtfData:
        """
        计算移动平均线
        
        Parameters:
        window: 移动窗口大小
        column: 计算移动平均的列名
        
        Returns:
        包含移动平均线的新EtfData对象
        """
        def _calc_ma(data):
            result = data.copy()
            result[f'ma_{window}'] = data[column].rolling(window=window).mean()
            return result
        
        return self._apply_operation(_calc_ma)
    
    def calculate_rsi(self, window: int = 14) -> EtfData:
        """
        计算相对强弱指数(RSI)
        
        Parameters:
        window: RSI计算窗口
        
        Returns:
        包含RSI的新EtfData对象
        """
        def _calc_rsi(data):
            close_prices = data['close']
            delta = close_prices.diff()
            
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            result = data.copy()
            result[f'rsi_{window}'] = rsi
            return result
        
        return self._apply_operation(_calc_rsi)
    
    def calculate_bollinger_bands(self, window: int = 20, 
                                 num_std: float = 2.0) -> EtfData:
        """
        计算布林带
        
        Parameters:
        window: 移动窗口大小
        num_std: 标准差倍数
        
        Returns:
        包含布林带的新EtfData对象
        """
        def _calc_bb(data):
            close_prices = data['close']
            rolling_mean = close_prices.rolling(window=window).mean()
            rolling_std = close_prices.rolling(window=window).std()
            
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            
            result = data.copy()
            result[f'bb_upper_{window}'] = upper_band
            result[f'bb_middle_{window}'] = rolling_mean
            result[f'bb_lower_{window}'] = lower_band
            return result
        
        return self._apply_operation(_calc_bb)
    
    def get_ohlc(self) -> pd.DataFrame:
        """获取OHLC数据"""
        return self._data[self.REQUIRED_COLUMNS].copy()
