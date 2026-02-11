from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np
from factors.base_factor import BaseFactor

from factors.new_high import NewHigh

class ChangeSinceNewHigh(BaseFactor):
    
    name = "ChangeSinceNewHigh"
    
    def __init__(self, long_period, short_period) -> None:
        super().__init__()
        self.long_period = long_period
        self.short_period = short_period
        self.add_dependency(NewHigh(long_period, short_period))
        
    def __call__(self, data: pd.DataFrame) -> pd.Series[Any]:
        merged_data = self.get_merged_dep_data(data)
        return self.calc_returns_from_first_high(merged_data)
        
    
    def get_merged_dep_data(self, data: pd.DataFrame) -> pd.DataFrame:
        merged_data = data.copy()
        merged_data["NewHigh"] = self.get_dependency_results(data)[self.dependencies[0]]
        return merged_data
    
    def calc_returns_from_first_high(self, df: pd.DataFrame) -> pd.Series:
        """
        对于df中NewHigh=1的行，计算自所在组第一个1以来的涨幅。
        参数:
            df : DataFrame，必须包含'NewHigh'和'close'列
        返回:
            Series，索引同df，对应位置为涨幅，非1行为NaN
        """
        df = df.copy()  # 避免修改原始数据
        
        # 1. 生成组号：每个-1之后开始新组，-1行本身不参与分组
        df['group'] = (df['NewHigh'] == -1).cumsum().shift(1).fillna(0)
        df.loc[df['NewHigh'] == -1, 'group'] = np.nan   # 剔除-1行
        
        # 2. 对NewHigh=1的行，按组取第一个收盘价作为该组基准
        first_close = (
            df[df['NewHigh'] == 1]
            .groupby('group')['close']
            .first()
        )
        
        # 3. 将基准价格映射回NewHigh=1的行
        mask_high = df['NewHigh'] == 1
        df['benchmark'] = np.nan
        df.loc[mask_high, 'benchmark'] = df.loc[mask_high, 'group'].map(first_close)
        
        # 4. 计算涨幅（今日 / 基准 - 1）
        change =  (df['close'] / df['benchmark'] - 1).where(mask_high)
        
        # 5. 格式化为2位小数的百分比字符串
        change = change.apply(lambda x: f"{x:.2%}" if pd.notna(x) else np.nan)
        return change
    