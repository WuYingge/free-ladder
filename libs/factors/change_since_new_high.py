from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np
from factors.base_factor import BaseFactor

from factors.new_high import NewHigh

class ChangeSinceNewHigh(BaseFactor):
    
    name = "ChangeSinceNewHigh"
    params = {
        "long_period": 50,
        "short_period": 25,
        "use_long_filter": False
    }
    
    def __init__(self, long_period: int, short_period: int, use_long_filter: bool = False) -> None:
        super().__init__()
        self.long_period = long_period
        self.short_period = short_period
        self.use_long_filter = use_long_filter
        self._set_params(long_period=long_period, short_period=short_period, use_long_filter=use_long_filter)
        self.add_dependency(NewHigh(long_period, short_period, use_long_filter=use_long_filter))
        
    def __call__(self, data: pd.DataFrame) -> pd.Series[Any]:
        merged_data = self.get_merged_dep_data(data)
        return self.calc_returns_from_first_high(merged_data)
        
    
    def get_merged_dep_data(self, data: pd.DataFrame) -> pd.DataFrame:
        merged_data = data.copy()
        # get_dependency_results 返回 list[(factor, series)]，需按 id 索引
        # （与 DerivedFactor.build_input_frame 的写法保持一致）
        dep_map = {id(dep): result for dep, result in self.get_dependency_results(data)}
        merged_data["NewHigh"] = dep_map[id(self.dependencies[0])]
        return merged_data
    
    def calc_returns_from_first_high(self, df: pd.DataFrame) -> pd.Series[Any]:
        """
        对于df中的每一行，计算自所在组第一个1以来的涨幅。
        如果遇到了-1则归零，并且不参与后续的涨幅计算。
        组的划分方式是：每遇到一个-1，开始一个新的组，-1行本身不参与分组。
        参数:
            df : DataFrame，必须包含'NewHigh'和'close'列
        返回:
            Series，索引同df，对应位置为涨幅，非1行为NaN
        """
        df = df.copy()
    
        # 1. 按 -1 进行分组（每个 -1 开始新组）
        group = (df["NewHigh"] == -1).cumsum()
        
        # 2. 标记每个组内的第一个 1
        df["is_one"] = df["NewHigh"] == 1
        first_one = df["is_one"] & (df.groupby(group)["is_one"].cumsum() == 1)
        
        # 3. 创建基准价格列：只在第一个 1 的位置填入 close
        base = df["close"].where(first_one)
        
        # 4. 在每组内向前填充基准（从第一个 1 向后传播）
        base_filled = base.groupby(group).ffill()
        
        # 5. 计算涨幅（除法，结果可能为 inf 但 close 通常非零）
        pct = (df["close"] - base_filled) / base_filled
        
        # 6. 数值化输出：仅当有基准时保留涨幅数值，其余为 NaN。
        #    框架约定因子输出必须为数值列（回测引擎用 to_numeric 解析排名列），
        #    不能返回格式化百分比字符串（会被解析为 NaN 导致全部标的被剔除）。
        df["change_since_new_high"] = pct.where(base_filled.notna())
        # Match the framework requirement: factor output Series name must equal factor output name.
        return df["change_since_new_high"].rename(self.get_output_name())
