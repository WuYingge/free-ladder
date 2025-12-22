from __future__ import annotations
import pandas as pd
from factors.base_cross_section_factor import BaseCrossSectionFactor
from core.models.etf_daily_data import EtfData

class CorrelationFactor(BaseCrossSectionFactor):
    """
    计算多个时间序列之间的相关系数矩阵的因子
    """
    
    name = "CorrelationFactor"
    
    def __call__(self, *data_models: EtfData) -> pd.DataFrame:
        if len(data_models) < 2:
            raise ValueError("至少需要两个数据来计算相关系数矩阵")
        
        columns = [data_model.symbol for data_model in data_models]
        combined_data = pd.concat([data_model.data.set_index("date")["gain"] for data_model in data_models], axis=1, keys=columns)
        if combined_data.shape[0] < 60:
            raise ValueError("数据长度不足以计算相关系数矩阵，至少需要60个数据点")
        correlation_matrix = combined_data.corr()
        return correlation_matrix
