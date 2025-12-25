from __future__ import annotations
import pandas as pd
import numpy as np
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
        return pd.DataFrame(self.risk_concentration(correlation_matrix), index=[0])

    def risk_concentration(self, corr_matrix: pd.DataFrame) -> dict[str, float]:
        """风险集中度分析"""
        # 平均相关系数（衡量整体相关性）
        avg_corr = corr_matrix.values[np.triu_indices(len(corr_matrix), 1)].mean()
        
        # 相关性矩阵的特征值分析（衡量多元化程度）
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalue_ratio = eigenvalues.max() / eigenvalues.sum()
        
        # 有效组合数（衡量真正独立的资产数量）
        if eigenvalues.sum() > 0:
            effective_n = (eigenvalues.sum()**2) / (eigenvalues**2).sum()
        else:
            effective_n = 0
        
        return {
            'average_correlation': avg_corr,
            'dominant_eigenvalue_ratio': eigenvalue_ratio,
            'effective_diversification_number': effective_n,
            'portfolio_concentration_score': eigenvalue_ratio * avg_corr
        }
