from __future__ import annotations
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Self, Mapping
import copy
import warnings
from factors.base_factor import BaseFactor

class FinancialData(ABC):
    """
    金融数据基类，提供数据保护机制和通用方法
    所有操作都不会修改原始数据，而是返回新对象
    """
    
    def __init__(self, data: pd.DataFrame, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        初始化金融数据对象
        
        Parameters:
        data: 包含金融数据的DataFrame
        metadata: 数据的元信息（如数据来源、更新时间等）
        """
        self._data = data.copy()  # 始终保存数据的副本
        self._metadata = metadata.copy() if metadata else {}
        self.factors: List[BaseFactor] = []
        
    @property
    def data(self) -> pd.DataFrame:
        """获取数据的副本（保护原始数据）"""
        return self._data.copy()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """获取元数据的副本"""
        return self._metadata.copy()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """返回数据形状"""
        return self._data.shape
    
    @property
    def columns(self) -> pd.Index:
        """返回列名"""
        return self._data.columns.copy()
    
    @property
    def index(self) -> pd.Index:
        """返回索引"""
        return self._data.index.copy()
    
    def __repr__(self) -> str:
        """对象表示"""
        return f"{self.__class__.__name__}(shape={self.shape})"
    
    def __getitem__(self, key) -> Self:
        """索引操作，返回新对象"""
        if isinstance(key, str):
            # 选择单列
            return self.__class__(self._data[[key]].copy(), self._metadata)
        else:
            # 选择多列或行切片
            return self.__class__(self._data[key].copy(), self._metadata)
    
    def _apply_operation(self, operation: Callable, *args, **kwargs) -> Self:
        """
        应用操作并返回新对象的通用方法
        
        Parameters:
        operation: 要应用的操作函数
        *args, **kwargs: 操作函数的参数
        
        Returns:
        包含操作结果的新FinancialData对象
        """
        # 创建数据的深拷贝
        new_data = self._data.copy()
        # 应用操作
        result = operation(new_data, *args, **kwargs)
        # 确保结果是DataFrame
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result)
        
        return self.__class__(result, self._metadata)
    
    def copy(self) -> Self:
        """创建对象的深拷贝"""
        return self.__class__(self._data.copy(), self._metadata.copy())
    
    def head(self, n: int = 5) -> Self:
        """返回前n行数据"""
        return self.__class__(self._data.head(n).copy(), self._metadata)
    
    def tail(self, n: int = 5) -> Self:
        """返回后n行数据"""
        return self.__class__(self._data.tail(n).copy(), self._metadata)
    
    def filter_dates(self, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> Self:
        """按日期范围过滤数据"""
        data_copy = self._data.copy()
        if start_date:
            data_copy = data_copy[data_copy.index >= start_date]
        if end_date:
            data_copy = data_copy[data_copy.index <= end_date]
        return self.__class__(data_copy, self._metadata)
    
    def add_metadata(self, key: str, value: Any) -> Self:
        """添加元数据"""
        new_metadata = self._metadata.copy()
        new_metadata[key] = value
        return self.__class__(self._data.copy(), new_metadata)
    
    def add_factors(self, factor: BaseFactor):
        self.factors.append(factor)
        
    def calc_factors(self) -> pd.DataFrame:
        factor_results = []
        for factor in self.factors:
            factor_results.append(pd.Series(factor(self.data), name=factor.name))
        return self.data.join(factor_results)
    
    @abstractmethod
    def validate_data(self) -> bool:
        """验证数据格式和完整性（子类必须实现）"""
        pass
