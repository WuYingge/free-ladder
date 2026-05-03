from __future__ import annotations
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Self, MutableMapping
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
        self.factor_results: MutableMapping[BaseFactor, pd.Series] = {}
        
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

    def _normalize_factor_series(
        self,
        factor: BaseFactor,
        result: Union[pd.Series, Any],
        data_index: pd.Index,
    ) -> pd.Series:
        if isinstance(result, pd.Series):
            series = result.copy()
        else:
            series = pd.Series(result, index=data_index)

        expected_name = factor.get_output_name()
        if series.name is None:
            series.name = expected_name
        elif series.name != expected_name:
            raise ValueError(
                f"Factor {factor!r} returned series named {series.name!r}, expected {expected_name!r}"
            )
        return series

    def _validate_unique_factor_output_names(self, factors: List[BaseFactor]) -> None:
        seen: Dict[str, BaseFactor] = {}
        for factor in factors:
            output_name = factor.get_output_name()
            existing = seen.get(output_name)
            if existing is not None and existing != factor:
                raise ValueError(
                    f"Duplicate factor output name {output_name!r} for {existing!r} and {factor!r}"
                )
            seen[output_name] = factor

    def _join_factor_series(self, factor_series: List[pd.Series]) -> pd.DataFrame:
        seen_names: set[str] = set()
        for series in factor_series:
            if series.name is None:
                raise ValueError("Factor result series must have a name before joining")
            if series.name in seen_names:
                raise ValueError(f"Duplicate factor result column {series.name!r}")
            seen_names.add(series.name)
        return self.data.join(factor_series)
        
    def calc_factors(self) -> pd.DataFrame:
        data = self.data
        sorted_factors = self.sort_factors_by_dependency()
        self._validate_unique_factor_output_names(sorted_factors)

        factor_results = []
        for factor in sorted_factors:
            series = self._normalize_factor_series(factor, factor(data), data.index)
            factor_results.append(series)
            self.factor_results[factor] = series
        return self._join_factor_series(factor_results)
    
    def sort_factors_by_dependency(self) -> List[BaseFactor]:
        """根据依赖关系对因子进行排序"""
        sorted_factors: List[BaseFactor] = []
        visited: Dict[BaseFactor, bool] = {}
        
        def visit(factor: BaseFactor):
            if factor in visited:
                if not visited[factor]:
                    raise ValueError(f"Circular dependency detected for factor {factor.name}")
                return
            visited[factor] = False
            for dep in factor.dependencies:
                visit(dep)
            visited[factor] = True
            sorted_factors.append(factor)
        
        for factor in self.factors:
            visit(factor)
        
        return sorted_factors
    
    def output_with_factors(self) -> pd.DataFrame:
        """输出包含因子结果的数据"""
        if not self.factor_results:
            self.calc_factors()
        factor_dfs = [
            self._normalize_factor_series(factor, series, self._data.index)
            for factor, series in self.factor_results.items()
        ]
        return self._join_factor_series(factor_dfs) # type: ignore
    
    @abstractmethod
    def validate_data(self) -> bool:
        """验证数据格式和完整性（子类必须实现）"""
        pass
