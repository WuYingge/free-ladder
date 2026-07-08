from __future__ import annotations

from abc import abstractmethod

import pandas as pd

from factors.base_factor import BaseFactor


class DerivedFactor(BaseFactor):
    """Base class for factors derived from raw columns plus dependency outputs."""

    def get_source_columns(self) -> tuple[str, ...]:
        return ()

    def get_dependency_column_map(self) -> list[str]:
        """返回依赖列名列表，按 self.dependencies 顺序。默认用 output_name。"""
        return [dep.get_output_name() for dep in self.dependencies]

    def build_input_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(index=data.index)

        for column in self.get_source_columns():
            if column not in data.columns:
                raise ValueError(
                    f"Derived factor {self.name} requires source column {column!r}, got columns {list(data.columns)}"
                )
            frame[column] = data[column].copy()

        col_names = self.get_dependency_column_map()
        dependency_results = self.get_dependency_results(data)
        # 构建 id-based 映射，避免 hash 碰撞（同参数不同依赖因子会冲突）
        dep_result_map = {id(dep): result for dep, result in dependency_results}
        for i, dependency in enumerate(self.dependencies):
            series = self._normalize_dependency_series(
                dependency=dependency,
                result=dep_result_map[id(dependency)],
                data_index=data.index,
            )
            column_name = col_names[i] if i < len(col_names) else series.name
            if column_name in frame.columns:
                raise ValueError(
                    f"Duplicate derived factor input column {column_name!r} for {self.name}"
                )
            frame[column_name] = series

        return frame

    def _normalize_dependency_series(
        self,
        dependency: BaseFactor,
        result: pd.Series | object,
        data_index: pd.Index,
    ) -> pd.Series:
        if isinstance(result, pd.Series):
            series = result.copy()
        else:
            series = pd.Series(result, index=data_index)

        expected_name = dependency.get_output_name()
        if series.name is None:
            series.name = expected_name
        elif series.name != expected_name:
            raise ValueError(
                f"Dependency {dependency!r} returned series named {series.name!r}, expected {expected_name!r}"
            )
        return series

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        frame = self.build_input_frame(data)
        result = self.compute_from_frame(frame)

        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=data.index)

        expected_name = self.get_output_name()
        if result.name is None:
            result = result.rename(expected_name)
        elif result.name != expected_name:
            raise ValueError(
                f"Derived factor {self!r} returned series named {result.name!r}, expected {expected_name!r}"
            )
        return result

    @abstractmethod
    def compute_from_frame(self, frame: pd.DataFrame) -> pd.Series:
        pass