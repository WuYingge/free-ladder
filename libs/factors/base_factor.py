from abc import abstractmethod, ABC
from typing import Any
import pandas as pd

class BaseFactor(ABC):
    
    name: str
    params: dict[str, Any] = {}
    warmup_period: int = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not isinstance(getattr(cls, "params", None), dict):
            raise TypeError(f"{cls.__name__}.params must be a dict")
    
    def __init__(self) -> None:
        # Always keep params at instance level so constructor overrides do not
        # leak across instances through mutable class attributes.
        self.params = dict(self.__class__.params)
        self._dependencies: list[BaseFactor] = []
        self._dep_res: dict[BaseFactor, pd.Series|None] = {}

    def _set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def get_output_name(self) -> str:
        """Return the canonical output column name for this factor instance."""
        return self.name
    
    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})[{', '.join(f'{k}={v}' for k, v in self.params.items())}]"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    @staticmethod
    def _hashable_params(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        """将 params 转为可哈希的 tuple，递归处理 list 等不可哈希类型。"""
        def _to_hashable(v: Any) -> Any:
            if isinstance(v, list):
                return tuple(_to_hashable(x) for x in v)
            if isinstance(v, dict):
                return tuple(sorted((k, _to_hashable(v2)) for k, v2 in v.items()))
            return v
        return tuple(sorted((k, _to_hashable(v)) for k, v in params.items()))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._hashable_params(self.params) == self._hashable_params(other.params)

    def __hash__(self) -> int:
        return hash((self.__class__, self._hashable_params(self.params)))

    @property
    def dependencies(self) -> list["BaseFactor"]:
        return self._dependencies

    def add_dependency(self, dependency: "BaseFactor") -> None:
        self._dependencies.append(dependency)
        self._dep_res[dependency] = None
        
    def get_dependency_results(self, data: pd.DataFrame) -> dict["BaseFactor", pd.Series]:
        results = {}
        for dependency in self._dependencies:
            results[dependency] = dependency(data)
        return results

    def get_warmup_period(self) -> int:
        """Return this factor's own warm-up period in bars."""
        warmup = int(self.warmup_period)
        if warmup < 0:
            raise ValueError(
                f"Invalid warmup_period for factor {self.name}: {warmup}. "
                "Expected a non-negative integer."
            )
        return warmup

    def get_max_warmup_period(self) -> int:
        """Return max warm-up across this factor and all dependencies."""
        max_warmup = self.get_warmup_period()
        for dependency in self._dependencies:
            dep_warmup = dependency.get_max_warmup_period()
            if dep_warmup > max_warmup:
                max_warmup = dep_warmup
        return max_warmup
