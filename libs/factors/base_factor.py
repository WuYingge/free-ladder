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
    
    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})[{', '.join(f'{k}={v}' for k, v in self.params.items())}]"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return tuple(sorted(self.params.items())) == tuple(sorted(other.params.items()))
    
    def __hash__(self) -> int:
        param_items = tuple(sorted(self.params.items()))
        return hash((self.__class__, param_items))

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
