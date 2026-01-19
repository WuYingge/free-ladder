from abc import abstractmethod, ABC
from typing import Any
import pandas as pd

class BaseFactor(ABC):
    
    name: str
    params: dict[str, Any] = {}
    
    def __init__(self) -> None:
        self._dependencies: list[BaseFactor] = []
        self._dep_res: dict[BaseFactor, pd.Series|None] = {}
    
    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.name}(name={self.name})"
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return hash(self) == hash(other)
    
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
