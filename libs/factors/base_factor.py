from abc import abstractmethod, ABC
from typing import Any
import pandas as pd

class BaseFactor(ABC):
    
    name: str
    
    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.name}(name={self.name})"
