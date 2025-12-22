from abc import abstractmethod, ABC
from typing import Any
import pandas as pd
from core.models.etf_daily_data import EtfData

class BaseCrossSectionFactor(ABC):
    
    name: str
    
    @abstractmethod
    def __call__(self, *data: EtfData) -> pd.DataFrame:
        pass
    
    def __repr__(self) -> str:
        return f"CrossSectionFactor:{self.__class__.name}(name={self.name})"
