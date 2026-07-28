from abc import abstractmethod, ABC
from typing import Any
import pandas as pd
from core.models.daily_quote_data import DailyQuoteData

class BaseCrossSectionFactor(ABC):
    
    name: str
    warmup_period: int = 0
    
    @abstractmethod
    def __call__(self, *data: DailyQuoteData) -> pd.DataFrame:
        pass
    
    def __repr__(self) -> str:
        return f"CrossSectionFactor:{self.__class__.name}(name={self.name})"

    def get_warmup_period(self) -> int:
        warmup = int(self.warmup_period)
        if warmup < 0:
            raise ValueError(
                f"Invalid warmup_period for cross section factor {self.name}: {warmup}. "
                "Expected a non-negative integer."
            )
        return warmup

    def get_max_warmup_period(self) -> int:
        return self.get_warmup_period()
