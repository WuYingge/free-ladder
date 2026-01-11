from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from typing_extensions import Self

class BaseProvider(ABC):
    
    def __new__(cls) -> Self:
        if not hasattr(cls, 'instance'):
            cls.instance = super(BaseProvider, cls).__new__(cls)
        return cls.instance
    
    def __init__(self) -> None:
        self.init()
        
    @abstractmethod
    def init(self) -> None:
        pass
        
    @classmethod
    @abstractmethod
    def get_instance(cls) -> Self:
        pass
