from __future__ import annotations
from typing_extensions import Self
import pandas as pd
from config import DataPath

class _ClusterProvider:
    
    def __new__(cls) -> Self:
        if not hasattr(cls, 'instance'):
            cls.instance = super(_ClusterProvider, cls).__new__(cls)
        return cls.instance
    
    def __init__(self) -> None:
        self._clusters: dict[str, int] = {}
        self._initialize_clusters()
        
    def _initialize_clusters(self) -> None:
        df = pd.read_excel(DataPath.CLUSTERING_DF, index_col=0)
        df.index = df.index.astype(str)
        self._clusters.update(df['ClusterLabel'].to_dict())
        
    def get_cluster(self, symbol: str) -> int:
        return self._clusters.get(symbol, -1)

ClusterInfo = _ClusterProvider()
