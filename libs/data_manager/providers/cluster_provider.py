from __future__ import annotations
from typing import override
from typing_extensions import Self
import pandas as pd
from config import DataPath
from data_manager.providers.base_provider import BaseProvider

class _ClusterProvider(BaseProvider):
    
    @override
    def init(self) -> None:
        self._clusters: dict[str, int] = {}
        self._initialize_clusters()
    
    @override
    @classmethod
    def get_instance(cls) -> Self:
        return cls()
        
    def _initialize_clusters(self) -> None:
        df = pd.read_excel(DataPath.CLUSTERING_DF, index_col=0)
        df.index = df.index.astype(str)
        self._clusters.update(df['ClusterLabel'].to_dict())
        
    def get_cluster(self, symbol: str) -> int:
        return self._clusters.get(symbol, -1)

ClusterInfo = _ClusterProvider.get_instance()
