from __future__ import annotations
from typing import override
from typing_extensions import Self, List, Set
import pandas as pd
from config import DataPath
from data_manager.providers.base_provider import BaseProvider

class _ClusterProvider(BaseProvider):
    
    @override
    def init(self) -> None:
        self._clusters: dict[str, int] = {}
        self._reverse_clusters: dict[int, list[str]] = {}
        self._initialize_clusters()
    
    @override
    @classmethod
    def get_instance(cls) -> Self:
        return cls()
        
    def _initialize_clusters(self) -> None:
        df = pd.read_excel(DataPath.CLUSTERING_DF, index_col=0)
        df.index = df.index.astype(str)
        self._clusters.update(df['ClusterLabel'].to_dict())
        
        # 构建反向字典
        self._reverse_clusters.clear()
        for symbol, cluster_label in self._clusters.items():
            if cluster_label not in self._reverse_clusters:
                self._reverse_clusters[cluster_label] = []
            self._reverse_clusters[cluster_label].append(symbol)
        
    def get_cluster(self, symbol: str) -> int:
        return self._clusters.get(symbol, -1)
    
    def items(self):
        for k, v in self._clusters.copy().items():
            yield k, v
            
    def labels(self):
        return self._reverse_clusters.keys()
    
    def get_symbols_for_cluster(self, cluster_label: int) -> List[str]:
        """获取指定聚类标签的所有股票代码列表"""
        return self._reverse_clusters.get(cluster_label, [])
        
ClusterInfo = _ClusterProvider.get_instance()
