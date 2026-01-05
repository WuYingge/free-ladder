from __future__ import annotations
from typing import override

import pandas as pd
import numpy as np
from typing import Iterable, List

from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from factors.base_cross_section_factor import BaseCrossSectionFactor
from core.models.etf_daily_data import EtfData

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from core.models.calandar_df import CALANDAR

class ClusterAnalysis(BaseCrossSectionFactor):
    name = "ClusterAnalysis"
    
    def __init__(
        self, 
        n:int=252,
        use_pca:bool=False,
        pca_component:float=0.95,
        pca_methods:str="standardize"
        ):
        self.n_days = n
        self.use_pca = use_pca
        self.pca_component = pca_component
        self.pca_methods = pca_methods
        self.scaler = StandardScaler()
        self.date_index: pd.Index = CALANDAR.get_last_n_trading_days_from(
            end_date=pd.Timestamp.today(), n=self.n_days
        )
        self.features: pd.DataFrame = pd.DataFrame()

    def __call__(self, *data: EtfData, name_dict: None|dict=None) -> pd.DataFrame:
        self._check(data)
        self._gen_feature_df(list(data))
        processed_data = self._preprocess()
        use_esp = self.estimate_eps_for_dbscan(processed_data)
        _, n_clusters, _ = self.dbscan_clustering(processed_data, use_esp)
        res_dict = self.kmeans_clustering(processed_data, n_clusters)
        final_labels = res_dict['final_labels']
        return pd.DataFrame(final_labels, index=self.features.index, columns=['ClusterLabel']).sort_values(by='ClusterLabel')
    
    def _check(self, data: Iterable[EtfData]) -> None:
        for etf_data in data:
            if len(etf_data.data) < self.n_days:
                raise ValueError(
                    f"Not enough data for ETF {etf_data.symbol}. "
                    f"Required: {self.n_days}, Available: {etf_data.data.shape[0]}"
                )
        
    def _gen_feature_df(self, datas: List[EtfData]):
        feature_series = []
        for etf in datas:
            data = etf.data.set_index("date", drop=True)
            data.index = pd.to_datetime(data.index)
            data = data.reindex(self.date_index)["gain"]
            data.name = etf.symbol
            data.ffill(inplace=True)
            data.bfill(inplace=True)
            if data.isna().any():
                raise ValueError(f"NaN values found in gain data for ETF {etf.symbol}")
            feature_series.append(data)
        self.features = pd.concat(feature_series, axis=1).T

    def _preprocess(self) -> pd.DataFrame:
        """
        数据预处理和降维
        """
        # 1. 标准化
        if self.pca_methods == 'standardize':
            processed_data = self.scaler.fit_transform(self.features)
        else:
            processed_data = self.features.copy()
        
        # 2. PCA降维（推荐）
        if self.use_pca:
            pca = PCA(n_components=self.pca_component)
            processed_data = pca.fit_transform(processed_data)
            print(f"PCA降维后维度: {processed_data.shape[1]}")
            print(f"PCA解释方差比例: {pca.explained_variance_ratio_.sum():.2%}")
        
        return pd.DataFrame(processed_data, index=self.features.index, columns=pd.RangeIndex(start=0, stop=processed_data.shape[1]))
    
    def estimate_eps_for_dbscan(self, processed_data, k=5) -> float:
        """
        使用k距离图估计DBSCAN的最佳eps参数
        """
        # 计算每个点到第k个最近邻的距离
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(processed_data)
        distances, indices = neighbors_fit.kneighbors(processed_data)
        
        # 取第k个最近邻的距离
        k_distances = distances[:, k-1]
        k_distances_sorted = np.sort(k_distances)
        
        # 绘制k距离图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(k_distances_sorted) + 1), k_distances_sorted)
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-th nearest neighbor distance')
        plt.title('k-Distance Graph for DBSCAN eps estimation')
        plt.grid(True, alpha=0.3)
        
        # 寻找拐点（肘点）
        # 计算曲率变化
        differences = np.diff(k_distances_sorted)
        second_derivatives = np.diff(differences)
        
        # 找到最大曲率变化点
        if len(second_derivatives) > 0:
            elbow_index = np.argmax(second_derivatives) + 1
            eps_suggestion = k_distances_sorted[elbow_index]
            plt.axhline(y=eps_suggestion, color='r', linestyle='--', 
                       label=f'Suggested eps: {eps_suggestion:.3f}')
            plt.legend()
        
        plt.show()
        
        # 返回建议的eps值
        if len(second_derivatives) > 0:
            return eps_suggestion # type: ignore
        else:
            # 如果没有明显拐点，使用中位数
            return np.median(k_distances_sorted).item()
        
    def dbscan_clustering(self, processed_data, eps, min_samples=5):
        """
        使用DBSCAN聚类
        """
        
        print(f"使用DBSCAN参数: eps={eps:.3f}, min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        dbscan_labels = dbscan.fit_predict(processed_data)
        
        # 分析结果
        unique_labels = np.unique(dbscan_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(dbscan_labels == -1)
        
        print(f"\nDBSCAN聚类结果:")
        print(f"聚类数量: {n_clusters}")
        print(f"噪声点数量: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)")
        
        # 计算评估指标（排除噪声点）
        core_samples_mask = np.zeros_like(dbscan_labels, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        
        if n_clusters > 1:
            # 只考虑核心点进行评估
            core_labels = dbscan_labels[core_samples_mask]
            core_data = processed_data[core_samples_mask]
            
            if len(np.unique(core_labels)) > 1:
                silhouette = silhouette_score(core_data, core_labels)
                calinski = calinski_harabasz_score(core_data, core_labels)
                db_score = davies_bouldin_score(core_data, core_labels)
                
                print(f"核心点轮廓系数: {silhouette:.3f}")
                print(f"核心点Calinski-Harabasz指数: {calinski:.2f}")
                print(f"核心点Davies-Bouldin指数: {db_score:.3f}")
        
        return dbscan_labels, n_clusters, dbscan
    
    def kmeans_clustering(self, processed_data: pd.DataFrame, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        final_labels = kmeans.fit_predict(processed_data)
        centers = kmeans.cluster_centers_
        
        # 评估KMeans聚类质量
        if n_clusters > 1:
            silhouette = silhouette_score(processed_data, final_labels)
            calinski = calinski_harabasz_score(processed_data, final_labels)
            db_score = davies_bouldin_score(processed_data, final_labels)

            print(f"KMeans聚类评估:")
            print(f"  轮廓系数: {silhouette:.3f}")
            print(f"  Calinski-Harabasz指数: {calinski:.2f}")
            print(f"  Davies-Bouldin指数: {db_score:.3f}")
        return {
            'final_labels': final_labels,
            'n_clusters': n_clusters,
            'centers': centers if 'centers' in locals() else None,
        }
    