from __future__ import annotations
from typing import override

import pandas as pd
import numpy as np
from typing import Iterable, List

import seaborn as sns
from itertools import combinations

from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from factors.base_cross_section_factor import BaseCrossSectionFactor
from core.models.etf_daily_data import EtfData

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

from core.models.calandar_df import CALANDAR

class ClusterAnalysis(BaseCrossSectionFactor):
    name = "ClusterAnalysis"
    
    APPROACHES = ["dbscan_kmeans", "corr_agglomerative"]
    STANDARDIZATIONS = ["standardize", "minmax", "none"]
    
    def __init__(
        self, 
        n:int=252,
        use_pca:bool=False,
        pca_component:float=0.95,
        standardization_methods:str="standardize",
        approach:str="dbscan_kmeans",
        min_clusters:int=10
        ):
        self.validate(standardization_methods, approach)
        self.n_days = n
        self.use_pca = use_pca
        self.pca_component = pca_component
        self.standardization_methods = standardization_methods
        self.scaler = StandardScaler()
        self.date_index: pd.Index = CALANDAR.get_last_n_trading_days_from(
            end_date=pd.Timestamp.today(), n=self.n_days
        )
        self.features: pd.DataFrame = pd.DataFrame()
        self.approach = approach
        self.min_clusters = min_clusters

    def validate(self, standardization_methods, approach):
        if standardization_methods not in self.STANDARDIZATIONS:
            raise ValueError(f"Invalid standardization method. Choose from {self.STANDARDIZATIONS}")
        if approach not in self.APPROACHES:
            raise ValueError(f"Invalid approach. Choose from {self.APPROACHES}")

    def __call__(self, *data: EtfData, name_dict: None|dict=None) -> pd.DataFrame:
        print("开始聚类分析...")
        self._check(data)
        print("生成特征矩阵...")
        self._gen_feature_df(list(data))
        print(f"特征矩阵维度: {self.features.shape}")
        processed_data = self._preprocess()
        print(f"预处理后数据维度: {processed_data.shape}")
        if self.approach == "dbscan_kmeans":
            final_labels = self.pca_dbscan_kmeans(processed_data)
        elif self.approach == "corr_agglomerative":
            print("使用相关性层次聚类方法...")
            final_labels = self.corr_agglomerative_clustering(processed_data)
        res_df = self.generate_result(name_dict, final_labels)
        return res_df

    def pca_dbscan_kmeans(self, processed_data:pd.DataFrame):
        processed_data = self._dimensionality_reduction(processed_data)
        use_esp = self.estimate_eps_for_dbscan(processed_data)
        _, n_clusters, _ = self.dbscan_clustering(processed_data, use_esp)          
        res_dict = self.kmeans_clustering(processed_data, n_clusters)
        final_labels = res_dict['final_labels']
        return final_labels

    def corr_agglomerative_clustering(self, processed_data:pd.DataFrame):
        print("计算相关性矩阵和距离矩阵...")
        corr_matrix = processed_data.T.corr()
        print("相关性矩阵样本:")
        dist_matrix = np.sqrt(1 - corr_matrix)
        print(corr_matrix.iloc[:5, :5])
        # np.fill_diagonal(dist_matrix, 0)
        print("距离矩阵样本:")
        dist_matrix = pd.DataFrame(dist_matrix, index=processed_data.index, columns=processed_data.index)
        print(dist_matrix.iloc[:5, :5])
        condensed_dist = squareform(dist_matrix)
        print("绘制层次聚类树状图...")
        linked = linkage(condensed_dist, method='ward')
        print("层次聚类树状图绘制完成.")
        plt.figure(figsize=(12, 6))
        plt.title('ETF Hierarchical Clustering Dendrogram')
        plt.xlabel('ETF Ticker')
        plt.ylabel('Distance')
        dendrogram(linked, 
                labels=dist_matrix.columns,
                leaf_rotation=90,
                leaf_font_size=10)
        plt.tight_layout()
        plt.show()
        print("before cutting the dendrogram:")
        distance = self.calc_elbow_point(linked)
        print(f"选择切割距离: {distance:.4f}")
        final_labels = fcluster(linked, t=distance, criterion='distance') - 1
        n_clusters = len(np.unique(final_labels))
        while n_clusters <= self.min_clusters:
            print(f"聚类数量 {n_clusters} 小于最小要求 {self.min_clusters}，尝试增加切割距离。")
            distance *= 0.8  # 增加切割距离
            final_labels = fcluster(linked, t=distance, criterion='distance') - 1
            n_clusters = len(np.unique(final_labels))
            print(f"调整后聚类数量: {n_clusters}，切割距离: {distance:.4f}")
        print(f"\n层次聚类结果:")
        print(f"聚类数量: {n_clusters}")
        return final_labels
        
    def calc_elbow_point(self, distances: np.ndarray) -> int:
        # 假设 Z 是您的 linkage 矩阵
        # 提取每次合并的距离
        last_merges = distances[-20:, 2]  # 查看最后20次合并的距离，通常变化发生在这里
        # 计算每次合并的距离差
        dist_increase = np.diff(last_merges)
        # 找到距离差最大的索引
        max_increase_index = np.argmax(dist_increase)

        # 推荐的切割距离就在这个“跳跃”发生之前
        # 因为索引是差值索引，所以对应到 Z 中的位置是 -20 + max_increase_index + 1
        recommended_distance = distances[-20 + max_increase_index, 2]

        print(f"检测到在距离 {recommended_distance:.4f} 处发生最大合并跳跃。")
        print(f"建议在此距离（或略低于此值）进行切割。")
        return recommended_distance

    def generate_result(self, name_dict: None|dict, final_labels: list) -> pd.DataFrame:
        res_df = pd.DataFrame(final_labels, index=self.features.index, columns=['ClusterLabel']).sort_values(by='ClusterLabel')
        if name_dict is None:
            name_dict = {}
        res_df["name"] = res_df.apply(lambda x: name_dict.get(x.name, "UNKNOWN"), axis=1)
        return res_df

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
        if self.standardization_methods == 'standardize':
            processed_data = self.scaler.fit_transform(self.features)
            return pd.DataFrame(processed_data, index=self.features.index, columns=pd.RangeIndex(start=0, stop=processed_data.shape[1]))
        elif self.standardization_methods == 'minmax':
            minmax_scaler = MinMaxScaler()
            processed_data = minmax_scaler.fit_transform(self.features)
            return pd.DataFrame(processed_data, index=self.features.index, columns=pd.RangeIndex(start=0, stop=processed_data.shape[1]))
        else:
            return self.features.copy()
        
    def _dimensionality_reduction(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        # 2. PCA降维（推荐）
        if self.use_pca:
            pca = PCA(n_components=self.pca_component)
            reduced = pca.fit_transform(processed_data)
            print(f"PCA降维后维度: {reduced.shape[1]}")
            print(f"PCA解释方差比例: {pca.explained_variance_ratio_.sum():.2%}")
            return pd.DataFrame(reduced, index=self.features.index, columns=pd.RangeIndex(start=0, stop=reduced.shape[1]))
        else:
            return processed_data
        
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

    def calculate_inter_cluster_correlation(self, group_df,
                                            corr_df: pd.DataFrame | None = None,
                                       group_col='ClusterLabel', 
                                       symbol_col='symbol',
                                       method='mean'):
        """
        计算聚类结果的组间平均相关性。
        
        参数:
        ----------
        corr_df : pandas.DataFrame
            相关性矩阵，行和列都是股票/ETF代码
        group_df : pandas.DataFrame
            分组结果，至少包含两列：股票/ETF代码和分组标签
        group_col : str
            group_df中分组标签的列名，默认为'group'
        symbol_col : str
            group_df中股票代码的列名，默认为'symbol'
        method : str
            计算方法，可选 'mean'（平均值）或 'median'（中位数），默认为'mean'
            
        返回:
        ----------
        inter_corr_matrix : pandas.DataFrame
            组间相关性矩阵，行和列为分组标签
        cluster_stats : pandas.DataFrame
            每个分组的统计信息
        """
        if corr_df is None:
            corr_df = self._preprocess().T.corr()
        
        # 1. 数据预处理和验证
        # 确保group_df中的代码在corr_df中存在
        valid_symbols = group_df[group_df[symbol_col].isin(corr_df.index) & 
                                group_df[symbol_col].isin(corr_df.columns)]
        
        if len(valid_symbols) < len(group_df):
            print(f"警告: {len(group_df) - len(valid_symbols)} 个代码在相关性矩阵中不存在，已过滤。")
        
        # 获取分组映射字典
        symbol_to_group = pd.Series(valid_symbols[group_col].values, 
                                index=valid_symbols[symbol_col]).to_dict()
        
        # 获取所有分组标签
        unique_groups = sorted(valid_symbols[group_col].unique())
        n_clusters = len(unique_groups)
        
        print(f"分析概况:")
        print(f"  有效分组数量: {n_clusters}")
        print(f"  有效股票数量: {len(valid_symbols)}")
        
        # 2. 初始化结果矩阵
        inter_corr_matrix = pd.DataFrame(
            np.zeros((n_clusters, n_clusters)),
            index=unique_groups,
            columns=unique_groups
        )
        
        # 3. 计算组间相关性
        for i, group_i in enumerate(unique_groups):
            # 获取组i的所有股票
            symbols_i = [s for s, g in symbol_to_group.items() if g == group_i]
            
            for j, group_j in enumerate(unique_groups):
                if i == j:
                    # 组内相关性设为1（或者可以计算实际组内平均相关性）
                    inter_corr_matrix.loc[group_i, group_j] = 1.0
                    continue
                    
                # 获取组j的所有股票
                symbols_j = [s for s, g in symbol_to_group.items() if g == group_j]
                
                # 提取两组之间的相关性子矩阵
                # 注意: 这里需要确保顺序一致，所以使用loc
                sub_corr = corr_df.loc[symbols_i, symbols_j]
                
                # 计算平均值或中位数（忽略NaN）
                if method == 'mean':
                    avg_corr = sub_corr.values.flatten().mean()
                elif method == 'median':
                    avg_corr = np.median(sub_corr.values.flatten())
                else:
                    raise ValueError("method参数必须是'mean'或'median'")
                
                inter_corr_matrix.loc[group_i, group_j] = avg_corr
        
        # 4. 计算每个分组的统计信息
        cluster_stats = []
        for group in unique_groups:
            symbols_in_group = [s for s, g in symbol_to_group.items() if g == group]
            n_symbols = len(symbols_in_group)
            
            # 获取组内相关性矩阵
            intra_corr_matrix = corr_df.loc[symbols_in_group, symbols_in_group]
            
            # 计算组内平均相关性（排除对角线）
            intra_corr_values = intra_corr_matrix.values.flatten()
            # 将对角线元素（值为1）设置为NaN以便排除
            intra_corr_values_flat = intra_corr_matrix.where(
                ~np.eye(intra_corr_matrix.shape[0], dtype=bool)
            ).values.flatten()
            intra_corr_values_flat = intra_corr_values_flat[~np.isnan(intra_corr_values_flat)]
            
            if len(intra_corr_values_flat) > 0:
                intra_mean = np.mean(intra_corr_values_flat)
                intra_median = np.median(intra_corr_values_flat)
            else:
                intra_mean = intra_median = 1.0  # 如果组内只有一个股票
                
            # 计算组外平均相关性（与其他所有组的平均相关性）
            other_groups = [g for g in unique_groups if g != group]
            inter_corrs = []
            for other_group in other_groups:
                inter_corrs.append(inter_corr_matrix.loc[group, other_group])
            
            inter_mean = np.mean(inter_corrs) if inter_corrs else 0
            
            cluster_stats.append({
                'group': group,
                'n_symbols': n_symbols,
                'intra_corr_mean': intra_mean,
                'intra_corr_median': intra_median,
                'inter_corr_mean': inter_mean,
                'corr_diff': intra_mean - inter_mean  # 组内与组间差异
            })
        
        cluster_stats_df = pd.DataFrame(cluster_stats)
        
        return inter_corr_matrix, cluster_stats_df


    def visualize_inter_cluster_correlation(self, inter_corr_matrix, cluster_stats_df=None):
        """
        可视化组间相关性矩阵。
        
        参数:
        ----------
        inter_corr_matrix : pandas.DataFrame
            组间相关性矩阵
        cluster_stats_df : pandas.DataFrame, optional
            分组统计信息
        """
        
        # 创建图形
        fig, axes = plt.subplots(1, 2 if cluster_stats_df is not None else 1, 
                                figsize=(14, 6) if cluster_stats_df is not None else (8, 6))
        
        if cluster_stats_df is not None:
            ax1, ax2 = axes
        else:
            ax1 = axes
        
        # 1. 绘制组间相关性热图
        mask = np.triu(np.ones_like(inter_corr_matrix, dtype=bool), k=1)
        
        # 为热图选择适当的颜色映射
        # 注意：相关性通常在-1到1之间，但我们这里计算的平均相关性可能范围不同
        vmax = max(1.0, np.nanmax(inter_corr_matrix.values))
        vmin = min(-1.0, np.nanmin(inter_corr_matrix.values))
        
        sns.heatmap(inter_corr_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt=".3f",
                    cmap="RdBu_r",  # 红色表示正相关，蓝色表示负相关
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": "Correlation"},
                    vmin=vmin, vmax=vmax,
                    ax=ax1)
        
        ax1.set_title('Inter-Cluster Average Correlation Matrix')
        
        # 2. 如果有分组统计信息，绘制条形图
        if cluster_stats_df is not None: # type: ignore
            # 设置位置和宽度
            x = range(len(cluster_stats_df))
            width = 0.35
            
            # 绘制组内和组间相关性对比
            ax2.bar([i - width/2 for i in x],  # type: ignore
                    cluster_stats_df['intra_corr_mean'], 
                    width, label='Intra-Cluster', alpha=0.7, color='skyblue')
            ax2.bar([i + width/2 for i in x], # type: ignore
                    cluster_stats_df['inter_corr_mean'], 
                    width, label='Inter-Cluster', alpha=0.7, color='lightcoral')
            
            # 添加数量标签
            for i, row in cluster_stats_df.iterrows():
                ax2.text(i, row['intra_corr_mean'] + 0.02, # type: ignore
                        str(row['n_symbols']), ha='center', fontsize=9)
            
            ax2.set_xlabel('Cluster')# type: ignore
            ax2.set_ylabel('Average Correlation')# type: ignore
            ax2.set_title('Intra vs Inter Cluster Correlation')# type: ignore
            ax2.set_xticks(x)# type: ignore
            ax2.set_xticklabels(cluster_stats_df['group'])# type: ignore
            ax2.legend()# type: ignore
            ax2.grid(True, alpha=0.3, axis='y')# type: ignore
            
            # 添加水平参考线（0相关线）
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)# type: ignore
        
        plt.tight_layout()
        plt.show()
        
        # 打印关键统计信息
        print("\n关键统计信息:")
        print("=" * 50)
        
        # 提取组间相关性（排除对角线）
        inter_values = inter_corr_matrix.values
        inter_values_flat = inter_values[~np.eye(len(inter_values), dtype=bool)]
        
        print(f"组间相关性范围: [{inter_values_flat.min():.3f}, {inter_values_flat.max():.3f}]")
        print(f"组间相关性平均值: {inter_values_flat.mean():.3f}")
        print(f"组间相关性中位数: {np.median(inter_values_flat):.3f}")
        print(f"组间相关性标准差: {inter_values_flat.std():.3f}")
        
        # 识别高相关性组对（可能需要关注）
        high_corr_threshold = 0.5  # 可以调整这个阈值
        high_corr_pairs = []
        
        for i in range(len(inter_corr_matrix)):
            for j in range(i+1, len(inter_corr_matrix)):
                corr = inter_corr_matrix.iloc[i, j]
                if corr > high_corr_threshold:
                    high_corr_pairs.append({
                        'group_i': inter_corr_matrix.index[i],
                        'group_j': inter_corr_matrix.columns[j],
                        'correlation': corr
                    })
        
        if high_corr_pairs:
            print(f"\n警告: 发现 {len(high_corr_pairs)} 对高相关组 (> {high_corr_threshold}):")
            for pair in sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)[:5]:  # 显示前5个
                print(f"  组 {pair['group_i']} 与 组 {pair['group_j']}: {pair['correlation']:.3f}")
        else:
            print(f"\n良好: 所有组间相关性均低于 {high_corr_threshold}")
            