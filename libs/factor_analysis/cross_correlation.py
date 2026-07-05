"""
因子截面相关性分析 (Factor Cross-Sectional Correlation Analysis)

计算 N 个因子两两之间的截面 Spearman rank 相关性矩阵。
包含两级指纹机制：矩阵级指纹（参数变化→全量重算）+
因子级指纹（因子变化→增量更新）。

核心入口:
    compute_or_load_correlation  — 指纹判断 + 缓存/增量/全量 一体化

使用方式:
    >>> from factor_analysis.cross_correlation import (
    ...     compute_or_load_correlation,
    ...     build_cross_corr_fingerprint,
    ... )
    >>>
    >>> # 日常运行（end_date 固定，数据追加不触发重算）
    >>> result = compute_or_load_correlation(
    ...     factor_values,
    ...     "output/factor_cross_correlation",
    ...     end_date="2026-06-30",
    ... )
    >>>
    >>> # 显式推进窗口
    >>> result = compute_or_load_correlation(
    ...     factor_values,
    ...     "output/factor_cross_correlation",
    ...     end_date="2026-07-03",
    ... )
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DataPath


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CrossCorrFingerprint:
    """因子截面相关性计算的输入指纹（两级结构）。

    Attributes
    ----------
    rebalance_interval : int
        调仓间隔（交易日），控制下采样密度。
    min_assets : int
        最小共同标的数，低于此值的日期被跳过。
    start_date : str | None
        起始日期 "YYYY-MM-DD"。
    end_date : str
        结束日期 "YYYY-MM-DD"，作为数据窗口锚点。
    factor_fingerprints : dict[str, str]
        因子名 → 该因子的独立指纹。用于增量判断。
    data_version : str
        底层 ETF 数据文件的版本 hash（基于 end_date + 文件列表）。
    """

    rebalance_interval: int
    min_assets: int
    start_date: str | None
    end_date: str
    factor_fingerprints: dict[str, str]
    data_version: str

    def compute_config_hash(self) -> str:
        """计算矩阵级（配置）指纹 hash。

        包含 rebalance_interval + min_assets + start_date + end_date + data_version。
        这些参数一变，整个矩阵的衍生过程就变了，旧结果无法修补。
        """
        payload = json.dumps(
            {
                "rebalance_interval": self.rebalance_interval,
                "min_assets": self.min_assets,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "data_version": self.data_version,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compute_full_hash(self) -> str:
        """计算完整指纹 hash（配置 + 因子）。"""
        # 因子指纹按 key 排序确保确定性
        sorted_ff = dict(sorted(self.factor_fingerprints.items()))
        payload = json.dumps(
            {
                "config_hash": self.compute_config_hash(),
                "factor_fingerprints": sorted_ff,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "rebalance_interval": self.rebalance_interval,
            "min_assets": self.min_assets,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "data_version": self.data_version,
            "factor_fingerprints": dict(sorted(self.factor_fingerprints.items())),
            "config_hash": self.compute_config_hash(),
            "full_hash": self.compute_full_hash(),
        }


@dataclass
class CrossCorrResult:
    """因子截面相关性计算结果。

    Attributes
    ----------
    factors : list[str]
        因子名列表（聚类后顺序）。
    mean_corr : np.ndarray
        N×N 均值 Spearman ρ 矩阵。
    std_corr : np.ndarray
        N×N 标准差矩阵。
    n_dates : np.ndarray
        N×N 有效日期计数矩阵（int）。
    cluster_labels : list[int] | None
        层次聚类标签。
    cluster_order : list[int] | None
        聚类排序索引。
    fingerprint : CrossCorrFingerprint | None
        本次计算使用的指纹。
    generated_at : str
        ISO 时间戳。
    """

    factors: list[str]
    mean_corr: np.ndarray
    std_corr: np.ndarray
    n_dates: np.ndarray
    cluster_labels: list[int] | None = None
    cluster_order: list[int] | None = None
    fingerprint: CrossCorrFingerprint | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    def mean_corr_df(self) -> pd.DataFrame:
        """返回 mean_corr 的 DataFrame 形式。"""
        return pd.DataFrame(self.mean_corr, index=self.factors, columns=self.factors)

    def std_corr_df(self) -> pd.DataFrame:
        """返回 std_corr 的 DataFrame 形式。"""
        return pd.DataFrame(self.std_corr, index=self.factors, columns=self.factors)

    def counts_df(self) -> pd.DataFrame:
        """返回 n_dates 的 DataFrame 形式。"""
        return pd.DataFrame(self.n_dates, index=self.factors, columns=self.factors)

    def find_highly_correlated(
        self, threshold: float = 0.95
    ) -> list[tuple[str, str, float]]:
        """找出 |ρ| >= threshold 的因子对（上三角去重，按 |ρ| 降序）。"""
        pairs = []
        n = self.n_factors
        for i in range(n):
            for j in range(i + 1, n):
                rho = float(self.mean_corr[i, j])
                if abs(rho) >= threshold:
                    pairs.append((self.factors[i], self.factors[j], rho))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    def find_lowly_correlated(
        self, threshold: float = 0.1
    ) -> list[tuple[str, str, float]]:
        """找出 |ρ| < threshold 的因子对（候选正交组合）。"""
        pairs = []
        n = self.n_factors
        for i in range(n):
            for j in range(i + 1, n):
                rho = float(self.mean_corr[i, j])
                if abs(rho) < threshold:
                    pairs.append((self.factors[i], self.factors[j], rho))
        pairs.sort(key=lambda x: abs(x[2]))
        return pairs

    def summary_text(self) -> str:
        """生成文本摘要。"""
        lines = []
        lines.append(f"因子数: {self.n_factors}")
        lines.append(f"生成时间: {self.generated_at}")

        high = self.find_highly_correlated(0.95)
        lines.append(f"\n高度冗余对 (|ρ|≥0.95): {len(high)} 对")
        for a, b, rho in high[:10]:
            lines.append(f"  {a} ↔ {b}: ρ={rho:.4f}")
        if len(high) > 10:
            lines.append(f"  ... 还有 {len(high) - 10} 对")

        low = self.find_lowly_correlated(0.1)
        lines.append(f"\n候选正交对 (|ρ|<0.1): {len(low)} 对")
        for a, b, rho in low[:10]:
            lines.append(f"  {a} ↔ {b}: ρ={rho:.4f}")
        if len(low) > 10:
            lines.append(f"  ... 还有 {len(low) - 10} 对")

        if self.cluster_labels is not None:
            unique_clusters = sorted(set(self.cluster_labels))
            lines.append(f"\n聚类: {len(unique_clusters)} 个簇")
            for cid in unique_clusters:
                members = [
                    self.factors[i]
                    for i, lbl in enumerate(self.cluster_labels)
                    if lbl == cid
                ]
                preview = members[:5]
                suffix = "..." if len(members) > 5 else ""
                lines.append(f"  簇 {cid} ({len(members)} 个): {preview}{suffix}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 指纹计算
# ═══════════════════════════════════════════════════════════════════════════════


def compute_data_version(
    symbols: list[str] | None = None,
    *,
    end_date: str | None = None,
) -> str:
    """计算底层 ETF 数据文件的版本 hash。

    核心原则：只依赖 end_date + ETF 文件列表，不依赖文件 mtime。
    - end_date 不变 → 指纹不变 → 日常数据追加不触发重算
    - end_date 推进 或 新增/删除 ETF 文件 → 指纹变化 → 触发重算

    Parameters
    ----------
    symbols : list[str] | None
        限定只检查这些标的。None = 检查所有。
    end_date : str | None
        数据窗口结束日期。None = 使用 "latest"。

    Returns
    -------
    str
        40 字符 SHA1 hex digest。
    """
    etf_dir = DataPath.DEFAULT_PATH
    if not os.path.isdir(etf_dir):
        return "no_data_dir"

    existing_files: list[str] = []
    try:
        for fname in sorted(os.listdir(etf_dir)):
            if not fname.endswith(".csv"):
                continue
            symbol = fname.replace(".csv", "")
            if symbols is not None and symbol not in symbols:
                continue
            existing_files.append(symbol)
    except FileNotFoundError:
        return "data_dir_missing"

    if not existing_files:
        return "no_data_files"

    existing_files.sort()
    payload = {
        "end_date": end_date or "latest",
        "n_etf_files": len(existing_files),
        "file_list_hash": hashlib.sha1(
            ",".join(existing_files).encode("utf-8")
        ).hexdigest()[:16],
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_cross_corr_fingerprint(
    factor_names: list[str],
    factor_fingerprints: dict[str, str],
    *,
    rebalance_interval: int = 5,
    min_assets: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | None = None,
) -> CrossCorrFingerprint:
    """构建完整的两级指纹。

    Parameters
    ----------
    factor_names : list[str]
        因子名列表（有序）。
    factor_fingerprints : dict[str, str]
        每个因子的独立指纹。key 必须在 factor_names 中。
    rebalance_interval : int
    min_assets : int
    start_date : str | None
    end_date : str | None
        结束日期。None 时取昨天。
    symbols : list[str] | None
        限定 data_version 的文件范围。

    Returns
    -------
    CrossCorrFingerprint
    """
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()

    # 验证 factor_fingerprints 覆盖所有 factor_names
    missing = set(factor_names) - set(factor_fingerprints.keys())
    if missing:
        raise ValueError(
            f"factor_fingerprints 缺少以下因子的指纹: {sorted(missing)}"
        )

    # 只保留 factor_names 中的因子指纹（按传入顺序）
    ordered_ff = {fname: factor_fingerprints[fname] for fname in factor_names}

    data_version = compute_data_version(symbols, end_date=end_date)

    return CrossCorrFingerprint(
        rebalance_interval=rebalance_interval,
        min_assets=min_assets,
        start_date=start_date,
        end_date=end_date,
        factor_fingerprints=ordered_ff,
        data_version=data_version,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 核心计算
# ═══════════════════════════════════════════════════════════════════════════════


def compute_cross_sectional_corr_matrix(
    factor_values: dict[str, pd.DataFrame],
    *,
    rebalance_interval: int = 5,
    min_assets: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    compute_clusters: bool = True,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    verbose: bool = True,
) -> CrossCorrResult:
    """计算因子截面 Spearman rank 相关性矩阵。

    对每个调仓日，取出当天所有 ETF 的 N 个因子值，
    逐对计算截面 Spearman rank correlation（沿 ETF 维度），
    然后沿时间维度汇总均值、标准差和有效日期计数。

    Parameters
    ----------
    factor_values : dict[str, pd.DataFrame]
        因子名 → date (DatetimeIndex) × symbol (columns) 的因子值矩阵。
    rebalance_interval : int
        调仓间隔（交易日），默认 5。
    min_assets : int
        最小共同标的数，默认 30。
    start_date : str | None
    end_date : str | None
    compute_clusters : bool
        是否做层次聚类。默认 True。
    n_clusters : int | None
        簇数。None = 自动。
    distance_threshold : float | None
        距离阈值。None = 默认 0.3（对应 |ρ|≈0.7）。
    verbose : bool

    Returns
    -------
    CrossCorrResult
    """
    if not factor_values:
        raise ValueError("factor_values 不能为空")

    factor_names = list(factor_values.keys())
    n = len(factor_names)

    # ── 1. 对齐日期 & 按 rebalance_interval 采样 ──────────────────────────
    rebalance_dates, aligned_fv = _align_and_sample(
        factor_values,
        rebalance_interval=rebalance_interval,
        start_date=start_date,
        end_date=end_date,
    )

    if len(rebalance_dates) == 0:
        raise ValueError(
            f"过滤后无可用日期（start={start_date}, end={end_date}）"
        )

    # 统一列：所有 factor 的 columns 并集
    all_symbols = sorted(
        set().union(*(set(fv.columns) for fv in aligned_fv.values()))
    )

    if verbose:
        print(f"因子数: {n}")
        print(f"标的数: {len(all_symbols)}")
        print(f"采样日期数: {len(rebalance_dates)} (间隔={rebalance_interval})")

    # ── 2. 逐日计算截面相关矩阵 ───────────────────────────────────────────
    sum_corr = np.zeros((n, n), dtype=np.float64)
    sum_sq_corr = np.zeros((n, n), dtype=np.float64)
    count_dates = np.zeros((n, n), dtype=np.int32)

    for date_idx, dt in enumerate(rebalance_dates):
        if verbose and (date_idx % 50 == 0 or date_idx == len(rebalance_dates) - 1):
            print(f"  进度: {date_idx + 1}/{len(rebalance_dates)} ({dt.date()})")

        # 提取该日各因子的值向量（对齐到 all_symbols）
        day_values: dict[str, np.ndarray] = {}
        for fname in factor_names:
            fv = aligned_fv[fname]
            if dt not in fv.index:
                day_values[fname] = np.full(len(all_symbols), np.nan)
                continue
            row = fv.loc[dt]
            # 对齐到 all_symbols
            vals = np.full(len(all_symbols), np.nan)
            common_cols = [s for s in all_symbols if s in row.index]
            if common_cols:
                col_idx = [all_symbols.index(s) for s in common_cols]
                vals[col_idx] = pd.to_numeric(row[common_cols], errors="coerce").values
            day_values[fname] = vals

        # 预计算各因子的 valid mask
        masks = {fname: ~np.isnan(v) for fname, v in day_values.items()}

        # 逐对计算
        for i in range(n):
            fi = factor_names[i]
            mi = masks[fi]
            ni_valid = int(mi.sum())
            if ni_valid < min_assets:
                continue

            vi = day_values[fi]

            for j in range(i, n):
                fj = factor_names[j]
                mj = masks[fj]
                nj_valid = int(mj.sum())
                if nj_valid < min_assets:
                    continue

                common_mask = mi & mj
                n_common = int(common_mask.sum())
                if n_common < min_assets:
                    continue

                xi = vi[common_mask]
                xj = day_values[fj][common_mask]

                # 常数序列无法计算相关
                if np.std(xi) < 1e-15 or np.std(xj) < 1e-15:
                    continue

                try:
                    rho, _ = spearmanr(xi, xj)
                except Exception:
                    continue

                if np.isnan(rho):
                    continue

                sum_corr[i, j] += rho
                sum_sq_corr[i, j] += rho * rho
                count_dates[i, j] += 1
                if i != j:
                    sum_corr[j, i] += rho
                    sum_sq_corr[j, i] += rho * rho
                    count_dates[j, i] += 1

    # ── 3. 汇总 ───────────────────────────────────────────────────────────
    safe_counts = np.where(count_dates > 0, count_dates, 1)
    mean_corr = sum_corr / safe_counts
    variance = sum_sq_corr / safe_counts - mean_corr**2
    std_corr = np.sqrt(np.maximum(variance, 0.0))
    std_corr = np.where(count_dates > 1, std_corr, np.nan)

    for i in range(n):
        mean_corr[i, i] = 1.0
        std_corr[i, i] = 0.0

    result = CrossCorrResult(
        factors=list(factor_names),
        mean_corr=mean_corr,
        std_corr=std_corr,
        n_dates=count_dates,
    )

    # ── 4. 层次聚类 ───────────────────────────────────────────────────────
    if compute_clusters and n >= 3:
        _compute_clustering(
            result, n_clusters=n_clusters, distance_threshold=distance_threshold
        )
        _reorder_by_clusters(result)

    return result


def _align_and_sample(
    factor_values: dict[str, pd.DataFrame],
    *,
    rebalance_interval: int,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """对齐日期、按 rebalance_interval 采样、日期范围切片。

    Returns
    -------
    (采样后的公共日期索引, 切片后的 factor_values)
    """
    # 取日期交集
    common_index = None
    for fv in factor_values.values():
        if common_index is None:
            common_index = fv.index
        else:
            common_index = common_index.intersection(fv.index)
    if common_index is None or len(common_index) == 0:
        raise ValueError("所有 factor_values 的日期索引无交集")

    common_dates = common_index.sort_values()

    # 日期范围切片
    if start_date is not None:
        common_dates = common_dates[common_dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        common_dates = common_dates[common_dates <= pd.Timestamp(end_date)]

    # 按 rebalance_interval 下采样
    rebalance_dates = common_dates[::rebalance_interval]

    # 截断各因子数据
    truncated: dict[str, pd.DataFrame] = {}
    for fname, fv in factor_values.items():
        common = fv.index.intersection(rebalance_dates)
        truncated[fname] = fv.loc[common].copy()

    return rebalance_dates, truncated


# ═══════════════════════════════════════════════════════════════════════════════
# 聚类
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_clustering(
    result: CrossCorrResult,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
) -> None:
    """对 mean_corr 矩阵做层次聚类（Ward linkage），原地填充 cluster_labels/order。

    距离 = 1 - |ρ|（正负相关均视为冗余）。
    """
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage

    n = result.n_factors
    dist_matrix = 1.0 - np.abs(result.mean_corr)
    np.fill_diagonal(dist_matrix, 0.0)

    condensed = np.array(
        [float(dist_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]
    )
    Z = linkage(condensed, method="ward")

    if n_clusters is not None:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    else:
        if distance_threshold is None:
            distance_threshold = 0.3
        labels = fcluster(Z, t=distance_threshold, criterion="distance")

    result.cluster_labels = [int(lbl) for lbl in labels]
    try:
        order = leaves_list(Z)
        result.cluster_order = [int(o) for o in order]
    except Exception:
        result.cluster_order = list(range(n))


def _reorder_by_clusters(result: CrossCorrResult) -> None:
    """按聚类顺序重排矩阵。"""
    if result.cluster_order is None:
        return
    order = result.cluster_order
    result.mean_corr = result.mean_corr[np.ix_(order, order)]
    result.std_corr = result.std_corr[np.ix_(order, order)]
    result.n_dates = result.n_dates[np.ix_(order, order)]
    result.factors = [result.factors[i] for i in order]
    if result.cluster_labels is not None:
        result.cluster_labels = [result.cluster_labels[i] for i in order]


# ═══════════════════════════════════════════════════════════════════════════════
# 增量更新
# ═══════════════════════════════════════════════════════════════════════════════


def _incremental_update(
    result_cached: CrossCorrResult,
    fp_cached: CrossCorrFingerprint,
    fp_new: CrossCorrFingerprint,
    factor_values: dict[str, pd.DataFrame],
    *,
    rebalance_interval: int,
    min_assets: int,
    start_date: str | None,
    end_date: str | None,
    verbose: bool,
) -> CrossCorrResult:
    """增量更新：只计算新增因子与已有因子的相关性。

    当前策略：
    - 纯新增因子（无修改/删除）→ 只计算增量部分，保留已有矩阵
    - 有修改/删除因子 → 回退到全量重算
    - 聚类在增量后重新计算

    Returns
    -------
    CrossCorrResult
        更新后的完整结果。
    """
    old_factors = result_cached.factors
    old_fp = fp_cached.factor_fingerprints if fp_cached else {}
    new_fp = fp_new.factor_fingerprints

    old_set = set(old_fp.keys())
    new_set = set(new_fp.keys())

    added = new_set - old_set
    removed = old_set - new_set
    changed = {f for f in old_set & new_set if old_fp.get(f) != new_fp.get(f)}

    if removed or changed:
        if verbose:
            parts = []
            if removed:
                parts.append(f"删除 {len(removed)} 个因子: {sorted(removed)}")
            if changed:
                parts.append(f"修改 {len(changed)} 个因子: {sorted(changed)}")
            print(f"检测到非新增变化 ({', '.join(parts)})，回退到全量重算")
        return compute_cross_sectional_corr_matrix(
            factor_values=factor_values,
            rebalance_interval=rebalance_interval,
            min_assets=min_assets,
            start_date=start_date,
            end_date=end_date,
            compute_clusters=True,
            verbose=verbose,
        )

    if not added:
        # 纯因子指纹变化但无新增/删除/修改（不应该到达这里）
        return result_cached

    if verbose:
        print(f"增量模式: 新增 {len(added)} 个因子，已有 {len(old_factors)} 个")
        print(f"  新增: {sorted(added)}")

    # ── 构建新的因子列表：旧因子保持原序 + 新因子追加 ──
    new_factor_list = list(old_factors)
    new_factor_list.extend(sorted(added))
    n_old = len(old_factors)
    n_new = len(new_factor_list)

    # ── 初始化新矩阵 ──
    mean_corr = np.full((n_new, n_new), np.nan)
    std_corr = np.full((n_new, n_new), np.nan)
    n_dates = np.zeros((n_new, n_new), dtype=np.int32)

    # 复制旧矩阵（旧因子 × 旧因子）
    mean_corr[:n_old, :n_old] = result_cached.mean_corr
    std_corr[:n_old, :n_old] = result_cached.std_corr
    n_dates[:n_old, :n_old] = result_cached.n_dates

    # ── 计算新因子 × 所有因子 ──
    new_names = sorted(added)
    block_mean, block_std, block_counts = _compute_new_vs_all(
        new_factor_names=new_names,
        all_factor_names=new_factor_list,
        factor_values=factor_values,
        rebalance_interval=rebalance_interval,
        min_assets=min_assets,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
    )

    # 填入矩阵右下角
    new_start = n_old
    for ni, new_fname in enumerate(new_names):
        row = new_start + ni
        for nj, all_fname in enumerate(new_factor_list):
            col = nj
            mean_corr[row, col] = block_mean[ni, nj]
            std_corr[row, col] = block_std[ni, nj]
            n_dates[row, col] = int(block_counts[ni, nj])
            # 对称填充（除了对角线）
            if row != col:
                mean_corr[col, row] = block_mean[ni, nj]
                std_corr[col, row] = block_std[ni, nj]
                n_dates[col, row] = int(block_counts[ni, nj])

    # 对角线
    for i in range(n_new):
        mean_corr[i, i] = 1.0
        std_corr[i, i] = 0.0

    # ── 重新聚类 ──
    result = CrossCorrResult(
        factors=new_factor_list,
        mean_corr=mean_corr,
        std_corr=std_corr,
        n_dates=n_dates,
    )

    if n_new >= 3:
        _compute_clustering(result)
        _reorder_by_clusters(result)

    return result


def _compute_new_vs_all(
    new_factor_names: list[str],
    all_factor_names: list[str],
    factor_values: dict[str, pd.DataFrame],
    *,
    rebalance_interval: int,
    min_assets: int,
    start_date: str | None,
    end_date: str | None,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算新增因子 × 所有因子 的截面相关性（长方形矩阵）。

    只计算 new × all 对，不计算 all × all 对。

    Returns
    -------
    (mean_corr_block, std_corr_block, n_dates_block)
        形状: (len(new_factor_names), len(all_factor_names))
    """
    n_new = len(new_factor_names)
    n_all = len(all_factor_names)

    if verbose:
        pairs_count = n_new * n_all
        print(f"    计算 {n_new} × {n_all} = {pairs_count} 对...")

    # 预处理日期
    rebalance_dates, aligned_fv = _align_and_sample(
        factor_values,
        rebalance_interval=rebalance_interval,
        start_date=start_date,
        end_date=end_date,
    )

    all_symbols = sorted(
        set().union(*(set(fv.columns) for fv in aligned_fv.values()))
    )

    sum_corr = np.zeros((n_new, n_all), dtype=np.float64)
    sum_sq_corr = np.zeros((n_new, n_all), dtype=np.float64)
    count_dates = np.zeros((n_new, n_all), dtype=np.int32)

    for date_idx, dt in enumerate(rebalance_dates):
        if verbose and date_idx % 100 == 0:
            print(f"      增量日期进度: {date_idx + 1}/{len(rebalance_dates)}")

        # 提取日值
        day_values: dict[str, np.ndarray] = {}
        for fname in all_factor_names:
            fv = aligned_fv.get(fname)
            if fv is None or dt not in fv.index:
                day_values[fname] = np.full(len(all_symbols), np.nan)
                continue
            row = fv.loc[dt]
            vals = np.full(len(all_symbols), np.nan)
            common_cols = [s for s in all_symbols if s in row.index]
            if common_cols:
                col_idx = [all_symbols.index(s) for s in common_cols]
                vals[col_idx] = pd.to_numeric(row[common_cols], errors="coerce").values
            day_values[fname] = vals

        masks = {fname: ~np.isnan(v) for fname, v in day_values.items()}

        # 只计算 new × all
        for ni, new_fname in enumerate(new_factor_names):
            mi = masks.get(new_fname)
            if mi is None or mi.sum() < min_assets:
                continue
            vi = day_values[new_fname]

            for nj, all_fname in enumerate(all_factor_names):
                mj = masks.get(all_fname)
                if mj is None or mj.sum() < min_assets:
                    continue

                common_mask = mi & mj
                n_common = int(common_mask.sum())
                if n_common < min_assets:
                    continue

                xi = vi[common_mask]
                xj = day_values[all_fname][common_mask]

                if np.std(xi) < 1e-15 or np.std(xj) < 1e-15:
                    continue

                try:
                    rho, _ = spearmanr(xi, xj)
                except Exception:
                    continue

                if np.isnan(rho):
                    continue

                sum_corr[ni, nj] += rho
                sum_sq_corr[ni, nj] += rho * rho
                count_dates[ni, nj] += 1

    # 汇总
    safe_counts = np.where(count_dates > 0, count_dates, 1)
    mean_corr = sum_corr / safe_counts
    variance = sum_sq_corr / safe_counts - mean_corr**2
    std_corr = np.sqrt(np.maximum(variance, 0.0))
    std_corr = np.where(count_dates > 1, std_corr, np.nan)

    # 对角线（new 因子对自己）
    for ni, new_fname in enumerate(new_factor_names):
        nj = all_factor_names.index(new_fname)
        mean_corr[ni, nj] = 1.0
        std_corr[ni, nj] = 0.0

    return mean_corr, std_corr, count_dates


# ═══════════════════════════════════════════════════════════════════════════════
# 持久化 / 缓存
# ═══════════════════════════════════════════════════════════════════════════════


def save_correlation_results(
    result: CrossCorrResult,
    fingerprint: CrossCorrFingerprint | None,
    output_dir: str | Path,
    *,
    save_heatmap: bool = True,
) -> dict[str, Path]:
    """保存因子相关性计算结果到磁盘。

    写入文件:
        metadata.json  — 配置指纹 + 因子指纹 + 聚类信息
        matrix.csv     — N×N 均值 Spearman ρ 矩阵
        std.csv        — N×N 标准差矩阵
        counts.csv     — N×N 有效日期计数矩阵
        heatmap.png    — 聚类热力图

    Returns
    -------
    dict[str, Path]
        各输出文件的路径映射。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # ── metadata.json ──
    metadata: dict[str, Any] = {
        "generated_at": result.generated_at,
        "n_factors": result.n_factors,
        "factors": result.factors,
        "fingerprint": fingerprint.to_dict() if fingerprint else None,
        "cluster_labels": result.cluster_labels,
        "cluster_order": result.cluster_order,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    saved["metadata"] = meta_path

    # ── CSV ──
    matrix_path = output_dir / "matrix.csv"
    result.mean_corr_df().to_csv(matrix_path, encoding="utf-8-sig", float_format="%.6f")
    saved["matrix"] = matrix_path

    std_path = output_dir / "std.csv"
    result.std_corr_df().to_csv(std_path, encoding="utf-8-sig", float_format="%.6f")
    saved["std"] = std_path

    counts_path = output_dir / "counts.csv"
    result.counts_df().to_csv(counts_path, encoding="utf-8-sig")
    saved["counts"] = counts_path

    # ── 热力图 ──
    if save_heatmap:
        try:
            heatmap_path = output_dir / "heatmap.png"
            _save_heatmap(result, heatmap_path)
            saved["heatmap"] = heatmap_path
        except Exception as e:
            print(f"警告: 热力图生成失败: {e}")

    return saved


def load_correlation_results(
    output_dir: str | Path,
) -> tuple[CrossCorrResult, CrossCorrFingerprint] | None:
    """从磁盘加载已缓存的计算结果。

    Returns
    -------
    (result, fingerprint) | None
    """
    output_dir = Path(output_dir)
    matrix_path = output_dir / "matrix.csv"
    meta_path = output_dir / "metadata.json"

    if not matrix_path.exists() or not meta_path.exists():
        return None

    try:
        df = pd.read_csv(matrix_path, index_col=0)
        factors = list(df.columns)
        mean_corr = df.values.astype(np.float64)

        std_path = output_dir / "std.csv"
        if std_path.exists():
            std_df = pd.read_csv(std_path, index_col=0)
            std_corr = std_df.values.astype(np.float64)
        else:
            std_corr = np.full_like(mean_corr, np.nan)

        counts_path = output_dir / "counts.csv"
        if counts_path.exists():
            counts_df = pd.read_csv(counts_path, index_col=0)
            n_dates = counts_df.values.astype(np.int32)
        else:
            n_dates = np.zeros_like(mean_corr, dtype=np.int32)

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        fp_dict = metadata.get("fingerprint") or {}
        fingerprint = CrossCorrFingerprint(
            rebalance_interval=fp_dict.get("rebalance_interval", 5),
            min_assets=fp_dict.get("min_assets", 30),
            start_date=fp_dict.get("start_date"),
            end_date=fp_dict.get("end_date", "unknown"),
            factor_fingerprints=fp_dict.get("factor_fingerprints", {}),
            data_version=fp_dict.get("data_version", "unknown"),
        )

        result = CrossCorrResult(
            factors=factors,
            mean_corr=mean_corr,
            std_corr=std_corr,
            n_dates=n_dates,
            cluster_labels=metadata.get("cluster_labels"),
            cluster_order=metadata.get("cluster_order"),
            fingerprint=fingerprint,
            generated_at=metadata.get("generated_at", ""),
        )

        return result, fingerprint

    except Exception:
        return None


def check_fingerprint_match(
    fingerprint: CrossCorrFingerprint,
    output_dir: str | Path,
) -> bool:
    """比较当前指纹与缓存指纹的 config_hash 是否一致。

    Returns
    -------
    bool
        True = 配置参数和数据窗口未变，可以复用缓存结构。
    """
    loaded = load_correlation_results(output_dir)
    if loaded is None:
        return False
    _, cached_fp = loaded
    if cached_fp is None:
        return False
    return fingerprint.compute_config_hash() == cached_fp.compute_config_hash()


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷入口
# ═══════════════════════════════════════════════════════════════════════════════


def compute_or_load_correlation(
    factor_values: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    factor_fingerprints: dict[str, str] | None = None,
    symbols_for_fingerprint: list[str] | None = None,
    rebalance_interval: int = 5,
    min_assets: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    force: bool = False,
    compute_clusters: bool = True,
    verbose: bool = True,
) -> CrossCorrResult:
    """一键式计算因子截面相关性（含指纹判断与增量缓存）。

    流程:
    1. 计算当前输入指纹（配置级 + 因子级）。
    2. 检查缓存:
       - 配置指纹一致 + 因子指纹完全一致 → 缓存命中
       - 配置指纹一致 + 因子指纹有纯新增 → 增量更新
       - 配置指纹不一致 或 有修改/删除 → 全量重算
    3. 保存结果。

    Parameters
    ----------
    factor_values : dict[str, pd.DataFrame]
        因子名 → date × symbol 因子值矩阵。
    output_dir : str | Path
        缓存/输出目录。
    factor_fingerprints : dict[str, str] | None
        每个因子的独立指纹。None 时退化到只用配置指纹（无因子级比较）。
    symbols_for_fingerprint : list[str] | None
        用于 data_version 的标的列表。None = 自动从 factor_values 获取。
    rebalance_interval : int
    min_assets : int
    start_date : str | None
    end_date : str | None
        结束日期。None 时取昨天。日常使用建议显式指定固定值。
    force : bool
        强制重算，跳过所有指纹检查。
    compute_clusters : bool
    verbose : bool

    Returns
    -------
    CrossCorrResult
    """
    factor_names = list(factor_values.keys())

    # 自动推导 symbols
    if symbols_for_fingerprint is None:
        sample_fv = next(iter(factor_values.values()))
        symbols_for_fingerprint = list(sample_fv.columns)

    # 默认 end_date
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()
        if verbose:
            print(f"end_date 未指定，取昨天: {end_date}")

    # 因子指纹
    if factor_fingerprints is None:
        # 退化：每个因子用因子名作为"指纹"
        factor_fingerprints = {fname: fname for fname in factor_names}
        if verbose:
            print(
                "factor_fingerprints 未提供，使用因子名作为退化指纹"
                "（无法检测因子参数变化）"
            )

    # 构建指纹
    fp_new = build_cross_corr_fingerprint(
        factor_names=factor_names,
        factor_fingerprints=factor_fingerprints,
        rebalance_interval=rebalance_interval,
        min_assets=min_assets,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols_for_fingerprint,
    )

    if verbose:
        print(f"当前配置指纹: {fp_new.compute_config_hash()[:16]}...")

    # 强制重算
    if force:
        if verbose:
            print("强制重算模式，跳过指纹检查")
        result = compute_cross_sectional_corr_matrix(
            factor_values=factor_values,
            rebalance_interval=rebalance_interval,
            min_assets=min_assets,
            start_date=start_date,
            end_date=end_date,
            compute_clusters=compute_clusters,
            verbose=verbose,
        )
        result.fingerprint = fp_new
        save_correlation_results(result, fp_new, output_dir)
        return result

    # 检查缓存
    cached = load_correlation_results(output_dir)

    if cached is None:
        if verbose:
            print("无缓存，开始全量计算...")
        result = compute_cross_sectional_corr_matrix(
            factor_values=factor_values,
            rebalance_interval=rebalance_interval,
            min_assets=min_assets,
            start_date=start_date,
            end_date=end_date,
            compute_clusters=compute_clusters,
            verbose=verbose,
        )
        result.fingerprint = fp_new
        save_correlation_results(result, fp_new, output_dir)
        return result

    result_cached, fp_cached = cached

    # 比较配置指纹
    if fp_new.compute_config_hash() != fp_cached.compute_config_hash():
        if verbose:
            print("配置指纹不匹配（参数或数据窗口变化），全量重算...")
            print(f"  旧 end_date: {fp_cached.end_date}")
            print(f"  新 end_date: {fp_new.end_date}")
        result = compute_cross_sectional_corr_matrix(
            factor_values=factor_values,
            rebalance_interval=rebalance_interval,
            min_assets=min_assets,
            start_date=start_date,
            end_date=end_date,
            compute_clusters=compute_clusters,
            verbose=verbose,
        )
        result.fingerprint = fp_new
        save_correlation_results(result, fp_new, output_dir)
        return result

    # 配置指纹一致 → 检查因子级指纹
    if fp_new.factor_fingerprints == fp_cached.factor_fingerprints:
        if verbose:
            print("指纹完全匹配，加载缓存结果")
        return result_cached

    # 因子级指纹不同 → 增量更新
    if verbose:
        print("配置指纹一致，因子指纹有变化，尝试增量更新...")

    result = _incremental_update(
        result_cached=result_cached,
        fp_cached=fp_cached,
        fp_new=fp_new,
        factor_values=factor_values,
        rebalance_interval=rebalance_interval,
        min_assets=min_assets,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
    )

    result.fingerprint = fp_new
    save_correlation_results(result, fp_new, output_dir)

    if verbose:
        high_pairs = result.find_highly_correlated(0.95)
        low_pairs = result.find_lowly_correlated(0.1)
        print(f"  高度冗余对 (|ρ|≥0.95): {len(high_pairs)}")
        print(f"  候选正交对 (|ρ|<0.1): {len(low_pairs)}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════════════════


def _save_heatmap(
    result: CrossCorrResult,
    output_path: Path,
    figsize: tuple[int, int] = (20, 18),
) -> None:
    """绘制并保存聚类热力图（RdBu 色阶）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = result.n_factors
    if n > 100:
        figsize = (24, 22)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        result.mean_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
    )

    fontsize = max(4, min(8, 160 // max(n, 1)))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(result.factors, rotation=90, fontsize=fontsize, ha="center")
    ax.set_yticklabels(result.factors, fontsize=fontsize)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Spearman ρ", fontsize=12)
    ax.set_title(
        f"Factor Cross-Sectional Correlation Matrix (n={n})", fontsize=14
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clustered_heatmap(
    result: CrossCorrResult,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (20, 18),
    show_dendrogram: bool = True,
) -> None:
    """绘制聚类热力图并可选保存。

    如果 show_dendrogram=True 且有聚类信息，会显示顶部和左侧树状图。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    n = result.n_factors

    if show_dendrogram and result.cluster_labels is not None and n >= 3:
        dist_matrix = 1.0 - np.abs(result.mean_corr)
        np.fill_diagonal(dist_matrix, 0.0)
        condensed = np.array(
            [float(dist_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]
        )
        Z = linkage(condensed, method="ward")

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(figsize[0] + 2, figsize[1]))
        gs = GridSpec(
            2,
            2,
            width_ratios=[1, 5],
            height_ratios=[1, 5],
            left=0.05,
            right=0.95,
            bottom=0.05,
            top=0.95,
            wspace=0.02,
            hspace=0.02,
        )

        ax_top = fig.add_subplot(gs[0, 1])
        dendrogram(
            Z, ax=ax_top, orientation="top", labels=None,
            link_color_func=lambda k: "black",
        )
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for spine in ax_top.spines.values():
            spine.set_visible(False)

        ax_left = fig.add_subplot(gs[1, 0])
        dendrogram(
            Z, ax=ax_left, orientation="left", labels=None,
            link_color_func=lambda k: "black",
        )
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        for spine in ax_left.spines.values():
            spine.set_visible(False)

        ax_hm = fig.add_subplot(gs[1, 1])
        im = ax_hm.imshow(
            result.mean_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
        )
        fontsize = max(4, min(8, 160 // max(n, 1)))
        ax_hm.set_xticks(range(n))
        ax_hm.set_yticks(range(n))
        ax_hm.set_xticklabels(
            result.factors, rotation=90, fontsize=fontsize, ha="center"
        )
        ax_hm.set_yticklabels(result.factors, fontsize=fontsize)

        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Mean Spearman ρ", fontsize=12)
        fig.suptitle(
            f"Factor Cross-Sectional Correlation Matrix (n={n})",
            fontsize=14,
            y=0.98,
        )
    else:
        _save_heatmap(result, output_path or Path("/dev/null"), figsize=figsize)
        return

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
