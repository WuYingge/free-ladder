from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass(slots=True)
class LowCorrelationSelectorConfig:
    target_size: int = 15
    cluster_count: int | None = None
    min_cluster_count: int = 10
    max_per_cluster: int = 2
    max_per_family: int | None = None
    linkage_method: str = "ward"
    min_assets: int = 2

    def resolve_cluster_count(self, asset_count: int) -> int:
        if asset_count < self.min_assets:
            raise ValueError(
                f"At least {self.min_assets} assets are required, got {asset_count}."
            )
        if self.target_size <= 0:
            raise ValueError(f"target_size must be > 0, got {self.target_size}")
        if self.max_per_cluster <= 0:
            raise ValueError(
                f"max_per_cluster must be > 0, got {self.max_per_cluster}"
            )
        if self.max_per_family is not None and self.max_per_family <= 0:
            raise ValueError(
                f"max_per_family must be > 0, got {self.max_per_family}"
            )

        if self.cluster_count is not None:
            cluster_count = int(self.cluster_count)
        else:
            cluster_count = max(self.min_cluster_count, int(round(self.target_size * 0.8)))
        return max(2, min(asset_count, cluster_count))


@dataclass(slots=True)
class ValidationReport:
    asset_count: int
    symmetry_error: float
    diagonal_error: float
    min_correlation: float
    max_correlation: float


@dataclass(slots=True)
class SelectionResult:
    selected_symbols: list[str]
    selected_frame: pd.DataFrame
    metrics: dict[str, float]
    cluster_membership: pd.Series
    cluster_summary: pd.DataFrame
    validation_report: ValidationReport


def load_named_correlation_matrix(csv_path: str | Path) -> pd.DataFrame:
    matrix = pd.read_csv(csv_path, index_col=0)
    matrix.index = matrix.index.map(str)
    matrix.columns = matrix.columns.map(str)
    return matrix


def validate_and_prepare_correlation_matrix(
    corr_matrix: pd.DataFrame,
    *,
    diagonal_tolerance: float = 1e-8,
    symmetry_tolerance: float = 1e-8,
) -> tuple[pd.DataFrame, ValidationReport]:
    if corr_matrix.empty:
        raise ValueError("Correlation matrix is empty.")
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError(
            f"Correlation matrix must be square, got shape={corr_matrix.shape}."
        )
    if corr_matrix.index.has_duplicates:
        duplicates = corr_matrix.index[corr_matrix.index.duplicated()].tolist()
        raise ValueError(f"Row labels contain duplicates: {duplicates}")
    if corr_matrix.columns.has_duplicates:
        duplicates = corr_matrix.columns[corr_matrix.columns.duplicated()].tolist()
        raise ValueError(f"Column labels contain duplicates: {duplicates}")
    if corr_matrix.index.tolist() != corr_matrix.columns.tolist():
        raise ValueError("Row labels and column labels must match in the same order.")

    numeric = corr_matrix.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        nan_locations = np.argwhere(numeric.isna().to_numpy())
        first_nan = nan_locations[0]
        row_name = numeric.index[int(first_nan[0])]
        col_name = numeric.columns[int(first_nan[1])]
        raise ValueError(f"Correlation matrix contains NaN at ({row_name}, {col_name}).")

    numeric = numeric.clip(lower=-1.0, upper=1.0)
    symmetry_error = float(np.abs(numeric.to_numpy() - numeric.to_numpy().T).max())
    if symmetry_error > symmetry_tolerance:
        raise ValueError(
            f"Correlation matrix must be symmetric. max_error={symmetry_error:.6g}"
        )

    prepared = (numeric + numeric.T) / 2.0
    diagonal = np.diag(prepared.to_numpy())
    diagonal_error = float(np.abs(diagonal - 1.0).max())
    if diagonal_error > diagonal_tolerance:
        raise ValueError(
            f"Correlation matrix diagonal must be 1. max_error={diagonal_error:.6g}"
        )
    np.fill_diagonal(prepared.to_numpy(), 1.0)

    report = ValidationReport(
        asset_count=int(prepared.shape[0]),
        symmetry_error=symmetry_error,
        diagonal_error=diagonal_error,
        min_correlation=float(prepared.to_numpy().min()),
        max_correlation=float(prepared.to_numpy().max()),
    )
    return prepared, report


def correlation_to_distance_matrix(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    clipped_corr = corr_matrix.clip(lower=-1.0, upper=1.0)
    dist_values = np.sqrt((1.0 - clipped_corr).clip(lower=0.0))
    dist_matrix = pd.DataFrame(
        dist_values,
        index=corr_matrix.index,
        columns=corr_matrix.columns,
    )
    np.fill_diagonal(dist_matrix.to_numpy(), 0.0)
    return dist_matrix


def cluster_from_correlation_matrix(
    corr_matrix: pd.DataFrame,
    *,
    cluster_count: int,
    linkage_method: str = "ward",
) -> pd.Series:
    if cluster_count <= 1:
        raise ValueError(f"cluster_count must be > 1, got {cluster_count}")

    dist_matrix = correlation_to_distance_matrix(corr_matrix)
    condensed = squareform(dist_matrix.to_numpy(), checks=False)
    linked = linkage(condensed, method=linkage_method)
    labels = fcluster(linked, t=cluster_count, criterion="maxclust") - 1
    return pd.Series(labels, index=corr_matrix.index, name="cluster_label")


def build_cluster_summary(
    corr_matrix: pd.DataFrame,
    cluster_membership: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    universe_avg_corr = _average_correlation_to_frame(corr_matrix, corr_matrix)

    for cluster_label in sorted(cluster_membership.unique().tolist()):
        members = cluster_membership[cluster_membership == cluster_label].index.tolist()
        cluster_corr = corr_matrix.loc[members, members]
        intra_avg_corr = _average_correlation_to_frame(
            corr_matrix.loc[members],
            cluster_corr,
            drop_self=True,
        )

        for symbol in members:
            rows.append(
                {
                    "symbol": symbol,
                    "cluster_label": int(cluster_label),
                    "cluster_size": int(len(members)),
                    "family_label": classify_symbol_family(symbol),
                    "avg_corr_to_universe": float(universe_avg_corr.loc[symbol]),
                    "avg_corr_in_cluster": float(intra_avg_corr.loc[symbol]),
                }
            )

    summary = pd.DataFrame(rows).set_index("symbol")
    summary["selection_score"] = (
        0.65 * summary["avg_corr_in_cluster"] + 0.35 * summary["avg_corr_to_universe"]
    )
    summary = summary.sort_values(
        ["cluster_label", "selection_score", "avg_corr_to_universe"],
        ascending=[True, True, True],
    )
    summary["cluster_rank"] = (
        summary.groupby("cluster_label").cumcount() + 1
    ).astype(int)
    return summary


def select_low_correlation_portfolio(
    corr_matrix: pd.DataFrame,
    config: LowCorrelationSelectorConfig | None = None,
) -> SelectionResult:
    selector_config = config or LowCorrelationSelectorConfig()
    prepared, validation_report = validate_and_prepare_correlation_matrix(corr_matrix)
    cluster_count = selector_config.resolve_cluster_count(len(prepared))
    cluster_membership = cluster_from_correlation_matrix(
        prepared,
        cluster_count=cluster_count,
        linkage_method=selector_config.linkage_method,
    )
    cluster_summary = build_cluster_summary(prepared, cluster_membership)
    selected_symbols = _greedy_select(
        prepared,
        cluster_summary,
        target_size=min(selector_config.target_size, len(prepared)),
        max_per_cluster=selector_config.max_per_cluster,
        max_per_family=selector_config.max_per_family,
    )
    selected_frame = build_selection_frame(prepared, cluster_summary, selected_symbols)
    metrics = summarize_portfolio_metrics(prepared.loc[selected_symbols, selected_symbols])

    return SelectionResult(
        selected_symbols=selected_symbols,
        selected_frame=selected_frame,
        metrics=metrics,
        cluster_membership=cluster_membership,
        cluster_summary=cluster_summary,
        validation_report=validation_report,
    )


def build_selection_frame(
    corr_matrix: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    selected_symbols: list[str],
    *,
    substitute_count: int = 3,
) -> pd.DataFrame:
    selected_corr = corr_matrix.loc[selected_symbols, selected_symbols]
    avg_to_pool = _average_correlation_to_frame(
        corr_matrix.loc[selected_symbols],
        selected_corr,
        drop_self=True,
    )

    rows: list[dict[str, Any]] = []
    for order, symbol in enumerate(selected_symbols, start=1):
        cluster_label = int(cluster_summary.loc[symbol, "cluster_label"])
        substitutes = _cluster_substitutes(cluster_summary, symbol, substitute_count)
        rows.append(
            {
                "selection_order": order,
                "symbol": symbol,
                "cluster_label": cluster_label,
                "family_label": str(cluster_summary.loc[symbol, "family_label"]),
                "cluster_rank": int(cluster_summary.loc[symbol, "cluster_rank"]),
                "cluster_size": int(cluster_summary.loc[symbol, "cluster_size"]),
                "avg_corr_to_selected_pool": float(avg_to_pool.loc[symbol]),
                "avg_corr_to_universe": float(cluster_summary.loc[symbol, "avg_corr_to_universe"]),
                "avg_corr_in_cluster": float(cluster_summary.loc[symbol, "avg_corr_in_cluster"]),
                "selection_score": float(cluster_summary.loc[symbol, "selection_score"]),
                "cluster_substitutes": ", ".join(substitutes),
            }
        )

    return pd.DataFrame(rows)


def summarize_portfolio_metrics(corr_matrix: pd.DataFrame) -> dict[str, float]:
    values = corr_matrix.to_numpy(dtype=float)
    upper = values[np.triu_indices(len(corr_matrix), 1)]
    if upper.size == 0:
        avg_pairwise = 0.0
        max_pairwise = 0.0
        min_pairwise = 0.0
    else:
        avg_pairwise = float(upper.mean())
        max_pairwise = float(upper.max())
        min_pairwise = float(upper.min())

    eigenvalues = np.linalg.eigvalsh(values)
    eigen_sum = float(eigenvalues.sum())
    if eigen_sum > 0:
        effective_n = float((eigen_sum**2) / float((eigenvalues**2).sum()))
        dominant_ratio = float(eigenvalues.max() / eigen_sum)
    else:
        effective_n = 0.0
        dominant_ratio = 0.0

    return {
        "asset_count": float(len(corr_matrix)),
        "average_pairwise_correlation": avg_pairwise,
        "max_pairwise_correlation": max_pairwise,
        "min_pairwise_correlation": min_pairwise,
        "effective_diversification_number": effective_n,
        "dominant_eigenvalue_ratio": dominant_ratio,
    }


def _greedy_select(
    corr_matrix: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    *,
    target_size: int,
    max_per_cluster: int,
    max_per_family: int | None,
) -> list[str]:
    if target_size <= 0:
        return []

    ranked = cluster_summary.sort_values(
        ["cluster_rank", "selection_score", "avg_corr_to_universe"],
        ascending=[True, True, True],
    )
    selected: list[str] = []
    cluster_counts: dict[int, int] = {}
    family_counts: dict[str, int] = {}

    first_round_candidates = ranked[ranked["cluster_rank"] == 1].index.tolist()
    seed = _choose_seed(cluster_summary.loc[first_round_candidates])
    selected.append(seed)
    cluster_counts[int(cluster_summary.loc[seed, "cluster_label"])] = 1
    seed_family = str(cluster_summary.loc[seed, "family_label"])
    family_counts[seed_family] = 1

    while len(selected) < target_size:
        candidates = _eligible_candidates(
            cluster_summary,
            selected,
            cluster_counts,
            family_counts,
            max_per_cluster=max_per_cluster,
            max_per_family=max_per_family,
            force_next_rank=1 if not _all_clusters_covered(cluster_summary, cluster_counts) else None,
        )
        if not candidates:
            candidates = _eligible_candidates(
                cluster_summary,
                selected,
                cluster_counts,
                family_counts,
                max_per_cluster=max_per_cluster,
                max_per_family=max_per_family,
                force_next_rank=None,
            )
        if not candidates:
            break

        next_symbol = min(
            candidates,
            key=lambda symbol: (
                _mean_corr_to_selected(corr_matrix, symbol, selected),
                float(cluster_summary.loc[symbol, "selection_score"]),
                float(cluster_summary.loc[symbol, "avg_corr_to_universe"]),
            ),
        )
        selected.append(next_symbol)
        cluster_label = int(cluster_summary.loc[next_symbol, "cluster_label"])
        cluster_counts[cluster_label] = cluster_counts.get(cluster_label, 0) + 1
        family_label = str(cluster_summary.loc[next_symbol, "family_label"])
        family_counts[family_label] = family_counts.get(family_label, 0) + 1

    return selected


def _choose_seed(cluster_candidates: pd.DataFrame) -> str:
    ordered = cluster_candidates.sort_values(
        ["selection_score", "avg_corr_to_universe", "avg_corr_in_cluster"],
        ascending=[True, True, True],
    )
    return str(ordered.index[0])


def _eligible_candidates(
    cluster_summary: pd.DataFrame,
    selected: list[str],
    cluster_counts: dict[int, int],
    family_counts: dict[str, int],
    *,
    max_per_cluster: int,
    max_per_family: int | None,
    force_next_rank: int | None,
) -> list[str]:
    selected_set = set(selected)
    candidates: list[str] = []
    for symbol, row in cluster_summary.iterrows():
        if symbol in selected_set:
            continue
        cluster_label = int(row["cluster_label"])
        current_count = cluster_counts.get(cluster_label, 0)
        if current_count >= max_per_cluster:
            continue
        family_label = str(row["family_label"])
        family_count = family_counts.get(family_label, 0)
        if max_per_family is not None and family_count >= max_per_family:
            continue
        if force_next_rank is not None and current_count == 0 and int(row["cluster_rank"]) != force_next_rank:
            continue
        candidates.append(str(symbol))
    return candidates


def classify_symbol_family(symbol: str) -> str:
    bond_keywords = ("国债", "债ETF", "地方债", "国开债", "短融", "公司债", "可转债")
    commodity_keywords = ("黄金", "石油", "油气", "豆粕", "粮食", "养殖", "矿业", "钢铁", "化工", "有色")
    overseas_keywords = (
        "纳指", "标普", "道琼斯", "德国", "法国", "日本", "东证", "日经", "亚太", "东南亚", "沙特", "中概", "恒生", "港股通", "香港", "中韩"
    )
    finance_keywords = ("银行", "证券", "非银", "金融", "保险")
    tech_keywords = ("半导体", "通信", "电信", "消费电子", "信息技术", "创新", "金融科技", "软件", "卫星")
    healthcare_keywords = ("医药", "医疗", "生物")
    consumption_keywords = ("消费", "酒", "旅游", "汽车", "家电", "教育")
    industry_keywords = ("基建", "建材", "电力", "电网", "工程机械", "交通运输", "长江保护", "房地产")

    if any(keyword in symbol for keyword in bond_keywords):
        return "bond"
    if any(keyword in symbol for keyword in commodity_keywords):
        return "commodity"
    if any(keyword in symbol for keyword in healthcare_keywords):
        return "healthcare"
    if any(keyword in symbol for keyword in tech_keywords):
        return "technology"
    if any(keyword in symbol for keyword in finance_keywords):
        return "finance"
    if any(keyword in symbol for keyword in consumption_keywords):
        return "consumption"
    if any(keyword in symbol for keyword in industry_keywords):
        return "industry"
    if any(keyword in symbol for keyword in overseas_keywords):
        return "overseas"
    return "diversified"


def _all_clusters_covered(
    cluster_summary: pd.DataFrame,
    cluster_counts: dict[int, int],
) -> bool:
    all_clusters = set(cluster_summary["cluster_label"].astype(int).tolist())
    return all_clusters.issubset(set(cluster_counts))


def _mean_corr_to_selected(
    corr_matrix: pd.DataFrame,
    symbol: str,
    selected_symbols: list[str],
) -> float:
    if not selected_symbols:
        return float(corr_matrix.loc[symbol].mean())
    return float(corr_matrix.loc[symbol, selected_symbols].mean())


def _cluster_substitutes(
    cluster_summary: pd.DataFrame,
    symbol: str,
    substitute_count: int,
) -> list[str]:
    cluster_label = int(cluster_summary.loc[symbol, "cluster_label"])
    same_cluster = cluster_summary[cluster_summary["cluster_label"] == cluster_label]
    substitutes = [candidate for candidate in same_cluster.index.tolist() if candidate != symbol]
    return substitutes[:substitute_count]


def _average_correlation_to_frame(
    row_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    drop_self: bool = False,
) -> pd.Series:
    result: dict[str, float] = {}
    target_columns = target_frame.columns.tolist()
    for symbol in row_frame.index.tolist():
        values = row_frame.loc[symbol, target_columns].astype(float)
        if drop_self and symbol in values.index:
            values = values.drop(index=symbol)
        result[str(symbol)] = float(values.mean()) if len(values) > 0 else 0.0
    return pd.Series(result)


__all__ = [
    "LowCorrelationSelectorConfig",
    "SelectionResult",
    "ValidationReport",
    "build_cluster_summary",
    "build_selection_frame",
    "classify_symbol_family",
    "cluster_from_correlation_matrix",
    "correlation_to_distance_matrix",
    "load_named_correlation_matrix",
    "select_low_correlation_portfolio",
    "summarize_portfolio_metrics",
    "validate_and_prepare_correlation_matrix",
]