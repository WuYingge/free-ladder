"""
因子相关性查询服务

从 data/factor_cross_correlation/matrix.csv 中读取相关性矩阵，
支持查询某个因子与其他因子的 Spearman rank 相关性。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import DataPath

# 相关性数据存储位置
CORR_DIR = Path(DataPath.DATA_DIR) / "factor_cross_correlation"
MATRIX_FILE = CORR_DIR / "matrix.csv"
METADATA_FILE = CORR_DIR / "metadata.json"


def _ensure_data_migrated() -> Path:
    """确保相关性数据在 data/ 下；首次调用时从 output/ 迁移。"""
    if MATRIX_FILE.exists() and METADATA_FILE.exists():
        return CORR_DIR

    old_dir = Path("output/factor_cross_correlation")
    if old_dir.exists():
        import shutil
        CORR_DIR.mkdir(parents=True, exist_ok=True)
        for f in old_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, CORR_DIR / f.name)
    return CORR_DIR


def load_correlation_matrix() -> dict[str, dict[str, float]]:
    """加载相关性矩阵。

    Returns
    -------
    dict: {factor_name: {other_factor: correlation, ...}}
    """
    _ensure_data_migrated()

    if not MATRIX_FILE.exists():
        return {}

    import csv
    matrix: dict[str, dict[str, float]] = {}
    with open(MATRIX_FILE, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        factor_names = [h.strip() for h in header[1:]]  # 第一列为空

        for row in reader:
            if not row:
                continue
            factor = row[0].strip()
            corr_dict = {}
            for i, val in enumerate(row[1:]):
                if i < len(factor_names):
                    try:
                        corr_dict[factor_names[i]] = float(val)
                    except (ValueError, IndexError):
                        pass
            matrix[factor] = corr_dict

    return matrix


def query_correlation(
    factor_name: str,
    threshold: float = 0.7,
    direction: str = "above",
    limit: int = 50,
) -> dict[str, Any]:
    """查询与指定因子相关/不相关的其他因子。

    Parameters
    ----------
    factor_name: 目标因子名
    threshold: 相关性阈值（绝对值）
    direction: "above" | "below"
    limit: 返回数量上限

    Returns
    -------
    dict: {factor, related: [{factor, correlation}], total_available}
    """
    matrix = load_correlation_matrix()
    if factor_name not in matrix:
        return {"factor": factor_name, "related": [], "total_available": len(matrix)}

    corr_row = matrix[factor_name]
    items: list[tuple[str, float]] = []

    for other, corr in corr_row.items():
        if other == factor_name:
            continue
        if np.isnan(corr):
            continue
        if direction == "above" and abs(corr) >= threshold:
            items.append((other, corr))
        elif direction == "below" and abs(corr) <= threshold:
            items.append((other, corr))

    # 按 |ρ| 降序
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    items = items[:limit]

    return {
        "factor": factor_name,
        "related": [{"factor": f, "correlation": round(c, 4)} for f, c in items],
        "total_available": len(matrix),
    }
