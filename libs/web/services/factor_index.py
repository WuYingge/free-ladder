"""
因子索引服务

- 构建：遍历 data/factors/*/report_*.json，提取关键指标 → data/factors/_index.json
- 查询：从索引文件中按筛选条件返回因子列表
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from config import DataPath

FACTORS_ROOT = Path(DataPath.DATA_DIR) / "factors"
INDEX_FILE = FACTORS_ROOT / "_index.json"


def _load_report_json(report_path: Path) -> dict | None:
    """安全加载 report.json。"""
    try:
        with open(report_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_ic_stats(rank_ic: dict, period: str) -> dict[str, Any]:
    """从 rank_ic/pearson_ic 的 period 中提取统计量。"""
    entry = rank_ic.get(period, {})
    summary = entry.get("summary", {})
    return {
        "mean": summary.get("mean"),
        "std": summary.get("std"),
        "ir": summary.get("ir"),
        "t_stat": summary.get("t_stat"),
        "pos_ratio": summary.get("pos_ratio"),
    }


def _extract_longshort_sharpe(grouping: dict, period: str = "5") -> float | None:
    """从 layer3_grouping 中提取 longshort sharpe。"""
    entry = grouping.get(period, {})
    longshort = entry.get("longshort", {})
    return longshort.get("sharpe")


def build_index(force: bool = False) -> dict:
    """构建因子索引。

    遍历 data/factors/ 下所有子目录，读取最新的 report_*.json，
    提取关键指标写入 FACTORS_ROOT/_index.json。

    Parameters
    ----------
    force: 是否强制重建（默认仅在索引不存在时构建）

    Returns
    -------
    dict: {"factors": [...], "generated_at": "..."}
    """
    if not force and INDEX_FILE.exists():
        with open(INDEX_FILE, encoding="utf-8") as f:
            return json.load(f)

    factors_data: list[dict[str, Any]] = []

    for entry in sorted(os.listdir(FACTORS_ROOT)):
        dir_path = FACTORS_ROOT / entry
        if not dir_path.is_dir() or entry.startswith(".") or entry.startswith("_"):
            continue

        # 找最新的 report_*.json
        reports = sorted(dir_path.glob("report_*.json"), reverse=True)
        if not reports:
            continue
        report = _load_report_json(reports[0])
        if report is None:
            continue

        meta = report.get("meta", {})
        panel = report.get("panel_summary", {})
        predictive = report.get("layer2_predictive", {})
        grouping = report.get("layer3_grouping", {})

        rank_ic = predictive.get("rank_ic", {})
        pearson_ic = predictive.get("pearson_ic", {})

        # 提取各周期 IC
        ic_rank = {p: _extract_ic_stats(rank_ic, p) for p in ("5", "10", "20", "60")}
        ic_pearson = {p: _extract_ic_stats(pearson_ic, p) for p in ("5", "10", "20", "60")}

        factor_entry = {
            "name": meta.get("factor_name", entry),
            "factor_type": meta.get("factor_type", ""),
            "params": meta.get("factor_params", {}),
            "n_symbols": panel.get("n_symbols", 0),
            "start_date": str(panel.get("start_date", "")),
            "end_date": str(panel.get("end_date", "")),
            "coverage_mean": panel.get("coverage_mean", 0),
            "ic_rank": ic_rank,
            "ic_pearson": ic_pearson,
            "longshort_sharpe_5d": _extract_longshort_sharpe(grouping, "5"),
            "analysis_date": meta.get("analysis_date", ""),
        }
        factors_data.append(factor_entry)

    index = {
        "generated_at": date.today().isoformat(),
        "n_factors": len(factors_data),
        "factors": factors_data,
    }

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return index


def load_index() -> dict:
    """加载因子索引。若不存在则自动构建。"""
    return build_index(force=False)


def _safe_float(v: Any) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return -999.0
    return float(v)


def query_factors(
    ic_period: int = 10,
    ic_type: str = "rank_ic",
    ic_min: float | None = None,
    ic_max: float | None = None,
    icir_min: float | None = None,
    icir_max: float | None = None,
    search: str | None = None,
    sort_by: str = "icir",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """查询因子列表（带筛选、排序、分页）。"""
    index = load_index()
    factors = index["factors"]

    ic_key = "ic_rank" if ic_type == "rank_ic" else "ic_pearson"
    period = str(ic_period)

    # 筛选
    filtered: list[dict] = []
    for f in factors:
        ic = f.get(ic_key, {}).get(period, {})
        ic_mean = ic.get("mean")
        ic_ir = ic.get("ir")

        if ic_mean is None and ic_ir is None:
            continue

        if ic_min is not None and _safe_float(ic_mean) < ic_min:
            continue
        if ic_max is not None and _safe_float(ic_mean) > ic_max:
            continue
        if icir_min is not None and _safe_float(ic_ir) < icir_min:
            continue
        if icir_max is not None and _safe_float(ic_ir) > icir_max:
            continue

        if search and search.lower() not in f["name"].lower():
            continue

        filtered.append(f)

    # 排序
    def _sort_key(f: dict) -> float:
        if sort_by == "icir":
            return _safe_float(f.get(ic_key, {}).get(period, {}).get("ir"))
        elif sort_by == "ic_mean":
            return _safe_float(f.get(ic_key, {}).get(period, {}).get("mean"))
        elif sort_by == "sharpe":
            return _safe_float(f.get("longshort_sharpe_5d"))
        elif sort_by == "coverage":
            return _safe_float(f.get("coverage_mean"))
        return 0.0

    reverse = sort_order == "desc"
    filtered.sort(key=_sort_key, reverse=reverse)

    # 分页
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = filtered[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "factors": page_items,
    }
