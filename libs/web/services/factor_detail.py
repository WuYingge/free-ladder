"""
因子详情服务 — 读取单个因子的完整 report.json 数据
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import DataPath

FACTORS_ROOT = Path(DataPath.DATA_DIR) / "factors"


def load_factor_detail(factor_name: str) -> dict[str, Any] | None:
    """加载单个因子的完整报告数据。

    只返回用户关注的部分：meta, panel_summary, layer2_predictive, layer3_grouping。
    layer1_quality（覆盖率等）不返回。
    """
    dir_path = FACTORS_ROOT / factor_name
    if not dir_path.is_dir():
        return None

    # 找最新的 report_*.json
    reports = sorted(dir_path.glob("report_*.json"), reverse=True)
    if not reports:
        return None

    try:
        with open(reports[0], encoding="utf-8") as f:
            full = json.load(f)
    except Exception:
        return None

    # 提取用户关心的字段
    result: dict[str, Any] = {
        "meta": full.get("meta", {}),
        "panel_summary": full.get("panel_summary", {}),
        "layer2_predictive": full.get("layer2_predictive", {}),
        "layer3_grouping": full.get("layer3_grouping", {}),
    }

    # 从 layer1_quality 中提取分布统计（用户关心的统计分布）
    layer1 = full.get("layer1_quality", {})
    daily_pct = layer1.get("daily_percentiles", {})
    dist_stats = layer1.get("distribution_stats", {})
    result["distribution"] = {
        "daily_percentiles": _extract_percentile_summary(daily_pct),
        "stats": dist_stats,
    }

    return result


def _extract_percentile_summary(daily_percentiles: list | dict) -> dict | None:
    """从每日分位数数据提取摘要（所有日期的分位数均值）。

    支持两种格式：
    - list of dicts: [{P5: val, P25: val, ...}, ...]
    - dict of dicts: {date_str: {p10: val, ...}, ...}
    """
    if not daily_percentiles:
        return None
    import numpy as np

    if isinstance(daily_percentiles, list):
        keys = ["P5", "P25", "P50", "P75", "P95"]
        summary = {}
        for k in keys:
            vals = []
            for pct_dict in daily_percentiles:
                v = pct_dict.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
            summary[k.lower()] = float(np.mean(vals)) if vals else None
        return summary
    else:
        # dict 格式
        keys = ["p10", "p25", "p50", "p75", "p90"]
        summary = {}
        for k in keys:
            vals = []
            for date_key, pct_dict in daily_percentiles.items():
                v = pct_dict.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
            summary[k] = float(np.mean(vals)) if vals else None
        return summary


def get_factor_report_path(factor_name: str) -> Path | None:
    """获取某个因子最新 report.json 的路径。"""
    dir_path = FACTORS_ROOT / factor_name
    if not dir_path.is_dir():
        return None
    reports = sorted(dir_path.glob("report_*.json"), reverse=True)
    return reports[0] if reports else None
