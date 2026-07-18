"""
报告生成器 (Report Generator)

将因子分析结果输出为结构化 JSON 报告，供 Web 后端读取。
同时负责将分析结果中的 DataFrame/Series 转换为可序列化格式。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_analysis.config import FactorAnalysisConfig
from factor_analysis.panel import FactorPanel


# ── 序列化辅助 ───────────────────────────────────────────────────────────────


class _FactorReportEncoder(json.JSONEncoder):
    """自定义 JSON 编码器：处理 numpy/pandas/Timestamp 类型。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        if isinstance(obj, (date,)):
            return obj.isoformat()
        if isinstance(obj, (pd.Series,)):
            return obj.to_dict()
        if isinstance(obj, (pd.DataFrame,)):
            return obj.to_dict(orient="records")
        return super().default(obj)


def _safe_json(obj: Any) -> Any:
    """将对象递归转换为 JSON 可序列化格式。"""
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (pd.Series,)):
        d = obj.dropna().to_dict()
        return {str(k): (v if not isinstance(v, float) or not np.isnan(v) else None) for k, v in d.items()}
    if isinstance(obj, (pd.DataFrame,)):
        return obj.where(pd.notna(obj), None).to_dict(orient="records")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (pd.Timestamp, date)):
        return obj.isoformat()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _strip_predictive_for_json(predictive: dict | None) -> dict | None:
    """裁剪 Layer 2 数据，去掉逐日时序序列，仅保留汇总指标用于 JSON。

    - rank_ic / pearson_ic / top_ic: 保留 summary，删掉 ic_series
    - rolling_ic / top_rolling_ic: 整体删除
    - ic_decay / top_ic_decay: 保留（已是汇总表）
    - param_grid: 保留
    """
    if predictive is None:
        return None
    stripped: dict[str, Any] = {}
    for key, value in predictive.items():
        if key in ("rank_ic", "pearson_ic", "top_ic"):
            stripped[key] = {
                str(period): {"summary": per["summary"]}
                for period, per in value.items()
                if isinstance(per, dict) and "summary" in per
            }
        elif key in ("rolling_ic", "top_rolling_ic"):
            continue
        else:
            stripped[key] = value
    return stripped


def _strip_grouping_for_json(grouping: dict | None) -> dict | None:
    """裁剪 Layer 3 数据，去掉逐日时序矩阵，仅保留汇总指标用于 JSON。

    - quantile_returns: 删除（已有 CSV 落盘）
    - quantile_cumret: 删除
    - longshort: 删除 ls_series（时序），保留标量指标
    - quantile_summary / monotonicity: 保留
    """
    if grouping is None:
        return None
    stripped: dict[str, Any] = {}
    for period, gr in grouping.items():
        per_stripped: dict[str, Any] = {}
        for key, value in gr.items():
            if key in ("quantile_returns", "quantile_cumret"):
                continue
            elif key == "longshort":
                per_stripped[key] = {
                    k: v for k, v in value.items() if k != "ls_series"
                }
            else:
                per_stripped[key] = value
        stripped[str(period)] = per_stripped
    return stripped


# ── 报告保存 ─────────────────────────────────────────────────────────────────


def generate_and_save_reports(
    panel: FactorPanel,
    quality_results: dict | None,
    predictive_results: dict | None,
    grouping_results: dict | None,
    config: FactorAnalysisConfig,
) -> Path:
    """生成 JSON 报告，保存到 Linux 端输出目录。

    Parameters
    ----------
    panel: 因子面板。
    quality_results / predictive_results / grouping_results: 各层结果。
    config: 分析配置。

    Returns
    -------
    Path
        json_path
    """
    output_root = config.resolve_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    output_date = config.resolve_output_date()

    json_data = _safe_json({
        "meta": {
            "factor_name": panel.factor_name,
            "factor_type": type(config.factor).__name__,
            "factor_params": config.factor.params,
            "analysis_date": output_date,
        },
        "panel_summary": panel.summary(),
        "layer1_quality": quality_results,
        "layer2_predictive": _strip_predictive_for_json(predictive_results),
        "layer3_grouping": _strip_grouping_for_json(grouping_results),
    })

    json_path = output_root / f"report_{output_date}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, cls=_FactorReportEncoder)

    return json_path


def copy_to_windows(
    linux_root: Path,
    windows_root: Path | None,
) -> bool:
    """将 Linux 端输出目录完整复制到 Windows 端。

    Returns
    -------
    bool
        True 如果复制成功，False 如果 Windows 目录为空或复制失败。
    """
    if windows_root is None:
        return False

    import shutil

    try:
        windows_root.mkdir(parents=True, exist_ok=True)
        for item in linux_root.iterdir():
            if item.is_file():
                shutil.copy2(item, windows_root / item.name)
        return True
    except Exception:
        return False
