"""
因子相关路由 — 因子列表、详情、相关性查询
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from services.factor_index import query_factors, load_index, build_index
from services.factor_detail import load_factor_detail
from services.correlation_svc import query_correlation

router = APIRouter(prefix="/api/factors", tags=["factors"])


@router.get("")
def list_factors(
    ic_period: int = Query(10, ge=5, le=60, description="IC 周期"),
    ic_type: str = Query("rank_ic", pattern="^(rank_ic|pearson_ic)$"),
    ic_min: float | None = Query(None, description="IC 最小值"),
    ic_max: float | None = Query(None, description="IC 最大值"),
    icir_min: float | None = Query(None, description="ICIR 最小值"),
    icir_max: float | None = Query(None, description="ICIR 最大值"),
    search: str | None = Query(None, description="因子名模糊搜索"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort_by: str = Query("icir", pattern="^(icir|ic_mean|sharpe|top_icir|coverage)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
):
    """因子列表（带筛选、排序、分页）。"""
    result = query_factors(
        ic_period=ic_period,
        ic_type=ic_type,
        ic_min=ic_min,
        ic_max=ic_max,
        icir_min=icir_min,
        icir_max=icir_max,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        page=page,
        page_size=page_size,
    )
    return result


@router.post("/_/rebuild-index")
def rebuild_index():
    """强制重建因子索引（扫描 data/factors/ 下所有报告）。"""
    idx = build_index(force=True)
    return {"n_factors": idx.get("n_factors", 0), "generated_at": idx.get("generated_at", "")}


@router.get("/_/index-status")
def index_status():
    """因子索引状态。"""
    idx = load_index()
    return {"n_factors": idx.get("n_factors", 0), "generated_at": idx.get("generated_at", "")}


# 注意：/{factor_name}/correlation 必须在 /{factor_name} 之前注册
@router.get("/{factor_name}/correlation")
def get_factor_correlation(
    factor_name: str,
    threshold: float = Query(0.7, ge=0, le=1.0),
    direction: str = Query("above", pattern="^(above|below)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """查询因子相关性。"""
    return query_correlation(factor_name=factor_name, threshold=threshold, direction=direction, limit=limit)


@router.get("/{factor_name}")
def get_factor_detail(factor_name: str) -> dict[str, Any]:
    """获取单个因子的完整详情。"""
    detail = load_factor_detail(factor_name)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"因子 {factor_name} 不存在")
    return detail
