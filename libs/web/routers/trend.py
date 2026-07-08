"""
因子走势路由 — 因子 + symbol 的时序走势数据
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from services.trend_svc import (
    get_trend_data,
    get_available_symbols,
    get_trendable_factors,
)

router = APIRouter(prefix="/api/trend", tags=["trend"])


@router.get("/_/symbols")
def list_symbols():
    """获取可用 ETF 标的列表。"""
    symbols = get_available_symbols()
    return {"symbols": symbols}


@router.get("/_/trendable-factors")
def list_trendable_factors():
    """获取可以做走势查询的因子列表。"""
    factors = get_trendable_factors()
    return {"factors": factors}


@router.get("/{factor_name}")
def trend(
    factor_name: str,
    symbol: str = Query(..., description="ETF 代码"),
    start_date: str | None = Query(None, description="起始日期 YYYY-MM-DD"),
    end_date: str | None = Query(None, description="截止日期 YYYY-MM-DD"),
):
    """获取因子在某 symbol 上的走势时序数据。"""
    try:
        result = get_trend_data(
            factor_name=factor_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
