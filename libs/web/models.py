"""
投资框架网站 API — Pydantic 数据模型
"""
from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel


# ── 因子列表查询参数 ──────────────────────────────────────────────────────────

class FactorListQuery(BaseModel):
    """因子列表筛选参数。"""
    ic_period: int = 10           # IC 周期: 5/10/20/60
    ic_type: str = "rank_ic"      # rank_ic | pearson_ic
    ic_min: float | None = None
    ic_max: float | None = None
    icir_min: float | None = None
    icir_max: float | None = None
    search: str | None = None     # 因子名模糊匹配
    page: int = 1
    page_size: int = 50
    sort_by: str = "icir"         # ic_mean | icir | sharpe | top_icir | coverage
    sort_order: str = "desc"      # asc | desc


# ── 因子列表响应 ──────────────────────────────────────────────────────────────

class ICStats(BaseModel):
    mean: float | None
    std: float | None
    ir: float | None
    t_stat: float | None
    pos_ratio: float | None

class FactorSummary(BaseModel):
    """因子列表中每个因子的摘要。"""
    name: str
    factor_type: str
    params: dict[str, Any]
    n_symbols: int
    start_date: str
    end_date: str
    coverage_mean: float
    ic: dict[str, ICStats]          # "5"/"10"/"20"/"60" → ICStats
    top_ic: dict[str, ICStats] | None = None
    longshort_sharpe_5d: float | None
    analysis_date: str


class FactorListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    factors: list[FactorSummary]


# ── 因子详情响应 ──────────────────────────────────────────────────────────────

class FactorDetail(BaseModel):
    """单个因子的完整数据（从 report.json 提取）。"""
    meta: dict[str, Any]
    panel_summary: dict[str, Any]
    layer2_predictive: dict[str, Any]     # rank_ic / pearson_ic / ic_decay
    layer3_grouping: dict[str, Any]       # quantile_summary / longshort / monotonicity
    # layer1_quality 不返回（用户不关心覆盖率等信息）


# ── 相关性查询 ────────────────────────────────────────────────────────────────

class CorrelationQuery(BaseModel):
    threshold: float = 0.7
    direction: str = "above"       # above | below
    limit: int = 50


class CorrelationItem(BaseModel):
    factor: str
    correlation: float


class CorrelationResponse(BaseModel):
    factor: str
    related: list[CorrelationItem]
    total_available: int           # 相关性矩阵中的因子总数


# ── 因子走势 API ──────────────────────────────────────────────────────────────

class TrendQuery(BaseModel):
    symbol: str                    # ETF 代码
    start_date: str | None = None  # YYYY-MM-DD，默认近一年
    end_date: str | None = None    # YYYY-MM-DD，默认最新


class TrendPoint(BaseModel):
    date: str
    factor_value: float | None
    close: float | None
    volume: float | None           # 成交额（元）


class TrendResponse(BaseModel):
    factor_name: str
    symbol: str
    symbol_name: str
    start_date: str
    end_date: str
    data: list[TrendPoint]
    stats: dict[str, Any]          # factor 均值/标准差/最新值等


# ── Symbol 列表 ────────────────────────────────────────────────────────────────

class SymbolItem(BaseModel):
    symbol: str
    name: str


class SymbolListResponse(BaseModel):
    symbols: list[SymbolItem]
