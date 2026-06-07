"""宽股票池动量基线回测。

这个模块按三个阶段组织：

1. 从 ETF 价格数据准备可交易股票池。
2. 在共享交易日历上运行一个或多个 top-N 组合变体。
3. 持久化汇总、诊断信息和分变体产物，便于后续分析。
"""

from __future__ import annotations

import inspect
import concurrent.futures
import os
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Optional

import pandas as pd

from config import DataPath
from core.models.etf_daily_data import EtfData
from data_manager.etf_data_manager import get_etf_data_by_symbol
from data_manager.providers.cluster_provider import ClusterInfo
from data_manager.providers.etf_list_provider import ETF_LIST
from factors.base_factor import BaseFactor
from factors.price_return import PriceReturn

from .performance import (
    annualised_return,
    annualised_volatility,
    cumulative_return,
    sharpe_ratio,
)


CandidateFilterCallable = Callable[..., bool]


@dataclass(slots=True, frozen=True)
class CandidateFilterSpec:
    """带可选名称的候选过滤器定义。"""

    filter_fn: CandidateFilterCallable
    name: Optional[str] = None
    params: Optional[dict[str, Any]] = None


@dataclass(slots=True, frozen=True)
class ThresholdFilter:
    """描述一个基于字段阈值的内置过滤器。

    和 `candidate_filters` 的自定义 callable 不同，这里的过滤参数
    是可序列化的，因此能自动出现在实验名和运行元数据中。
    """

    VALID_OPERATORS: ClassVar[tuple[str, ...]] = (">=", "<=", ">", "<", "==")

    field: str
    operator: str
    value: float
    name: Optional[str] = None


# 操作符标签映射，与 ThresholdFilter.VALID_OPERATORS 保持同步
_OPERATOR_LABELS: dict[str, str] = {
    ">=": "ge",
    "<=": "le",
    ">": "gt",
    "<": "lt",
    "==": "eq",
}


@dataclass(slots=True)
class WideMomentumBaselineConfig:
    """控制股票池准备、回测模拟和结果命名的用户配置。"""

    top_n_values: tuple[int, ...] = (20,)
    min_listing_days: int = 1200
    momentum_window: int = 20
    momentum_skip_recent: int = 1
    min_momentum_value: Optional[float] = None
    builtin_filters: tuple[ThresholdFilter, ...] = field(default_factory=tuple)
    ranking_factor: Optional[BaseFactor] = None
    factor_pipeline: tuple[BaseFactor, ...] = field(default_factory=tuple)
    candidate_filters: tuple[CandidateFilterCallable | CandidateFilterSpec, ...] = field(
        default_factory=tuple
    )
    rebalance_interval: int = 5
    cash: float = 100_000.0
    commission: float = 0.00025
    risk_free_rate: float = 0.02
    stable_pool_size: int = 100
    start_date: Optional[str | pd.Timestamp] = "2023-12-04"
    end_date: Optional[str | pd.Timestamp] = "2026-05-29"
    # 集群约束 —— 每个 cluster 最多持有多少个标的。
    # cluster_limit_enabled=False 时完全不参与选股逻辑。
    cluster_limit_enabled: bool = False
    # 未分类标的（cluster_label == -1）不计数也不受限制。
    cluster_max_per_group: int = 3
    # 排除的 cluster 标签 —— 这些 cluster 的标的不会进入候选池。
    # 常用于剔除债券类标的（cluster 43/44），与集群约束无关。
    exclude_clusters: tuple[int, ...] = field(default_factory=tuple)
    # 实验名称和指定标的
    symbols: Optional[tuple[str, ...]] = None
    experiment_name: Optional[str] = None
    # 调仓时保留重叠标的（已在持仓中且下期仍选中的标的不卖出，继续持有）
    hold_overlap: bool = False

    def __post_init__(self) -> None:
        """尽早校验配置，减少下游流程中的防御性判断。"""

        if self.experiment_name is not None:
            normalized_experiment_name = str(self.experiment_name).strip()
            self.experiment_name = normalized_experiment_name or None
        if self.symbols is not None:
            self.symbols = tuple(str(symbol) for symbol in self.symbols)
        self.factor_pipeline = tuple(self.factor_pipeline or ())
        self.candidate_filters = tuple(self.candidate_filters or ())
        self.builtin_filters = tuple(self.builtin_filters or ())
        if not self.top_n_values:
            raise ValueError("top_n_values cannot be empty")
        if any(int(top_n) <= 0 for top_n in self.top_n_values):
            raise ValueError("top_n_values must all be >= 1")
        if int(self.momentum_window) <= 0:
            raise ValueError("momentum_window must be >= 1")
        if int(self.momentum_skip_recent) < 0:
            raise ValueError("momentum_skip_recent must be >= 0")
        if self.min_momentum_value is not None and math.isnan(float(self.min_momentum_value)):
            raise ValueError("min_momentum_value must be a real number when provided")
        # 校验内置阈值过滤器
        for idx, bf in enumerate(self.builtin_filters):
            if not isinstance(bf, ThresholdFilter):
                raise TypeError(f"builtin_filters[{idx}] must be ThresholdFilter")
            if not isinstance(bf.field, str) or not bf.field.strip():
                raise ValueError(f"builtin_filters[{idx}].field must be a non-empty string")
            if bf.operator not in ThresholdFilter.VALID_OPERATORS:
                raise ValueError(
                    f"builtin_filters[{idx}].operator must be one of "
                    f"{ThresholdFilter.VALID_OPERATORS}"
                )
            if math.isnan(float(bf.value)):
                raise ValueError(f"builtin_filters[{idx}].value must be a real number")
        # 校验集群约束
        if not isinstance(self.cluster_limit_enabled, bool):
            self.cluster_limit_enabled = bool(self.cluster_limit_enabled)
        if not isinstance(self.cluster_max_per_group, int):
            self.cluster_max_per_group = int(self.cluster_max_per_group)
        if self.cluster_limit_enabled and int(self.cluster_max_per_group) <= 0:
            raise ValueError("cluster_max_per_group must be >= 1 when cluster_limit_enabled")
        self.exclude_clusters = tuple(
            int(c) for c in (self.exclude_clusters or ())
        )

        if int(self.rebalance_interval) <= 0:
            raise ValueError("rebalance_interval must be >= 1")
        if float(self.cash) <= 0:
            raise ValueError("cash must be > 0")
        if float(self.commission) < 0:
            raise ValueError("commission must be >= 0")
        if float(self.risk_free_rate) < 0:
            raise ValueError("risk_free_rate must be >= 0")
        if int(self.stable_pool_size) <= 0:
            raise ValueError("stable_pool_size must be >= 1")
        if int(self.min_listing_days) < 0:
            raise ValueError("min_listing_days must be >= 0")
        if self.ranking_factor is not None and not isinstance(self.ranking_factor, BaseFactor):
            raise TypeError("ranking_factor must inherit from BaseFactor")
        if any(not isinstance(factor, BaseFactor) for factor in self.factor_pipeline):
            raise TypeError("factor_pipeline must contain only BaseFactor instances")
        for candidate_filter in self.candidate_filters:
            resolved_callable = (
                candidate_filter.filter_fn
                if isinstance(candidate_filter, CandidateFilterSpec)
                else candidate_filter
            )
            if not callable(resolved_callable):
                raise TypeError("candidate_filters must contain callables or CandidateFilterSpec")

        # 将 min_momentum_value 转换为 ThresholdFilter，保持向后兼容
        if self.min_momentum_value is not None:
            momentum_filter = ThresholdFilter(
                field="score", operator=">=", value=float(self.min_momentum_value)
            )
            self.builtin_filters = self.builtin_filters + (momentum_filter,)


@dataclass(slots=True)
class SymbolBaselineData:
    """候选筛选和交易执行阶段需要的单标的数据。"""

    symbol: str
    listing_proxy_date: pd.Timestamp
    cluster_label: int
    frame: pd.DataFrame
    etf_data: Optional[EtfData] = None
    ranking_output_name: str = "momentum"


@dataclass(slots=True)
class PreparedWideMomentumUniverse:
    """供所有 top-N 变体共享的已准备股票池及其诊断信息。"""

    symbol_data_map: dict[str, SymbolBaselineData]
    calendar: pd.DatetimeIndex
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    recent_complete_date: pd.Timestamp
    monthly_pool_diagnostics: pd.DataFrame
    source_symbol_count: int
    load_errors: list[dict[str, str]] = field(default_factory=list)
    excluded_symbols: dict[str, str] = field(default_factory=dict)

    @property
    def stable_start_month(self) -> Optional[pd.Timestamp]:
        """返回股票池首次达到稳定规模目标的月份。"""

        if self.monthly_pool_diagnostics.empty:
            return None
        stable_rows = self.monthly_pool_diagnostics[
            self.monthly_pool_diagnostics["is_stable_start"]
        ]
        if stable_rows.empty:
            return None
        return pd.to_datetime(stable_rows.iloc[0]["month_end_date"])


@dataclass(slots=True)
class WideMomentumVariantResult:
    """单个 top-N 参数对应的模拟结果。"""

    top_n: int
    summary: dict[str, Any]
    equity_curve: pd.DataFrame
    annual_returns: pd.DataFrame
    rebalance_log: pd.DataFrame


@dataclass(slots=True)
class WideMomentumBaselineResult:
    """一次基线运行在一个或多个变体上的完整输出。"""

    config: WideMomentumBaselineConfig
    prepared_universe: PreparedWideMomentumUniverse
    variant_results: dict[int, WideMomentumVariantResult]


@dataclass(slots=True, frozen=True)
class BaselineCandidate:
    """信号日用于排序的可交易标的、打分值及因子值。"""

    symbol: str
    score: float
    etf_data: Optional[EtfData] = None
    factor_values: dict[str, float] = field(default_factory=dict)


def _resolve_ranking_factor(config: WideMomentumBaselineConfig) -> BaseFactor:
    """解析当前回测用于排序的主因子。"""

    if config.ranking_factor is not None:
        return config.ranking_factor
    return PriceReturn(
        window=int(config.momentum_window),
        skip_recent=int(config.momentum_skip_recent),
    )


def _resolve_factor_pipeline(config: WideMomentumBaselineConfig) -> tuple[BaseFactor, ...]:
    """解析完整因子管线，并确保排序因子始终包含在其中。"""

    ranking_factor = _resolve_ranking_factor(config)
    resolved_factors: list[BaseFactor] = []
    seen_factors: set[BaseFactor] = set()

    for factor in (ranking_factor, *config.factor_pipeline):
        if factor in seen_factors:
            continue
        seen_factors.add(factor)
        resolved_factors.append(factor)

    return tuple(resolved_factors)


def _resolve_candidate_filter_name(
    candidate_filter: CandidateFilterCallable | CandidateFilterSpec,
) -> str:
    """为候选过滤器解析一个稳定且可读的名称。"""

    if isinstance(candidate_filter, CandidateFilterSpec):
        if candidate_filter.name:
            return str(candidate_filter.name)
        candidate_filter = candidate_filter.filter_fn

    filter_name = getattr(candidate_filter, "__name__", None)
    if filter_name and filter_name != "<lambda>":
        return str(filter_name)
    return candidate_filter.__class__.__name__


def _resolve_candidate_filter_callable(
    candidate_filter: CandidateFilterCallable | CandidateFilterSpec,
) -> CandidateFilterCallable:
    """提取候选过滤器里真正可调用的函数对象。"""

    if isinstance(candidate_filter, CandidateFilterSpec):
        return candidate_filter.filter_fn
    return candidate_filter


def _serialize_factor(factor: BaseFactor) -> dict[str, Any]:
    """将因子配置整理为可写入元数据的字典。"""

    return {
        "class_name": factor.__class__.__name__,
        "output_name": factor.get_output_name(),
        "params": dict(factor.params),
    }


def _serialize_candidate_filters(config: WideMomentumBaselineConfig) -> list[dict[str, Any]]:
    """将启用的候选过滤条件整理成适合写入元数据的 JSON 结构。"""

    filters: list[dict[str, Any]] = []
    for bf in config.builtin_filters:
        filters.append(
            {
                "name": bf.name or bf.field,
                "field": bf.field,
                "operator": bf.operator,
                "value": float(bf.value),
            }
        )
    for candidate_filter in config.candidate_filters:
        entry: dict[str, Any] = {
            "name": _resolve_candidate_filter_name(candidate_filter),
            "kind": "callable",
        }
        if isinstance(candidate_filter, CandidateFilterSpec) and candidate_filter.params:
            entry["params"] = candidate_filter.params
        filters.append(entry)
    return filters


def _format_experiment_value(value: Any) -> str:
    """将参数值格式化为稳定且紧凑的实验名片段。"""

    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def _resolve_experiment_name(config: WideMomentumBaselineConfig) -> str:
    """在调用方未显式提供时，构造可自解释的实验名。"""

    if config.experiment_name is not None:
        return config.experiment_name

    serialized_filters = _serialize_candidate_filters(config)
    if not serialized_filters:
        return "wide_momentum_baseline"

    filter_labels: list[str] = []
    for item in serialized_filters:
        if item.get("kind") == "callable":
            filter_labels.append(str(item.get("name")))
            continue
        operator = _OPERATOR_LABELS.get(str(item.get("operator")), "filter")
        filter_labels.append(
            f"{item.get('name')}_{operator}_{_format_experiment_value(item.get('value'))}"
        )

    # 集群约束标签
    if config.cluster_limit_enabled:
        filter_labels.append(f"cluster_max{config.cluster_max_per_group}")

    # 排除 cluster 标签
    if config.exclude_clusters:
        sorted_clusters = sorted(config.exclude_clusters)
        filter_labels.append("no_cl" + "_".join(str(c) for c in sorted_clusters))

    if not filter_labels:
        return "wide_momentum_baseline"
    return "wide_momentum_baseline__" + "__".join(filter_labels)


def _normalize_price_frame(
    frame: pd.DataFrame,
    symbol: str,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """将原始 ETF 行情整理为按日期排序的特征表。"""

    local_df = frame.copy()
    if "date" in local_df.columns:
        local_df["date"] = pd.to_datetime(local_df["date"], errors="coerce")
        local_df = local_df.set_index("date")
    else:
        local_df.index = pd.to_datetime(local_df.index, errors="coerce")

    local_df = local_df.loc[local_df.index.notna()].sort_index()
    if local_df.empty:
        raise ValueError(f"{symbol} has no valid rows")

    required_columns = {"open", "close", *extra_columns}
    missing_columns = required_columns.difference(local_df.columns)
    if missing_columns:
        raise ValueError(f"{symbol} missing required columns: {sorted(missing_columns)}")

    for column in required_columns:
        local_df[column] = pd.to_numeric(local_df[column], errors="coerce")
    local_df = local_df.dropna(subset=["open", "close"])
    local_df = local_df[local_df["open"] > 0]
    local_df = local_df[local_df["close"] > 0]
    if local_df.empty:
        raise ValueError(f"{symbol} has no valid open/close rows after normalization")

    ordered_columns = ["open", "close", *extra_columns]
    return local_df.loc[:, ordered_columns].copy()


def _resolve_recent_complete_date(
    last_dates: list[pd.Timestamp],
    explicit_end_date: Optional[str | pd.Timestamp],
) -> pd.Timestamp:
    """在未显式指定结束日期时，选择最近的共同完整日期。"""

    if explicit_end_date is not None:
        return pd.to_datetime(explicit_end_date)

    counts = pd.Series(pd.to_datetime(last_dates)).value_counts()
    if counts.empty:
        raise ValueError("No last_dates available to resolve end date")

    top_count = int(counts.iloc[0])
    candidates = sorted(pd.to_datetime(counts[counts == top_count].index))
    return pd.Timestamp(candidates[-1])


def _load_trading_calendar(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    """加载仓库中的交易日历，并裁剪到指定时间窗口。"""

    calendar_df = pd.read_csv(
        DataPath.CALANDAR_DF,
        parse_dates=["trade_date"],
    )
    trade_dates = pd.to_datetime(calendar_df["trade_date"], errors="coerce")
    trade_dates = trade_dates.dropna().sort_values().drop_duplicates()
    mask = (trade_dates >= start_date) & (trade_dates <= end_date)
    calendar = pd.DatetimeIndex(trade_dates.loc[mask])
    if calendar.empty:
        raise ValueError(
            f"No trading days found between {start_date.date()} and {end_date.date()}"
        )
    return calendar


def _resolve_cluster_label(
    symbol: str,
    cluster_lookup: Optional[Mapping[str, int] | Callable[[str], int]],
) -> int:
    """优先使用覆盖参数，否则从默认 provider 中解析标的的 cluster。"""

    if cluster_lookup is None:
        return int(ClusterInfo.get_cluster(symbol))
    if callable(cluster_lookup):
        return int(cluster_lookup(symbol))
    return int(cluster_lookup.get(symbol, -1))


def _month_end_dates(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """返回交易日历中每个月最后一个可交易日。"""

    if len(calendar) == 0:
        return []
    month_end_series = pd.Series(calendar, index=calendar)
    return [pd.Timestamp(value) for value in month_end_series.groupby(calendar.to_period("M")).max()]


def _build_monthly_pool_diagnostics(
    symbol_data_map: Mapping[str, SymbolBaselineData],
    calendar: pd.DatetimeIndex,
    stable_pool_size: int,
) -> pd.DataFrame:
    """汇总股票池在各月月底的可选范围变化。"""

    rows: list[dict[str, Any]] = []
    stable_marked = False

    for month_end_date in _month_end_dates(calendar):
        eligible_symbols: list[str] = []
        covered_clusters: set[int] = set()
        unassigned_cluster_symbols = 0

        for symbol, symbol_data in symbol_data_map.items():
            eligible = bool(
                symbol_data.frame["eligible_signal"].reindex([month_end_date], fill_value=False).iloc[0]
            )
            if not eligible:
                continue
            eligible_symbols.append(symbol)
            if symbol_data.cluster_label >= 0:
                covered_clusters.add(int(symbol_data.cluster_label))
            else:
                unassigned_cluster_symbols += 1

        pool_size = len(eligible_symbols)
        is_pool_stable = pool_size >= stable_pool_size
        is_stable_start = bool(is_pool_stable and not stable_marked)
        if is_stable_start:
            stable_marked = True

        rows.append(
            {
                "month": month_end_date.strftime("%Y-%m"),
                "month_end_date": month_end_date,
                "eligible_symbol_count": pool_size,
                "covered_cluster_count": len(covered_clusters),
                "unassigned_cluster_symbol_count": int(unassigned_cluster_symbols),
                "is_pool_stable": is_pool_stable,
                "is_stable_start": is_stable_start,
            }
        )

    return pd.DataFrame(rows)


def _clone_etf_data(etf_data: EtfData, symbol: str) -> EtfData:
    """复制 EtfData，避免在原对象上残留因子计算状态。"""

    return EtfData(
        etf_data.data,
        metadata=etf_data.metadata,
        symbol=symbol,
        name=getattr(etf_data, "name", ""),
    )


def _coerce_to_etf_data(symbol: str, source: pd.DataFrame | EtfData) -> EtfData:
    """将 DataFrame 或 EtfData 统一转换为 EtfData。"""

    if isinstance(source, EtfData):
        return _clone_etf_data(source, symbol=symbol)
    if isinstance(source, pd.DataFrame):
        return EtfData(source.copy(), symbol=symbol)
    raise TypeError(f"Unsupported source type for {symbol}: {type(source)!r}")


def _load_etf_data_map(
    symbol_source_map: Mapping[str, pd.DataFrame | EtfData],
) -> tuple[dict[str, EtfData], list[dict[str, str]]]:
    """将输入统一转换为 EtfData，并验证可用于回测的行情列。"""

    etf_data_map: dict[str, EtfData] = {}
    load_errors: list[dict[str, str]] = []

    for symbol, source in symbol_source_map.items():
        try:
            etf_data = _coerce_to_etf_data(symbol=symbol, source=source)
            _normalize_price_frame(frame=etf_data.data, symbol=symbol)
            etf_data_map[symbol] = etf_data
        except Exception as exc:
            load_errors.append({"symbol": symbol, "error": str(exc)})

    return etf_data_map, load_errors


def _prepare_factor_ready_etf_data(
    etf_data: EtfData,
    symbol: str,
    factor_pipeline: tuple[BaseFactor, ...],
) -> EtfData:
    """在 EtfData 上注册并计算因子，返回带因子结果的副本。"""

    prepared_etf_data = _clone_etf_data(etf_data, symbol=symbol)
    prepared_etf_data.factors.clear()
    prepared_etf_data.factor_results.clear()

    for factor in factor_pipeline:
        prepared_etf_data.add_factors(factor)

    prepared_etf_data.calc_factors()
    return prepared_etf_data


def _build_feature_frame(
    etf_data: EtfData,
    symbol: str,
    factor_pipeline: tuple[BaseFactor, ...],
    ranking_output_name: str,
) -> pd.DataFrame:
    """将带因子的 EtfData 物化为供手工回测使用的特征表。"""

    factor_output_names = tuple(factor.get_output_name() for factor in factor_pipeline)
    feature_frame = _normalize_price_frame(
        frame=etf_data.output_with_factors(),
        symbol=symbol,
        extra_columns=factor_output_names,
    )
    if ranking_output_name in feature_frame.columns:
        feature_frame["ranking_value"] = pd.to_numeric(
            feature_frame[ranking_output_name],
            errors="coerce",
        ).astype(float)
    if ranking_output_name != "momentum" and ranking_output_name in feature_frame.columns:
        feature_frame["momentum"] = feature_frame[ranking_output_name].astype(float)
    return feature_frame


def _process_single_symbol(args: tuple) -> tuple[
    str, Optional[SymbolBaselineData], Optional[str], Optional[str]
]:
    """处理单个标的，供多进程并行使用。

    此函数必须是模块级函数（而非闭包/嵌套函数），才能被 pickle 传递给子进程。
    """
    (
        symbol,
        etf_data_df,
        etf_metadata,
        etf_name,
        recent_complete_str,
        min_listing_days,
        ranking_output_name,
        factor_pipeline,
        cluster_label,
    ) = args

    # 在子进程中重建 EtfData
    raw_etf_data = EtfData(
        etf_data_df,
        metadata=etf_metadata or {},
        symbol=symbol,
        name=etf_name or "",
    )

    recent_complete_date = pd.Timestamp(recent_complete_str)

    try:
        raw_price_frame = _normalize_price_frame(
            frame=raw_etf_data.data, symbol=symbol
        )
        if raw_price_frame.index[-1] < recent_complete_date:
            return symbol, None, None, "stale_before_recent_complete_date"

        prepared_etf_data = _prepare_factor_ready_etf_data(
            etf_data=raw_etf_data,
            symbol=symbol,
            factor_pipeline=factor_pipeline,
        )
        prepared_etf_data = prepared_etf_data.slice_date_range(
            end_date=str(recent_complete_date.date())
        )
        local_frame = _build_feature_frame(
            etf_data=prepared_etf_data,
            symbol=symbol,
            factor_pipeline=factor_pipeline,
            ranking_output_name=ranking_output_name,
        )

        listing_proxy_date = pd.Timestamp(local_frame.index[0])
        listing_cutoff = listing_proxy_date + pd.Timedelta(days=int(min_listing_days))
        ranking_series = pd.to_numeric(local_frame[ranking_output_name], errors="coerce")
        eligible_signal = (local_frame.index > listing_cutoff) & ranking_series.notna()

        if ranking_output_name != "momentum":
            local_frame["momentum"] = ranking_series.astype(float)
        local_frame["eligible_signal"] = eligible_signal.astype(bool)

        if not bool(local_frame["eligible_signal"].any()):
            return symbol, None, None, "never_eligible_in_window"

        sd = SymbolBaselineData(
            symbol=symbol,
            listing_proxy_date=listing_proxy_date,
            cluster_label=cluster_label,
            frame=local_frame,
            etf_data=prepared_etf_data,
            ranking_output_name=ranking_output_name,
        )
        return symbol, sd, None, None

    except Exception as exc:
        return symbol, None, str(exc), None


def _build_symbol_data_map(
    etf_data_map: Mapping[str, EtfData],
    config: WideMomentumBaselineConfig,
    recent_complete_date: pd.Timestamp,
    cluster_lookup: Optional[Mapping[str, int] | Callable[[str], int]],
) -> tuple[dict[str, SymbolBaselineData], dict[str, str]]:
    """计算每个标的的动量和可交易资格，并排除无法交易的标的。

    因子计算阶段使用多进程并行，这部分是回测中最耗时的环节。
    """

    ranking_factor = _resolve_ranking_factor(config)
    factor_pipeline = _resolve_factor_pipeline(config)
    ranking_output_name = ranking_factor.get_output_name()
    symbol_data_map: dict[str, SymbolBaselineData] = {}
    excluded_symbols: dict[str, str] = {}
    worker_jobs: list[tuple] = []

    # 阶段 1：过滤数据过期的标的，打包为 worker 参数。
    for symbol, raw_etf_data in etf_data_map.items():
        raw_price_frame = _normalize_price_frame(frame=raw_etf_data.data, symbol=symbol)
        if raw_price_frame.index[-1] < recent_complete_date:
            excluded_symbols[symbol] = "stale_before_recent_complete_date"
            continue

        cluster_label = _resolve_cluster_label(symbol, cluster_lookup)

        worker_jobs.append((
            symbol,
            raw_etf_data.data.copy(),
            raw_etf_data.metadata or {},
            getattr(raw_etf_data, "name", ""),
            str(recent_complete_date.date()),
            int(config.min_listing_days),
            ranking_output_name,
            factor_pipeline,
            cluster_label,
        ))

    # 阶段 2：多进程并行因子计算。
    max_workers = min(os.cpu_count() or 4, len(worker_jobs))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_process_single_symbol, job): job[0]
            for job in worker_jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            try:
                symbol, sd, error, exclusion_reason = future.result()
            except Exception as exc:
                symbol = future_map[future]
                excluded_symbols[symbol] = f"worker_failed: {exc}"
                continue

            if exclusion_reason:
                excluded_symbols[symbol] = exclusion_reason
                continue
            if error:
                excluded_symbols[symbol] = f"error: {error}"
                continue

            symbol_data_map[symbol] = sd

    return symbol_data_map, excluded_symbols


def _resolve_universe_start_date(
    symbol_data_map: Mapping[str, SymbolBaselineData],
    config: WideMomentumBaselineConfig,
) -> pd.Timestamp:
    """使用首个可交易信号日和用户起始日期中较晚的那个日期。"""

    computed_start_date = min(
        pd.Timestamp(symbol_data.frame.index[symbol_data.frame["eligible_signal"]][0])
        for symbol_data in symbol_data_map.values()
    )
    if config.start_date is None:
        return computed_start_date
    return max(computed_start_date, pd.to_datetime(config.start_date))


def _resolve_calendar_window(
    calendar: Optional[pd.DatetimeIndex],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    """标准化交易日历，并裁剪到最终模拟窗口。"""

    if calendar is None:
        resolved_calendar = _load_trading_calendar(start_date=start_date, end_date=end_date)
    else:
        resolved_calendar = pd.DatetimeIndex(pd.to_datetime(calendar))

    resolved_calendar = resolved_calendar[
        (resolved_calendar >= start_date) & (resolved_calendar <= end_date)
    ]
    resolved_calendar = pd.DatetimeIndex(resolved_calendar.sort_values().drop_duplicates())
    if resolved_calendar.empty:
        raise ValueError("Resolved trading calendar is empty")
    return resolved_calendar


def prepare_wide_momentum_universe_from_frames(
    symbol_frame_map: Mapping[str, pd.DataFrame | EtfData],
    config: WideMomentumBaselineConfig,
    *,
    calendar: Optional[pd.DatetimeIndex] = None,
    cluster_lookup: Optional[Mapping[str, int] | Callable[[str], int]] = None,
) -> PreparedWideMomentumUniverse:
    """基于已加载的 symbol 数据准备可复用的回测股票池。"""

    # 阶段 1：将输入统一转换为 EtfData，并验证基本行情字段可用。
    etf_data_map, load_errors = _load_etf_data_map(symbol_frame_map)

    if not etf_data_map:
        raise ValueError("No valid ETF frames available for baseline prep")

    recent_complete_date = _resolve_recent_complete_date(
        last_dates=[
            _normalize_price_frame(frame=etf_data.data, symbol=symbol).index[-1]
            for symbol, etf_data in etf_data_map.items()
        ],
        explicit_end_date=config.end_date,
    )

    # 阶段 2：在 EtfData 上计算因子，并过滤掉在窗口内永远无法交易的标的。
    symbol_data_map, excluded_symbols = _build_symbol_data_map(
        etf_data_map=etf_data_map,
        config=config,
        recent_complete_date=recent_complete_date,
        cluster_lookup=cluster_lookup,
    )

    if not symbol_data_map:
        raise ValueError("No symbols remain after eligibility filtering")

    # 阶段 3：将所有标的对齐到共同的模拟窗口，并生成诊断信息。
    start_date = _resolve_universe_start_date(symbol_data_map=symbol_data_map, config=config)

    resolved_calendar = _resolve_calendar_window(
        calendar=calendar,
        start_date=start_date,
        end_date=recent_complete_date,
    )

    aligned_start_date = pd.Timestamp(resolved_calendar[0])
    monthly_pool_diagnostics = _build_monthly_pool_diagnostics(
        symbol_data_map=symbol_data_map,
        calendar=resolved_calendar,
        stable_pool_size=int(config.stable_pool_size),
    )

    return PreparedWideMomentumUniverse(
        symbol_data_map=symbol_data_map,
        calendar=resolved_calendar,
        start_date=aligned_start_date,
        end_date=pd.Timestamp(resolved_calendar[-1]),
        recent_complete_date=recent_complete_date,
        monthly_pool_diagnostics=monthly_pool_diagnostics,
        source_symbol_count=len(symbol_frame_map),
        load_errors=load_errors,
        excluded_symbols=excluded_symbols,
    )


def prepare_wide_momentum_universe_from_etf_data_map(
    etf_data_map: Mapping[str, EtfData],
    config: WideMomentumBaselineConfig,
    *,
    calendar: Optional[pd.DatetimeIndex] = None,
    cluster_lookup: Optional[Mapping[str, int] | Callable[[str], int]] = None,
) -> PreparedWideMomentumUniverse:
    """基于已准备好的 EtfData 映射准备可复用的回测股票池。"""

    return prepare_wide_momentum_universe_from_frames(
        symbol_frame_map=etf_data_map,
        config=config,
        calendar=calendar,
        cluster_lookup=cluster_lookup,
    )


def prepare_wide_momentum_universe(
    config: WideMomentumBaselineConfig,
    *,
    symbols: Optional[list[str]] = None,
    cluster_lookup: Optional[Mapping[str, int] | Callable[[str], int]] = None,
) -> PreparedWideMomentumUniverse:
    """从存储中加载 ETF 数据，并为所有变体准备共享股票池。"""

    target_symbols = symbols or list(config.symbols or tuple(ETF_LIST.get_all_symbol()))
    etf_data_map: dict[str, EtfData] = {}

    for symbol in target_symbols:
        try:
            # 这里使用 data_manager 返回的 EtfData 作为 ETF 历史数据交互入口。
            etf_data_map[symbol] = get_etf_data_by_symbol(symbol)
        except Exception:
            continue

    return prepare_wide_momentum_universe_from_etf_data_map(
        etf_data_map=etf_data_map,
        config=config,
        cluster_lookup=cluster_lookup,
    )


def _portfolio_value(
    positions: Mapping[str, float],
    cash: float,
    date: pd.Timestamp,
    symbol_data_map: Mapping[str, SymbolBaselineData],
    price_column: str,
) -> float:
    """使用指定价格列对当前组合进行市值重估。"""

    total_value = float(cash)
    for symbol, shares in positions.items():
        if abs(float(shares)) <= 1e-12:
            continue
        price = symbol_data_map[symbol].frame[price_column].asof(date)
        if pd.isna(price) or float(price) <= 0:
            raise ValueError(f"Missing {price_column} price for {symbol} on {date.date()}")
        total_value += float(shares) * float(price)
    return float(total_value)


def _portfolio_weights(
    positions: Mapping[str, float],
    cash: float,
    date: pd.Timestamp,
    symbol_data_map: Mapping[str, SymbolBaselineData],
    price_column: str,
) -> dict[str, float]:
    """按指定估值价格将持仓换算为组合权重。"""

    portfolio_value = _portfolio_value(
        positions=positions,
        cash=cash,
        date=date,
        symbol_data_map=symbol_data_map,
        price_column=price_column,
    )
    if portfolio_value <= 0:
        return {}

    weights: dict[str, float] = {}
    for symbol, shares in positions.items():
        if abs(float(shares)) <= 1e-12:
            continue
        price = float(symbol_data_map[symbol].frame[price_column].asof(date))
        weights[symbol] = float(shares) * price / float(portfolio_value)
    return weights


def _collect_raw_candidates(
    symbol_data_map: Mapping[str, SymbolBaselineData],
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    *,
    exclude_clusters: tuple[int, ...] = (),
) -> list[BaselineCandidate]:
    """收集在信号日满足资格且可在下一开盘执行的标的。"""

    candidates: list[BaselineCandidate] = []

    for symbol, symbol_data in symbol_data_map.items():
        if symbol_data.cluster_label in exclude_clusters:
            continue

        frame = symbol_data.frame
        signal_row = frame.reindex([signal_date])
        execution_row = frame.reindex([execution_date])
        if signal_row.empty or execution_row.empty:
            continue

        ranking_column = symbol_data.ranking_output_name
        if ranking_column not in frame.columns:
            if "ranking_value" in frame.columns:
                ranking_column = "ranking_value"
            elif "momentum" in frame.columns:
                ranking_column = "momentum"
            else:
                raise KeyError(
                    f"Missing ranking column for {symbol}; expected one of "
                    f"{symbol_data.ranking_output_name!r}, 'ranking_value', or 'momentum'"
                )

        eligible = bool(signal_row["eligible_signal"].iloc[0])
        score = signal_row[ranking_column].iloc[0]
        next_open = execution_row["open"].iloc[0]
        if not eligible or pd.isna(score) or pd.isna(next_open):
            continue
        if float(next_open) <= 0:
            continue
        infrastructure_cols = {"open", "close", "eligible_signal"}
        factor_values: dict[str, float] = {}
        for col in frame.columns:
            if col in infrastructure_cols:
                continue
            val = signal_row[col].iloc[0]
            if not pd.isna(val):
                factor_values[col] = float(val)

        # 将 score 也注入 factor_values，使 ThresholdFilter(field="score", ...) 能正常工作。
        factor_values["score"] = float(score)

        candidates.append(
            BaselineCandidate(
                symbol=symbol,
                score=float(score),
                etf_data=symbol_data.etf_data,
                factor_values=factor_values,
            )
        )

    return candidates


def _build_candidate_filters(
    config: WideMomentumBaselineConfig,
) -> list[tuple[str, Callable[[BaselineCandidate], bool]]]:
    """根据当前配置构造内置候选过滤条件。

    遍历 `builtin_filters`，为每个 ThresholdFilter 生成一个闭包，
    闭包会从 `BaselineCandidate.factor_values` 中按 field 名取值并比较。
    """

    filters: list[tuple[str, Callable[[BaselineCandidate], bool]]] = []
    for bf in config.builtin_filters:
        threshold = float(bf.value)
        field = bf.field
        operator = bf.operator
        if operator == ">=":
            predicate = (
                lambda c, f=field, t=threshold: c.factor_values.get(f, float("-inf")) >= t
            )
        elif operator == "<=":
            predicate = (
                lambda c, f=field, t=threshold: c.factor_values.get(f, float("inf")) <= t
            )
        elif operator == ">":
            predicate = (
                lambda c, f=field, t=threshold: c.factor_values.get(f, float("-inf")) > t
            )
        elif operator == "<":
            predicate = (
                lambda c, f=field, t=threshold: c.factor_values.get(f, float("inf")) < t
            )
        elif operator == "==":
            predicate = (
                lambda c, f=field, t=threshold: c.factor_values.get(f, float("nan")) == t
            )
        else:
            raise ValueError(f"Unsupported operator: {operator!r}")
        label = bf.name or f"{field}{operator}{threshold}"
        filters.append((label, predicate))
    return filters


def _build_signal_filter_view_map(
    symbol_data_map: Mapping[str, SymbolBaselineData],
    signal_date: pd.Timestamp,
) -> dict[str, EtfData]:
    """为自定义过滤函数构建截断到信号日的 EtfData 视图。"""

    signal_view_map: dict[str, EtfData] = {}
    for symbol, symbol_data in symbol_data_map.items():
        if symbol_data.etf_data is None:
            raise ValueError("Custom candidate filters require SymbolBaselineData.etf_data")
        signal_view_map[symbol] = symbol_data.etf_data.slice_date_range(
            end_date=str(signal_date.date())
        )
    return signal_view_map


def _evaluate_custom_candidate_filter(
    candidate_filter: CandidateFilterCallable | CandidateFilterSpec,
    candidate_etf_data: EtfData,
    signal_view_map: Mapping[str, EtfData],
) -> bool:
    """执行自定义 callable 过滤器，支持一参或两参形式。"""

    resolved_callable = _resolve_candidate_filter_callable(candidate_filter)
    try:
        signature = inspect.signature(resolved_callable)
    except (TypeError, ValueError):
        return bool(resolved_callable(candidate_etf_data, signal_view_map))

    params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if has_varargs or len(params) >= 2:
        return bool(resolved_callable(candidate_etf_data, signal_view_map))
    if len(params) == 1:
        return bool(resolved_callable(candidate_etf_data))
    raise TypeError(
        "candidate filter callable must accept (candidate_etf_data) or "
        "(candidate_etf_data, signal_etf_data_map)"
    )


def _apply_candidate_filters(
    candidates: list[BaselineCandidate],
    config: WideMomentumBaselineConfig,
    symbol_data_map: Mapping[str, SymbolBaselineData],
    signal_date: pd.Timestamp,
) -> tuple[list[BaselineCandidate], list[str]]:
    """应用候选过滤条件，并返回过滤后的结果及其标签。"""

    filtered_candidates = list(candidates)
    active_filters = _build_candidate_filters(config)
    active_filter_names: list[str] = []

    for filter_name, candidate_filter in active_filters:
        active_filter_names.append(filter_name)
        filtered_candidates = [candidate for candidate in filtered_candidates if candidate_filter(candidate)]

    if config.candidate_filters:
        signal_view_map = _build_signal_filter_view_map(
            symbol_data_map=symbol_data_map,
            signal_date=signal_date,
        )
        for candidate_filter in config.candidate_filters:
            filter_name = _resolve_candidate_filter_name(candidate_filter)
            active_filter_names.append(filter_name)
            filtered_candidates = [
                candidate
                for candidate in filtered_candidates
                if _evaluate_custom_candidate_filter(
                    candidate_filter=candidate_filter,
                    candidate_etf_data=signal_view_map[candidate.symbol],
                    signal_view_map=signal_view_map,
                )
            ]

    return filtered_candidates, active_filter_names


def _select_target_weights(
    symbol_data_map: Mapping[str, SymbolBaselineData],
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    top_n: int,
    config: WideMomentumBaselineConfig,
) -> tuple[dict[str, float], int, int, list[str]]:
    """对过滤后的候选标的进行排序，并为 top-N 结果分配等权重。"""

    candidates = _collect_raw_candidates(
        symbol_data_map=symbol_data_map,
        signal_date=signal_date,
        execution_date=execution_date,
        exclude_clusters=config.exclude_clusters,
    )
    filtered_candidates, active_candidate_filters = _apply_candidate_filters(
        candidates=candidates,
        config=config,
        symbol_data_map=symbol_data_map,
        signal_date=signal_date,
    )

    # 先做可选过滤，再排序，这样日志里才能同时保留原始数量和过滤后数量。
    filtered_candidates.sort(key=lambda item: (-item.score, item.symbol))

    # 集群约束（可选）：按 score 降序遍历，每个 cluster 最多取 max_per_group 个。
    if config.cluster_limit_enabled:
        cluster_counts: dict[int, int] = {}
        constrained: list = []
        for candidate in filtered_candidates:
            cluster_label = symbol_data_map[candidate.symbol].cluster_label
            if cluster_label < 0:
                # 未分类标的跳过计数，但允许入选。
                constrained.append(candidate)
                continue
            count = cluster_counts.get(cluster_label, 0)
            if count < int(config.cluster_max_per_group):
                constrained.append(candidate)
                cluster_counts[cluster_label] = count + 1
        selected = constrained[: int(top_n)]
    else:
        selected = filtered_candidates[: int(top_n)]

    if not selected:
        return {}, len(candidates), len(filtered_candidates), active_candidate_filters

    weight = 1.0 / float(len(selected))
    return (
        {candidate.symbol: weight for candidate in selected},
        len(candidates),
        len(filtered_candidates),
        active_candidate_filters,
    )


def _solve_target_exposures(
    portfolio_value: float,
    current_exposures: Mapping[str, float],
    target_weights: Mapping[str, float],
    commission: float,
) -> tuple[dict[str, float], float, float, float]:
    """在保证手续费可支付的前提下，求解目标持仓金额。"""

    if portfolio_value <= 0:
        return {}, 0.0, 0.0, 0.0

    if not target_weights:
        trade_notional = float(sum(abs(float(value)) for value in current_exposures.values()))
        commission_paid = trade_notional * float(commission)
        cash_after = float(portfolio_value) - commission_paid
        return {}, commission_paid, trade_notional, cash_after

    total_target_weight = float(sum(target_weights.values()))
    normalized_weights = {
        symbol: float(weight) / total_target_weight
        for symbol, weight in target_weights.items()
        if float(weight) > 0.0
    }

    symbols = set(current_exposures).union(normalized_weights)
    low = 0.0
    high = 1.0

    def _cash_after(scale: float) -> float:
        desired_total = float(portfolio_value) * float(scale)
        trade_notional = 0.0
        for symbol in symbols:
            desired = desired_total * normalized_weights.get(symbol, 0.0)
            current = float(current_exposures.get(symbol, 0.0))
            trade_notional += abs(desired - current)
        return float(portfolio_value) - desired_total - float(commission) * trade_notional

    # 用二分搜索找到既能尽量满仓、又不会让现金变成负数的投资比例。
    for _ in range(60):
        mid = (low + high) / 2.0
        if _cash_after(mid) >= 0.0:
            low = mid
        else:
            high = mid

    desired_total = float(portfolio_value) * low
    desired_exposures = {
        symbol: desired_total * weight
        for symbol, weight in normalized_weights.items()
        if desired_total * weight > 1e-12
    }
    trade_notional = 0.0
    for symbol in symbols:
        trade_notional += abs(
            desired_exposures.get(symbol, 0.0) - float(current_exposures.get(symbol, 0.0))
        )
    commission_paid = float(commission) * trade_notional
    cash_after = float(portfolio_value) - sum(desired_exposures.values()) - commission_paid
    if abs(cash_after) < 1e-10:
        cash_after = 0.0
    return desired_exposures, commission_paid, trade_notional, cash_after


def _execute_rebalance(
    *,
    positions: Mapping[str, float],
    cash: float,
    execution_date: pd.Timestamp,
    target_weights: Mapping[str, float],
    symbol_data_map: Mapping[str, SymbolBaselineData],
    commission: float,
    hold_overlap: bool = False,
    ) -> tuple[dict[str, float], float, dict[str, Any]]:
    """在执行日开盘价上将目标权重转换为实际持仓，并返回执行诊断信息。"""

    open_prices: dict[str, float] = {}
    involved_symbols = set(positions).union(target_weights)

    for symbol in involved_symbols:
        open_price = symbol_data_map[symbol].frame["open"].asof(execution_date)
        if pd.isna(open_price) or float(open_price) <= 0:
            raise ValueError(f"Missing execution open price for {symbol} on {execution_date.date()}")
        open_prices[symbol] = float(open_price)

    current_exposures = {
        symbol: float(shares) * open_prices[symbol]
        for symbol, shares in positions.items()
        if abs(float(shares)) > 1e-12
    }

    if hold_overlap:
        return _execute_rebalance_hold_overlap(
            positions=positions,
            cash=cash,
            execution_date=execution_date,
            target_weights=target_weights,
            open_prices=open_prices,
            current_exposures=current_exposures,
            commission=commission,
        )

    portfolio_value_before_trade = float(cash) + sum(current_exposures.values())
    desired_exposures, commission_paid, trade_notional, cash_after = _solve_target_exposures(
        portfolio_value=portfolio_value_before_trade,
        current_exposures=current_exposures,
        target_weights=target_weights,
        commission=commission,
    )

    new_positions: dict[str, float] = {}
    for symbol, desired_exposure in desired_exposures.items():
        shares = desired_exposure / open_prices[symbol]
        if abs(shares) > 1e-12:
            new_positions[symbol] = float(shares)

    portfolio_value_after_trade = float(cash_after) + sum(desired_exposures.values())
    executed_weights_open = {
        symbol: desired_exposure / portfolio_value_after_trade
        for symbol, desired_exposure in desired_exposures.items()
        if portfolio_value_after_trade > 0
    }

    execution_info = {
        "execution_date": execution_date,
        "portfolio_value_before_trade": float(portfolio_value_before_trade),
        "portfolio_value_after_trade": float(portfolio_value_after_trade),
        "commission_paid": float(commission_paid),
        "trade_notional": float(trade_notional),
        "cash_after_trade": float(cash_after),
        "executed_weights_open": executed_weights_open,
    }
    return new_positions, float(cash_after), execution_info


def _execute_rebalance_hold_overlap(
    *,
    positions: Mapping[str, float],
    cash: float,
    execution_date: pd.Timestamp,
    target_weights: Mapping[str, float],
    open_prices: dict[str, float],
    current_exposures: dict[str, float],
    commission: float,
    ) -> tuple[dict[str, float], float, dict[str, Any]]:
    """hold_overlap 模式下调仓：重叠标的保留不动，仅对新增标的等权分配可支配资金。"""

    overlap_symbols = set(positions.keys()) & set(target_weights.keys())
    sell_symbols = set(positions.keys()) - set(target_weights.keys())
    new_symbols = set(target_weights.keys()) - set(positions.keys())

    overlap_value = sum(
        float(positions.get(s, 0.0)) * open_prices[s]
        for s in overlap_symbols
    )
    sell_value = sum(
        float(positions.get(s, 0.0)) * open_prices[s]
        for s in sell_symbols
    )

    portfolio_value_before_trade = cash + overlap_value + sell_value
    sell_commission = sell_value * commission
    cash_after_sell = cash + sell_value - sell_commission

    new_positions: dict[str, float] = {}
    for s in overlap_symbols:
        shares = float(positions[s])
        if abs(shares) > 1e-12:
            new_positions[s] = shares

    if not new_symbols or cash_after_sell <= 0:
        new_total = 0.0
        buy_commission = 0.0
    else:
        # new_total + new_total * commission = cash_after_sell
        new_total = cash_after_sell / (1.0 + commission)
        buy_commission = new_total * commission
        per_symbol = new_total / float(len(new_symbols))
        for s in new_symbols:
            shares = per_symbol / open_prices[s]
            if abs(shares) > 1e-12:
                new_positions[s] = float(shares)

    commission_paid = sell_commission + buy_commission
    trade_notional = sell_value + new_total
    cash_after = cash_after_sell - new_total - buy_commission
    if abs(cash_after) < 1e-10:
        cash_after = 0.0

    portfolio_value_after_trade = (
        cash_after
        + sum(new_positions.get(s, 0.0) * open_prices[s] for s in new_positions)
    )
    if portfolio_value_after_trade > 0:
        executed_weights_open = {
            s: new_positions[s] * open_prices[s] / portfolio_value_after_trade
            for s in new_positions
            if abs(new_positions.get(s, 0.0)) > 1e-12
        }
    else:
        executed_weights_open = {}

    execution_info = {
        "execution_date": execution_date,
        "portfolio_value_before_trade": float(portfolio_value_before_trade),
        "portfolio_value_after_trade": float(portfolio_value_after_trade),
        "commission_paid": float(commission_paid),
        "trade_notional": float(trade_notional),
        "cash_after_trade": float(cash_after),
        "executed_weights_open": executed_weights_open,
    }
    return new_positions, cash_after, execution_info


def _compute_weight_turnover(
    previous_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
) -> float:
    """计算相邻两次目标权重之间的单边换手率。"""

    symbols = set(previous_weights).union(current_weights)
    return 0.5 * float(
        sum(
            abs(float(current_weights.get(symbol, 0.0)) - float(previous_weights.get(symbol, 0.0)))
            for symbol in symbols
        )
    )


def _finalize_rebalance_log(
    rebalance_entries: list[dict[str, Any]],
) -> pd.DataFrame:
    """为调仓日志补充换手率和持有期收益。"""

    previous_target_weights: Optional[dict[str, float]] = None
    for entry in rebalance_entries:
        current_target_weights = dict(entry.get("target_weights", {}))
        if previous_target_weights is None:
            entry["turnover"] = None
        else:
            entry["turnover"] = _compute_weight_turnover(
                previous_weights=previous_target_weights,
                current_weights=current_target_weights,
            )
        previous_target_weights = current_target_weights

    for idx, entry in enumerate(rebalance_entries[:-1]):
        next_entry = rebalance_entries[idx + 1]
        after_trade_value = entry.get("portfolio_value_after_trade")
        next_before_trade = next_entry.get("portfolio_value_before_trade")
        if after_trade_value is None or next_before_trade is None:
            entry["period_return"] = None
            continue
        entry["period_return"] = float(next_before_trade) / float(after_trade_value) - 1.0

    if rebalance_entries:
        rebalance_entries[-1]["period_return"] = None

    rebalance_df = pd.DataFrame(rebalance_entries)
    if rebalance_df.empty:
        return rebalance_df
    for column in ("signal_date", "execution_date"):
        if column in rebalance_df.columns:
            rebalance_df[column] = pd.to_datetime(rebalance_df[column])
    return rebalance_df


def _compute_drawdown_details(equity: pd.Series) -> dict[str, Any]:
    """返回最大回撤及其对应的峰值和谷值日期。"""

    if equity.empty:
        return {
            "max_drawdown": float("nan"),
            "peak_date": None,
            "trough_date": None,
        }

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    trough_date = pd.Timestamp(drawdown.idxmin())
    peak_date = pd.Timestamp(equity.loc[:trough_date].idxmax())
    max_drawdown = abs(float(drawdown.min()))
    return {
        "max_drawdown": max_drawdown,
        "peak_date": peak_date,
        "trough_date": trough_date,
    }


def _build_annual_returns(equity_curve_df: pd.DataFrame) -> pd.DataFrame:
    """将日度净值收益聚合为自然年收益。"""

    if equity_curve_df.empty:
        return pd.DataFrame(columns=["year", "annual_return_pct"])

    returns = equity_curve_df["equity"].pct_change().dropna()
    if returns.empty:
        return pd.DataFrame(columns=["year", "annual_return_pct"])

    return_years = pd.DatetimeIndex(returns.index).year
    annual = (1.0 + returns.astype(float)).groupby(return_years).prod() - 1.0
    annual_df = annual.rename("annual_return_pct").mul(100.0).reset_index()
    annual_df = annual_df.rename(columns={"date": "year", "index": "year"})
    annual_df["annual_return_pct"] = annual_df["annual_return_pct"].round(4)
    return annual_df


def _build_variant_summary(
    *,
    top_n: int,
    equity_curve_df: pd.DataFrame,
    rebalance_df: pd.DataFrame,
    config: WideMomentumBaselineConfig,
    prepared: PreparedWideMomentumUniverse,
) -> dict[str, Any]:
    """将单个 top-N 变体汇总为可落盘的绩效指标。"""

    equity = equity_curve_df["equity"] if not equity_curve_df.empty else pd.Series(dtype=float)
    returns = equity.pct_change().dropna()

    cumulative = cumulative_return(returns) if not returns.empty else float("nan")
    annualised = annualised_return(returns) if not returns.empty else float("nan")
    volatility = annualised_volatility(returns) if len(returns) >= 2 else float("nan")
    sharpe = sharpe_ratio(returns, risk_free_rate=float(config.risk_free_rate)) if len(returns) >= 2 else None
    drawdown_details = _compute_drawdown_details(equity)
    max_drawdown = drawdown_details["max_drawdown"]
    calmar = None
    if not math.isnan(annualised) and not math.isnan(max_drawdown) and max_drawdown > 0:
        calmar = annualised / max_drawdown

    period_returns = (
        rebalance_df["period_return"].dropna()
        if "period_return" in rebalance_df.columns
        else pd.Series(dtype=float)
    )
    turnover_series = (
        rebalance_df["turnover"].dropna()
        if "turnover" in rebalance_df.columns
        else pd.Series(dtype=float)
    )
    stable_start = prepared.stable_start_month

    return {
        "top_n": int(top_n),
        "experiment_name": _resolve_experiment_name(config),
        "date_range_start": str(prepared.start_date.date()),
        "date_range_end": str(prepared.end_date.date()),
        "recent_complete_date": str(prepared.recent_complete_date.date()),
        "eligible_symbol_count": int(len(prepared.symbol_data_map)),
        "pool_stable_start_month": str(stable_start.date()) if stable_start is not None else None,
        "risk_free_rate_pct": round(float(config.risk_free_rate) * 100.0, 4),
        "commission_pct": round(float(config.commission) * 100.0, 4),
        "rebalance_interval_days": int(config.rebalance_interval),
        "cumulative_return_pct": round(float(cumulative) * 100.0, 4) if not math.isnan(cumulative) else None,
        "annualised_return_pct": round(float(annualised) * 100.0, 4) if not math.isnan(annualised) else None,
        "annualised_volatility_pct": round(float(volatility) * 100.0, 4) if not math.isnan(volatility) else None,
        "sharpe": round(float(sharpe), 4) if sharpe is not None else None,
        "max_drawdown_pct": round(float(max_drawdown) * 100.0, 4) if not math.isnan(max_drawdown) else None,
        "max_drawdown_peak_date": (
            str(drawdown_details["peak_date"].date())
            if drawdown_details["peak_date"] is not None
            else None
        ),
        "max_drawdown_trough_date": (
            str(drawdown_details["trough_date"].date())
            if drawdown_details["trough_date"] is not None
            else None
        ),
        "calmar": round(float(calmar), 4) if calmar is not None else None,
        "rebalance_win_rate_pct": (
            round(float((period_returns > 0.0).mean()) * 100.0, 4)
            if not period_returns.empty
            else None
        ),
        "monthly_turnover_pct": (
            round(float(turnover_series.mean()) * 100.0, 4)
            if not turnover_series.empty
            else None
        ),
        "rebalance_count": int(len(rebalance_df)),
        "completed_period_count": int(len(period_returns)),
    }


def _maybe_execute_pending_rebalance(
    *,
    current_date: pd.Timestamp,
    pending_rebalance: Optional[dict[str, Any]],
    positions: dict[str, float],
    cash: float,
    symbol_data_map: Mapping[str, SymbolBaselineData],
    commission: float,
    rebalance_entries: list[dict[str, Any]],
    hold_overlap: bool = False,
    ) -> tuple[dict[str, float], float, Optional[dict[str, Any]]]:
    """当计划执行日到来时，执行上一信号日安排的调仓。"""

    if pending_rebalance is None or current_date != pending_rebalance["execution_date"]:
        return positions, cash, pending_rebalance

    positions, cash, execution_info = _execute_rebalance(
        positions=positions,
        cash=cash,
        execution_date=current_date,
        target_weights=pending_rebalance["target_weights"],
        symbol_data_map=symbol_data_map,
        commission=commission,
        hold_overlap=hold_overlap,
    )
    rebalance_entries[pending_rebalance["entry_index"]].update(execution_info)
    return positions, cash, None


def _append_equity_row(
    equity_rows: list[dict[str, Any]],
    current_date: pd.Timestamp,
    equity_value: float,
    cash: float,
) -> None:
    """记录一条日终净值观测，用于输出净值曲线。"""

    equity_rows.append(
        {
            "date": current_date,
            "equity": float(equity_value),
            "cash": float(cash),
        }
    )


def _is_signal_bar(bar_index: int, rebalance_interval: int, calendar_length: int) -> bool:
    """判断当前 bar 是否应产生一个在下一开盘执行的信号。"""

    return (bar_index % rebalance_interval == 0) and (bar_index + 1 < calendar_length)


def _create_rebalance_plan(
    *,
    prepared: PreparedWideMomentumUniverse,
    config: WideMomentumBaselineConfig,
    top_n: int,
    bar_index: int,
    signal_date: pd.Timestamp,
    positions: Mapping[str, float],
    cash: float,
    signal_equity: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """为单个信号日生成调仓日志项和延迟执行计划。"""

    execution_date = pd.Timestamp(prepared.calendar[bar_index + 1])
    target_weights, candidate_count, filtered_candidate_count, active_candidate_filters = (
        _select_target_weights(
            symbol_data_map=prepared.symbol_data_map,
            signal_date=signal_date,
            execution_date=execution_date,
            top_n=top_n,
            config=config,
        )
    )
    current_weights = _portfolio_weights(
        positions=positions,
        cash=cash,
        date=signal_date,
        symbol_data_map=prepared.symbol_data_map,
        price_column="close",
    )
    rebalance_entry = {
        "signal_date": signal_date,
        "execution_date": execution_date,
        "bar_index": int(bar_index),
        "candidate_count": int(candidate_count),
        "filtered_candidate_count": int(filtered_candidate_count),
        "active_candidate_filters": active_candidate_filters,
        "selected_symbols": sorted(target_weights.keys()),
        "current_weights": current_weights,
        "target_weights": target_weights,
        "signal_equity": float(signal_equity),
    }
    pending_rebalance = {
        "execution_date": execution_date,
        "target_weights": target_weights,
    }
    return rebalance_entry, pending_rebalance


def _build_equity_curve_frame(
    equity_rows: list[dict[str, Any]],
    initial_cash: float,
) -> pd.DataFrame:
    """将收集到的日终观测转换为标准净值曲线表。"""

    equity_curve_df = pd.DataFrame(equity_rows)
    if equity_curve_df.empty:
        return pd.DataFrame(columns=["equity", "cash", "daily_return", "cumulative_return_pct"])

    equity_curve_df = equity_curve_df.set_index("date")
    equity_curve_df.index = pd.to_datetime(equity_curve_df.index)
    equity_curve_df["daily_return"] = equity_curve_df["equity"].pct_change()
    equity_curve_df["cumulative_return_pct"] = (
        equity_curve_df["equity"] / float(initial_cash) - 1.0
    ) * 100.0
    return equity_curve_df


def _run_top_n_variant(
    *,
    prepared: PreparedWideMomentumUniverse,
    config: WideMomentumBaselineConfig,
    top_n: int,
) -> WideMomentumVariantResult:
    """运行单个 top-N 组合规模的逐日模拟。"""

    positions: dict[str, float] = {}
    cash = float(config.cash)
    rebalance_entries: list[dict[str, Any]] = []
    pending_rebalance: Optional[dict[str, Any]] = None
    equity_rows: list[dict[str, Any]] = []
    hold_overlap = bool(config.hold_overlap)

    for bar_index, current_date in enumerate(prepared.calendar):
        current_date = pd.Timestamp(current_date)

        # 调仓信号在信号 bar 生成，并在下一根 bar 的开盘执行。
        positions, cash, pending_rebalance = _maybe_execute_pending_rebalance(
            current_date=current_date,
            pending_rebalance=pending_rebalance,
            positions=positions,
            cash=cash,
            symbol_data_map=prepared.symbol_data_map,
            commission=float(config.commission),
            rebalance_entries=rebalance_entries,
            hold_overlap=hold_overlap,
        )

        # 先按收盘价做日终估值，再判断今天是否产生新信号。
        close_value = _portfolio_value(
            positions=positions,
            cash=cash,
            date=current_date,
            symbol_data_map=prepared.symbol_data_map,
            price_column="close",
        )
        _append_equity_row(
            equity_rows=equity_rows,
            current_date=current_date,
            equity_value=float(close_value),
            cash=float(cash),
        )

        if not _is_signal_bar(
            bar_index=bar_index,
            rebalance_interval=int(config.rebalance_interval),
            calendar_length=len(prepared.calendar),
        ):
            continue

        rebalance_entry, pending_rebalance = _create_rebalance_plan(
            prepared=prepared,
            config=config,
            top_n=top_n,
            bar_index=bar_index,
            signal_date=current_date,
            positions=positions,
            cash=cash,
            signal_equity=float(close_value),
        )
        rebalance_entries.append(rebalance_entry)
        pending_rebalance["entry_index"] = len(rebalance_entries) - 1

    equity_curve_df = _build_equity_curve_frame(equity_rows=equity_rows, initial_cash=float(config.cash))

    rebalance_df = _finalize_rebalance_log(rebalance_entries)
    annual_returns = _build_annual_returns(equity_curve_df)
    summary = _build_variant_summary(
        top_n=top_n,
        equity_curve_df=equity_curve_df,
        rebalance_df=rebalance_df,
        config=config,
        prepared=prepared,
    )
    return WideMomentumVariantResult(
        top_n=top_n,
        summary=summary,
        equity_curve=equity_curve_df,
        annual_returns=annual_returns,
        rebalance_log=rebalance_df,
    )


def run_wide_momentum_baseline_from_prepared(
    prepared: PreparedWideMomentumUniverse,
    config: WideMomentumBaselineConfig,
) -> WideMomentumBaselineResult:
    """基于已准备好的共享股票池运行所有请求的 top-N 变体。"""

    variant_results = {
        int(top_n): _run_top_n_variant(prepared=prepared, config=config, top_n=int(top_n))
        for top_n in config.top_n_values
    }
    return WideMomentumBaselineResult(
        config=config,
        prepared_universe=prepared,
        variant_results=variant_results,
    )


def run_wide_momentum_baseline(
    config: WideMomentumBaselineConfig,
    *,
    symbols: Optional[list[str]] = None,
    cluster_lookup: Optional[Mapping[str, int] | Callable[[str], int]] = None,
) -> WideMomentumBaselineResult:
    """完成数据加载、股票池准备，并执行全部基线变体。"""

    prepared = prepare_wide_momentum_universe(
        config=config,
        symbols=symbols,
        cluster_lookup=cluster_lookup,
    )
    return run_wide_momentum_baseline_from_prepared(prepared=prepared, config=config)


def _serialize_csv_cell(value: Any) -> Any:
    """将 dict/list 单元格编码为 JSON，避免嵌套结构在 CSV 中丢失。"""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _prepare_rebalance_log_for_csv(rebalance_df: pd.DataFrame) -> pd.DataFrame:
    """将调仓日志中的嵌套字段转换为适合写入 CSV 的标量值。"""

    if rebalance_df.empty:
        return rebalance_df.copy()

    csv_df = rebalance_df.copy()
    for column in (
        "selected_symbols",
        "current_weights",
        "target_weights",
        "executed_weights_open",
        "active_candidate_filters",
    ):
        if column in csv_df.columns:
            csv_df[column] = csv_df[column].apply(_serialize_csv_cell)
    return csv_df


def _plot_equity_curve(equity_curve_df: pd.DataFrame, output_path: Path, title: str) -> None:
    """将单个变体的累计收益曲线渲染为 PNG 图片。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = equity_curve_df.reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["date"], plot_df["cumulative_return_pct"], linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_run_metadata(result: WideMomentumBaselineResult) -> dict[str, Any]:
    """构造描述配置、股票池和排除结果的元数据负载。"""

    ranking_factor = _resolve_ranking_factor(result.config)

    return {
        "config": {
            "top_n_values": [int(value) for value in result.config.top_n_values],
            "experiment_name": _resolve_experiment_name(result.config),
            "min_listing_days": int(result.config.min_listing_days),
            "momentum_window": int(result.config.momentum_window),
            "momentum_skip_recent": int(result.config.momentum_skip_recent),
            "min_momentum_value": (
                float(result.config.min_momentum_value)
                if result.config.min_momentum_value is not None
                else None
            ),
            "builtin_filters": [
                {
                    "field": bf.field,
                    "operator": bf.operator,
                    "value": float(bf.value),
                }
                for bf in result.config.builtin_filters
            ],
            "candidate_filters": _serialize_candidate_filters(result.config),
            "rebalance_interval": int(result.config.rebalance_interval),
            "cash": float(result.config.cash),
            "commission": float(result.config.commission),
            "risk_free_rate": float(result.config.risk_free_rate),
            "stable_pool_size": int(result.config.stable_pool_size),
            "cluster_limit_enabled": bool(result.config.cluster_limit_enabled),
            "cluster_max_per_group": int(result.config.cluster_max_per_group),
            "exclude_clusters": [int(c) for c in result.config.exclude_clusters],
            "hold_overlap": bool(result.config.hold_overlap),
            "ranking_factor": _serialize_factor(ranking_factor),
            "factor_pipeline": [
                _serialize_factor(factor)
                for factor in _resolve_factor_pipeline(result.config)
            ],
            "start_date": (
                str(result.config.start_date) if result.config.start_date is not None else None
            ),
            "end_date": str(result.config.end_date) if result.config.end_date is not None else None,
        },
        "prepared_universe": {
            "source_symbol_count": int(result.prepared_universe.source_symbol_count),
            "eligible_symbol_count": int(len(result.prepared_universe.symbol_data_map)),
            "load_error_count": int(len(result.prepared_universe.load_errors)),
            "excluded_symbol_count": int(len(result.prepared_universe.excluded_symbols)),
            "start_date": str(result.prepared_universe.start_date.date()),
            "end_date": str(result.prepared_universe.end_date.date()),
            "recent_complete_date": str(result.prepared_universe.recent_complete_date.date()),
            "stable_start_month": (
                str(result.prepared_universe.stable_start_month.date())
                if result.prepared_universe.stable_start_month is not None
                else None
            ),
        },
        "load_errors": result.prepared_universe.load_errors,
        "excluded_symbols": result.prepared_universe.excluded_symbols,
    }


def _save_variant_result(
    output_path: Path,
    top_n: int,
    variant_result: WideMomentumVariantResult,
) -> None:
    """将单个 top-N 变体的全部产物写入对应输出目录。"""

    variant_dir = output_path / f"top_{top_n}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant_result.equity_curve.to_csv(
        variant_dir / "equity_curve.csv",
        index=True,
        encoding="utf-8-sig",
    )
    variant_result.annual_returns.to_csv(
        variant_dir / "annual_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _prepare_rebalance_log_for_csv(variant_result.rebalance_log).to_csv(
        variant_dir / "rebalance_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (variant_dir / "summary.json").write_text(
        json.dumps(variant_result.summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _plot_equity_curve(
        equity_curve_df=variant_result.equity_curve,
        output_path=variant_dir / "equity_curve.png",
        title=f"Wide Momentum Baseline Top {top_n}",
    )


def save_wide_momentum_baseline_result(
    result: WideMomentumBaselineResult,
    output_dir: str | Path,
) -> Path:
    """将运行级和变体级结果写入目标目录。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        variant_result.summary
        for _, variant_result in sorted(result.variant_results.items(), key=lambda item: item[0])
    ]
    pd.DataFrame(summary_rows).to_csv(
        output_path / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result.prepared_universe.monthly_pool_diagnostics.to_csv(
        output_path / "monthly_pool_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = _build_run_metadata(result)
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 共享输出写在根目录；更细的交易日志和净值曲线按 top-N 分目录存放。
    for top_n, variant_result in sorted(result.variant_results.items(), key=lambda item: item[0]):
        _save_variant_result(output_path=output_path, top_n=top_n, variant_result=variant_result)

    return output_path


__all__ = [
    "BaselineCandidate",
    "CandidateFilterSpec",
    "PreparedWideMomentumUniverse",
    "SymbolBaselineData",
    "ThresholdFilter",
    "WideMomentumBaselineConfig",
    "WideMomentumBaselineResult",
    "WideMomentumVariantResult",
    "prepare_wide_momentum_universe",
    "prepare_wide_momentum_universe_from_etf_data_map",
    "prepare_wide_momentum_universe_from_frames",
    "run_wide_momentum_baseline",
    "run_wide_momentum_baseline_from_prepared",
    "save_wide_momentum_baseline_result",
]