"""
时序扩展数据集注册表 (Time-Series Dataset Registry)

每个"与某资产同代码集、按日对齐"的扩展数据集注册一项, 统一 getter
(`get_stock_data_by_symbol` 等) 通过 `with_xxx` 开关按需加载合并。
已注册:
  * basic     (with_basic)      daily_basic 市值/股本: 逐日 CSV, 按日期精确 left join
  * adj_factor (with_adj_factor) 复权因子/不复权收盘: 逐日 CSV, 按日期精确 left join
                                (行情价格是后复权, 成交额成交量是不复权口径,
                                 alpha101 的 vwap 需用它搬回同一尺度)
  * industry  (with_industry)   申万行业归属: 变动事件帧 (date=生效日),
                                按日向后取最近生效事件 (point-in-time, 防未来函数)
  * 财务七表  (with_financial 一键 / with_income / with_balance / with_cashflow /
              with_forecast / with_share_capital / with_dividend / with_holder_num)
               data/financial/<表>.csv: 事件帧 (date=ann_date 首次公告日),
               按日向后取最近**已公告**的报告期 → 公告日之前拿不到该数据 (PIT)。
               同一 ann_date 多行时 on_duplicate="first" (首版优先, 更正不覆盖)。

合并语义: 以主行情日期为轴, 缺失行留 NaN/数据集指定填充值不强填;
`df.attrs["datasets"]` 记录本次已加载的数据集 key。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping

import pandas as pd


@dataclass(frozen=True, slots=True)
class TimeSeriesDataset:
    key: str                          # 数据集标识 (如 "basic")
    flag: str                         # getter 参数名 (如 "with_basic")
    columns: tuple[str, ...]          # 加载后应包含的列
    loader: Callable[[str], pd.DataFrame]  # symbol -> date 索引 DataFrame
    # True: loader 返回"变动事件帧"(index=生效日), 合并时按日期向后取最近事件
    #       (point-in-time); False: 逐日帧, 按日期精确 left join
    point_in_time: bool = False
    # 合并后用于填充缺失列的标量 (默认 NaN, 保持与旧版一致)
    fill_value: object | None = None
    # 同一事件日多行时的保留策略 (仅 point_in_time 生效):
    # "last" = 保留最后一行 (行业分类等"最新生效"语义);
    # "first" = 保留首行 (财务表: 首次公告版本优先, 防更正版覆盖首版)
    on_duplicate: Literal["first", "last"] = "last"
    # 是否属于"财务数据"一族 (with_financial 一键开关)
    financial: bool = False


def _empty_columns(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=list(columns),
        index=pd.DatetimeIndex([], name="date"),
    )


def _lazy_basic_loader(symbol: str) -> pd.DataFrame:
    # 延迟导入避免模块级循环 (daily_basic_manager -> stock_data_manager -> datasets)
    from data_manager.daily_basic_manager import load_daily_basic

    return load_daily_basic(symbol)


def _lazy_adj_factor_loader(symbol: str) -> pd.DataFrame:
    # 延迟导入避免模块级循环 (adj_factor_manager -> providers -> ...)
    from data_manager.providers.adj_factor_provider import ADJ_FACTOR

    return ADJ_FACTOR.get_events(symbol)


def _lazy_industry_loader(symbol: str) -> pd.DataFrame:
    # 延迟导入避免模块级循环 (sw_industry_manager -> providers -> ...)
    from data_manager.providers.sw_industry_provider import SW_INDUSTRY

    return SW_INDUSTRY.get_events(symbol)


def _financial_loader(table: str) -> Callable[[str], pd.DataFrame]:
    """财务表 loader 工厂: 按 ann_date 时点合并, 默认只取首次公告版本。"""

    def _load(symbol: str) -> pd.DataFrame:
        # 延迟导入避免模块级循环 (financial_provider -> financial_manager -> ...)
        from data_manager.providers.financial_provider import FINANCIAL

        return FINANCIAL.get_events(symbol, table)

    return _load


def _financial_dataset(table: str, flag: str, columns: tuple[str, ...]) -> TimeSeriesDataset:
    return TimeSeriesDataset(
        key=table,
        flag=flag,
        columns=columns,
        loader=_financial_loader(table),
        point_in_time=True,
        fill_value=None,          # 未公告/无数据 → NaN, 不强填
        on_duplicate="first",     # 首版优先 (更正行不覆盖)
        financial=True,
    )


DATASETS: dict[str, TimeSeriesDataset] = {
    "basic": TimeSeriesDataset(
        key="basic",
        flag="with_basic",
        columns=("circ_mv", "total_mv", "float_share"),
        loader=_lazy_basic_loader,
    ),
    "adj_factor": TimeSeriesDataset(
        key="adj_factor",
        flag="with_adj_factor",
        columns=("close_raw", "adj_factor"),
        loader=_lazy_adj_factor_loader,
    ),
    "industry": TimeSeriesDataset(
        key="industry",
        flag="with_industry",
        columns=("industry_code", "level1_name", "level2_name", "level3_name"),
        loader=_lazy_industry_loader,
        point_in_time=True,
        fill_value="",  # 行业归属缺失/未上市 → 空串, 而非 NaN
    ),
    # ---- 财务七表 (data/financial/<表>.csv, 事件日 = ann_date) ----
    "income_q": _financial_dataset(
        "income_q",
        "with_income",
        (
            "net_profit_attr_p",
            "revenue",
            "eps_basic",
            "operating_profit",
            "net_profit",
            "net_profit_deducted",
            "minority_interest_profit",
            "rd_expense",
            "sell_admin_expense",
        ),
    ),
    "balance_q": _financial_dataset(
        "balance_q",
        "with_balance",
        (
            "total_assets",
            "total_equity_attr_p",
            "total_liabilities",
            "total_equity",
            "total_share",
            "monetary_funds",
            "accounts_receivable",
            "inventory",
            "goodwill",
            "minority_interest_equity",
        ),
    ),
    "cashflow_q": _financial_dataset("cashflow_q", "with_cashflow", ("ocf_net",)),
    "forecast": _financial_dataset(
        "forecast",
        "with_forecast",
        (
            "forecast_indicator_cn",
            "announce_type",
            "announce_type_en",
            "forecast_net_profit_low",
            "forecast_net_profit_high",
            "forecast_net_profit_mid",
            "yoy_low",
            "yoy_high",
        ),
    ),
    "share_capital": _financial_dataset(
        "share_capital",
        "with_share_capital",
        ("total_share", "circ_share", "reason"),
    ),
    "dividend": _financial_dataset(
        "dividend",
        "with_dividend",
        ("cash_div_per_share", "bonus_ratio", "transfer_ratio"),
    ),
    "holder_num": _financial_dataset("holder_num", "with_holder_num", ("holder_num",)),
}

#: with_financial 一键开关覆盖的数据集 key
FINANCIAL_DATASET_KEYS: tuple[str, ...] = tuple(
    key for key, dataset in DATASETS.items() if dataset.financial
)


def resolve_enabled_datasets(**kwargs: bool) -> set[str]:
    """从 getter 关键字参数解析启用的数据集 key 集合。

    ``with_financial=True`` 展开为全部财务表; 单个 ``with_xxx`` 可细化覆盖。
    """
    enabled: set[str] = set()
    blanket = bool(kwargs.get("with_financial", False))
    for dataset in DATASETS.values():
        if dataset.financial and blanket:
            enabled.add(dataset.key)
            continue
        if kwargs.get(dataset.flag, False):
            enabled.add(dataset.key)
    return enabled


def merge_point_in_time(
    base_df: pd.DataFrame,
    events_df: pd.DataFrame,
    columns: tuple[str, ...],
    on_duplicate: Literal["first", "last"] = "last",
) -> pd.DataFrame:
    """把事件帧按日向后合并 (point-in-time) 到主行情, 返回新 DataFrame。

    :param base_df: date 索引、按日升序的主行情
    :param events_df: date 索引(生效日)、升序的事件帧 (列含 columns)
    :param columns: 需并入的列
    :param on_duplicate: 同一事件日多行时保留哪一行 ("last" 为旧行为)
    :return: 与 base_df 等长的合并结果 (含全部原列); 生效日早于首事件的
             日期/无任何事件 → 该行列缺失 (调用方按 fill_value 处理)
    """
    result = base_df.copy()
    new_cols = [c for c in columns if c not in result.columns]
    if not new_cols:
        return result
    result = result.sort_index()
    events = events_df.copy()
    if "start_date" in events.columns:
        # 允许"生效日在列中"的宽松输入; 常规 loader 返回的已是日期索引
        events["start_date"] = pd.to_datetime(events["start_date"], errors="coerce")
        events = events.set_index("start_date")
    events = events.sort_index()
    events = events[~events.index.duplicated(keep=on_duplicate)]
    if events.empty:
        return result
    merged = pd.merge_asof(
        result,
        events[new_cols],
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return merged


def merge_extra_datasets(
    base_df: pd.DataFrame,
    symbol: str,
    enabled: set[str],
    providers: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """
    把启用的扩展数据集按 date 索引合并到主行情。

    :param base_df: date 索引的主行情 DataFrame
    :param symbol: 标的代码
    :param enabled: 启用的数据集 key 集合
    :param providers: 数据集 key → provider 覆盖 (测试注入用, 需提供 ``get_events``);
                      默认走注册表内 loader (真实 provider)
    :return: 合并后的 DataFrame; 数据集缺文件时补缺失列 (NaN 或数据集 fill_value),
             保证下游列访问安全。
    """
    result = base_df.copy()
    for key in enabled:
        dataset = DATASETS.get(key)
        if dataset is None:
            raise ValueError(f"未知数据集: {key}")
        override = (providers or {}).get(key)
        if override is not None:
            extra = override.get_events(symbol)  # type: ignore[attr-defined]
        else:
            extra = dataset.loader(symbol)
        if dataset.point_in_time:
            if extra is None or extra.empty:
                for col in dataset.columns:
                    if col not in result.columns:
                        result[col] = dataset.fill_value
                continue
            merged = merge_point_in_time(result, extra, dataset.columns, on_duplicate=dataset.on_duplicate)
            result = merged
            if dataset.fill_value is not None:
                for col in dataset.columns:
                    if col in result.columns:
                        result[col] = result[col].fillna(dataset.fill_value)
            continue
        if extra is None or extra.empty:
            for col in dataset.columns:
                if col not in result.columns:
                    result[col] = float("nan")
            continue
        new_cols = [c for c in dataset.columns if c not in result.columns]
        if new_cols:
            result = result.join(extra[new_cols], how="left")
    return result
