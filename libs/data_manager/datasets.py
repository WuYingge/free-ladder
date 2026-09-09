"""
时序扩展数据集注册表 (Time-Series Dataset Registry)

每个"与某资产同代码集、按日对齐"的扩展数据集注册一项, 统一 getter
(`get_stock_data_by_symbol` 等) 通过 `with_xxx` 开关按需加载合并。
本次注册: basic (daily_basic 市值/股本)。

合并语义: 以主行情日期为轴 left join, 缺失行留 NaN 不强填;
`df.attrs["datasets"]` 记录本次已加载的数据集 key。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True, slots=True)
class TimeSeriesDataset:
    key: str                          # 数据集标识 (如 "basic")
    flag: str                         # getter 参数名 (如 "with_basic")
    columns: tuple[str, ...]          # 加载后应包含的列
    loader: Callable[[str], pd.DataFrame]  # symbol -> date 索引 DataFrame


def _empty_columns(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=list(columns),
        index=pd.DatetimeIndex([], name="date"),
    )


def _lazy_basic_loader(symbol: str) -> pd.DataFrame:
    # 延迟导入避免模块级循环 (daily_basic_manager -> stock_data_manager -> datasets)
    from data_manager.daily_basic_manager import load_daily_basic

    return load_daily_basic(symbol)


DATASETS: dict[str, TimeSeriesDataset] = {
    "basic": TimeSeriesDataset(
        key="basic",
        flag="with_basic",
        columns=("circ_mv", "total_mv", "float_share"),
        loader=_lazy_basic_loader,
    ),
}


def resolve_enabled_datasets(**kwargs: bool) -> set[str]:
    """从 getter 关键字参数解析启用的数据集 key 集合。"""
    enabled: set[str] = set()
    for dataset in DATASETS.values():
        if kwargs.get(dataset.flag, False):
            enabled.add(dataset.key)
    return enabled


def merge_extra_datasets(
    base_df: pd.DataFrame,
    symbol: str,
    enabled: set[str],
) -> pd.DataFrame:
    """
    把启用的扩展数据集按 date 索引 left join 到主行情。

    :param base_df: date 索引的主行情 DataFrame
    :param symbol: 标的代码
    :param enabled: 启用的数据集 key 集合
    :return: 合并后的 DataFrame; 数据集缺文件时补 NaN 列, 保证下游列访问安全。
    """
    result = base_df.copy()
    for key in enabled:
        dataset = DATASETS.get(key)
        if dataset is None:
            raise ValueError(f"未知数据集: {key}")
        extra = dataset.loader(symbol)
        if extra is None or extra.empty:
            for col in dataset.columns:
                if col not in result.columns:
                    result[col] = float("nan")
            continue
        new_cols = [c for c in dataset.columns if c not in result.columns]
        if new_cols:
            result = result.join(extra[new_cols], how="left")
    return result
