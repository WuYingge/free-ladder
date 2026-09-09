"""
Alpha101 抽象数据源 (Universe Abstraction)

将「标的池 + 数据加载」抽象为统一接口，使 alpha101 既能跑 ETF 也能跑股票，
底层各自封装 `data_manager` / `EtfData` / `StockDailyData`。

设计：
  * Alpha101Universe.list_symbols()  -> 标的代码列表
  * Alpha101Universe.load(symbol)    -> date 索引的 DataFrame(open/high/low/close/volume/value)

数据访问遵守仓库约束：一律走 data_manager 的 `get_*_data_by_symbol`，
绝不直接 `pd.read_csv` 读 data/etf_data/*.csv。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

__all__ = ["Alpha101Universe", "EtfAlpha101Universe", "StockAlpha101Universe"]

# 保证 Alpha101 算子可用的 OHLCV 必需列。
_REQUIRED_COLS = ["open", "high", "low", "close", "volume", "value"]


def _to_datetime_indexed_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """把 data_manager 读入的 DataFrame（date 为字符串列）转成 DatetimeIndex。

    与 libs/factor_analysis/panel.py 中 _calc_factor_worker 的清洗逻辑保持一致。
    """
    df = raw_df.copy()
    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index(date_series)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Universe.load 缺少必需列 {missing}（symbol 数据可能不完整）")

    df = df[_REQUIRED_COLS].astype(float)
    # 成交额/成交量为非正数的视为无效交易日（无分析意义）
    df["value"] = df["value"].where(df["value"] > 0, other=float("nan"))
    df["volume"] = df["volume"].where(df["volume"] > 0, other=float("nan"))
    return df.sort_index()


class Alpha101Universe(ABC):
    """Alpha101 数据源抽象基类。"""

    kind: str = "abstract"

    def __init__(self, symbols: list[str] | None = None) -> None:
        # 显式传入的标的是首选；为 None 时按数据集默认池。
        self._symbols = list(symbols) if symbols is not None else None

    def list_symbols(self) -> list[str]:
        """返回标的代码列表（显式传入或默认池）。"""
        if self._symbols is not None:
            return list(self._symbols)
        return self._default_symbols()

    @abstractmethod
    def _default_symbols(self) -> list[str]:
        """返回该数据源的默认标的池。"""

    @abstractmethod
    def load(self, symbol: str) -> pd.DataFrame:
        """加载单标的，返回 date 索引的 OHLCV DataFrame。"""


class EtfAlpha101Universe(Alpha101Universe):
    """ETF 数据源：默认池来自 ETF_INDEX_MAP（约 484 只代表 ETF）。"""

    kind = "etf"

    def _default_symbols(self) -> list[str]:
        from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP

        return ETF_INDEX_MAP.get_all_symbols()

    def load(self, symbol: str) -> pd.DataFrame:
        from data_manager.etf_data_manager import get_etf_data_by_symbol

        return _to_datetime_indexed_frame(get_etf_data_by_symbol(symbol).data)


class StockAlpha101Universe(Alpha101Universe):
    """股票数据源：默认池来自 STOCK_LIST。"""

    kind = "stock"

    def _default_symbols(self) -> list[str]:
        from data_manager.providers.stock_list_provider import STOCK_LIST

        return STOCK_LIST.get_all_symbol()

    def load(self, symbol: str) -> pd.DataFrame:
        from data_manager.stock_data_manager import get_stock_data_by_symbol

        return _to_datetime_indexed_frame(get_stock_data_by_symbol(symbol).data)


__all__ += ["_to_datetime_indexed_frame"]
