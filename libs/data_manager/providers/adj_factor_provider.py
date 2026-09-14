"""复权因子 Provider（不复权收盘 + 逐日复权因子）。

从 ``data/adj_factor/<code>.csv``（由 ``libs/data_manager/adj_factor_manager.py``
抓取落盘）按需读取，提供：

- ``get_dataframe(symbol)``   : date 索引的 close_raw/adj_factor 两列
- ``get_events(symbol)``      : 同 get_dataframe，供 ``data_manager/datasets.py``
                                的 ``with_adj_factor`` 按日精确合并
- ``get_factor_series(symbol)``: 单列 adj_factor（日期索引）
- ``reload(symbol=None)``     : 热重载（文件变化/落盘后）

为什么需要它：本地行情价格列是后复权，而成交额/成交量是不复权口径，二者相差
一个逐股、逐日漂移的复权因子；alpha101 的 vwap 必须乘上该因子才能与 OHLC 同尺度
（见 ``adj_factor_manager.compute_vwap``）。

文件数量与标的同量级（数千个），故**不预载全量**，改为逐标的按需读 + 进程内缓存。
"""

from __future__ import annotations

from typing import override

import pandas as pd
from typing_extensions import Self

from data_manager.providers.base_provider import BaseProvider


class _AdjFactorProvider(BaseProvider):
    """复权因子 Provider（单例，逐标的懒加载 + 缓存）。"""

    @override
    def init(self) -> None:
        self._cache: dict[str, pd.DataFrame] = {}

    @override
    @classmethod
    def get_instance(cls) -> Self:
        return cls()

    def reload(self, symbol: str | None = None) -> None:
        """清空缓存：symbol=None 清全部，否则只清该标的（测试注入/落盘后热更新）。"""
        if symbol is None:
            self._cache.clear()
        else:
            self._cache.pop(str(symbol).zfill(6), None)

    def get_dataframe(self, symbol: str) -> pd.DataFrame:
        """返回 date 索引的 close_raw/adj_factor；无数据返回同列空帧（不抛错）。"""
        code = str(symbol).zfill(6)
        cached = self._cache.get(code)
        if cached is None:
            from data_manager.adj_factor_manager import load_adj_factor

            cached = load_adj_factor(code)
            self._cache[code] = cached
        return cached.copy()

    def get_events(self, symbol: str) -> pd.DataFrame:
        """datasets.py 注册表统一入口（本数据集为逐日帧，非事件帧）。"""
        return self.get_dataframe(symbol)

    def get_factor_series(self, symbol: str) -> pd.Series:
        """返回 adj_factor 单列（date 索引）；无数据返回空 Series。"""
        df = self.get_dataframe(symbol)
        if df.empty or "adj_factor" not in df.columns:
            return pd.Series(dtype="float64", name="adj_factor")
        return df["adj_factor"].rename("adj_factor")

    def get_close_raw_series(self, symbol: str) -> pd.Series:
        """返回不复权收盘单列（date 索引）；无数据返回空 Series。"""
        df = self.get_dataframe(symbol)
        if df.empty or "close_raw" not in df.columns:
            return pd.Series(dtype="float64", name="close_raw")
        return df["close_raw"].rename("close_raw")


ADJ_FACTOR = _AdjFactorProvider.get_instance()
