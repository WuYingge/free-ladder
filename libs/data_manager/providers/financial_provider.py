"""财务数据 Provider (Financial Data Provider, 单例 ``FINANCIAL``)

从 ``data/financial/<表>.csv`` 加载七张财务表, 提供按 ``ann_date`` (首次公告日)
时点对齐的事件帧与横截面取数, 供:

- ``libs/data_manager/datasets.py`` 的 ``with_income`` / ``with_balance`` /
  ``with_cashflow`` / ``with_forecast`` / ``with_share_capital`` / ``with_dividend`` /
  ``with_holder_num`` (以及一键 ``with_financial``) 按日时点合并;
- 横截面研究: ``get_dataframe(table, asof)`` / ``get_latest(symbol, table, asof)``。

口径约定 (与需求清单 §8.3 一致):
  * 金额=元, 股本=股, 比率=小数, 累计口径;
  * ``basis`` 过滤默认 ``first_reported``: 只保留"首次公告版本"的行。
    实测东财 F10 当前版本无法还原首次公告数值, 被追溯调整过的行标为
    ``revised`` (数值可能已是调整后版本) —— 需要完整数值链的研究应显式
    传 ``basis=None`` 并自行披露该风险 (详见 docs/financial_data_pipeline.md);
  * 事件帧行级门禁在 ``financial_manager.load_financial_events`` 内完成
    (ann_date 缺失 / ann_date < report_date / 越出行情区间的行一律排除并计数)。

用法示例::

    from data_manager.providers.financial_provider import FINANCIAL
    events = FINANCIAL.get_events("600519", "income_q")      # date=ann_date
    panel = FINANCIAL.get_dataframe("income_q", asof="2024-12-31")
    FINANCIAL.reload()                                        # 落盘后热更新
"""

from __future__ import annotations

from typing import override

import pandas as pd
from typing_extensions import Self

from config import DataPath
from data_manager.financial_schema import (
    BASIS_FIRST_REPORTED,
    FINANCIAL_TABLES,
    FinancialTableSpec,
    TableQuality,
)
from data_manager.providers.base_provider import BaseProvider


class _FinancialProvider(BaseProvider):
    """财务七表 Provider (单例), 见模块 docstring。"""

    @override
    def init(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        self._quality: dict[str, TableQuality] = {}
        self._initialize()

    @override
    @classmethod
    def get_instance(cls) -> Self:
        return cls()

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """按当前 ``DataPath.FINANCIAL_DIR`` 重新加载 (落盘后热更新 / 测试注入)。"""
        self._initialize()

    def _initialize(self) -> None:
        from data_manager.financial_manager import load_table

        frames: dict[str, pd.DataFrame] = {}
        quality: dict[str, TableQuality] = {}
        for table in FINANCIAL_TABLES:
            try:
                frame, report = load_table(table)
            except Exception as err:  # noqa: BLE001 — 缺文件/损坏时退化为空数据
                print(f"[financial] {table} 加载失败: {type(err).__name__}: {err}")
                frame, report = pd.DataFrame(columns=list(FINANCIAL_TABLES[table].columns)), TableQuality(
                    table=table,
                    filename=FINANCIAL_TABLES[table].filename,
                    schema_ok=False,
                    missing_columns=[],
                )
            frames[table] = frame
            quality[table] = report
        self._frames = frames
        self._quality = quality

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def data_dir(self) -> str:
        return DataPath.FINANCIAL_DIR

    def list_tables(self) -> list[str]:
        return list(FINANCIAL_TABLES)

    def spec(self, table: str) -> FinancialTableSpec:
        spec = FINANCIAL_TABLES.get(table)
        if spec is None:
            raise ValueError(f"未知财务表: {table!r}; 可用: {sorted(FINANCIAL_TABLES)}")
        return spec

    def quality(self, table: str) -> TableQuality:
        self.spec(table)
        return self._quality.get(table, TableQuality(table=table, filename=""))

    def is_empty(self, table: str) -> bool:
        frame = self._frames.get(table)
        return frame is None or frame.empty

    def available_tables(self) -> list[str]:
        return [table for table in FINANCIAL_TABLES if not self.is_empty(table)]

    def to_dataframe(self, table: str) -> pd.DataFrame:
        """返回整表副本 (含全部行与 basis; 不改内部状态)。"""
        self.spec(table)
        frame = self._frames.get(table)
        if frame is None:
            return pd.DataFrame(columns=list(FINANCIAL_TABLES[table].columns))
        return frame.copy()

    def get_events(
        self,
        symbol: str,
        table: str,
        basis: str | None = None,
    ) -> pd.DataFrame:
        """单表单股事件帧 (index = 事件日, 升序, 已过行级门禁)。

        :param basis: **默认 None = 全部行**; 传 ``first_reported`` 进入严格模式
            (只保留首次公告版本 —— 实测会滤掉 2010-2024 年 87%~97% 的行, 慎用)
        :return: 无数据/无该股时返回同列空帧 (不抛错), 保证下游列访问安全
        """
        spec = self.spec(table)
        from data_manager.financial_manager import load_financial_events

        frames, _ = load_financial_events(
            symbol,
            basis=basis,
            tables=[table],
            apply_quote_bounds=True,
        )
        frame = frames.get(table)
        if frame is None:
            return pd.DataFrame(columns=list(spec.columns))
        return frame

    def get_symbol_dataframe(
        self,
        symbol: str,
        table: str,
        basis: str | None = None,
    ) -> pd.DataFrame:
        """单表单股原始行 (报告期序, 不过行情区间门禁, 供审计/口径核查)。

        ``basis`` 默认 None (全部行, 含 ``revised``)。"""
        spec = self.spec(table)
        frame = self._frames.get(table)
        code = str(symbol).zfill(6)
        if frame is None or frame.empty:
            return pd.DataFrame(columns=list(spec.columns))
        subset = frame[frame["symbol"].astype(str).str.zfill(6) == code].copy()
        if basis is not None and "basis" in subset.columns:
            subset = subset[subset["basis"].astype(str).str.startswith(basis)]
        if "report_date" in subset.columns:
            subset = subset.sort_values(["report_date", "ann_date"], kind="stable")
        return subset.reset_index(drop=True)

    def get_dataframe(
        self,
        table: str,
        asof: str | pd.Timestamp | None = None,
        basis: str | None = None,
    ) -> pd.DataFrame:
        """整表横截面视图; ``asof`` 给定则只保留 ann_date <= asof 的行 (防前视)。

        ``basis`` 默认 None (全部行); ``first_reported`` 为严格模式 (会滤掉大部分历史行)。"""
        frame = self.to_dataframe(table)
        if frame.empty:
            return frame
        if basis is not None and "basis" in frame.columns:
            frame = frame[frame["basis"].astype(str).str.startswith(basis)]
        if asof is not None and "ann_date" in frame.columns:
            cutoff = pd.Timestamp(asof)
            ann = pd.to_datetime(frame["ann_date"], errors="coerce")
            frame = frame[ann <= cutoff]
        return frame.reset_index(drop=True)

    def get_latest(
        self,
        symbol: str,
        table: str,
        asof: str | pd.Timestamp | None = None,
        basis: str | None = None,
    ) -> dict[str, object]:
        """该股截至 asof 的最新一条财务记录 (无数据返回空 dict)。"""
        frame = self.get_symbol_dataframe(symbol, table, basis=basis)
        if frame.empty:
            return {}
        if asof is not None and "ann_date" in frame.columns:
            ann = pd.to_datetime(frame["ann_date"], errors="coerce")
            frame = frame[ann <= pd.Timestamp(asof)]
        if frame.empty:
            return {}
        row = frame.sort_values(["report_date", "ann_date"], kind="stable").iloc[-1]
        return {key: value for key, value in row.items()}

    def symbols(self, table: str | None = None) -> list[str]:
        """落盘数据出现过的全部股票代码 (含退市股)。"""
        tables = [table] if table else list(FINANCIAL_TABLES)
        codes: set[str] = set()
        for name in tables:
            frame = self._frames.get(name)
            if frame is None or frame.empty or "symbol" not in frame.columns:
                continue
            codes |= set(frame["symbol"].astype(str).str.zfill(6))
        return sorted(codes)


FINANCIAL = _FinancialProvider.get_instance()
