"""申万行业分类 Provider（个股 → 申万 2021 版 6 位行业代码 + 中文名称）。

从 data/const/stock_sw_industry_clf.csv 加载（由 libs/fetcher/industry.py 抓取、
libs/data_manager/sw_industry_manager.py 刷新并用 sw_industry_standard_2021.csv 附名），提供：

- get_mapping(asof): 截至某日的 symbol → 行业代码 时点映射（无 asof 取各股最新归属）
- get_industry(symbol, asof): 单只股票归属查询
- get_group_series(symbols, asof, level): symbol → 行业标签 Series，
  直接作为 alpha101 operators.indneutralize 的 group 输入
- get_events(symbol): 单股分类变动事件帧（date=生效日索引），
  供 libs/data_manager/datasets.py 的 with_industry 按日时点合并
- effective_industry_mapping(df, asof): 模块级纯函数，便于单测
- reload(): 文件变化后按当前 DataPath 热重载

行业代码语义：6 位申万分类代码按前缀嵌套，前 2/4/6 位分别为申万一/二/三级档位
（如 480301 → 48 一级）。level1/2/3_name 由刷新流程按行生效时代匹配官方标准表：
2021-07-30 起用 2021 版表（sw_industry_standard_2021.csv）、2014-02-21 起用 2014 版表
（sw_industry_standard_2014.csv）、更早时代按代码跨版近似回退；
两表皆无的代码（如 2001 版特有）名称列为空串，代码仍可作分组键。

用法示例（alpha101 行业中性化）：

    from data_manager.providers.sw_industry_provider import SW_INDUSTRY
    group = SW_INDUSTRY.get_group_series(symbols=universe, asof="2024-12-31", level=1)
    neut = indneutralize(alpha, group)
"""

from __future__ import annotations

from typing import override
from typing_extensions import Self
import pandas as pd
from config import DataPath
from data_manager.providers.base_provider import BaseProvider

_LEVEL_PREFIX_LEN = {1: 2, 2: 4, 3: 6}  # 申万一/二/三级 = 6 位代码的前 2/4/6 位

# 标准列
_SYMBOL = "symbol"
_START_DATE = "start_date"
_INDUSTRY_CODE = "industry_code"

# 名称列 (由标准表 enrichment 追加; 旧版时代代码无对应名称时为空串)
_NAME_COLUMNS = ("level1_name", "level2_name", "level3_name")


def effective_industry_mapping(
    df: pd.DataFrame,
    asof: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """求截至 asof 的 symbol → 行业代码 时点映射（模块级纯函数）。

    :param df: 含 symbol/start_date/industry_code 列的原始分类历史
    :param asof: 截止日期；为 None 时取每只股票最新一条归属
    :return: DataFrame[symbol, industry_code, start_date]，按 symbol 升序，
             start_date 为生效行日期；asof 之前无记录的 symbol 不出现
    """
    parsed = pd.DataFrame(
        {
            _SYMBOL: df[_SYMBOL].astype(str).str.strip(),
            _START_DATE: pd.to_datetime(df[_START_DATE], errors="coerce"),
            _INDUSTRY_CODE: df[_INDUSTRY_CODE].astype(str).str.strip(),
        }
    )
    parsed = parsed.dropna(subset=[_START_DATE])
    if asof is not None:
        cutoff = pd.Timestamp(asof)
        parsed = parsed[parsed[_START_DATE] <= cutoff]
    parsed = parsed.sort_values([_SYMBOL, _START_DATE])
    effective = parsed.drop_duplicates(subset=[_SYMBOL], keep="last")
    return effective.reset_index(drop=True)[[_SYMBOL, _INDUSTRY_CODE, _START_DATE]]


class _SWIndustryProvider(BaseProvider):
    """申万行业分类 Provider（单例），见模块 docstring。"""

    @override
    def init(self) -> None:
        self._df: pd.DataFrame = pd.DataFrame()
        self._initialize()

    @override
    @classmethod
    def get_instance(cls) -> Self:
        return cls()

    def reload(self) -> None:
        """按当前 DataPath 重新加载 (测试注入临时文件 / 刷新落盘后热更新)。"""
        self._initialize()

    def _initialize(self) -> None:
        path = DataPath.STOCK_SW_INDUSTRY_CLF_CSV
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except Exception:  # noqa: BLE001 — 文件尚未生成/损坏时退化为空数据
            return
        if df.empty:
            return
        df.columns = df.columns.str.strip().str.lstrip("\ufeff")
        required = {_SYMBOL, _START_DATE, _INDUSTRY_CODE}
        if not required.issubset(set(df.columns)):
            return
        df[_START_DATE] = pd.to_datetime(df[_START_DATE], errors="coerce")
        df = df.dropna(subset=[_START_DATE])
        # 保留名称列 (存在时), 缺失时补空列保证下游列访问安全
        for col in _NAME_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        keep = [_SYMBOL, _START_DATE, _INDUSTRY_CODE] + list(_NAME_COLUMNS)
        self._df = df[keep].sort_values(
            [_SYMBOL, _START_DATE], ignore_index=True
        )

    def to_dataframe(self) -> pd.DataFrame:
        """返回原始分类历史 DataFrame (symbol/start_date/industry_code + 名称列)。"""
        return self._df.copy()

    def get_events(self, symbol: str) -> pd.DataFrame:
        """返回单只股票的行业归属变动事件帧, 供按日时点合并。

        :return: date 索引 (生效日, 升序) 的 DataFrame,
                 列 industry_code/level1_name/level2_name/level3_name;
                 无记录时返回同列空帧 (不抛错)。
        """
        sym = str(symbol).zfill(6)
        ev = self._df[self._df[_SYMBOL] == sym]
        if ev.empty:
            cols = [_INDUSTRY_CODE] + list(_NAME_COLUMNS)
            return pd.DataFrame(
                columns=cols, index=pd.DatetimeIndex([], name="date")
            )
        ev = ev.sort_values(_START_DATE).drop_duplicates(subset=[_START_DATE], keep="last")
        ev = ev.set_index(_START_DATE)[[_INDUSTRY_CODE] + list(_NAME_COLUMNS)]
        ev.index.name = "date"
        return ev

    def get_mapping(self, asof: str | pd.Timestamp | None = None) -> pd.DataFrame:
        """返回截至 asof 的 symbol → industry_code 映射表（见 effective_industry_mapping）。"""
        if self._df.empty:
            return pd.DataFrame(columns=[_SYMBOL, _INDUSTRY_CODE, _START_DATE])
        return effective_industry_mapping(self._df, asof=asof)

    def get_industry(self, symbol: str, asof: str | pd.Timestamp | None = None) -> str:
        """查询单只股票截至 asof 的行业代码，无记录返回空串。"""
        mapping = self.get_mapping(asof=asof)
        row = mapping[mapping[_SYMBOL] == symbol.zfill(6)]
        if row.empty:
            return ""
        return str(row[_INDUSTRY_CODE].values[0])

    def get_group_series(
        self,
        symbols: list[str] | None = None,
        asof: str | pd.Timestamp | None = None,
        level: int = 1,
    ) -> pd.Series:
        """返回 symbol → 行业标签 Series（供 indneutralize 使用）。

        :param symbols: 目标标的列表；None 表示映射表内全部 symbol
        :param asof: 时点日期；None 取各股最新归属
        :param level: 行业档位 1/2/3（代码前 2/4/6 位）
        :return: index=symbol、value=行业代码前缀标签 的 Series（升序）
        """
        prefix_len = _LEVEL_PREFIX_LEN.get(level)
        if prefix_len is None:
            raise ValueError(f"level 必须是 {sorted(_LEVEL_PREFIX_LEN)} 之一，收到 {level!r}")
        mapping = self.get_mapping(asof=asof)
        if symbols is not None:
            wanted = {s.zfill(6) for s in symbols}
            mapping = mapping[mapping[_SYMBOL].isin(wanted)]
        labels = {s: c[:prefix_len] for s, c in zip(mapping[_SYMBOL], mapping[_INDUSTRY_CODE])}
        return pd.Series(labels, dtype=str, name=f"sw_level{level}")


SW_INDUSTRY = _SWIndustryProvider.get_instance()
